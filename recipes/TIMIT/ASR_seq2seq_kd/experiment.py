#!/usr/bin/env python3
import os
import sys
import torch
import speechbrain as sb

import speechbrain.data_io.wer as wer_io
import speechbrain.utils.edit_distance as edit_distance
from speechbrain.data_io.data_io import convert_index_to_lab
from speechbrain.data_io.data_io import prepend_bos_token

from speechbrain.decoders.seq2seq import S2SRNNGreedySearcher
from speechbrain.decoders.seq2seq import S2SRNNBeamSearcher
from speechbrain.decoders.decoders import undo_padding
from speechbrain.utils.checkpoints import ckpt_recency
from speechbrain.utils.train_logger import summarize_error_rate

# This hack needed to import data preparation script from ..
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from timit_prepare import prepare_timit  # noqa E402

# Load hyperparameters file with command-line overrides
params_file, overrides = sb.core.parse_arguments(sys.argv[1:])
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, overrides)

# Create experiment directory
sb.core.create_experiment_directory(
    experiment_directory=params.output_folder,
    params_to_save=params_file,
    overrides=overrides,
)

modules = torch.nn.ModuleList(
    [params.enc, params.emb, params.dec, params.ctc_lin, params.seq_lin]
)

tea0_modules = torch.nn.ModuleList(
    [
        params.tea0_enc,
        params.tea0_emb,
        params.tea0_dec,
        params.tea0_ctc_lin,
        params.tea0_seq_lin,
    ]
)

tea1_modules = torch.nn.ModuleList(
    [
        params.tea1_enc,
        params.tea1_emb,
        params.tea1_dec,
        params.tea1_ctc_lin,
        params.tea1_seq_lin,
    ]
)

greedy_searcher = S2SRNNGreedySearcher(
    modules=[params.emb, params.dec, params.seq_lin, params.log_softmax],
    bos_index=params.bos_index,
    eos_index=params.eos_index,
    min_decode_ratio=0,
    max_decode_ratio=1,
)
beam_searcher = S2SRNNBeamSearcher(
    modules=[params.emb, params.dec, params.seq_lin, params.log_softmax],
    bos_index=params.bos_index,
    eos_index=params.eos_index,
    min_decode_ratio=0,
    max_decode_ratio=1,
    beam_size=params.beam_size,
    length_penalty=params.length_penalty,
    eos_threshold=params.eos_threshold,
)

checkpointer = sb.utils.checkpoints.Checkpointer(
    checkpoints_dir=params.save_folder,
    recoverables={
        "model": modules,
        "optimizer": params.optimizer,
        "scheduler": params.lr_annealing,
        "normalizer": params.normalize,
        "counter": params.epoch_counter,
    },
)


# Define training procedure
class ASR_kd(sb.core.Brain):
    def __init__(
        self,
        modules=None,
        tea0_modules=None,
        tea1_modules=None,
        optimizer=None,
        first_inputs=None,
    ):

        # Initialize teacher parameters
        self.tea0_modules = torch.nn.ModuleList(tea0_modules)
        self.tea1_modules = torch.nn.ModuleList(tea1_modules)
        if first_inputs is not None:
            self.compute_forward_tea0(*first_inputs, init_params=True)
            self.compute_forward_tea1(*first_inputs, init_params=True)

        super(ASR_kd, self).__init__(
            modules=modules, optimizer=optimizer, first_inputs=first_inputs
        )

    def compute_forward_tea0(self, x, y, stage="train", init_params=False):
        ids, wavs, wav_lens = x
        ids, phns, phn_lens = y

        wavs, wav_lens = wavs.to(params.device), wav_lens.to(params.device)
        phns, phn_lens = phns.to(params.device), phn_lens.to(params.device)

        if hasattr(params, "augmentation"):
            wavs = params.augmentation(wavs, wav_lens, init_params)
        feats = params.compute_features(wavs, init_params)
        feats = params.normalize(feats, wav_lens)

        self.tea0_modules.eval()
        with torch.no_grad():
            x_tea0 = params.tea0_enc(feats, init_params=init_params)
            ctc_logits_tea0 = params.tea0_ctc_lin(x_tea0, init_params)
            p_ctc_tea0 = params.log_softmax(ctc_logits_tea0)

            y_in_tea0 = prepend_bos_token(phns, bos_index=params.bos_index)
            e_in_tea0 = params.tea0_emb(y_in_tea0, init_params=init_params)
            h_tea0, _ = params.tea0_dec(
                e_in_tea0, x_tea0, wav_lens, init_params
            )
            seq_logits_tea0 = params.tea0_seq_lin(h_tea0, init_params)
            m = torch.nn.Softmax(dim=-1)
            p_seq_tea0 = m(seq_logits_tea0)

            wer_ctc_tea0 = params.compute_wer_list_ctc(
                p_ctc_tea0, phns, wav_lens, phn_lens
            )
            wer_tea0 = params.compute_wer_list(p_seq_tea0, phns, phn_lens)
            wer_ctc_tea0, wer_tea0 = (
                wer_ctc_tea0.to(params.device),
                wer_tea0.to(params.device),
            )

        return p_ctc_tea0, p_seq_tea0, wer_ctc_tea0, wer_tea0

    def compute_forward_tea1(self, x, y, stage="train", init_params=False):
        ids, wavs, wav_lens = x
        ids, phns, phn_lens = y

        wavs, wav_lens = wavs.to(params.device), wav_lens.to(params.device)
        phns, phn_lens = phns.to(params.device), phn_lens.to(params.device)

        if hasattr(params, "augmentation"):
            wavs = params.augmentation(wavs, wav_lens, init_params)
        feats = params.compute_features(wavs, init_params)
        feats = params.normalize(feats, wav_lens)

        self.tea1_modules.eval()
        with torch.no_grad():
            x_tea1 = params.tea1_enc(feats, init_params=init_params)
            ctc_logits_tea1 = params.tea1_ctc_lin(x_tea1, init_params)
            p_ctc_tea1 = params.log_softmax(ctc_logits_tea1)

            y_in_tea1 = prepend_bos_token(phns, bos_index=params.bos_index)
            e_in_tea1 = params.tea1_emb(y_in_tea1, init_params=init_params)
            h_tea1, _ = params.tea1_dec(
                e_in_tea1, x_tea1, wav_lens, init_params
            )
            seq_logits_tea1 = params.tea1_seq_lin(h_tea1, init_params)
            m = torch.nn.Softmax(dim=-1)
            p_seq_tea1 = m(seq_logits_tea1)

            wer_ctc_tea1 = params.compute_wer_list_ctc(
                p_ctc_tea1, phns, wav_lens, phn_lens
            )
            wer_tea1 = params.compute_wer_list(p_seq_tea1, phns, phn_lens)
            wer_ctc_tea1, wer_tea1 = (
                wer_ctc_tea1.to(params.device),
                wer_tea1.to(params.device),
            )

        return p_ctc_tea1, p_seq_tea1, wer_ctc_tea1, wer_tea1

    def compute_forward(self, x, y, stage="train", init_params=False):
        ids, wavs, wav_lens = x
        ids, phns, phn_lens = y

        wavs, wav_lens = wavs.to(params.device), wav_lens.to(params.device)
        phns, phn_lens = phns.to(params.device), phn_lens.to(params.device)

        if hasattr(params, "augmentation"):
            wavs = params.augmentation(wavs, wav_lens, init_params)
        feats = params.compute_features(wavs, init_params)
        feats = params.normalize(feats, wav_lens)

        x = params.enc(feats, init_params=init_params)

        logits = params.ctc_lin(x, init_params)
        p_ctc = params.log_softmax(logits)

        y_in = prepend_bos_token(phns, bos_index=params.bos_index)
        e_in = params.emb(y_in, init_params=init_params)
        h, _ = params.dec(e_in, x, wav_lens, init_params)
        logits = params.seq_lin(h, init_params)
        p_seq = params.log_softmax(logits)

        if stage == "valid":
            hyps, scores = greedy_searcher(x, wav_lens)
            return p_ctc, p_seq, wav_lens, hyps

        elif stage == "test":
            hyps, scores = beam_searcher(x, wav_lens)
            return p_ctc, p_seq, wav_lens, hyps

        return p_ctc, p_seq, wav_lens

    def compute_objectives(
        self, predictions, tea0_output, tea1_output, targets, stage="train"
    ):
        if stage == "train":
            p_ctc, p_seq, wav_lens = predictions
        else:
            p_ctc, p_seq, wav_lens, hyps = predictions

        p_ctc_tea0, p_seq_tea0, wer_ctc_tea0, wer_tea0 = tea0_output
        p_ctc_tea1, p_seq_tea1, wer_ctc_tea1, wer_tea1 = tea1_output

        ids, phns, phn_lens = targets
        phns, phn_lens = phns.to(params.device), phn_lens.to(params.device)
        # add one for eos
        abs_length = torch.round(phn_lens * phns.shape[1])

        rel_length = (abs_length + 1) / phns.shape[1]

        # wer
        tea_wer = torch.cat([wer_tea0, wer_tea1], 0)
        tea_wer_ctc = torch.cat([wer_ctc_tea0, wer_ctc_tea1], 0)

        # tea_ctc for kd
        tea_wer_ctc_mean = tea_wer_ctc.mean(1)
        tea_acc_main = 100 - tea_wer_ctc_mean
        m0 = torch.nn.Softmax(dim=0)
        tea_acc_softmax = m0(tea_acc_main)

        # tea_ce for kd
        tea_wer_mean = tea_wer.mean(1)
        tea_acc_ce_main = 100 - tea_wer_mean
        tea_acc_ce_softmax = m0(tea_acc_ce_main)

        # loss0
        loss_ctc0 = params.ctc_cost_kd(p_ctc, p_ctc_tea0, wav_lens)
        loss_ctc0 = torch.unsqueeze(loss_ctc0, 0)
        loss_seq0 = params.seq_cost_kd(p_seq, p_seq_tea0, rel_length)
        loss_seq0 = torch.unsqueeze(loss_seq0, 0)
        # loss1
        loss_ctc1 = params.ctc_cost_kd(p_ctc, p_ctc_tea1, wav_lens)
        loss_ctc1 = torch.unsqueeze(loss_ctc1, 0)
        loss_seq1 = params.seq_cost_kd(p_seq, p_seq_tea1, rel_length)
        loss_seq1 = torch.unsqueeze(loss_seq1, 0)

        ctc_loss_list = torch.cat([loss_ctc0, loss_ctc1], 0)
        ce_loss_list = torch.cat([loss_seq0, loss_seq1], 0)

        # kd loss
        ctc_loss_kd = (tea_acc_softmax * ctc_loss_list).sum(0)
        seq2seq_loss_kd = (tea_acc_ce_softmax * ce_loss_list).sum(0)

        loss_ctc = ctc_loss_kd
        loss_seq = seq2seq_loss_kd

        loss = params.ctc_weight * loss_ctc + (1 - params.ctc_weight) * loss_seq

        stats = {}
        if stage != "train":
            ind2lab = params.train_loader.label_dict["phn"]["index2lab"]
            sequence = convert_index_to_lab(hyps, ind2lab)
            phns = undo_padding(phns, phn_lens)
            phns = convert_index_to_lab(phns, ind2lab)
            per_stats = edit_distance.wer_details_for_batch(
                ids, phns, sequence, compute_alignments=True
            )
            stats["PER"] = per_stats
        return loss, stats

    def fit_batch(self, batch):
        inputs, targets = batch
        predictions = self.compute_forward(inputs, targets)
        tea0_output = self.compute_forward_tea0(inputs, targets)
        tea1_output = self.compute_forward_tea1(inputs, targets)
        loss, stats = self.compute_objectives(
            predictions, tea0_output, tea1_output, targets
        )
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        stats["loss"] = loss.detach()
        return stats

    def evaluate_batch(self, batch, stage="valid"):
        inputs, targets = batch
        predictions = self.compute_forward(inputs, targets, stage=stage)
        tea0_output = self.compute_forward_tea0(inputs, targets)
        tea1_output = self.compute_forward_tea1(inputs, targets)
        loss, stats = self.compute_objectives(
            predictions, tea0_output, tea1_output, targets, stage=stage
        )
        stats["loss"] = loss.detach()
        return stats

    def on_epoch_end(self, epoch, train_stats, valid_stats=None):
        per = summarize_error_rate(valid_stats["PER"])
        old_lr, new_lr = params.lr_annealing([params.optimizer], epoch, per)
        epoch_stats = {"epoch": epoch, "lr": old_lr}
        params.train_logger.log_stats(epoch_stats, train_stats, valid_stats)

        checkpointer.save_and_keep_only(
            meta={"PER": per},
            importance_keys=[ckpt_recency, lambda c: -c.meta["PER"]],
        )


# Prepare data
prepare_timit(
    data_folder=params.data_folder,
    splits=["train", "dev", "test"],
    save_folder=params.data_folder,
)
train_set = params.train_loader()
valid_set = params.valid_loader()
first_x, first_y = next(iter(train_set))

if hasattr(params, "augmentation"):
    modules.append(params.augmentation)
asr_brain = ASR_kd(
    modules=modules,
    tea0_modules=tea0_modules,
    tea1_modules=tea1_modules,
    optimizer=params.optimizer,
    first_inputs=[first_x, first_y],
)

# load teacher models
tea0_path = "/nfs-share/yan/speechbrain_newArch/recipes/TIMIT/ASR_seq2seq/results/CRDNN/1234/save/CKPT+2020-06-04+19-41-48+00/model.ckpt"
tea0_modules.load_state_dict(torch.load(tea0_path))

tea1_path = "/nfs-share/yan/speechbrain_newArch/recipes/TIMIT/ASR_seq2seq_tea2/results/CRDNN/1234/save/CKPT+2020-06-09+14-59-28+00/model.ckpt"
tea1_modules.load_state_dict(torch.load(tea1_path))

# Load latest checkpoint to resume training
checkpointer.recover_if_possible()
asr_brain.fit(params.epoch_counter, train_set, valid_set)

# Load best checkpoint for evaluation
checkpointer.recover_if_possible(lambda c: -c.meta["PER"])
test_stats = asr_brain.evaluate(params.test_loader())
params.train_logger.log_stats(
    stats_meta={"Epoch loaded": params.epoch_counter.current},
    test_stats=test_stats,
)

# Write alignments to file
per_summary = edit_distance.wer_summary(test_stats["PER"])
with open(params.wer_file, "w") as fo:
    wer_io.print_wer_summary(per_summary, fo)
    wer_io.print_alignments(test_stats["PER"], fo)
