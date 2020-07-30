#!/usr/bin/env python3
import os
import sys
import torch
import speechbrain as sb
import numpy as np
from tqdm.contrib import tqdm

import speechbrain.data_io.wer as wer_io
import speechbrain.utils.edit_distance as edit_distance
from speechbrain.data_io.data_io import convert_index_to_lab
from speechbrain.data_io.data_io import prepend_bos_token
from speechbrain.data_io.data_io import append_eos_token

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


def load_teachers():
    if hasattr(params, "augmentation"):
        path = current_dir + "/tea_infer_{}batch.npz".format(params.batch_size)
    else:
        path = current_dir + "/tea_infer_noAug_{}batch.npz".format(
            params.batch_size
        )
    data = np.load(path, allow_pickle=True)
    train_dict = data["train_dict"][()]
    valid_dict = data["valid_dict"][()]
    test_dict = data["test_dict"][()]
    return [train_dict, valid_dict], test_dict


# Define training procedure
class ASR(sb.core.Brain):
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

        # output layer for ctc log-probabilities
        logits = params.ctc_lin(x, init_params)
        p_ctc = params.log_softmax(logits / params.T)

        # Prepend bos token at the beginning
        y_in = prepend_bos_token(phns, bos_index=params.bos_index)
        e_in = params.emb(y_in, init_params=init_params)
        h, _ = params.dec(e_in, x, wav_lens, init_params)

        # output layer for seq2seq log-probabilities
        logits = params.seq_lin(h, init_params)
        p_seq = params.log_softmax(logits / params.T)

        if stage == "valid":
            hyps, scores = greedy_searcher(x, wav_lens)
            return p_ctc, p_seq, wav_lens, hyps

        elif stage == "test":
            hyps, scores = beam_searcher(x, wav_lens)
            return p_ctc, p_seq, wav_lens, hyps

        return p_ctc, p_seq, wav_lens

    def load_teacher(self, ids, data_dict):
        tea_list = [None, None, None, None]
        for b_num in range(len(ids)):
            id = ids[b_num]
            tea_infer = data_dict[id]
            item_tea_list = [None, None, None, None]
            for tea_num in range(params.num_tea):
                tea_dict = tea_infer[tea_num]
                for i in range(4):

                    item_tea = tea_dict[tea_keys[i]]
                    if tea_keys[i].startswith("wer"):
                        item_tea = torch.tensor(item_tea)
                    else:
                        item_tea = torch.from_numpy(item_tea)

                    item_tea = item_tea.to(params.device)
                    item_tea = torch.unsqueeze(item_tea, 0)
                    if tea_num == 0:
                        item_tea_list[i] = item_tea
                    else:
                        item_tea_list[i] = torch.cat(
                            [item_tea_list[i], item_tea], 0
                        )
            for j in range(4):
                item_tea_list[j] = torch.unsqueeze(item_tea_list[j], 1)

                if b_num == 0:
                    tea_list[j] = item_tea_list[j]
                else:
                    tea_list[j] = torch.cat([tea_list[j], item_tea_list[j]], 1)
        return tea_list

    def compute_objectives(
        self, predictions, targets, data_dict, stage="train"
    ):
        if stage == "train":
            p_ctc, p_seq, wav_lens = predictions
        else:
            p_ctc, p_seq, wav_lens, hyps = predictions

        ids, phns, phn_lens = targets
        phns, phn_lens = phns.to(params.device), phn_lens.to(params.device)

        # Add phn_lens by one for eos token
        abs_length = torch.round(phn_lens * phns.shape[1])

        # Append eos token at the end of the label sequences
        phns_with_eos = append_eos_token(
            phns, length=abs_length, eos_index=params.eos_index
        )

        # convert to speechbrain-style relative length
        rel_length = (abs_length + 1) / phns.shape[1]

        # normal supervised training
        loss_ctc_nor = params.ctc_cost(p_ctc, phns, wav_lens, phn_lens)
        loss_seq_nor = params.seq_cost(p_seq, phns_with_eos, length=rel_length)

        # load teacher inference results
        tea_list = self.load_teacher(ids, data_dict)
        p_ctc_tea = tea_list[0]
        p_seq_tea = tea_list[1]
        wer_ctc_tea = tea_list[2]
        wer_tea = tea_list[3]

        if params.strategy == "s1":
            # tea_ce for kd
            wer_scores, indx = torch.min(wer_tea, dim=0)
            indx = list(indx.cpu().numpy())

            tea_seq2seq_pout = None
            for stn_indx, tea_indx in enumerate(indx):
                s2s_one = p_seq_tea[tea_indx][stn_indx]
                s2s_one = torch.unsqueeze(s2s_one, 0)
                if stn_indx == 0:
                    tea_seq2seq_pout = s2s_one
                else:
                    tea_seq2seq_pout = torch.cat([tea_seq2seq_pout, s2s_one], 0)

        if params.strategy == "s1" or "s2":
            # mean wer for ctc
            tea_wer_ctc_mean = wer_ctc_tea.mean(1)
            tea_acc_main = 100 - tea_wer_ctc_mean
            m0 = torch.nn.Softmax(dim=0)
            tea_acc_softmax = m0(tea_acc_main)

        if params.strategy == "s2":
            # mean wer for ce
            tea_wer_mean = wer_tea.mean(1)
            tea_acc_ce_main = 100 - tea_wer_mean
            tea_acc_ce_softmax = m0(tea_acc_ce_main)

        # kd loss
        ctc_loss_list = None
        ce_loss_list = None
        for tea_num in range(params.num_tea):
            # ctc
            p_ctc_tea_one = p_ctc_tea[tea_num]
            loss_ctc_one = params.ctc_cost_kd(p_ctc, p_ctc_tea_one, wav_lens)
            loss_ctc_one = torch.unsqueeze(loss_ctc_one, 0)
            if tea_num == 0:
                ctc_loss_list = loss_ctc_one
            else:
                ctc_loss_list = torch.cat([ctc_loss_list, loss_ctc_one])

            # ce
            p_seq_tea_one = p_seq_tea[tea_num]
            loss_seq_one = params.seq_cost_kd(p_seq, p_seq_tea_one, rel_length)
            loss_seq_one = torch.unsqueeze(loss_seq_one, 0)
            if tea_num == 0:
                ce_loss_list = loss_seq_one
            else:
                ce_loss_list = torch.cat([ce_loss_list, loss_seq_one])

        # kd loss
        if params.strategy == "baseline":
            ctc_loss_kd = ctc_loss_list.mean(0)
            seq2seq_loss_kd = ce_loss_list.mean(0)
        else:
            ctc_loss_kd = (tea_acc_softmax * ctc_loss_list).sum(0)
            if params.strategy == "s1":
                seq2seq_loss_kd = params.seq_cost_kd(
                    p_seq, tea_seq2seq_pout, rel_length
                )
            if params.strategy == "s2":
                seq2seq_loss_kd = (tea_acc_ce_softmax * ce_loss_list).sum(0)

        # total loss
        loss_ctc = (
            params.T * params.T * params.alpha * ctc_loss_kd
            + (1 - params.alpha) * loss_ctc_nor
        )
        loss_seq = (
            params.T * params.T * params.alpha * seq2seq_loss_kd
            + (1 - params.alpha) * loss_seq_nor
        )

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

    def fit_batch(self, batch, train_dict):
        inputs, targets = batch
        predictions = self.compute_forward(inputs, targets)
        loss, stats = self.compute_objectives(predictions, targets, train_dict)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        stats["loss"] = loss.detach()
        return stats

    def evaluate_batch(self, batch, data_dict=None, stage="valid"):
        inputs, targets = batch
        predictions = self.compute_forward(inputs, targets, stage=stage)
        loss, stats = self.compute_objectives(
            predictions, targets, data_dict, stage=stage
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

    def fit(self, epoch_counter, save_dict, train_set, valid_set=None):
        train_dict, valid_dict = save_dict
        for epoch in epoch_counter:
            self.modules.train()
            train_stats = {}
            with tqdm(train_set, dynamic_ncols=True) as t:
                for i, batch in enumerate(t):
                    stats = self.fit_batch(batch, train_dict)
                    self.add_stats(train_stats, stats)
                    average = self.update_average(stats, iteration=i + 1)
                    t.set_postfix(train_loss=average)

            valid_stats = {}
            if valid_set is not None:
                self.modules.eval()
                with torch.no_grad():
                    for batch in tqdm(valid_set, dynamic_ncols=True):
                        stats = self.evaluate_batch(
                            batch, valid_dict, stage="valid"
                        )
                        self.add_stats(valid_stats, stats)

            self.on_epoch_end(epoch, train_stats, valid_stats)

    def evaluate(self, test_set, test_dict):
        test_stats = {}
        self.modules.eval()
        with torch.no_grad():
            for batch in tqdm(test_set, dynamic_ncols=True):
                stats = self.evaluate_batch(batch, test_dict, stage="test")
                self.add_stats(test_stats, stats)

        return test_stats


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
asr_brain = ASR(
    modules=modules,
    optimizer=params.optimizer,
    first_inputs=[first_x, first_y],
)

# load teacher models
save_dict, test_dict = load_teachers()
tea_keys = ["p_ctc_tea", "p_seq_tea", "wer_ctc_tea", "wer_tea"]

# initialization strategy
if params.retrain:
    # load pre-trained student model except last layer
    if params.epoch_counter.current == 0:
        save_dir = (
            current_dir
            + "/teacher_models_training/results/tea0/{}/save/".format(
                params.seed
            )
        )
        ckpt_dir_list = os.listdir(save_dir)
        chpt_path = save_dir + ckpt_dir_list[0] + "/model.ckpt"
        weight_dict = torch.load(chpt_path)
        # del the last layer
        key_list = []
        for k in weight_dict.keys():
            key_list.append(k)
        for k in key_list:
            if k.startswith("1") or k.startswith("2"):
                del weight_dict[k]

        modules.load_state_dict(weight_dict, strict=False)
    else:
        # Load latest checkpoint to resume training
        checkpointer.recover_if_possible()
else:
    checkpointer.recover_if_possible()

# training
asr_brain.fit(params.epoch_counter, save_dict, train_set, valid_set)

# Load best checkpoint for evaluation
checkpointer.recover_if_possible(lambda c: -c.meta["PER"])
test_stats = asr_brain.evaluate(params.test_loader(), test_dict)
params.train_logger.log_stats(
    stats_meta={"Epoch loaded": params.epoch_counter.current},
    test_stats=test_stats,
)

# Write alignments to file
per_summary = edit_distance.wer_summary(test_stats["PER"])
with open(params.wer_file, "w") as fo:
    wer_io.print_wer_summary(per_summary, fo)
    wer_io.print_alignments(test_stats["PER"], fo)
