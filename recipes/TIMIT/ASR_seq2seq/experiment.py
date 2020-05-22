#!/usr/bin/env python3
import os
import sys
import torch
import speechbrain as sb

import speechbrain.data_io.wer as wer_io
import speechbrain.utils.edit_distance as edit_distance
from speechbrain.data_io.data_io import convert_index_to_lab
from speechbrain.data_io.data_io import put_bos_token

from speechbrain.decoders.seq2seq import RNNGreedySearcher
from speechbrain.decoders.seq2seq import RNNBeamSearcher
from speechbrain.decoders.decoders import undo_padding
from speechbrain.utils.checkpoints import ckpt_recency
from speechbrain.utils.train_logger import (
    FileTrainLogger,
    summarize_average,
    summarize_error_rate,
)

# This hack needed to import data preparation script from ..
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from timit_prepare import TIMITPreparer  # noqa E402

# Load hyperparameters file with command-line overrides
params_file, overrides = sb.core.parse_arguments(sys.argv[1:])
if "seed" in overrides:
    torch.manual_seed(overrides["seed"])
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, overrides)

# Create experiment directory
sb.core.create_experiment_directory(
    experiment_directory=params.output_folder,
    params_to_save=params_file,
    overrides=overrides,
)

train_logger = FileTrainLogger(
    save_file=params.train_log,
    summary_fns={
        "loss": summarize_average,
        "loss_ctc": summarize_average,
        "loss_seq": summarize_average,
        "PER": summarize_error_rate,
    },
)

modules = torch.nn.ModuleList(
    [params.enc, params.emb, params.dec, params.ctc_lin, params.seq_lin]
)
if params.using_greedy:
    searcher = RNNGreedySearcher(
        modules=[params.emb, params.dec, params.seq_lin, params.log_softmax],
        bos_index=params.bos_index,
        eos_index=params.eos_index,
        min_decode_ratio=0,
        max_decode_ratio=1,
    )
else:
    searcher = RNNBeamSearcher(
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

# TODO: temporary
params.emb = params.emb.to(params.device)


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, x, y, train_mode=True, init_params=False):
        id, wavs, wav_lens = x
        id, phns, phn_lens = y

        wavs, wav_lens = wavs.to(params.device), wav_lens.to(params.device)
        phns, phn_lens = phns.to(params.device), phn_lens.to(params.device)

        if hasattr(params, "augmentation"):
            wavs = params.augmentation(wavs, wav_lens, init_params)

        feats = params.compute_features(wavs, init_params)
        feats = params.normalize(feats, wav_lens)
        x = params.enc(feats, init_params=init_params)

        logits = params.ctc_lin(x, init_params)
        p_ctc = params.log_softmax(logits)

        y_in = put_bos_token(phns, bos_index=params.bos_index)
        e_in = params.emb(y_in)
        h, _ = params.dec(e_in, x, wav_lens, init_params)
        logits = params.seq_lin(h, init_params)
        p_seq = params.log_softmax(logits)

        if not train_mode:
            hyps, scores = searcher(x, wav_lens)
            return p_ctc, p_seq, wav_lens, hyps

        return p_ctc, p_seq, wav_lens

    def compute_objectives(self, predictions, targets, train_mode=True):
        if train_mode:
            p_ctc, p_seq, wav_lens = predictions
        else:
            p_ctc, p_seq, wav_lens, hyps = predictions

        ids, phns, phn_lens = targets
        phns, phn_lens = phns.to(params.device), phn_lens.to(params.device)

        loss_ctc = params.ctc_cost(p_ctc, phns, [wav_lens, phn_lens])
        loss_seq = params.seq_cost(p_seq, phns, phn_lens)
        loss = params.ctc_weight * loss_ctc + (1 - params.ctc_weight) * loss_seq

        if not train_mode:
            ind2lab = params.train_loader.label_dict["phn"]["index2lab"]
            sequence = convert_index_to_lab(hyps, ind2lab)
            phns = undo_padding(phns, phn_lens)
            phns = convert_index_to_lab(phns, ind2lab)
            stats = edit_distance.wer_details_for_batch(
                ids, phns, sequence, compute_alignments=True
            )
            stats = {"PER": stats}
            return loss, loss_ctc, loss_seq, stats

        return loss, loss_ctc, loss_seq

    def fit_batch(self, batch):
        inputs, targets = batch
        predictions = self.compute_forward(inputs, targets)
        loss, loss_ctc, loss_seq = self.compute_objectives(predictions, targets)
        loss.backward()
        self.optimizer(self.modules)
        return {
            "loss": loss.detach(),
            "loss_ctc": loss_ctc.detach(),
            "loss_seq": loss_seq.detach(),
        }

    def evaluate_batch(self, batch):
        inputs, targets = batch
        predictions = self.compute_forward(inputs, targets, train_mode=False)
        loss, loss_ctc, loss_seq, stats = self.compute_objectives(
            predictions, targets, train_mode=False
        )

        loss_dict = {
            "loss": loss.item(),
            "loss_ctc": loss_ctc.item(),
            "loss_seq": loss_seq.item(),
        }

        return {**loss_dict, **stats}

    def on_epoch_end(self, epoch, train_stats, valid_stats=None):
        per = summarize_error_rate(valid_stats["PER"])
        old_lr, new_lr = params.lr_annealing([params.optimizer], epoch, per)
        epoch_stats = {"epoch": epoch, "lr": old_lr}
        train_logger.log_stats(epoch_stats, train_stats, valid_stats)

        checkpointer.save_and_keep_only(
            meta={"PER": per},
            importance_keys=[ckpt_recency, lambda c: -c.meta["PER"]],
        )


prepare = TIMITPreparer(
    data_folder=params.data_folder,
    splits=["train", "dev", "test"],
    save_folder=params.data_folder,
)
prepare()
train_set = params.train_loader()
valid_set = params.valid_loader()

if hasattr(params, "augmentation"):
    modules.append(params.augmentation)
first_x, first_y = next(zip(*train_set))
asr_brain = ASR(
    modules=modules,
    optimizer=params.optimizer,
    first_inputs=[first_x, first_y],
)

# Load latest checkpoint to resume training
checkpointer.recover_if_possible()
asr_brain.fit(params.epoch_counter, train_set, valid_set)

# Load best checkpoint for evaluation
checkpointer.recover_if_possible(lambda c: -c.meta["PER"])
test_stats = asr_brain.evaluate(params.test_loader())
train_logger.log_stats(
    stats_meta={"Epoch loaded": params.epoch_counter.current},
    test_stats=test_stats,
)

# Write alignments to file
per_summary = edit_distance.wer_summary(test_stats["PER"])
with open(params.wer_file, "w") as fo:
    wer_io.print_wer_summary(per_summary, fo)
    wer_io.print_alignments(test_stats["PER"], fo)
