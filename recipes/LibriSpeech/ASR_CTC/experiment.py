#!/usr/bin/env python3
import os
import sys
import torch
import speechbrain as sb

import speechbrain.data_io.wer as wer_io
import speechbrain.utils.edit_distance as edit_distance
from speechbrain.data_io.data_io import convert_index_to_lab
from speechbrain.data_io.data_io import merge_char

from speechbrain.decoders.ctc import ctc_greedy_decode
from speechbrain.decoders.decoders import undo_padding
from speechbrain.utils.checkpoints import ckpt_recency
from speechbrain.utils.train_logger import summarize_error_rate

# This hack needed to import data preparation script from ..
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from librispeech_prepare import prepare_librispeech  # noqa E402

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

modules = torch.nn.ModuleList([params.enc, params.output])

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
class ASR(sb.core.Brain):
    def compute_forward(self, x, stage="train", init_params=False):
        ids, wavs, wav_lens = x

        wavs, wav_lens = wavs.to(params.device), wav_lens.to(params.device)
        if hasattr(params, "env_corrupt"):
            wavs_noise = params.env_corrupt(wavs, wav_lens, init_params)
            wavs = torch.cat([wavs, wavs_noise], dim=0)
            wav_lens = torch.cat([wav_lens, wav_lens])

        if hasattr(params, "augmentation"):
            wavs = params.augmentation(wavs, wav_lens, init_params)
        feats = params.compute_features(wavs, init_params)
        feats = params.normalize(feats, wav_lens)
        out = params.enc(feats, init_params=init_params)
        out = params.output(out, init_params=init_params)
        pout = params.log_softmax(out)

        return pout, wav_lens

    def compute_objectives(self, predictions, targets, stage="train"):
        pout, pout_lens = predictions
        ids, chars, char_lens = targets
        chars, char_lens = chars.to(params.device), char_lens.to(params.device)

        if hasattr(params, "env_corrupt"):
            chars = torch.cat([chars, chars], dim=0)
            char_lens = torch.cat([char_lens, char_lens], dim=0)
        loss = params.ctc_cost(pout, chars, pout_lens, char_lens)

        stats = {}
        if stage != "train":
            ind2lab = params.train_loader.label_dict["char"]["index2lab"]
            sequence = ctc_greedy_decode(pout, pout_lens, blank_id=-1)
            char_seq = convert_index_to_lab(sequence, ind2lab)
            word_seq = merge_char(char_seq)
            chars = undo_padding(chars, char_lens)
            chars = convert_index_to_lab(chars, ind2lab)
            words = merge_char(chars)
            cer_stats = edit_distance.wer_details_for_batch(
                ids, chars, char_seq, compute_alignments=True
            )
            wer_stats = edit_distance.wer_details_for_batch(
                ids, words, word_seq, compute_alignments=True
            )
            stats["CER"] = cer_stats
            stats["WER"] = wer_stats
        return loss, stats

    def fit_batch(self, batch):
        inputs, targets = batch
        predictions = self.compute_forward(inputs)
        loss, stats = self.compute_objectives(predictions, targets)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        stats["loss"] = loss.detach()
        return stats

    def evaluate_batch(self, batch, stage="valid"):
        inputs, targets = batch
        predictions = self.compute_forward(inputs, stage=stage)
        loss, stats = self.compute_objectives(predictions, targets, stage=stage)
        stats["loss"] = loss.detach()
        return stats

    def on_epoch_end(self, epoch, train_stats, valid_stats=None):
        wer = summarize_error_rate(valid_stats["WER"])
        old_lr, new_lr = params.lr_annealing([params.optimizer], epoch, wer)
        epoch_stats = {"epoch": epoch, "lr": old_lr}
        params.train_logger.log_stats(epoch_stats, train_stats, valid_stats)

        checkpointer.save_and_keep_only(
            meta={"WER": wer},
            importance_keys=[ckpt_recency, lambda c: -c.meta["WER"]],
        )


# Prepare data
prepare_librispeech(
    data_folder=params.data_folder,
    splits=["train-clean-100", "dev-clean", "test-clean"],
    save_folder=params.data_folder,
)
train_set = params.train_loader()
valid_set = params.valid_loader()
first_x, _ = next(iter(train_set))

if hasattr(params, "augmentation"):
    modules.append(params.augmentation)
asr_brain = ASR(
    modules=modules, optimizer=params.optimizer, first_inputs=[first_x],
)

# Load latest checkpoint to resume training
checkpointer.recover_if_possible()
asr_brain.fit(params.epoch_counter, train_set, valid_set)

# Load best checkpoint for evaluation
checkpointer.recover_if_possible(lambda c: -c.meta["WER"])
test_stats = asr_brain.evaluate(params.test_loader())
params.train_logger.log_stats(
    stats_meta={"Epoch loaded": params.epoch_counter.current},
    test_stats=test_stats,
)

# Write alignments to file
wer_summary = edit_distance.wer_summary(test_stats["WER"])
with open(params.wer_file, "w") as fo:
    wer_io.print_wer_summary(wer_summary, fo)
    wer_io.print_alignments(test_stats["WER"], fo)
