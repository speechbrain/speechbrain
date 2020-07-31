#!/usr/bin/env python3
import os
import sys
import torch
import speechbrain as sb

import speechbrain.data_io.wer as wer_io
import speechbrain.utils.edit_distance as edit_distance
from speechbrain.data_io.data_io import convert_index_to_lab
from speechbrain.data_io.data_io import prepend_bos_token
from speechbrain.data_io.data_io import append_eos_token
from speechbrain.data_io.data_io import merge_char

from speechbrain.decoders.seq2seq import S2SRNNGreedySearcher
from speechbrain.decoders.seq2seq import S2SRNNBeamSearcher
from speechbrain.decoders.decoders import undo_padding
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
    hyperparams_to_save=params_file,
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
    using_max_attn_shift=params.using_max_attn_shift,
    max_attn_shift=params.max_attn_shift,
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
class ASR(sb.core.Brain):
    def compute_forward(self, x, y, stage="train", init_params=False):
        ids, wavs, wav_lens = x
        ids, chars, char_lens = y

        wavs, wav_lens = wavs.to(params.device), wav_lens.to(params.device)
        chars, char_lens = chars.to(params.device), char_lens.to(params.device)

        if hasattr(params, "augmentation"):
            wavs = params.augmentation(wavs, wav_lens, init_params)
        feats = params.compute_features(wavs, init_params)
        feats = params.normalize(feats, wav_lens)
        x = params.enc(feats, init_params=init_params)

        # Prepend bos token at the beginning
        y_in = prepend_bos_token(chars, bos_index=params.bos_index)
        e_in = params.emb(y_in, init_params=init_params)
        h, _ = params.dec(e_in, x, wav_lens, init_params)

        # output layer for seq2seq log-probabilities
        logits = params.seq_lin(h, init_params)
        p_seq = params.log_softmax(logits)

        if (
            stage == "train"
            and params.epoch_counter.current <= params.number_of_ctc_epochs
        ):
            # output layer for ctc log-probabilities
            logits = params.ctc_lin(x, init_params)
            p_ctc = params.log_softmax(logits)
            return p_ctc, p_seq, wav_lens
        elif stage == "train":
            return p_seq, wav_lens
        elif stage == "valid":
            hyps, scores = greedy_searcher(x, wav_lens)
            return p_seq, wav_lens, hyps
        elif stage == "test":
            hyps, scores = beam_searcher(x, wav_lens)
            return p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, targets, stage="train"):
        if (
            stage == "train"
            and params.epoch_counter.current <= params.number_of_ctc_epochs
        ):
            p_ctc, p_seq, wav_lens = predictions
        elif stage == "train":
            p_seq, wav_lens = predictions
        else:
            p_seq, wav_lens, hyps = predictions

        ids, chars, char_lens = targets
        chars, char_lens = chars.to(params.device), char_lens.to(params.device)

        # Add char_lens by one for eos token
        abs_length = torch.round(char_lens * chars.shape[1])

        # Append eos token at the end of the label sequences
        chars_with_eos = append_eos_token(
            chars, length=abs_length, eos_index=params.eos_index
        )

        # convert to speechbrain-style relative length
        rel_length = (abs_length + 1) / chars.shape[1]
        loss_seq = params.seq_cost(p_seq, chars_with_eos, length=rel_length)

        if (
            stage == "train"
            and params.epoch_counter.current <= params.number_of_ctc_epochs
        ):
            loss_ctc = params.ctc_cost(p_ctc, chars, wav_lens, char_lens)
            loss = (
                params.ctc_weight * loss_ctc
                + (1 - params.ctc_weight) * loss_seq
            )
        else:
            loss = loss_seq

        stats = {}
        if stage != "train":
            ind2lab = params.train_loader.label_dict["char"]["index2lab"]
            char_seq = convert_index_to_lab(hyps, ind2lab)
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
        predictions = self.compute_forward(inputs, targets)
        loss, stats = self.compute_objectives(predictions, targets)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        stats["loss"] = loss.detach()
        return stats

    def evaluate_batch(self, batch, stage="valid"):
        inputs, targets = batch
        predictions = self.compute_forward(inputs, targets, stage=stage)
        loss, stats = self.compute_objectives(predictions, targets, stage=stage)
        stats["loss"] = loss.detach()
        return stats

    def on_epoch_end(self, epoch, train_stats, valid_stats=None):
        wer = summarize_error_rate(valid_stats["WER"])
        old_lr, new_lr = params.lr_annealing([params.optimizer], epoch, wer)
        epoch_stats = {"epoch": epoch, "lr": old_lr}
        params.train_logger.log_stats(epoch_stats, train_stats, valid_stats)

        checkpointer.save_and_keep_only(meta={"WER": wer}, min_keys=["WER"])


# Prepare data
prepare_librispeech(
    data_folder=params.data_folder,
    splits=["train-clean-100", "dev-clean", "test-clean"],
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

# Check if the model should be trained on multiple GPUs.
# Important: DataParallel MUST be called after the ASR (Brain) class init.
if params.multigpu:
    params.enc = torch.nn.DataParallel(params.enc)
    params.emb = torch.nn.DataParallel(params.emb)
    params.dec = torch.nn.DataParallel(params.dec)
    params.ctc_lin = torch.nn.DataParallel(params.ctc_lin)
    params.seq_lin = torch.nn.DataParallel(params.seq_lin)

# Load latest checkpoint to resume training
checkpointer.recover_if_possible()
asr_brain.fit(params.epoch_counter, train_set, valid_set)

# Load best checkpoint for evaluation
checkpointer.recover_if_possible(min_key="WER")
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
