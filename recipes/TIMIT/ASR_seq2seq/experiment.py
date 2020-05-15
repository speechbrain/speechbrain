#!/usr/bin/env python3
import os
import sys
import torch
import speechbrain as sb

import speechbrain.data_io.wer as wer_io
import speechbrain.utils.edit_distance as edit_distance
from speechbrain.data_io.data_io import convert_index_to_lab

# from speechbrain.decoders.seq2seq import RNNGreedySearcher
from speechbrain.decoders.seq2seq import RNNBeamSearcher
from speechbrain.decoders.decoders import undo_padding
from speechbrain.utils.checkpoints import ckpt_recency
from speechbrain.utils.train_logger import (
    FileTrainLogger,
    summarize_average,
    summarize_error_rate,
)
from speechbrain.nnet.sequential import Sequential
from speechbrain.nnet.CNN import Conv
from speechbrain.nnet.normalization import Normalize
from speechbrain.nnet.pooling import Pooling


def cnn_block(out_channels, kernel_size, pooling_size):

    block = Sequential(
        Conv(
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=True,
        ),
        Normalize(norm_type="batchnorm"),
        torch.nn.ReLU(),
        Conv(
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=True,
        ),
        Normalize(norm_type="batchnorm"),
        torch.nn.ReLU(),
        Pooling(
            pool_type="max",
            kernel_size=[pooling_size, pooling_size],
            stride=[pooling_size, pooling_size],
            pool_axis=[1, 2],
        ),
    )
    return block


def get_cnn(out_channels, kernel_size, pooling_size, n_blocks=2):

    blocks = [
        cnn_block(out_channels, kernel_size, pooling_size)
        for _ in range(n_blocks)
    ]
    return Sequential(*blocks)


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

cnn = get_cnn(
    out_channels=params.out_channels,
    kernel_size=params.kernel_size,
    pooling_size=params.pooling_size,
    n_blocks=params.n_blocks,
)

modules = torch.nn.ModuleList(
    [cnn, params.rnn_enc, params.rnn_dec, params.ctc_linear, params.seq_linear]
)

# searcher = RNNGreedySearcher(
#    modules=[params.rnn_dec, params.seq_linear],
#    bos_index=params.bos_index,
#    eos_index=params.eos_index,
#    min_decode_ratio=0,
#    max_decode_ratio=1,
# )
searcher = RNNBeamSearcher(
    modules=[params.rnn_dec, params.seq_linear],
    bos_index=params.bos_index,
    eos_index=params.eos_index,
    min_decode_ratio=0,
    max_decode_ratio=1,
    beam_size=params.beam_size,
    length_penalty=params.length_penalty,
    eos_penalty=params.eos_penalty,
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
    def forward(self, inputs, init_params=False):
        (_, wavs, wav_lens), (_, ys, y_lens), train = inputs

        wavs, wav_lens = wavs.to(params.device), wav_lens.to(params.device)
        ys, y_lens = ys.to(params.device), y_lens.to(params.device)

        if hasattr(params, "augmentation"):
            wavs = params.augmentation(wavs, wav_lens, init_params)

        feats = params.compute_features(wavs, init_params)
        feats = params.normalize(feats, wav_lens)

        x = cnn(feats, init_params)
        x = params.rnn_enc(x, init_params)

        logits = params.ctc_linear(x, init_params)
        p_ctc = params.softmax(logits)

        dec, _ = params.rnn_dec(x, wav_lens, ys, init_params)
        logits = params.seq_linear(dec, init_params)
        p_seq = params.softmax(logits)

        if not train:
            hyps = searcher(x, wav_lens)
            return p_ctc, p_seq, wav_lens, hyps

        return p_ctc, p_seq, wav_lens

    def compute_objectives(self, predictions, targets, train=True):
        if train:
            p_ctc, p_seq, wav_lens = predictions
        else:
            p_ctc, p_seq, wav_lens, (hyps, scores) = predictions

        ids, phns, phn_lens = targets
        phns, phn_lens = phns.to(params.device), phn_lens.to(params.device)
        loss_ctc = params.ctc_cost(p_ctc, phns, [wav_lens, phn_lens])
        loss_seq = params.seq_cost(p_seq, phns, phn_lens)
        loss = params.ctc_weight * loss_ctc + (1 - params.ctc_weight) * loss_seq

        if not train:
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
        train = True
        predictions = self.forward([inputs, targets, train])
        loss, loss_ctc, loss_seq = self.compute_objectives(predictions, targets)
        loss.backward()
        self.optimizer(self.modules)
        return {
            "loss": loss.item(),
            "loss_ctc": loss_ctc.item(),
            "loss_seq": loss_seq.item(),
        }

    def evaluate_batch(self, batch):
        inputs, targets = batch
        train = False
        predictions = self.forward([inputs, targets, train])
        loss, loss_ctc, loss_seq, stats = self.compute_objectives(
            predictions, targets, train=train
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
first_input = next(iter(train_set[0]))
asr_brain = ASR(
    modules=modules,
    optimizer=params.optimizer,
    first_input=(next(iter(train_set[0])), next(iter(train_set[1])), True),
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
