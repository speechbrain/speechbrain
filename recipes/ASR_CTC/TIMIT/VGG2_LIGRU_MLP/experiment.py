#!/usr/bin/env python3
import sys
import torch
import speechbrain as sb
import speechbrain.data_io.wer as wer_io
import speechbrain.utils.edit_distance as edit_distance
from speechbrain.data_io.data_io import convert_index_to_lab
from speechbrain.decoders.ctc import ctc_greedy_decode
from speechbrain.decoders.decoders import undo_padding
from speechbrain.utils.checkpoints import ckpt_recency
from speechbrain.utils.train_logger import TrainLogger

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

train_logger = TrainLogger(save_file=params.train_log)
checkpointer = sb.utils.checkpoints.Checkpointer(
    checkpoints_dir=params.save_folder,
    recoverables={
        "model": params.model,
        "optimizer": params.optimizer,
        "scheduler": params.lr_annealing,
        "normalizer": params.normalize,
        "counter": params.epoch_counter,
    },
)


# Define training procedure
class ASR(sb.core.Brain):
    def forward(self, x, init_params=False):
        ids, wavs, wav_lens = x
        wavs, wav_lens = wavs.to(params.device), wav_lens.to(params.device)
        feats = params.compute_features(wavs, init_params)
        feats = params.normalize(feats, wav_lens)
        return params.model(feats, init_params), wav_lens

    def compute_objectives(self, predictions, targets, train=True):
        pout, pout_lens = predictions
        ids, phns, phn_lens = targets
        phns, phn_lens = phns.to(params.device), phn_lens.to(params.device)
        loss = params.compute_cost(pout, phns, [pout_lens, phn_lens])

        if not train:
            ind2lab = params.train_loader.label_dict["phn"]["index2lab"]
            sequence = ctc_greedy_decode(pout, pout_lens, blank_id=-1)
            sequence = convert_index_to_lab(sequence, ind2lab)
            phns = undo_padding(phns, phn_lens)
            phns = convert_index_to_lab(phns, ind2lab)
            stats = edit_distance.wer_details_for_batch(
                ids, phns, sequence, compute_alignments=True
            )
            stats = {"PER": stats}
            return loss, stats

        return loss

    def summarize(self, stats, test=False):
        summary = {"loss": float(sum(stats["loss"]) / len(stats["loss"]))}
        if "PER" in stats:
            per_summary = edit_distance.wer_summary(stats["PER"])
            summary["PER"] = per_summary["WER"]
            if test:
                with open(params.wer_file, "w") as fo:
                    wer_io.print_wer_summary(per_summary, fo)
                    wer_io.print_alignments(stats["PER"], fo)
        return summary

    def on_epoch_end(self, epoch, train_stats, valid_stats=None):
        old_lr, new_lr = params.lr_annealing(
            [params.optimizer], epoch, valid_stats["PER"],
        )
        epoch_stats = {"epoch": epoch, "lr": old_lr}
        train_logger.log_epoch(epoch_stats, train_stats, valid_stats)

        checkpointer.save_and_keep_only(
            meta=valid_stats,
            importance_keys=[ckpt_recency, lambda c: -c.meta["PER"]],
        )


# Experiment setup
params.prepare_timit()
train_set = params.train_loader()
valid_set = params.valid_loader()
asr_brain = ASR(
    modules=[params.model],
    optimizer=params.optimizer,
    first_input=next(iter(train_set[0])),
)

# Load latest checkpoint to resume training
checkpointer.recover_if_possible()
asr_brain.fit(params.epoch_counter, train_set, valid_set)

# Load best checkpoint for evaluation
test_set = params.test_loader()
checkpointer.recover_if_possible(lambda c: -c.meta["PER"])
asr_brain.evaluate(params.test_loader())
