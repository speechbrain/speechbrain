#!/usr/bin/env python3
import os
import sys
import torch
import speechbrain as sb
import speechbrain.data_io.wer as wer_io
import speechbrain.utils.edit_distance as edit_distance
from speechbrain.data_io.data_io import convert_index_to_lab
from speechbrain.decoders.ctc import ctc_greedy_decode
from speechbrain.decoders.decoders import undo_padding
from speechbrain.utils.checkpoints import Checkpointer, ckpt_recency
from speechbrain.utils.train_logger import summarize_error_rate

# This hack needed to import data preparation script from ..
experiment_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(experiment_dir))
from timit_prepare import TIMITPreparer  # noqa E402

# Command-line overrides for reading two yaml files: common and hyperparams
params_file, common_params, hyperparams = sb.core.parse_arguments(sys.argv[1:])
with open(os.path.join(experiment_dir, "asr_ctc.yaml")) as fin:
    experiment = sb.yaml.load_extended_yaml(fin, common_params)

# Set up experiment
torch.manual_seed(experiment.seed)
sb.core.create_experiment_directory(
    experiment_directory=experiment.output_folder,
    params_to_save=params_file,
    overrides=hyperparams,
)
prepare = TIMITPreparer(
    data_folder=experiment.data_folder,
    splits=["train", "dev", "test"],
    save_folder=experiment.data_folder,
)
prepare()
train_set = experiment.train_loader()
valid_set = experiment.valid_loader()
test_set = experiment.test_loader()
device = experiment.device
ind2lab = experiment.train_loader.label_dict["phn"]["index2lab"]


# Load hyperparameters for this experiment
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, hyperparams)

# Checkpointer
checkpointer = Checkpointer(
    checkpoints_dir=experiment.save_folder,
    recoverables={
        "model": params.model,
        "output": params.output,
        "scheduler": params.lr_annealing,
        "normalizer": params.normalize,
        "counter": params.epoch_counter,
    },
)


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, x, train_mode=True, init_params=False):
        ids, wavs, wav_lens = x
        wavs, wav_lens = wavs.to(device), wav_lens.to(device)
        if hasattr(params, "augmentation"):
            wavs = params.augmentation(wavs, wav_lens, init_params)
        feats = params.compute_features(wavs, init_params)
        feats = params.normalize(feats, wav_lens)
        out = params.model(feats, init_params)
        out = params.output(out, init_params)
        pout = params.log_softmax(out)
        return pout, wav_lens

    def compute_objectives(self, predictions, targets, train_mode=True):
        pout, pout_lens = predictions
        ids, phns, phn_lens = targets
        phns, phn_lens = phns.to(device), phn_lens.to(device)
        loss = params.compute_cost(pout, phns, [pout_lens, phn_lens])

        if not train_mode:
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

    def on_epoch_end(self, epoch, train_stats, valid_stats=None):
        per = summarize_error_rate(valid_stats["PER"])
        old_lr, new_lr = params.lr_annealing([params.optimizer], epoch, per)
        experiment.train_logger.log_stats(
            stats_meta={"epoch": epoch, "lr": old_lr},
            train_stats=train_stats,
            valid_stats=valid_stats,
        )

        checkpointer.save_and_keep_only(
            meta={"PER": per},
            importance_keys=[ckpt_recency, lambda c: -c.meta["PER"]],
        )


# Modules are passed to optimizer and have train/eval called on them
modules = [params.model, params.output]
if hasattr(params, "augmentation"):
    modules.append(params.augmentation)

# Create brain object for training
first_x, first_y = next(zip(*train_set))
asr_brain = ASR(
    modules=modules, optimizer=params.optimizer, first_inputs=[first_x],
)

# Load latest checkpoint to resume training
checkpointer.recover_if_possible()
asr_brain.fit(params.epoch_counter, train_set, valid_set)

# Load best checkpoint for evaluation
checkpointer.recover_if_possible(lambda c: -c.meta["PER"])
test_stats = asr_brain.evaluate(test_set)
experiment.train_logger.log_stats(
    stats_meta={"Epoch loaded": params.epoch_counter.current},
    test_stats=test_stats,
)

# Write alignments to file
per_summary = edit_distance.wer_summary(test_stats["PER"])
with open(params.wer_file, "w") as fo:
    wer_io.print_wer_summary(per_summary, fo)
    wer_io.print_alignments(test_stats["PER"], fo)
