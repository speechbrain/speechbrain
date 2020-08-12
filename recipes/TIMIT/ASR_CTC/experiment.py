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
    hyperparams_to_save=params_file,
    overrides=overrides,
)


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, x, stage="train", init_params=False):
        ids, wavs, wav_lens = x
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Adding environmental corruption if specified (i.e., noise+rev)
        if "env_corrupt" in self.features and stage == "train":
            wavs_noise = self.features["env_corrupt"](
                wavs, wav_lens, init_params
            )
            wavs = torch.cat([wavs, wavs_noise], dim=0)
            wav_lens = torch.cat([wav_lens, wav_lens])

        # Adding time-domain SpecAugment if specified
        if "augmentation" in self.features:
            wavs = self.features["augmentation"](wavs, wav_lens, init_params)

        feats = self.features["compute_features"](wavs, init_params)
        feats = self.features["normalize"](feats, wav_lens)
        out = self.model["model"](feats, init_params)
        out = self.model["output"](out, init_params)
        pout = self.model["log_softmax"](out)

        return pout, wav_lens

    def compute_objectives(self, predictions, targets, stage="train"):
        pout, pout_lens = predictions
        ids, phns, phn_lens = targets
        phns, phn_lens = phns.to(self.device), phn_lens.to(self.device)
        stats = {}

        if stage == "train":
            if "env_corrupt" in self.features:
                phns = torch.cat([phns, phns], dim=0)
                phn_lens = torch.cat([phn_lens, phn_lens], dim=0)
            loss = params.compute_cost(pout, phns, pout_lens, phn_lens)
        else:
            loss = params.compute_cost(pout, phns, pout_lens, phn_lens)
            ind2lab = params.train_loader.label_dict["phn"]["index2lab"]
            sequence = ctc_greedy_decode(pout, pout_lens, blank_id=-1)
            sequence = convert_index_to_lab(sequence, ind2lab)
            phns = undo_padding(phns, phn_lens)
            phns = convert_index_to_lab(phns, ind2lab)
            per_stats = edit_distance.wer_details_for_batch(
                ids, phns, sequence, compute_alignments=True
            )
            stats["PER"] = per_stats
        return loss, stats

    def on_epoch_end(self, epoch, train_stats, valid_stats=None):
        per = summarize_error_rate(valid_stats["PER"])
        old_lr, new_lr = params.lr_annealing(self.optimizers, epoch, per)
        epoch_stats = {"epoch": epoch, "lr": old_lr}
        params.train_logger.log_stats(epoch_stats, train_stats, valid_stats)

        params.checkpointer.save_and_keep_only(
            meta={"PER": per}, min_keys=["PER"],
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

models = [
    ("features", params.features, None),
    ("model", params.model, params.optimizer),
]

# Create brain object for training
asr_brain = ASR(models=models, device=params.device, first_inputs=[first_x])

# Load latest checkpoint to resume training
params.checkpointer.recover_if_possible()
asr_brain.fit(params.epoch_counter, train_set, valid_set)

# Load best checkpoint for evaluation
params.checkpointer.recover_if_possible(min_key="PER")
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
