#!/usr/bin/env python3
import os
import sys
import torch
import speechbrain as sb
from speechbrain.decoders.ctc import ctc_greedy_decode

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
        wavs, wav_lens = wavs.to(params.device), wav_lens.to(params.device)

        # Adding environmental corruption if specified (i.e., noise+rev)
        if hasattr(params, "env_corrupt") and stage == "train":
            wavs_noise = params.env_corrupt(wavs, wav_lens, init_params)
            wavs = torch.cat([wavs, wavs_noise], dim=0)
            wav_lens = torch.cat([wav_lens, wav_lens])

        # Adding time-domain SpecAugment if specified
        if hasattr(params, "augmentation"):
            wavs = params.augmentation(wavs, wav_lens, init_params)

        feats = params.compute_features(wavs, init_params)
        feats = params.normalize(feats, wav_lens)
        out = params.model(feats, init_params)
        out = params.output(out, init_params)
        pout = params.log_softmax(out)
        return pout, wav_lens

    def compute_objectives(self, predictions, targets, stage="train"):
        pout, pout_lens = predictions
        ids, phns, phn_lens = targets
        phns, phn_lens = phns.to(params.device), phn_lens.to(params.device)

        if stage == "train":
            if hasattr(params, "env_corrupt"):
                phns = torch.cat([phns, phns], dim=0)
                phn_lens = torch.cat([phn_lens, phn_lens], dim=0)
            loss = params.compute_cost(pout, phns, pout_lens, phn_lens)
        else:
            loss = params.compute_cost(pout, phns, pout_lens, phn_lens)
            ind2lab = params.train_loader.label_dict["phn"]["index2lab"]
            sequence = ctc_greedy_decode(pout, pout_lens, blank_id=-1)
            self.stats[stage]["PER"].append(
                ids, sequence, phns, target_len=phn_lens, ind2lab=ind2lab
            )

        self.stats[stage]["loss"].append(ids, pout, phns, pout_lens, phn_lens)
        return loss

    def on_epoch_start(self, epoch):
        self.stats = {
            "train": {"loss": params.ctc_stats()},
            "valid": {"loss": params.ctc_stats(), "PER": params.per_stats()},
        }

    def on_epoch_end(self, epoch, train_loss, valid_loss=None):
        per = self.stats["valid"]["PER"].summarize()
        old_lr, new_lr = params.lr_annealing([params.optimizer], epoch, per)
        params.train_logger.log_stats(
            stats_meta={"epoch": epoch, "lr": old_lr},
            train_stats={"loss": train_loss},
            valid_stats={"loss": valid_loss, "PER": per},
        )
        params.checkpointer.save_and_keep_only(
            meta={"PER": per}, min_keys=["PER"],
        )

    def on_eval_start(self):
        self.stats = {
            "test": {"loss": params.ctc_stats(), "PER": params.per_stats()}
        }

    def on_eval_end(self, test_loss):
        print(test_loss)


# Prepare data
prepare_timit(
    data_folder=params.data_folder,
    splits=["train", "dev", "test"],
    save_folder=params.data_folder,
)
train_set = params.train_loader()
valid_set = params.valid_loader()
first_x, first_y = next(iter(train_set))

# Modules are passed to optimizer and have train/eval called on them
modules = [params.model, params.output]

# We need to pass the augmentation module too.
# This way we do augment only in traning mode.
if hasattr(params, "augmentation"):
    modules.append(params.augmentation)

# Create brain object for training
asr_brain = ASR(
    modules=modules, optimizer=params.optimizer, first_inputs=[first_x],
)

# Load latest checkpoint to resume training
params.checkpointer.recover_if_possible()
asr_brain.fit(params.epoch_counter, train_set, valid_set)

# Load best checkpoint for evaluation
params.checkpointer.recover_if_possible(min_key="PER")
test_loss = asr_brain.evaluate(params.test_loader())
test_per = asr_brain.stats["test"]["PER"].summarize()
params.train_logger.log_stats(
    stats_meta={"Epoch loaded": params.epoch_counter.current},
    test_stats={"loss": test_loss, "PER": test_per},
)

# Write alignments to file
asr_brain.stats["test"]["loss"].write_stats("test.txt")
asr_brain.stats["test"]["PER"].write_stats(params.wer_file)
