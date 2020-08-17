#!/usr/bin/python
import os
import sys
import torch
import speechbrain as sb

# This hack needed to import data preparation script from ..
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from voxceleb_prepare import prepare_voxceleb  # noqa E402

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

# Prepare data from dev of Voxceleb1
prepare_voxceleb(
    data_folder=params.data_folder,
    save_folder=params.save_folder,
    splits=["train", "dev"],
    split_ratio=[90, 10],
    seg_dur=300,
    vad=False,
    rand_seed=params.seed,
)


# Trains xvector model
class XvectorBrain(sb.core.Brain):
    def compute_forward(self, x, stage="train", init_params=False):
        id, wavs, lens = x

        wavs, lens = wavs.to(params.device), lens.to(params.device)

        if stage == "train":
            # Addding noise and reverberation
            wavs_aug = params.env_corrupt(wavs, lens, init_params)

            # Adding time-domain augmentation
            wavs_aug = params.augmentation(wavs_aug, lens, init_params)

            # Concatenate noisy and clean batches
            wavs = torch.cat([wavs, wavs_aug], dim=0)
            lens = torch.cat([lens, lens], dim=0)

        # Feature extraction and normalization
        feats = params.compute_features(wavs, init_params)
        feats = params.mean_var_norm(feats, lens)

        # Xvector + speaker classifier
        x_vect = params.xvector_model(feats, init_params=init_params)
        outputs = params.classifier(x_vect, init_params)

        return outputs, lens

    def compute_objectives(self, predictions, targets, stage="train"):
        predictions, lens = predictions
        uttid, spkid, _ = targets

        spkid, lens = spkid.to(params.device), lens.to(params.device)

        # Concatenate labels
        if stage == "train":
            spkid = torch.cat([spkid, spkid], dim=0)

        loss = params.compute_cost(predictions, spkid, lens)

        stats = {}
        if stage != "train":
            stats["error"] = params.compute_error(predictions, spkid, lens)

        return loss, stats

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        old_lr, new_lr = params.lr_annealing(
            [params.optimizer], epoch, valid_stats["error"]
        )
        epoch_stats = {"epoch": epoch, "lr": old_lr}
        params.train_logger.log_stats(epoch_stats, train_stats, valid_stats)
        params.checkpointer.save_and_keep_only()


# Data loaders
train_set = params.train_loader()
valid_set = params.valid_loader()

# Xvector Model
modules = [params.xvector_model, params.classifier]
first_x, first_y = next(iter(train_set))


# Object initialization for training xvector model
xvect_brain = XvectorBrain(
    modules=modules, optimizer=params.optimizer, first_inputs=[first_x],
)

# Recover checkpoints
params.checkpointer.recover_if_possible()

# Train the Xvector model
xvect_brain.fit(
    params.epoch_counter, train_set=train_set, valid_set=valid_set,
)
print("Xvector model training completed!")
