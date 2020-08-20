#!/usr/bin/python
import os
import sys
import torch
import speechbrain as sb
import torch.nn.functional as F

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
    split_speaker=True,
    source=params.voxceleb_source
    if hasattr(params, "voxceleb_source")
    else None,
)


# Trains xvector model
class XvectorBrain(sb.core.Brain):
    def compute_forward(self, x, stage="train", init_params=False):
        id, wavs, lens = x

        wavs, lens = wavs.to(params.device), lens.to(params.device)

        if stage == "train" and hasattr(params, "num_corrupts"):
            G = params.batch_size // params.group_size
            s = list(wavs.shape)
            wavs_aug_list = []
            for i in range(params.num_corrupts):
                # Addding noise and reverberation
                wavs_aug = params.env_corrupt(wavs, lens, init_params)

                # Adding time-domain augmentation
                wavs_aug = params.augmentation(wavs_aug, lens, init_params)

                wavs_aug = wavs_aug.reshape([G, params.group_size] + s[1:])
                wavs_aug_list.append(wavs_aug)
            wavs_aug = torch.cat(wavs_aug_list, dim=1)
            # Concatenate corrupted batches as addtional query samples
            wavs = wavs.reshape([G, params.group_size] + s[1:])
            wavs = torch.cat([wavs, wavs_aug], dim=1)
            wavs = wavs.reshape(
                [params.batch_size * (1 + params.num_corrupts)] + s[1:]
            )
            lens = lens.repeat_interleave(1 + params.num_corrupts)

        # Feature extraction and normalization
        feats = params.compute_features(wavs, init_params)
        feats = params.mean_var_norm(feats, lens)

        # Xvector
        x_vect = params.xvector_model(feats, init_params=init_params)

        return x_vect, lens

    def compute_objectives(self, outputs, targets, stage="train"):
        outputs, _ = outputs
        uttid, _, _ = targets

        if stage == "train":
            G = params.batch_size // params.group_size
            K = params.group_size - params.query_size
            Q = params.query_size
            if hasattr(params, "num_corrupts"):
                Q += params.group_size * params.num_corrupts
        else:
            G = params.eval_batch_size // params.eval_group_size
            K = params.eval_group_size - params.eval_query_size
            Q = params.eval_query_size
        # target is one-hot for brevity
        targets = F.one_hot(torch.arange(G)).to(params.device)
        loss, predictions = params.meta_wrapper(outputs, targets.float(), K, Q)

        stats = {}
        if stage != "train":
            # speaker id is not absolute
            spkid = torch.arange(G).repeat_interleave(params.eval_query_size)
            spkid = spkid.to(params.device)
            stats["error"] = params.compute_error(predictions, spkid)

        return loss, stats

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        old_lr, new_lr = params.lr_annealing(
            [params.optimizer], epoch, valid_stats["error"]
        )
        epoch_stats = {"epoch": epoch, "lr": old_lr}
        params.train_logger.log_stats(epoch_stats, train_stats, valid_stats)
        params.checkpointer.save_and_keep_only()


# Data loaders
train_loader = params.train_loader
valid_loader = params.valid_loader

# Xvector Model
modules = [params.xvector_model, params.meta_loss]
first_x, first_y = next(iter(train_loader()))
if hasattr(params, "augmentation"):
    modules.append(params.augmentation)

# Object initialization for training xvector model
xvect_brain = XvectorBrain(
    modules=modules, optimizer=params.optimizer, first_inputs=[first_x],
)
xvect_brain = XvectorBrain(modules=modules, optimizer=params.optimizer,)

# Recover checkpoints
params.checkpointer.recover_if_possible()

# Train the Xvector model
xvect_brain.fit_resample(
    params.epoch_counter, train_loader=train_loader, valid_loader=valid_loader,
)
print("Xvector model training completed!")
