#!/usr/bin/python
import os
import sys
import torch
import logging
import speechbrain as sb

logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True

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
    seg_dur=params.max_frames if hasattr(params, "max_frames") else 300,
    random_segment=params.random_segment
    if hasattr(params, "random_segment")
    else False,
    source=params.voxceleb_source
    if hasattr(params, "voxceleb_source")
    else None,
)


def _nan_to_zero(t, msg, ids):
    isnan = t.isnan().long()
    if int(isnan.sum()) > 0:
        for idx in isnan.nonzero(as_tuple=True)[0]:
            err_msg = f"{msg}: {ids[idx]}"
            logger.error(err_msg, exc_info=True)
        t = torch.where(t.isnan(), torch.zeros_like(t), t)
    return t


# Trains embedding model
class EmbeddingBrain(sb.core.Brain):
    def compute_forward(self, x, stage="train", init_params=False):
        ids, wavs, lens = x

        wavs, lens = wavs.to(params.device), lens.to(params.device)
        wavs = _nan_to_zero(wavs, "NaN in wav", ids)

        if stage == "train":
            # Addding noise and reverberation
            wavs_aug = params.env_corrupt(wavs, lens, init_params)
            wavs_aug = _nan_to_zero(wavs, "NaN after env_corrupt", ids)

            # Adding time-domain augmentation
            wavs_aug = params.augmentation(wavs_aug, lens, init_params)
            wavs_aug = _nan_to_zero(wavs_aug, "NaN after augmentation", ids)

            # Concatenate noisy and clean batches
            ids = ids + ids
            wavs = torch.cat([wavs, wavs_aug], dim=0)
            lens = torch.cat([lens, lens], dim=0)

        # Feature extraction and normalization
        feats = params.compute_features(wavs, init_params)
        feats = _nan_to_zero(feats, "NaN after compute_features", ids)

        feats = params.mean_var_norm(feats, lens)
        feats = _nan_to_zero(feats, "NaN after mean_var_norm", ids)

        # Embedding + speaker classifier
        emb = params.embedding_model(feats, init_params=init_params)
        emb = _nan_to_zero(emb, "NaN after embedding_model", ids)

        outputs = params.classifier(emb, init_params)
        outputs = _nan_to_zero(outputs, "NaN after classifier", ids)

        return outputs, lens

    def compute_objectives(self, predictions, targets, stage="train"):
        predictions, lens = predictions
        uttid, spkid, _ = targets

        spkid, lens = spkid.to(params.device), lens.to(params.device)

        # Concatenate labels
        if stage == "train":
            uttid = uttid + uttid
            spkid = torch.cat([spkid, spkid], dim=0)

        loss = params.compute_cost(predictions, spkid)
        loss = _nan_to_zero(loss, "NaN after classifier", uttid)

        stats = {}
        if stage != "train":
            stats["error"] = params.compute_error(predictions, spkid, lens)

        params.lr_annealing.on_batch_end([params.optimizer])
        return loss, stats

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        old_lr = params.lr_annealing(
            [params.optimizer], epoch, valid_stats["error"]
        )
        epoch_stats = {"epoch": epoch, "lr": old_lr}
        params.train_logger.log_stats(epoch_stats, train_stats, valid_stats)
        params.checkpointer.save_and_keep_only()


# Data loaders
train_set = params.train_loader()
valid_set = params.valid_loader()

# Embedding Model
modules = [params.embedding_model, params.classifier]
first_x, first_y = next(iter(train_set))
if hasattr(params, "augmentation"):
    modules.append(params.augmentation)

# Object initialization for training embedding model
embedding_brain = EmbeddingBrain(
    modules=modules, optimizer=params.optimizer, first_inputs=[first_x],
)

# Recover checkpoints
params.checkpointer.recover_if_possible()

# Train the Embedding model
embedding_brain.fit(
    params.epoch_counter, train_set=train_set, valid_set=valid_set,
)
print("Embedding model training completed!")
