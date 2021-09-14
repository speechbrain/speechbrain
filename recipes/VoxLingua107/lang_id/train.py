#!/usr/bin/python3
"""Recipe for training language embeddings using the VoxLingua107 Dataset.

This recipe is heavily inspired by this: https://github.com/nikvaessen/speechbrain/tree/sharded-voxceleb/my-recipes/SpeakerRec

To run this recipe, use the following command:
> python train_lang_embeddings_wds.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:

    hparams/train_epaca_tdnn_wds.yaml (for the ecapa+tdnn system)

Author
    * Mirco Ravanelli 2020
    * Hwidong Na 2020
    * Nauman Dawalatabad 2020
    * Tanel Alumäe 2021
    * @nikvaessen
"""
import os
import sys
import random
from typing import Dict
import json
from functools import partial
import webdataset as wds
import logging

import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.batch import PaddedBatch

logger = logging.getLogger(__name__)


class LanguageBrain(sb.core.Brain):
    """Class for language ID training"
    """

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + speaker classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        if stage == sb.Stage.TRAIN:

            # Applying the augmentation pipeline
            wavs_aug_tot = []
            wavs_aug_tot.append(wavs)
            for count, augment in enumerate(self.hparams.augment_pipeline):

                # Apply augment
                wavs_aug = augment(wavs, lens)

                # Managing speed change
                if wavs_aug.shape[1] > wavs.shape[1]:
                    wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
                else:
                    zero_sig = torch.zeros_like(wavs)
                    zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                    wavs_aug = zero_sig

                if self.hparams.concat_augment:
                    wavs_aug_tot.append(wavs_aug)
                else:
                    wavs = wavs_aug
                    wavs_aug_tot[0] = wavs

            wavs = torch.cat(wavs_aug_tot, dim=0)
            self.n_augment = len(wavs_aug_tot)
            lens = torch.cat([lens] * self.n_augment)

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        # Embeddings + speaker classifier
        embeddings = self.modules.embedding_model(feats, lens)
        outputs = self.modules.classifier(embeddings)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        predictions, lens = predictions
        uttid = batch.id
        langid = batch.lang_id_encoded

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN:
            langid = torch.cat([langid] * self.n_augment, dim=0)

        # breakpoint()
        loss = self.hparams.compute_cost(predictions, langid.unsqueeze(1), lens)

        if hasattr(self.hparams.lr_annealing, "on_batch_end"):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(
                uttid, predictions, langid.unsqueeze(1), lens
            )

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )


def dataio_prep_shards(hparams):

    # load the meta info json file
    with wds.gopen.gopen(hparams["train_meta"], "rb") as f:
        train_meta = json.load(f)
    with wds.gopen.gopen(hparams["val_meta"], "rb") as f:
        val_meta = json.load(f)

    # define the mapping functions in the data pipeline
    snt_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_iterables=[train_meta["language_ids"]],
        output_key="lang_id",
    )
    # breakpoint()

    def audio_pipeline(sample_dict: Dict, random_chunk=True):
        key = sample_dict["__key__"]
        language_id = sample_dict["language_id"].decode("ascii")
        audio_tensor = sample_dict["audio.pth"]

        # determine what part of audio sample to use
        audio_tensor = audio_tensor.squeeze()

        if random_chunk:
            if len(audio_tensor) - snt_len_sample - 1 <= 0:
                start = 0
            else:
                start = random.randint(
                    0, len(audio_tensor) - snt_len_sample - 1
                )

            stop = start + snt_len_sample
        else:
            start = 0
            stop = len(audio_tensor)

        sig = audio_tensor[start:stop]

        # determine the language ID of the sample
        lang_id_idx = label_encoder.encode_label(language_id)

        return {
            "sig": sig,
            "lang_id_encoded": lang_id_idx,
            "id": key,
        }

    train_data = (
        wds.WebDataset(
            hparams["train_shards"], cache_dir=hparams["shard_cache_dir"],
        )
        .repeat()
        .shuffle(1000)
        .decode("pil")
        .map(partial(audio_pipeline, random_chunk=True))
    )
    logger.info(
        f"Training data consist of {train_meta['num_data_samples']} samples"
    )

    valid_data = (
        wds.WebDataset(
            hparams["val_shards"], cache_dir=hparams["shard_cache_dir"],
        )
        .decode("pil")
        .map(partial(audio_pipeline, random_chunk=False))
    )
    logger.info(
        f"Validation data consist of {val_meta['num_data_samples']} samples"
    )

    return (
        train_data,
        valid_data,
        train_meta["num_data_samples"],
        val_meta["num_data_samples"],
    )


if __name__ == "__main__":

    logger.info("Starting training...")
    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    (
        train_data,
        valid_data,
        num_train_samples,
        num_valid_samples,
    ) = dataio_prep_shards(hparams)

    # add collate_fn to dataloader options
    hparams["train_dataloader_options"]["collate_fn"] = PaddedBatch
    hparams["val_dataloader_options"]["collate_fn"] = PaddedBatch

    hparams["train_dataloader_options"]["looped_nominal_epoch"] = (
        num_train_samples // hparams["train_dataloader_options"]["batch_size"]
    )

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Brain class initialization
    language_brain = LanguageBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    language_brain.fit(
        language_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_options"],
        valid_loader_kwargs=hparams["val_dataloader_options"],
    )
