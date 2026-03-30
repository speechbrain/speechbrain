#!/usr/bin/env python3
"""Recipe for training a w2v-BERT-based SENSE model on Common Voice.

The system fine-tunes a w2v-BERT encoder with an attention-pooling head
to predict BGE-M3 sentence embeddings for each utterance, so that speech
and text share a common semantic space.

To run this recipe, do the following:
> python train.py hparams/train_sense.yaml

Authors
 * Maryem Bouziane 2025
 * Salima Mdhaffar 2025
 * Haroun Elleuch 2025
 * Yannick Estève 2025
 * Ha Nguyen 2023
"""

import sys

import pandas as pd
import torch
import torch.nn.functional as F
from common_voice_sense_prepare import prepare_sense
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.sampler import ReproducibleWeightedRandomSampler
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


# Define training procedure
class SenseBrain(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from waveform batches to speech and text embeddings."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        # Student speech encoder: w2v-BERT
        feats = self.modules.wav2vec2(wavs, wav_lens)
        uttr_embeddings = self.modules.attn_pooling(feats)
        # L2-normalise speech embeddings.
        uttr_embeddings = F.normalize(uttr_embeddings, p=2, dim=-1)

        # Teacher text encoder: BGE-M3
        src_text = batch.wrd
        text_embeddings = self.modules.bge_model(src_text)

        return uttr_embeddings, text_embeddings

    def compute_objectives(self, predictions, batch, stage):
        """Cosine-based loss used for semantic alignment between speech and text."""
        uttr_embeddings, text_embeddings = predictions

        cosine_sim = torch.sum(
            uttr_embeddings.float() * text_embeddings.float(), dim=-1
        )

        loss = 1.0 - cosine_sim
        loss = loss.sum()
        loss *= self.hparams.loss_scale
        return loss

    def init_optimizers(self):
        """Initializes optimizers for the attention head and the w2v-BERT encoder."""
        # Optimizer for the attention pooling
        self.adam_optimizer = self.hparams.adam_opt_class(
            self.hparams.model.parameters()
        )
        self.optimizers_dict = {"model_optimizer": self.adam_optimizer}

        # Separate optimizer for w2v-BERT if not frozen
        if not self.hparams.wav2vec2_frozen:
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.wav2vec2.parameters()
            )
            self.optimizers_dict["wav2vec_optimizer"] = self.wav2vec_optimizer

    def freeze_optimizers(self, optimizers):
        """Freezes the wav2vec2 optimizer according to the warmup steps."""
        valid_optimizers = {}
        if not self.hparams.wav2vec2_frozen:
            valid_optimizers["wav2vec_optimizer"] = optimizers[
                "wav2vec_optimizer"
            ]
        return valid_optimizers

    def on_stage_start(self, stage, epoch):
        """Gets called when a stage (either training or validation) starts."""
        return

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of each stage.

        For validation, applies learning rate scheduling and handles
        logging and checkpointing.
        """
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_loss
            return

        # VALID
        stage_stats = {"loss": stage_loss}
        current_epoch = self.hparams.epoch_counter.current

        if stage == sb.Stage.VALID:
            # Scheduler for the attention pooling
            old_lr_adam, new_lr_adam = self.hparams.lr_annealing_adam(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.adam_optimizer, new_lr_adam
            )

            stats_meta = {
                "epoch": current_epoch,
                "lr_adam": old_lr_adam,
            }

            # Scheduler for w2v-BERT if not frozen
            if not self.hparams.wav2vec2_frozen:
                (
                    old_lr_wav2vec,
                    new_lr_wav2vec,
                ) = self.hparams.lr_annealing_wav2vec(stage_stats["loss"])
                sb.nnet.schedulers.update_learning_rate(
                    self.wav2vec_optimizer, new_lr_wav2vec
                )
                stats_meta["lr_wav2vec"] = old_lr_wav2vec

            # Log validation statistics and save the checkpoint
            self.hparams.train_logger.log_stats(
                stats_meta=stats_meta,
                train_stats={"loss": self.train_stats},
                valid_stats=stage_stats,
            )

            meta = {"loss": stage_stats["loss"], "epoch": current_epoch}
            name = "checkpoint_epoch" + str(current_epoch)
            self.checkpointer.save_and_keep_only(
                meta=meta, name=name, num_to_keep=10, min_keys=["loss"]
            )


def dataio_prepare(hparams):
    """Prepares the datasets and data pipelines used by the SenseBrain."""

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Audio pipeline: reads the waveform."""
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    datasets = {}

    # TRAIN
    train_csv = hparams["train_csv"]
    datasets["train"] = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=train_csv,
        dynamic_items=[audio_pipeline],
        output_keys=[
            "id",
            "lang",
            "sig",
            "duration",
            "wrd",
        ],
    )

    # VALID
    valid_csv = hparams["valid_csv"]
    datasets["valid"] = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=valid_csv,
        dynamic_items=[audio_pipeline],
        output_keys=[
            "id",
            "lang",
            "sig",
            "duration",
            "wrd",
        ],
    )

    return datasets


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create ddp_group
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset preparation (TSV -> multilingual train/dev CSV)
    if not hparams["skip_prep"]:
        run_on_main(
            prepare_sense,
            kwargs={
                "data_folder": hparams["data_folder"],
                "output_folder": hparams["output_folder"],
                "languages": hparams["languages"],
                "sampling_alpha": hparams["sampling_alpha"],
                "language_ratios_file": hparams["language_ratios_file"],
                "train_csv": hparams["train_csv"],
                "valid_csv": hparams["valid_csv"],
                "convert_to_wav": hparams["convert_to_wav"],
            },
        )

    # Create main experiment class
    sense_brain = SenseBrain(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Datasets
    datasets = dataio_prepare(hparams)

    # Load sampling ratios from train.csv
    logger.info("Loading language ratios from train.csv ...")
    manifest = pd.read_csv(hparams["train_csv"])
    if "ratio" not in manifest.columns:
        raise RuntimeError(
            "Column 'ratio' is missing in train.csv. "
            "Check that the preparation step ran correctly."
        )

    sample_ratios = list(manifest["ratio"])
    num_samples = len(sample_ratios)

    # Create weighted sampler for the training dataloader
    train_sampler = ReproducibleWeightedRandomSampler(
        sample_ratios,
        replacement=True,
        num_samples=num_samples,
    )

    # inject the sampler in the training dataloader
    train_loader_kwargs = dict(hparams["dataloader_options"])
    train_loader_kwargs["sampler"] = train_sampler

    # Training
    logger.info("Start of model training:")
    sense_brain.fit(
        sense_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=train_loader_kwargs,
        valid_loader_kwargs=hparams["dataloader_options"],
    )
