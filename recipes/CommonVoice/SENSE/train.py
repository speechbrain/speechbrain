#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recipe for fine-tuning a wav2vec / w2v-bert model for semantic enrichment (SENSE).

Training logic aligned with the original SENSE code:
- Same ST Brain structure (wav2vec2 → attention pooling → linear layer → L2 norm)
- Same loss: sum_b (1 - dot(speech_b, text_b)) * loss_scale
- Same ReproducibleWeightedRandomSampler using language ratios

Differences:
- Text embeddings are computed on-the-fly with BGE (FlagEmbedding)
  from the "text" column of the CSV files (train.csv / dev.csv).
- Data preparation starts directly from multilingual Common Voice TSV files
  via `prep.prepare_sense`, with:
    * filtering duration < 10 s,
    * exclusion of empty files,
    * computation of language ratios (alpha),
    * optional conversion to .wav and resampling (convert_to_wav, sample_rate).

Authors
    * Maryem Bouziane 2025
    * Salima Mdhaffar 2025
    * Yannick Estève 2025
    * Haroun Elleuch 2025
    * Ha Nguyen 2025
"""

import logging
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.sampler import ReproducibleWeightedRandomSampler
from speechbrain.utils.distributed import run_on_main

import speechbrain as sb

from preparation import prepare_sense

logger = logging.getLogger(__name__)

# Global variable (same behaviour as the original code)
sample_ratios = None


# ============================
# Brain ST (adapted original logic)
# ============================
class ST(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from waveform batches to speech and text embeddings."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        # wav2vec / w2v-bert module (W2VBert expects (wav, wav_lens))
        feats = self.modules.wav2vec2(wavs, wav_lens)

        # Self-attention pooling + linear projection
        uttr_embeddings = self.modules.attn_pooling(feats)
        linear = self.modules.lin(uttr_embeddings)

        # L2-normalisation
        uttr_embeddings = F.normalize(linear, p=2)

        # ======== TEXT: BGE embeddings on-the-fly ========
        # batch.text comes from the "text" column in CSV files (train/dev)
        src_text = batch.text

        # BGE → NumPy (B, dim); hparams.bge_model is a BGEM3FlagModel
        bge_out = self.hparams.bge_model.encode(src_text)["dense_vecs"]

        # → torch.Tensor(float32) on the correct device
        text_embeddings = torch.from_numpy(bge_out).to(self.device).float()

        return uttr_embeddings, text_embeddings

    def compute_objectives(self, predictions, batch, stage):
        """Computes the cosine-based SENSE loss."""
        uttr_embeddings, text_embeddings = predictions

        B, S = uttr_embeddings.shape
        loss = 0.0
        for b in range(B):
            cosine_sim = torch.dot(
                uttr_embeddings[b].float(), text_embeddings[b].float()
            )
            loss += 1.0 - cosine_sim

        loss *= self.hparams.loss_scale
        return loss

    def init_optimizers(self):
        """Initialises optimizers as in the original SENSE recipe."""
        self.adam_optimizer = self.hparams.adam_opt_class(
            self.hparams.model.parameters()
        )

        self.optimizers_dict = {"model_optimizer": self.adam_optimizer}

        # Initializes the wav2vec2 optimizer if the encoder is not frozen
        if not self.hparams.wav2vec2_frozen:
            self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
                self.modules.wav2vec2.parameters()
            )
            self.optimizers_dict["wav2vec_optimizer"] = self.wav2vec_optimizer

    def freeze_optimizers(self, optimizers):
        """Optionally freezes the wav2vec2 optimizer according to warm-up steps."""
        valid_optimizers = {}
        if not self.hparams.wav2vec2_frozen:
            valid_optimizers["wav2vec_optimizer"] = optimizers["wav2vec_optimizer"]
        return valid_optimizers

    def on_stage_start(self, stage, epoch):
        """Called at the beginning of each stage (train/valid/test)."""
        return

    def on_stage_end(self, stage, stage_loss, epoch):
        """Called at the end of each epoch or test stage."""
        # Store main stats
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_loss

        else:  # valid or test
            stage_stats = {"loss": stage_loss}
            current_epoch = self.hparams.epoch_counter.current

        # Log stats and save checkpoint at the end of validation
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():
            current_epoch = self.hparams.epoch_counter.current
            old_lr_adam, new_lr_adam = self.hparams.lr_annealing_adam(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(self.adam_optimizer, new_lr_adam)

            stats_meta = {
                "epoch": current_epoch,
                "lr_adam": old_lr_adam,
            }

            if not self.hparams.wav2vec2_frozen:
                (
                    old_lr_wav2vec,
                    new_lr_wav2vec,
                ) = self.hparams.lr_annealing_wav2vec(stage_stats["loss"])
                sb.nnet.schedulers.update_learning_rate(
                    self.wav2vec_optimizer, new_lr_wav2vec
                )
                stats_meta["lr_wav2vec"] = old_lr_wav2vec

            self.hparams.train_logger.log_stats(
                stats_meta=stats_meta,
                train_stats={"loss": self.train_stats},
                valid_stats=stage_stats,
            )

            # Create checkpoint
            meta = {"loss": stage_stats["loss"], "epoch": current_epoch}
            name = "checkpoint_epoch" + str(current_epoch)

            self.checkpointer.save_and_keep_only(
                meta=meta, name=name, num_to_keep=10, min_keys=["loss"]
            )

        elif stage == sb.Stage.TEST:
            # Only loss is logged on the evaluation set
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def make_dataloader(
        self, dataset, stage, ckpt_prefix="dataloader-", **loader_kwargs
    ):
        """
        Overrides dataloader creation to obtain a balanced language sampling
        in each batch during training (as in the original SENSE recipe).

        The global variable `sample_ratios` must be filled from train.csv.
        """
        if stage == sb.Stage.TRAIN:
            global sample_ratios
            if sample_ratios is None:
                raise RuntimeError(
                    "sample_ratios is not initialised. "
                    "Check the preparation step and train.csv."
                )
            num_samples = len(sample_ratios)
            sampler = ReproducibleWeightedRandomSampler(
                sample_ratios,
                epoch=self.hparams.epoch_counter.current,
                replacement=True,
                num_samples=num_samples,
            )
            loader_kwargs["sampler"] = sampler

        # Standard dataloader creation
        return super().make_dataloader(dataset, stage, ckpt_prefix, **loader_kwargs)


# ============================
# Data I/O
# ============================
def dataio_prepare(hparams):
    """
    Prepares the datasets used by the ST Brain.

    The original SENSE structure is preserved, but precomputed embeddings
    are replaced by raw text in the CSV files, which is then encoded
    on-the-fly by BGE in compute_forward.
    """

    # Audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Loads the audio signal from disk."""
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def sp_audio_pipeline(wav):
        """Version with speed perturbation (only used if enabled in hparams)."""
        sig = sb.dataio.dataio.read_audio(wav)
        sig = sig.unsqueeze(0)
        sig = hparams["speed_perturb"](sig)
        sig = sig.squeeze(0)
        return sig

    # Text pipeline: pass raw transcription (CSV "text" column)
    @sb.utils.data_pipeline.takes("text")
    @sb.utils.data_pipeline.provides("text")
    def text_pipeline(text):
        yield text

    datasets = {}

    # ===== TRAIN =====
    train_csv = hparams["train_csv"]
    is_use_sp = "speed_perturb" in hparams
    audio_pipeline_func = sp_audio_pipeline if is_use_sp else audio_pipeline

    datasets["train"] = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=train_csv,
        dynamic_items=[
            audio_pipeline_func,
            text_pipeline,
        ],
        output_keys=[
            "id",       # logical SpeechBrain key (CSV column is ID)
            "lang",
            "sig",
            "duration",
            "text",     # used for BGE embeddings
        ],
    )

    # ===== VALID =====
    valid_csv = hparams["valid_csv"]
    datasets["valid"] = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=valid_csv,
        dynamic_items=[
            audio_pipeline,
            text_pipeline,
        ],
        output_keys=[
            "id",
            "lang",
            "sig",
            "duration",
            "text",
        ],
    )

    return datasets


# ============================
# Main
# ============================
if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # ============================
    # Data preparation: TSV → combined train.csv / dev.csv
    # ============================
    if not hparams["skip_prep"]:
        run_on_main(
            prepare_sense,
            kwargs={
                "data_folder": hparams["data_folder"],
                # CSV files and JSON ratios are written directly under output_folder
                "preparation_output_folder": hparams["output_folder"],
                "language_ratios_file": hparams["language_ratios_file"],
                "alpha": hparams.get("sampling_alpha", 0.05),
                "skip_prep": hparams.get("skip_prep", False),
                "convert_to_wav": hparams.get("convert_to_wav", False),
                "sample_rate": hparams.get("sample_rate", 16000),
            },
        )

    # Create main experiment class
    st_brain = ST(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Datasets
    datasets = dataio_prepare(hparams)

    # Load language ratios from train.csv
    logger.info("Loading language ratios from train.csv ...")
    manifest = pd.read_csv(hparams["train_csv"])
    if "ratio" not in manifest.columns:
        raise RuntimeError(
            "Column 'ratio' is missing in train.csv. "
            "Check that the preparation step ran correctly."
        )

    # Global variable used by the sampler (same logic as the original code)
    sample_ratios = list(manifest["ratio"])

    # Training
    logger.info("Start of model training:")
    st_brain.fit(
        st_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )

    # Final evaluation on the validation split (loss only, no WER file)
    logger.info("Final evaluation on the validation split:")
    st_brain.evaluate(
        datasets["valid"],
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
