#!/usr/bin/python3
"""The recipe to train an NMF model with amortized inference on ESC50 data.

To run this recipe, use the following command:
> python train_nmf.py hparams/nmf.yaml --data_folder /yourpath/ESC-50-master

Authors
    * Cem Subakan 2022, 2023
    * Francesco Paissan 2022, 2023
"""
import os
import sys

import matplotlib.pyplot as plt
import torch
from esc50_prepare import prepare_esc50
from hyperpyyaml import load_hyperpyyaml
from train_l2i import dataio_prep

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main


class NMFBrain(sb.core.Brain):
    """
    The SpeechBrain class to train Non-Negative Factorization with Amortized Inference
    """

    def compute_forward(self, batch, stage=sb.Stage.TRAIN):
        """
        This function calculates the forward pass for NMF
        """

        batch = batch.to(self.device)
        wavs, _ = batch.sig

        X_stft = self.hparams.compute_stft(wavs)
        X_stft_power = self.hparams.compute_stft_mag(X_stft)
        X_stft_tf = torch.log1p(X_stft_power)
        z = self.hparams.nmf_encoder(X_stft_tf.permute(0, 2, 1))
        Xhat = self.hparams.nmf_decoder(z)

        # returning wavs because they are augmented
        return Xhat, wavs

    def compute_objectives(self, predictions, batch, stage=sb.Stage.TRAIN):
        """
        this function computes the l2-error to train the NMF model.
        """
        # extracting augmented wavs
        predictions, wavs = predictions

        X_stft = self.hparams.compute_stft(wavs)
        X_stft_power = self.hparams.compute_stft_mag(X_stft)
        target = torch.log1p(X_stft_power).permute(0, 2, 1)

        loss = ((target.squeeze() - predictions) ** 2).mean()

        with torch.no_grad():
            if (
                self.hparams.epoch_counter.current % self.hparams.save_period
                == 0
                and stage == sb.Stage.VALID
            ):
                os.makedirs("nmf_rec", exist_ok=True)
                for idx in range(X_stft.shape[0]):
                    tmp = os.path.join("nmf_rec", f"{idx}.png")
                    plt.subplot(121)
                    plt.imshow(target[idx].cpu(), origin="lower")

                    plt.subplot(122)
                    plt.imshow(predictions[idx].cpu(), origin="lower")

                    plt.tight_layout()
                    plt.savefig(tmp)

        return loss

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            self.train_stats = {
                "loss": self.train_loss,
            }
        # Summarize Valid statistics from the stage for record-keeping.
        elif stage == sb.Stage.VALID:
            valid_stats = {
                "loss": stage_loss,
            }
        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch},
                train_stats=self.train_stats,
                valid_stats=valid_stats,
            )
            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=valid_stats, min_keys=["loss"]
            )


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    assert hparams["signal_length_s"] == 5, "Fix wham sig length!"

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    run_on_main(
        prepare_esc50,
        kwargs={
            "data_folder": hparams["data_folder"],
            "audio_data_folder": hparams["audio_data_folder"],
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "save_json_test": hparams["test_annotation"],
            "train_fold_nums": hparams["train_fold_nums"],
            "valid_fold_nums": hparams["valid_fold_nums"],
            "test_fold_nums": hparams["test_fold_nums"],
            "skip_manifest_creation": hparams["skip_manifest_creation"],
        },
    )

    datasets, _ = dataio_prep(hparams)

    nmfbrain = NMFBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    nmfbrain.fit(
        epoch_counter=nmfbrain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    test_stats = nmfbrain.evaluate(
        test_set=datasets["test"],
        min_key="loss",
        progressbar=True,
        test_loader_kwargs=hparams["dataloader_options"],
    )

    if hparams["save_nmfdictionary"]:
        torch.save(hparams["nmf_decoder"].return_W(), hparams["nmf_savepath"])
