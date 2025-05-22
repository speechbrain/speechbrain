#!/usr/bin/python3

"""This recipe trains PIQ to interpret an audio classifier.

To run this recipe, use the following command:
> python train_piq.py hparams/<piq-config>.yaml --data_folder /yourpath/ESC-50-master

Authors
    * Cem Subakan 2022, 2023
    * Francesco Paissan 2022, 2023, 2024
    * Luca Della Libera 2024
"""

import sys

import torch
from esc50_prepare import dataio_prep
from hyperpyyaml import load_hyperpyyaml
from interpreter_brain import InterpreterBrain
from torch.nn import functional as F

import speechbrain as sb
from speechbrain.processing.NMF import spectral_phase
from speechbrain.utils.distributed import run_on_main

eps = 1e-10


class PIQ(InterpreterBrain):
    """Class for interpreter training."""

    def interpret_computation_steps(self, wavs, print_probability=False):
        """Computation steps to get the interpretation spectrogram."""
        X_stft_logpower, X_mel, X_stft, X_stft_power = self.preprocess(wavs)
        X_stft_phase = spectral_phase(X_stft)

        hcat, embeddings, predictions, class_pred = self.classifier_forward(
            X_stft_logpower
        )
        if print_probability:
            predictions = F.softmax(predictions, dim=1)
            class_prob = predictions[0, class_pred].item()
            print(f"classifier_prob: {class_prob}")

        if self.hparams.use_vq:
            xhat, hcat, _ = self.modules.psi(hcat, class_pred)
        else:
            xhat = self.modules.psi.decoder(hcat)
        xhat = xhat.squeeze(1)

        Tmax = xhat.shape[1]
        if self.hparams.use_mask_output:
            xhat = F.sigmoid(xhat)
            X_int = xhat * X_stft_logpower[:, :Tmax, :]
        else:
            xhat = F.softplus(xhat)
            th = xhat.max() * self.hparams.mask_th
            X_int = (xhat > th) * X_stft_logpower[:, :Tmax, :]

        return X_int.permute(0, 2, 1), xhat.permute(0, 2, 1), X_stft_phase

    def compute_forward(self, batch, stage):
        """Computation pipeline based on an encoder + sound classifier."""
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        X_stft_logpower, X_mel, X_stft, X_stft_power = self.preprocess(wavs)

        # Embeddings + sound classifier
        hcat, embeddings, predictions, class_pred = self.classifier_forward(
            X_stft_logpower
        )

        if self.hparams.use_vq:
            xhat, hcat, z_q_x = self.modules.psi(hcat, class_pred)
        else:
            xhat = self.modules.psi.decoder(hcat)
            z_q_x = None

        xhat = xhat.squeeze(1)

        if self.hparams.use_mask_output:
            xhat = F.sigmoid(xhat)
        else:
            xhat = F.softplus(xhat)

        garbage = 0

        if stage == sb.Stage.VALID:
            # Save some samples
            if (
                self.hparams.epoch_counter.current
                % self.hparams.interpret_period
            ) == 0 and self.hparams.save_interpretations:
                self.viz_ints(X_stft, X_stft_logpower, batch, wavs)

        return predictions, xhat, hcat, z_q_x, garbage

    def compute_objectives(self, pred, batch, stage):
        """Helper function to compute the objectives."""
        predictions, xhat, hcat, z_q_x, garbage = pred

        batch = batch.to(self.device)
        wavs, lens = batch.sig

        uttid = batch.id
        classid, _ = batch.class_string_encoded

        X_stft_logpower, X_mel, X_stft, X_stft_power = self.preprocess(wavs)

        Tmax = xhat.shape[1]

        hcat_theta, embeddings, theta_out, _ = self.classifier_forward(
            xhat * X_stft_logpower[:, :Tmax, :]
        )
        mask_in_preds = theta_out
        mask_out_preds = self.classifier_forward(
            (1 - xhat) * X_stft_logpower[:, :Tmax, :]
        )[2]

        # If there is a separator, we need to add sigmoid to the sum
        loss_fid = 0

        if self.hparams.use_mask_output:
            eps = 1e-10
            target_spec = X_stft_logpower[:, : xhat.shape[1], :]
            target_mask = target_spec > (
                target_spec.max(keepdim=True, dim=-1)[0].max(
                    keepdim=True, dim=-2
                )[0]
                * self.hparams.mask_th
            )
            target_mask = target_mask.float()
            rec_loss = (
                -target_mask * torch.log(xhat + eps)
                - (1 - target_mask) * torch.log(1 - xhat + eps)
            ).mean()
        else:
            rec_loss = (
                (X_stft_logpower[:, : xhat.shape[1], :] - xhat).pow(2).mean()
            )

        if self.hparams.use_vq:
            loss_vq = F.mse_loss(z_q_x, hcat.detach())
            loss_commit = F.mse_loss(hcat, z_q_x.detach())
        else:
            loss_vq = 0
            loss_commit = 0

        self.acc_metric.append(uttid, predict=predictions, target=classid)
        if stage == sb.Stage.VALID or stage == sb.Stage.TEST:
            self.inp_fid.append(
                uttid,
                mask_in_preds.softmax(1),
                predictions.softmax(1),
            )

            self.AD.append(
                uttid,
                mask_in_preds.softmax(1),
                predictions.softmax(1),
            )
            self.AI.append(
                uttid,
                mask_in_preds.softmax(1),
                predictions.softmax(1),
            )
            self.AG.append(
                uttid,
                mask_in_preds.softmax(1),
                predictions.softmax(1),
            )
            self.sps.append(uttid, wavs, X_stft_logpower, classid)
            self.comp.append(uttid, wavs, X_stft_logpower, classid)
            self.faithfulness.append(
                uttid,
                predictions.softmax(1),
                mask_out_preds.softmax(1),
            )

        if stage != sb.Stage.TEST:
            if hasattr(self.hparams.lr_annealing, "on_batch_end"):
                self.hparams.lr_annealing.on_batch_end(self.optimizer)

        return (
            self.hparams.rec_loss_coef * rec_loss
            + loss_vq
            + loss_commit
            + loss_fid
        )


if __name__ == "__main__":
    # This flag enables the built-in cuDNN auto-tuner
    # torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    print("Inherited hparams:")
    print("use_melspectra_log1p=", hparams["use_melspectra_log1p"])

    print(
        "Interpreter class is inheriting the train_logger",
        hparams["train_logger"],
    )

    # Classifier is fixed here
    hparams["embedding_model"].eval()
    hparams["classifier"].eval()

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Tensorboard logging
    if hparams["use_tensorboard"]:
        from speechbrain.utils.train_logger import TensorboardLogger

        hparams["tensorboard_train_logger"] = TensorboardLogger(
            hparams["tensorboard_logs_folder"]
        )

    from esc50_prepare import prepare_esc50

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

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    datasets, label_encoder = dataio_prep(hparams)
    hparams["label_encoder"] = label_encoder

    class_labels = list(label_encoder.ind2lab.values())
    print("Class Labels:", class_labels)

    Interpreter_brain = PIQ(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    if "pretrained_esc50" in hparams and hparams["use_pretrained"]:
        print("Loading model...")
        run_on_main(hparams["pretrained_esc50"].collect_files)
        hparams["pretrained_esc50"].load_collected()

    hparams["embedding_model"].to(run_opts["device"])
    hparams["classifier"].to(run_opts["device"])
    hparams["embedding_model"].eval()

    Interpreter_brain.fit(
        epoch_counter=Interpreter_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    Interpreter_brain.checkpointer.recover_if_possible(
        max_key="valid_top-3_fid",
    )

    test_stats = Interpreter_brain.evaluate(
        test_set=datasets["test"],
        min_key="loss",
        progressbar=True,
        test_loader_kwargs=hparams["dataloader_options"],
    )
