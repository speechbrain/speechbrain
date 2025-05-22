#!/usr/bin/python3
"""This recipe to train L2I (https://arxiv.org/abs/2202.11479) to interpret audio classifiers.

The command to run this recipe:
    python train_l2i.py hparams/l2i_cnn14.yaml --data_folder /yourpath/ESC50

Authors
    * Cem Subakan 2022, 2023
    * Francesco Paissan 2022, 2023, 2024
"""
import sys

import torch
import torch.nn.functional as F
from esc50_prepare import dataio_prep, prepare_esc50
from hyperpyyaml import load_hyperpyyaml
from interpreter_brain import InterpreterBrain
from wham_prepare import combine_batches, prepare_wham

import speechbrain as sb
from speechbrain.processing.NMF import spectral_phase
from speechbrain.utils.distributed import run_on_main

eps = 1e-10


class L2I(InterpreterBrain):
    """Class for sound class embedding training" """

    def interpret_computation_steps(self, wavs_batch):
        """computation steps to get the interpretation spectrogram"""
        X_stft = self.modules.compute_stft(wavs_batch[0:1])
        ret_X_int = torch.empty(
            wavs_batch.shape[0], X_stft.shape[2], X_stft.shape[1]
        ).to(wavs_batch.device)
        ret_mask = torch.empty(
            wavs_batch.shape[0], X_stft.shape[2], X_stft.shape[1]
        ).to(wavs_batch.device)
        ret_X_stft_phase = torch.empty(
            wavs_batch.shape[0], X_stft.shape[1], X_stft.shape[2]
        ).to(wavs_batch.device)
        for idx, wavs in enumerate(wavs_batch):
            # compute stft and logmel, and phase
            wavs = wavs[None]
            X_stft_logpower, X_mel, X_stft, _ = self.preprocess(wavs)

            net_input = X_stft_logpower
            if self.hparams.use_melspectra_log1p:
                net_input = X_mel

            X_stft_phase = spectral_phase(X_stft)

            # get the classifier embeddings
            temp = self.hparams.embedding_model(net_input)

            if isinstance(
                temp, tuple
            ):  # if embeddings are not used for interpretation
                embeddings, f_I = temp
            else:
                embeddings, f_I = temp, temp

            # get the nmf activations
            psi_out = self.modules.psi(f_I)

            if isinstance(psi_out, tuple):
                psi_out = psi_out[0]
                psi_out = psi_out.squeeze(1).permute(0, 2, 1)

            # cut the length of psi in case necessary
            psi_out = psi_out[:, :, : net_input.shape[1]]

            # get the classifier output
            if embeddings.ndim == 4:
                embeddings = embeddings.mean((-1, -2))

            predictions = self.hparams.classifier(embeddings).squeeze(1)
            pred_cl = torch.argmax(predictions, dim=1)[0].item()

            nmf_dictionary = self.hparams.nmf_decoder.return_W()

            # computes time activations per component
            # FROM NOW ON WE FOLLOW THE PAPER'S NOTATION
            psi_out = psi_out.squeeze()
            z = self.modules.theta.hard_att(psi_out).squeeze()
            theta_c_w = self.modules.theta.classifier[0].weight[pred_cl]

            # some might be negative, relevance of component
            r_c_x = theta_c_w * z / torch.abs(theta_c_w * z).max()

            # define selected components by thresholding
            L = (
                torch.arange(r_c_x.shape[0])
                .to(r_c_x.device)[r_c_x > self.hparams.relevance_th]
                .tolist()
            )

            # get the log power spectra, this is needed as NMF is trained on log-power spectra
            X_stft_logpower = X_stft_logpower.transpose(1, 2).squeeze(0)

            X_withselected = nmf_dictionary[:, L] @ psi_out[L, :]
            Xhat = nmf_dictionary @ psi_out

            X_stft_power_log = X_stft_logpower[..., : Xhat.shape[1]]

            # need the eps for the denominator
            mask = X_withselected / (Xhat + eps)
            X_int = mask * X_stft_power_log

            pad_time = X_stft_logpower.shape[1] - Xhat.shape[1]
            X_int = F.pad(X_int, (0, pad_time))
            mask = F.pad(mask, (0, pad_time))

            ret_X_int[idx] = X_int
            ret_mask[idx] = mask
            ret_X_stft_phase[idx] = X_stft_phase

        return ret_X_int, ret_mask, ret_X_stft_phase, X_stft_logpower[None]

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + sound classifier.
        Data augmentation and environmental corruption are applied to the
        input sound.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        if self.hparams.add_wham_noise:
            # augment batch with WHAM!
            wavs = combine_batches(wavs, iter(self.hparams.wham_dataset))

        net_input, X_mel, X_stft, _ = self.preprocess(wavs)

        if self.hparams.use_melspectra_log1p:
            net_input = X_mel

        # Embeddings + sound classifier
        temp = self.hparams.embedding_model(net_input)
        if isinstance(temp, tuple):
            embeddings, f_I = temp
        else:
            embeddings, f_I = temp, temp

        if embeddings.ndim == 4:
            embeddings = embeddings.mean((-1, -2))

        predictions = self.hparams.classifier(embeddings).squeeze(1)

        psi_out = self.modules.psi(f_I)  # generate nmf activations

        if isinstance(psi_out, tuple):
            psi_out = psi_out[0]
            psi_out = psi_out.squeeze(1).permute(0, 2, 1)

        # cut the length of psi
        psi_out = psi_out[:, :, : net_input.shape[1]]

        #  generate log-mag spectrogram
        reconstructed = self.hparams.nmf_decoder(psi_out).transpose(1, 2)

        # generate classifications from time activations
        theta_out = self.modules.theta(psi_out)

        if stage == sb.Stage.VALID:
            # save some samples
            if (
                self.hparams.epoch_counter.current
                % self.hparams.interpret_period
            ) == 0 and self.hparams.save_interpretations:
                self.viz_ints(X_stft, net_input, batch, wavs)

        if stage == sb.Stage.TEST and self.hparams.save_interpretations:
            # During TEST save always, if required
            self.viz_ints(X_stft, net_input, batch, wavs)

        return (reconstructed, psi_out), (predictions, theta_out), wavs

    def compute_objectives(self, pred, batch, stage):
        """Computes the loss using class-id as label."""
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        (
            (reconstructions, time_activations),
            (classification_out, theta_out),
            # take augmented wavs
            wavs,
        ) = pred

        uttid = batch.id
        classid, _ = batch.class_string_encoded

        X_stft = self.modules.compute_stft(wavs).to(self.device)
        X_stft_power = sb.processing.features.spectral_magnitude(
            X_stft, power=self.hparams.spec_mag_power
        )
        X_stft_logpower = torch.log1p(X_stft_power)

        with torch.no_grad():
            tmp, _, _, _ = self.interpret_computation_steps(
                wavs
            )  # returns log1p
            interpretations = torch.expm1(tmp).transpose(2, 1)

            if self.hparams.use_melspectra_log1p:
                interpretations = self.hparams.compute_fbank(interpretations)
                interpretations = torch.log1p(interpretations)

            # Embeddings + sound classifier
            temp = self.hparams.embedding_model(interpretations)
            if isinstance(temp, tuple):
                embeddings, _ = temp
            else:
                embeddings, _ = temp, temp

            if embeddings.ndim == 4:
                embeddings = embeddings.mean((-1, -2))

            maskin_preds = (
                self.hparams.classifier(embeddings).squeeze(1).softmax(1)
            )

            X_stft_logpower = X_stft_logpower[:, : interpretations.shape[-2], :]
            if self.hparams.use_melspectra_log1p:
                xx_temp = torch.log1p(self.hparams.compute_fbank(X_stft_power))
                temp = self.hparams.embedding_model(xx_temp - interpretations)
            else:
                temp = self.hparams.embedding_model(
                    X_stft_logpower - interpretations
                )

            if isinstance(temp, tuple):
                embeddings, _ = temp
            else:
                embeddings, _ = temp, temp

            if embeddings.ndim == 4:
                embeddings = embeddings.mean((-1, -2))

            maskout_preds = (
                self.hparams.classifier(embeddings).squeeze(1).softmax(1)
            )
        self.l2i_fid.append(uttid, theta_out, classid)
        self.inp_fid.append(uttid, maskin_preds, classid)

        self.acc_metric.append(
            uttid,
            predict=classification_out,
            target=classid,
        )

        self.AD.append(
            uttid,
            maskin_preds,
            classification_out.softmax(1),
        )
        self.AI.append(
            uttid,
            maskin_preds,
            classification_out.softmax(1),
        )
        self.AG.append(
            uttid,
            maskin_preds,
            classification_out.softmax(1),
        )
        self.faithfulness.append(
            uttid,
            classification_out.softmax(1),
            maskout_preds,
        )

        if stage == sb.Stage.VALID or stage == sb.Stage.TEST:
            try:
                self.sps.append(uttid, wavs, X_stft_logpower, classid)
            except ValueError:
                print("zero sps entry!")

            try:
                self.comp.append(uttid, wavs, X_stft_logpower, classid)
            except ValueError:
                print("zero comp entry!")

        X_stft_logpower = X_stft_logpower[:, : reconstructions.shape[1], :]

        loss_nmf = ((reconstructions - X_stft_logpower) ** 2).mean()
        self.recons_err.append(uttid, loss_nmf)

        loss_nmf = self.hparams.alpha * loss_nmf
        prev = loss_nmf.clone().detach()

        loss_nmf += self.hparams.beta * (time_activations).abs().mean()
        self.reg_loss.append(uttid, loss_nmf - prev)

        if stage != sb.Stage.TEST:
            if hasattr(self.hparams.lr_annealing, "on_batch_end"):
                self.hparams.lr_annealing.on_batch_end(self.optimizer)

        self.last_batch = batch
        self.batch_to_plot = (reconstructions.clone(), X_stft_logpower.clone())

        theta_out = -torch.log(theta_out)
        loss_fdi = (
            F.softmax(classification_out / self.hparams.classifier_temp, dim=1)
            * theta_out
        ).mean()

        self.fid_loss.append(uttid, loss_fdi)

        return loss_nmf + loss_fdi

    def extra_metrics(self):
        """This function defines the extra metrics required for the L2I method"""

        @torch.no_grad()
        def l2i_fid(predict, target):
            """Computes Accuracy"""
            predict = predict.argmax(1)
            return (predict.unsqueeze(1) == target).float().squeeze(1)

        @torch.no_grad()
        def save(x):
            return x[None]

        return {
            "l2i_fid": l2i_fid,
            "recons_err": save,
            "reg_loss": save,
            "fid_loss": save,
        }

    def pretrained_interpreter(self):
        """This function enables us to use hparams.pretrained_interpreter inside train.py"""
        print(
            f"pretrained_interpreter path {self.hparams.pretrained_interpreter}"
        )


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    print("Eval only hparams:")
    print("overlap_type=", hparams["overlap_type"])
    print("int_method=", hparams["int_method"])
    print("ljspeech_path=", hparams["ljspeech_path"])
    print("single_sample=", hparams["single_sample"])

    print(
        "Interpreter class is inheriting the train_logger",
        hparams["train_logger"],
    )

    # classifier is fixed here
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

    # create WHAM dataset according to hparams
    if "wham_folder" in hparams:
        hparams["wham_dataset"] = prepare_wham(
            hparams["wham_folder"],
            hparams["add_wham_noise"],
            hparams["sample_rate"],
            hparams["signal_length_s"],
            hparams["wham_audio_folder"],
        )

    class_labels = list(label_encoder.ind2lab.values())
    print("Class Labels:", class_labels)

    assert hparams["signal_length_s"] == 5, "Fix wham sig length!"
    # assert hparams["out_n_neurons"] == 50, "Fix number of outputs classes!"

    Interpreter_brain = L2I(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    if "pretrained_esc50" in hparams and hparams["use_pretrained"]:
        run_on_main(hparams["pretrained_esc50"].collect_files)
        hparams["pretrained_esc50"].load_collected()

    # transfer the frozen parts to the model to the device
    hparams["embedding_model"].to(run_opts["device"])
    hparams["classifier"].to(run_opts["device"])
    hparams["nmf_decoder"].to(run_opts["device"])
    hparams["embedding_model"].eval()

    Interpreter_brain.fit(
        epoch_counter=Interpreter_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    test_stats = Interpreter_brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        progressbar=True,
        test_loader_kwargs=hparams["dataloader_options"],
    )
