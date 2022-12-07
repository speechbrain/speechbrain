#!/usr/bin/python3
import os
import sys
import torch
import torchaudio
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from esc50_prepare import prepare_esc50
from train_classifier import dataio_prep
from speechbrain.utils.metric_stats import MetricStats
from os import makedirs
import torch.nn.functional as F
from speechbrain.processing.NMF import spectral_phase


eps = 1e-10


class InterpreterESC50Brain(sb.core.Brain):
    """Class for sound class embedding training" """

    def interpret_computation_steps(self, wavs):
        """computation steps to get the interpretation spectrogram"""
        # compute stft and logmel, and phase
        X_stft = self.modules.compute_stft(wavs)
        X_stft_phase = spectral_phase(X_stft)
        X_stft_power = sb.processing.features.spectral_magnitude(
            X_stft, power=self.hparams.spec_mag_power
        )
        X_logmel = self.modules.compute_fbank(X_stft_power)

        # get the classifier embeddings
        embeddings, f_I = self.hparams.embedding_model(X_logmel)

        # get the nmf activations
        psi_out = self.modules.psi(f_I)

        # cut the length of psi in case necessary
        psi_out = psi_out[:, :, : X_stft_power.shape[1]]

        # get the classifier output
        predictions = self.hparams.classifier(embeddings).squeeze(1)
        pred_cl = torch.argmax(predictions, dim=1)[0].item()
        # print(pred_cl)

        nmf_dictionary = self.hparams.nmf_decoder.return_W()

        # computes time activations per component
        # FROM NOW ON WE FOLLOW THE PAPER'S NOTATION
        psi_out = psi_out.squeeze()
        z = self.modules.theta.hard_att(psi_out).squeeze()
        theta_c_w = self.modules.theta.classifier[0].weight[pred_cl]

        # some might be negative, relevance of component
        r_c_x = theta_c_w * z / torch.abs(theta_c_w * z).max()
        # define selected components by thresholding
        L = torch.arange(r_c_x.shape[0])[r_c_x > 0.2].tolist()

        # get the log power spectra, this is needed as NMF is trained on log-power spectra
        X_stft_power_log = (
            torch.log(X_stft_power + 1).transpose(1, 2).squeeze(0)
        )

        # cem : for the denominator we need to sum over all K, not just the selected ones.
        X_withselected = nmf_dictionary[:, L] @ psi_out[L, :]
        Xhat = nmf_dictionary @ psi_out

        # need the eps for the denominator
        eps = 1e-10
        # X_int = (X_ks / (sum_X_k.unsqueeze(0)+eps)).sum(0) * X_stft_power_log
        X_int = (X_withselected / (Xhat + eps)) * X_stft_power_log

        # get back to the standard stft
        X_int = torch.exp(X_int) - 1
        return X_int, X_stft_phase, pred_cl

    def interpret_sample(self, wavs, batch=None):
        """ get the interpratation for a given wav file."""

        # get the interpretation spectrogram, phase, and the predicted class
        X_int, X_stft_phase, pred_cl = self.interpret_computation_steps(wavs)
        if not (batch is None):
            X_stft_phase_sb = torch.cat(
                (
                    torch.cos(X_stft_phase).unsqueeze(-1),
                    torch.sin(X_stft_phase).unsqueeze(-1),
                ),
                dim=-1,
            )

            temp = X_int.transpose(0, 1).unsqueeze(0).unsqueeze(-1)

            X_wpsb = temp * X_stft_phase_sb
            x_int_sb = self.modules.compute_istft(X_wpsb)

            # save reconstructed and original spectrograms
            makedirs(
                os.path.join(
                    self.hparams.output_folder, f"audios_from_interpretation",
                ),
                exist_ok=True,
            )

            current_class_ind = batch.class_string_encoded.data[0].item()
            current_class_name = self.hparams.label_encoder.ind2lab[
                current_class_ind
            ]
            predicted_class_name = self.hparams.label_encoder.ind2lab[pred_cl]
            torchaudio.save(
                os.path.join(
                    self.hparams.output_folder,
                    f"audios_from_interpretation",
                    f"original_tc_{current_class_name}_pc_{predicted_class_name}.wav",
                ),
                wavs[0].unsqueeze(0),
                self.hparams.sample_rate,
            )

            torchaudio.save(
                os.path.join(
                    self.hparams.output_folder,
                    f"audios_from_interpretation",
                    f"interpretation_tc_{current_class_name}_pc_{predicted_class_name}.wav",
                ),
                x_int_sb,
                self.hparams.sample_rate,
            )

        return X_int

    def overlap_test(self, batch):
        """interpration test with overlapped audio"""
        wavs, _ = batch.sig
        wavs = wavs.to(self.device)

        s1 = wavs[0]
        s2 = wavs[1]

        # create the mixture with s2 being the noise (lower gain)
        mix = (s1 + (s2 * 0.2)).unsqueeze(0)

        # get the interpretation spectrogram, phase, and the predicted class
        X_int, X_stft_phase, pred_cl = self.interpret_computation_steps(mix)

        X_stft_phase_sb = torch.cat(
            (
                torch.cos(X_stft_phase).unsqueeze(-1),
                torch.sin(X_stft_phase).unsqueeze(-1),
            ),
            dim=-1,
        )

        temp = X_int.transpose(0, 1).unsqueeze(0).unsqueeze(-1)

        X_wpsb = temp * X_stft_phase_sb
        x_int_sb = self.modules.compute_istft(X_wpsb)

        # save reconstructed and original spectrograms
        # epoch = self.hparams.epoch_counter.current
        current_class_ind = batch.class_string_encoded.data[0].item()
        current_class_name = self.hparams.label_encoder.ind2lab[
            current_class_ind
        ]
        predicted_class_name = self.hparams.label_encoder.ind2lab[pred_cl]

        noise_class_ind = batch.class_string_encoded.data[1].item()
        noise_class_name = self.hparams.label_encoder.ind2lab[noise_class_ind]

        out_folder = os.path.join(
            self.hparams.output_folder,
            f"overlap_test",
            f"tc_{current_class_name}_nc_{noise_class_name}_pc_{predicted_class_name}",
        )
        makedirs(
            out_folder, exist_ok=True,
        )

        torchaudio.save(
            os.path.join(out_folder, "mixture.wav"),
            mix,
            self.hparams.sample_rate,
        )

        torchaudio.save(
            os.path.join(out_folder, "source.wav"),
            s1.unsqueeze(0),
            self.hparams.sample_rate,
        )

        torchaudio.save(
            os.path.join(out_folder, "noise.wav"),
            s2.unsqueeze(0),
            self.hparams.sample_rate,
        )

        torchaudio.save(
            os.path.join(out_folder, "interpretation.wav"),
            x_int_sb,
            self.hparams.sample_rate,
        )

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + sound classifier.
        Data augmentation and environmental corruption are applied to the
        input sound.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        X_stft = self.modules.compute_stft(wavs)
        X_stft_power = sb.processing.features.spectral_magnitude(
            X_stft, power=self.hparams.spec_mag_power
        )
        X_logmel = self.modules.compute_fbank(X_stft_power)

        # Embeddings + sound classifier
        embeddings, f_I = self.hparams.embedding_model(X_logmel)
        predictions = self.hparams.classifier(embeddings).squeeze(1)

        psi_out = self.modules.psi(f_I)  # generate nmf activations
        # cut the length of psi
        psi_out = psi_out[:, :, : X_stft_power.shape[1]]

        #  generate log-mag spectrogram
        reconstructed = self.hparams.nmf_decoder(psi_out)

        # generate classifications from time activations
        theta_out = self.modules.theta(psi_out)

        if stage == sb.Stage.VALID:
            # save some samples
            if (
                self.hparams.epoch_counter.current
                % self.hparams.interpret_period
            ) == 0 and self.hparams.save_interpretations:
                wavs = wavs[0].unsqueeze(0)
                self.interpret_sample(wavs, batch)
                self.overlap_test(batch)

        return (reconstructed, psi_out), (predictions, theta_out)

    def compute_objectives(self, pred, batch, stage):
        """Computes the loss using class-id as label."""
        (
            (reconstructions, time_activations),
            (classification_out, theta_out),
        ) = pred

        uttid = batch.id
        classid, _ = batch.class_string_encoded

        batch = batch.to(self.device)
        wavs, lens = batch.sig

        X_stft = self.modules.compute_stft(wavs).to(self.device)
        X_stft_power = sb.processing.features.spectral_magnitude(
            X_stft, power=self.hparams.spec_mag_power
        )
        X_stft_logpower = torch.log(X_stft_power + 1).transpose(1, 2)

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.VALID or stage == sb.Stage.TEST:
            self.top_3_fidelity.append(batch.id, theta_out, classification_out)
            self.faithfulness.append(batch.id, wavs, classification_out)
        self.acc_metric.append(
            uttid, predict=classification_out, target=classid, length=lens
        )

        loss_nmf = ((reconstructions - X_stft_logpower) ** 2).mean()
        # loss_nmf = loss_nmf / reconstructions.shape[0]  # avg on batches
        loss_nmf = self.hparams.alpha * loss_nmf
        loss_nmf += self.hparams.beta * (time_activations).abs().mean()

        if stage != sb.Stage.TEST:
            if hasattr(self.hparams.lr_annealing, "on_batch_end"):
                self.hparams.lr_annealing.on_batch_end(self.optimizer)

        self.last_batch = batch
        self.batch_to_plot = (reconstructions.clone(), X_stft_logpower.clone())

        theta_out = -torch.log(theta_out)
        loss_fdi = (F.softmax(classification_out, dim=0) * theta_out).mean()

        return loss_nmf + loss_fdi

    def on_stage_start(self, stage, epoch=None):
        def accuracy_value(predict, target, length):
            """Computes Accuracy"""
            # predict = predict.argmax(1, keepdim=True)
            nbr_correct, nbr_total = sb.utils.Accuracy.Accuracy(
                predict.unsqueeze(1), target, length
            )
            acc = torch.tensor([nbr_correct / nbr_total])
            return acc

        @torch.no_grad()
        def compute_fidelity(theta_out, predictions):
            """ Computes top-`k` fidelity of interpreter. """
            predictions = F.softmax(predictions, dim=1)
            theta_out = F.softmax(theta_out, dim=1)

            pred_cl = torch.argmax(predictions, dim=1)
            k_top = torch.topk(theta_out, k=self.hparams.k_fidelity, dim=1)[1]

            # 1 element for each sample in batch, is 0 if pred_cl is in top k
            temp = (k_top - pred_cl.unsqueeze(1) == 0).sum(1)

            return temp

        @torch.no_grad()
        def compute_faithfulness(wavs, predictions):
            X_stft = self.modules.compute_stft(wavs).to(self.device)
            X_stft_power = sb.processing.features.spectral_magnitude(
                X_stft, power=self.hparams.spec_mag_power
            ).transpose(1, 2)

            X2 = torch.zeros_like(X_stft_power)
            for (i, wav) in enumerate(wavs):
                X2[i] = X_stft_power[i] - self.interpret_sample(
                    wav.unsqueeze(0)
                )

            X2_logmel = self.modules.compute_fbank(X2.transpose(1, 2))

            embeddings, _ = self.hparams.embedding_model(X2_logmel)
            predictions_masked = self.hparams.classifier(embeddings).squeeze(1)

            predictions = F.softmax(predictions, dim=1)
            predictions_masked = F.softmax(predictions_masked, dim=1)

            # get the prediction indices
            pred_cl = predictions.argmax(dim=1, keepdim=True)

            # get the corresponding output probabilities
            predictions_selected = torch.gather(
                predictions, dim=1, index=pred_cl
            )
            predictions_masked_selected = torch.gather(
                predictions_masked, dim=1, index=pred_cl
            )

            faithfulness = (
                predictions_selected - predictions_masked_selected
            ).squeeze()

            return faithfulness

        self.top_3_fidelity = MetricStats(metric=compute_fidelity)
        self.faithfulness = MetricStats(metric=compute_faithfulness)
        self.acc_metric = sb.utils.metric_stats.MetricStats(
            metric=accuracy_value, n_jobs=1
        )
        return super().on_stage_start(stage, epoch)

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.
        Plots in subplots the values of `self.batch_to_plot` and saves the
        plot to the experiment folder. `self.hparams.output_folder`"""

        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            self.train_stats = {
                "loss": self.train_loss,
                "acc": self.acc_metric.summarize("average"),
            }

        if stage == sb.Stage.VALID:
            current_fid = self.top_3_fidelity.summarize("average")
            old_lr, new_lr = self.hparams.lr_annealing(
                [self.optimizer], epoch, -current_fid
            )
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            valid_stats = {
                "loss": stage_loss,
                "acc": self.acc_metric.summarize("average"),
                "top-3_fid": current_fid,
                "faithfulness_median": torch.Tensor(
                    self.faithfulness.scores
                ).median(),
                "faithfulness_mean": torch.Tensor(
                    self.faithfulness.scores
                ).mean(),
            }

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=valid_stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=valid_stats, max_keys=["top-3_fid"]
            )

        if stage == sb.Stage.TEST:
            current_fid = self.top_3_fidelity.summarize("average")
            test_stats = {
                "loss": stage_loss,
                "acc": self.acc_metric.summarize("average"),
                "top-3_fid": current_fid,
                "faithfulness_median": torch.Tensor(
                    self.faithfulness.scores
                ).median(),
                "faithfulness_mean": torch.Tensor(
                    self.faithfulness.scores
                ).mean(),
            }

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch}, test_stats=test_stats
            )


if __name__ == "__main__":

    # # This flag enables the inbuilt cudnn auto-tuner
    # torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

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

    class_labels = list(label_encoder.ind2lab.values())
    print("Class Labels:", class_labels)

    Interpreter_brain = InterpreterESC50Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    if "pretrained_esc50" in hparams:
        run_on_main(hparams["pretrained_esc50"].collect_files)
        hparams["pretrained_esc50"].load_collected()

    # transfer the frozen parts to the model to the device
    hparams["embedding_model"].to(hparams["device"])
    hparams["classifier"].to(hparams["device"])
    hparams["nmf_decoder"].to(hparams["device"])
    hparams["embedding_model"].eval()

    if not hparams["test_only"]:
        Interpreter_brain.fit(
            epoch_counter=Interpreter_brain.hparams.epoch_counter,
            train_set=datasets["train"],
            valid_set=datasets["valid"],
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )
    else:
        # Load the best checkpoint for evaluation
        test_stats = Interpreter_brain.evaluate(
            test_set=datasets["test"],
            min_key="error",
            progressbar=True,
            test_loader_kwargs=hparams["dataloader_options"],
        )
