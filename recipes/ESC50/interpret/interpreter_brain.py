"""This is a parent class for the interpretability recipes.

Authors
    * Francesco Paissan 2022, 2023, 2024
    * Cem Subakan 2022, 2023, 2024
    * Luca Della Libera 2024
"""

import os

import matplotlib.pyplot as plt
import torch
import torchaudio
import torchvision
from torch.nn import functional as F

import speechbrain as sb
from speechbrain.processing.NMF import spectral_phase
from speechbrain.utils.metric_stats import MetricStats

eps = 1e-10


class InterpreterBrain(sb.core.Brain):
    """Class for interpreter training."""

    def invert_stft_with_phase(self, X_int, X_stft_phase):
        """Inverts STFT spectra given phase."""
        X_stft_phase_sb = torch.cat(
            (
                torch.cos(X_stft_phase).unsqueeze(-1),
                torch.sin(X_stft_phase).unsqueeze(-1),
            ),
            dim=-1,
        )

        X_stft_phase_sb = X_stft_phase_sb[:, : X_int.shape[1], :, :]
        if X_int.ndim == 3:
            X_int = X_int.unsqueeze(-1)
        X_wpsb = X_int * X_stft_phase_sb
        x_int_sb = self.modules.compute_istft(X_wpsb)

        return x_int_sb

    def preprocess(self, wavs):
        """Pre-process wavs."""
        X_stft = self.modules.compute_stft(wavs)
        X_stft_power = sb.processing.features.spectral_magnitude(
            X_stft, power=self.hparams.spec_mag_power
        )
        X_stft_logpower = torch.log1p(X_stft_power)

        return X_stft_logpower, X_stft, X_stft_power

    @torch.no_grad()
    def classifier_forward(self, X_stft_logpower):
        """The forward pass for the classifier."""
        if hasattr(self.hparams.embedding_model, "config"):
            # Hugging Face model
            config = self.hparams.embedding_model.config
            # Resize to match expected resolution
            net_input = torchvision.transforms.functional.resize(
                X_stft_logpower, (config.image_size, config.image_size)
            )
            # Expand to have 3 channels
            net_input = net_input[:, None, ...].expand(-1, 3, -1, -1)
            if config.model_type == "focalnet":
                hcat = self.hparams.embedding_model(net_input).feature_maps[-1]
                embeddings = hcat.mean(dim=(-1, -2))
                # Upsample spatial dimensions by 2x to avoid OOM (otherwise the psi model is too large)
                hcat = torchvision.transforms.functional.resize(
                    hcat, (2 * hcat.shape[-2], 2 * hcat.shape[-1])
                )
            elif config.model_type == "vit":
                hcat = self.hparams.embedding_model(
                    net_input
                ).last_hidden_state.movedim(-1, -2)
                embeddings = hcat.mean(dim=-1)
                # Reshape to have 2 spatial dimensions (remove CLS token)
                num_patches = (
                    self.hparams.embedding_model.config.image_size
                    // self.hparams.embedding_model.config.patch_size
                )
                hcat = hcat[..., 1:].reshape(
                    len(hcat), -1, num_patches, num_patches
                )
            else:
                raise NotImplementedError
        else:
            hcat = self.hparams.embedding_model(X_stft_logpower)
            embeddings = hcat.mean((-1, -2))

        predictions = self.hparams.classifier(embeddings).squeeze(1)
        class_pred = predictions.argmax(1)

        return hcat, embeddings, predictions, class_pred

    def interpret_computation_steps(self, wavs, print_probability=False):
        """Computation steps to get the interpretation spectrogram."""

    def extra_metrics(self):
        return {}

    def viz_ints(self, X_stft, xhat, X_stft_logpower, batch, wavs):
        """Helper function to visualize images."""
        X_stft_phase = spectral_phase(X_stft)
        temp = xhat[0].transpose(0, 1).unsqueeze(0).unsqueeze(-1)
        Xspec_est = torch.expm1(temp.permute(0, 2, 1, 3))
        xhat_tm = self.invert_stft_with_phase(Xspec_est, X_stft_phase)

        Tmax = Xspec_est.shape[1]
        if self.hparams.use_mask_output:
            X_masked = xhat[0] * X_stft_logpower[0, :Tmax, :]
        else:
            th = xhat[0].max() * 0.15
            X_masked = (xhat[0] > th) * X_stft_logpower[0, :Tmax, :]

        X_est_masked = torch.expm1(X_masked).unsqueeze(0).unsqueeze(-1)
        xhat_tm_masked = self.invert_stft_with_phase(X_est_masked, X_stft_phase)

        plt.figure(figsize=(10, 5), dpi=100)

        plt.subplot(141)
        X_target = X_stft_logpower[0].permute(1, 0)[:, : xhat.shape[1]].cpu()
        plt.imshow(X_target, origin="lower")
        plt.title("input")
        plt.colorbar(fraction=0.05)

        plt.subplot(142)
        input_masked = X_target > (
            X_target.max(keepdim=True, dim=-1)[0].max(keepdim=True, dim=-2)[0]
            * self.hparams.mask_th
        )
        plt.imshow(input_masked, origin="lower")
        plt.title("input masked")
        plt.colorbar(fraction=0.05)

        plt.subplot(143)
        if self.hparams.use_mask_output:
            mask = xhat[0]
        else:
            mask = xhat[0] > th

        X_masked = mask * X_stft_logpower[0, :Tmax, :]
        plt.imshow(X_masked.permute(1, 0).data.cpu(), origin="lower")
        plt.colorbar(fraction=0.05)
        plt.title("interpretation")

        plt.subplot(144)
        plt.imshow(mask.permute(1, 0).data.cpu(), origin="lower")
        plt.colorbar(fraction=0.05)
        plt.title("estimated mask")

        out_folder = os.path.join(
            self.hparams.output_folder,
            "interpretations",
            f"{batch.id[0]}",
        )
        os.makedirs(
            out_folder,
            exist_ok=True,
        )

        plt.subplots_adjust()
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_folder, "viz.png"),
            bbox_inches="tight",
        )
        plt.close()

        torchaudio.save(
            os.path.join(out_folder, "xhat.wav"),
            xhat_tm.data.cpu(),
            self.hparams.sample_rate,
        )

        torchaudio.save(
            os.path.join(out_folder, "int.wav"),
            xhat_tm_masked.data.cpu(),
            self.hparams.sample_rate,
        )

        torchaudio.save(
            os.path.join(out_folder, "inp.wav"),
            wavs[0:1].data.cpu(),
            self.hparams.sample_rate,
        )

    def compute_forward(self, batch, stage):
        """Interpreter training forward step."""

    def compute_objectives(self, pred, batch, stage):
        """Defines and computes the optimization objectives."""

    def on_stage_start(self, stage, epoch=None):
        """Steps taken before stage start."""

        @torch.no_grad()
        def accuracy_value(predict, target, length):
            """Computes accuracy."""
            nbr_correct, nbr_total = sb.utils.Accuracy.Accuracy(
                predict.unsqueeze(1), target, length
            )
            acc = torch.tensor([nbr_correct / nbr_total])
            return acc

        @torch.no_grad()
        def compute_fidelity(theta_out, predictions):
            """Computes top-k fidelity of interpreter."""
            predictions = F.softmax(predictions, dim=1)
            theta_out = F.softmax(theta_out, dim=1)

            pred_cl = torch.argmax(predictions, dim=1)
            k_top = torch.argmax(theta_out, dim=1)

            # 1 element for each sample in batch, is 0 if pred_cl is in top k
            temp = (k_top == pred_cl).float()

            return temp

        @torch.no_grad()
        def compute_faithfulness(wavs, predictions):
            """Computes the faithfulness metric."""
            X2 = self.interpret_computation_steps(wavs)[0]

            _, _, predictions_masked, _ = self.classifier_forward(X2)

            predictions = F.softmax(predictions, dim=1)
            predictions_masked = F.softmax(predictions_masked, dim=1)

            # Get the prediction indices
            pred_cl = predictions.argmax(dim=1, keepdim=True)

            # Get the corresponding output probabilities
            predictions_selected = torch.gather(
                predictions, dim=1, index=pred_cl
            )
            predictions_masked_selected = torch.gather(
                predictions_masked, dim=1, index=pred_cl
            )

            faithfulness = (
                predictions_selected - predictions_masked_selected
            ).squeeze(1)

            return faithfulness

        @torch.no_grad()
        def compute_rec_error(preds, specs, length=None):
            """Computes the reconstruction error."""
            if self.hparams.use_mask_output:
                preds = specs * preds

            return (specs - preds).pow(2).mean((-2, -1))

        @torch.no_grad()
        def compute_bern_ll(xhat, target_mask, length=None):
            """Computes Bernoulli likelihood."""
            eps = 1e-10
            rec_loss = (
                -target_mask * torch.log(xhat + eps)
                - (1 - target_mask) * torch.log(1 - xhat + eps)
            ).mean((-2, -1))
            return rec_loss

        self.inp_fid = MetricStats(metric=compute_fidelity)
        self.faithfulness = MetricStats(metric=compute_faithfulness)
        self.acc_metric = sb.utils.metric_stats.MetricStats(
            metric=accuracy_value, n_jobs=1
        )
        self.recons_err = sb.utils.metric_stats.MetricStats(
            metric=compute_rec_error
        )

        for metric_name, metric_fn in self.extra_metrics():
            setattr(
                self,
                metric_name,
                sb.utils.metric_stats.MetricStats(metric=metric_fn),
            )

        return super().on_stage_start(stage, epoch)

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.
        Plots in subplots the values of `self.batch_to_plot` and saves the
        plot to the experiment folder `self.hparams.output_folder`."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            self.train_stats = {
                "loss": self.train_loss,
                "acc": self.acc_metric.summarize("average"),
                # "rec_error": self.recons_err.summarize("average"),
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
                "input_fidelity": current_fid,
                "rec_error": self.recons_err.summarize("average"),
                "faithfulness_median": torch.Tensor(
                    self.faithfulness.scores
                ).median(),
                "faithfulness_mean": torch.Tensor(
                    self.faithfulness.scores
                ).mean(),
            }

            # The train_logger writes a summary to stdout and to the log file
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=valid_stats,
            )

            # Save the current checkpoint and delete previous checkpoints
            self.checkpointer.save_and_keep_only(
                meta=valid_stats, max_keys=["top-3_fid"]
            )

        if stage == sb.Stage.TEST:
            current_fid = self.top_3_fidelity.summarize("average")
            test_stats = {
                "loss": stage_loss,
                "acc": self.acc_metric.summarize("average"),
                "input_fidelity": current_fid,
                "faithfulness_median": torch.Tensor(
                    self.faithfulness.scores
                ).median(),
                "faithfulness_mean": torch.Tensor(
                    self.faithfulness.scores
                ).mean(),
            }

            # The train_logger writes a summary to stdout and to the log file
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch}, test_stats=test_stats
            )
