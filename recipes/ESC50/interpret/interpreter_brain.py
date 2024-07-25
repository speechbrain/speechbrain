"""This is a parent class for the interpretability recipes.

Authors
    * Francesco Paissan 2022, 2023, 2024
    * Cem Subakan 2022, 2023, 2024
    * Luca Della Libera 2024
"""

import os

import matplotlib.pyplot as plt
import quantus
import torch
import torchaudio
import torchvision
from torch.nn import functional as F

import speechbrain as sb
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
            X_stft, power=0.5
        )

        X_mel, X_mel_log1p = [None] * 2
        if self.hparams.use_melspectra_log1p:
            X_mel = self.hparams.compute_fbank(X_stft_power)
            X_mel_log1p = torch.log1p(X_mel)

        X_stft_logpower = torch.log1p(X_stft_power)

        return X_stft_logpower, X_mel_log1p, X_stft, X_stft_power

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
            if hasattr(self.hparams, "return_reps"):
                embeddings, hs = self.hparams.embedding_model(X_stft_logpower)
                hcat = hs
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

    def viz_ints(self, X_stft, X_stft_logpower, batch, wavs):
        """The helper function to create debugging images"""
        X_int, _, X_stft_phase, _ = self.interpret_computation_steps(wavs)

        X_int = torch.expm1(X_int)

        X_int = X_int[..., None]
        X_int = X_int.permute(0, 2, 1, 3)

        X_stft_phase = X_stft_phase[:, : X_int.shape[1], :]

        xhat_tm = self.invert_stft_with_phase(X_int, X_stft_phase)

        plt.figure(figsize=(10, 5), dpi=100)

        plt.subplot(121)
        plt.imshow(X_stft_logpower[0].squeeze().cpu().t(), origin="lower")
        plt.title("input")
        plt.colorbar()

        plt.subplot(122)
        plt.imshow(X_int[0].squeeze().cpu().t(), origin="lower")
        plt.colorbar()
        plt.title("interpretation")

        out_folder = os.path.join(
            self.hparams.output_folder,
            "interpretations/" f"{batch.id[0]}",
        )
        os.makedirs(
            out_folder,
            exist_ok=True,
        )

        plt.savefig(
            os.path.join(out_folder, "spectra.png"),
            format="png",
        )
        plt.close()

        torchaudio.save(
            os.path.join(out_folder, "interpretation.wav"),
            xhat_tm.data.cpu(),
            self.hparams.sample_rate,
        )

        torchaudio.save(
            os.path.join(out_folder, "original.wav"),
            wavs.data.cpu(),
            self.hparams.sample_rate,
        )

    def compute_forward(self, batch, stage):
        """Interpreter training forward step."""

    def compute_objectives(self, pred, batch, stage):
        """Defines and computes the optimization objectives."""

    def on_stage_start(self, stage, epoch=None):
        """Steps taken before stage start."""

        @torch.no_grad()
        def compute_fidelity(theta_out, predictions):
            """Computes top-`k` fidelity of interpreter."""
            pred_cl = torch.argmax(predictions, dim=1)
            k_top = torch.topk(theta_out, k=1, dim=1)[1]

            # 1 element for each sample in batch, is 0 if pred_cl is in top k
            temp = (k_top - pred_cl.unsqueeze(1) == 0).sum(1)

            return temp

        @torch.no_grad()
        def compute_faithfulness(predictions, predictions_masked):
            "This function implements the faithful metric (FF) used in the L-MAC paper."
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
            ).squeeze(dim=1)

            return faithfulness

        @torch.no_grad()
        def compute_AD(theta_out, predictions):
            """Computes top-`k` fidelity of interpreter."""
            predictions = F.softmax(predictions, dim=1)
            theta_out = F.softmax(theta_out, dim=1)

            pc = torch.gather(
                predictions, dim=1, index=predictions.argmax(1, keepdim=True)
            ).squeeze()
            oc = torch.gather(
                theta_out, dim=1, index=predictions.argmax(1, keepdim=True)
            ).squeeze(dim=1)

            # 1 element for each sample in batch, is 0 if pred_cl is in top k
            temp = (F.relu(pc - oc) / (pc + eps)) * 100

            return temp

        @torch.no_grad()
        def compute_AI(theta_out, predictions):
            """Computes top-`k` fidelity of interpreter."""
            pc = torch.gather(
                predictions, dim=1, index=predictions.argmax(1, keepdim=True)
            ).squeeze()
            oc = torch.gather(
                theta_out, dim=1, index=predictions.argmax(1, keepdim=True)
            ).squeeze(dim=1)

            # 1 element for each sample in batch, is 0 if pred_cl is in top k
            temp = (pc < oc).float() * 100

            return temp

        @torch.no_grad()
        def compute_AG(theta_out, predictions):
            """Computes top-`k` fidelity of interpreter."""
            pc = torch.gather(
                predictions, dim=1, index=predictions.argmax(1, keepdim=True)
            ).squeeze()
            oc = torch.gather(
                theta_out, dim=1, index=predictions.argmax(1, keepdim=True)
            ).squeeze(dim=1)

            # 1 element for each sample in batch, is 0 if pred_cl is in top k
            temp = (F.relu(oc - pc) / (1 - pc + eps)) * 100

            return temp

        @torch.no_grad()
        def compute_sparseness(wavs, X, y):
            """Computes the SPS metric used in the L-MAC paper."""
            self.sparseness = quantus.Sparseness(
                return_aggregate=True, abs=True
            )
            device = X.device
            attr = (
                self.interpret_computation_steps(wavs)[1]
                .transpose(1, 2)
                .unsqueeze(1)
                .clone()
                .detach()
                .cpu()
                .numpy()
            )
            if attr.sum() > 0:
                X = X[:, : attr.shape[2], :]
                X = X.unsqueeze(1)
                quantus_inp = {
                    "model": None,
                    "x_batch": X.clone()
                    .detach()
                    .cpu()
                    .numpy(),  # quantus expects the batch dim
                    "a_batch": attr,
                    "y_batch": y.squeeze(dim=1).clone().detach().cpu().numpy(),
                    "softmax": False,
                    "device": device,
                }
                return torch.Tensor([self.sparseness(**quantus_inp)[0]]).float()
            else:
                print("all zeros saliency map")
                return torch.zeros([0])

        @torch.no_grad()
        def compute_complexity(wavs, X, y):
            """Computes the COMP metric used in L-MAC paper"""
            self.complexity = quantus.Complexity(
                return_aggregate=True, abs=True
            )
            device = X.device
            attr = (
                self.interpret_computation_steps(wavs)[1]
                .transpose(1, 2)
                .unsqueeze(1)
                .clone()
                .detach()
                .cpu()
                .numpy()
            )
            if attr.sum() > 0:
                X = X[:, : attr.shape[2], :]
                X = X.unsqueeze(1)
                quantus_inp = {
                    "model": None,
                    "x_batch": X.clone()
                    .detach()
                    .cpu()
                    .numpy(),  # quantus expects the batch dim
                    "a_batch": attr,
                    "y_batch": y.squeeze(dim=1).clone().detach().cpu().numpy(),
                    "softmax": False,
                    "device": device,
                }

                return torch.Tensor([self.complexity(**quantus_inp)[0]]).float()
            else:
                print("all zeros saliency map")
                return torch.zeros([0])

        @torch.no_grad()
        def accuracy_value(predict, target):
            """Computes Accuracy"""
            predict = predict.argmax(1)

            return (predict.unsqueeze(1) == target).float().squeeze(1)

        self.AD = MetricStats(metric=compute_AD)
        self.AI = MetricStats(metric=compute_AI)
        self.AG = MetricStats(metric=compute_AG)
        self.sps = MetricStats(metric=compute_sparseness)
        self.comp = MetricStats(metric=compute_complexity)
        self.inp_fid = MetricStats(metric=compute_fidelity)
        self.faithfulness = MetricStats(metric=compute_faithfulness)
        self.acc_metric = MetricStats(metric=accuracy_value)

        for metric_name, metric_fn in self.extra_metrics().items():
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
            }

        extra_m = {
            k: torch.Tensor(getattr(self, k).scores).mean()
            for k in self.extra_metrics().keys()
        }

        # this is needed to eliminate comp values which are nan
        comp_tensor = torch.Tensor(self.comp.scores)
        comp_tensor = comp_tensor[~torch.isnan(comp_tensor)]
        tmp = {
            "SPS": torch.Tensor(self.sps.scores).mean(),
            "COMP": comp_tensor.mean(),
        }
        quantus_metrics = {}
        for m in tmp:
            if not tmp[m].isnan():
                quantus_metrics[m] = tmp[m]

        if stage == sb.Stage.VALID:
            current_fid = torch.Tensor(self.inp_fid.scores).mean()
            old_lr, new_lr = self.hparams.lr_annealing(
                [self.optimizer], epoch, -current_fid
            )
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            valid_stats = {
                "loss": stage_loss,
                "acc": self.acc_metric.summarize("average"),
                "input_fidelity": current_fid,
                "AI": torch.Tensor(self.AI.scores).mean(),
                "AD": torch.Tensor(self.AD.scores).mean(),
                "AG": torch.Tensor(self.AG.scores).mean(),
                "faithfulness_mean": torch.Tensor(
                    self.faithfulness.scores
                ).mean(),
            }
            valid_stats.update(extra_m)
            valid_stats.update(quantus_metrics)

            # The train_logger writes a summary to stdout and to the log file
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=valid_stats,
            )

            # Save the current checkpoint and delete previous checkpoints
            self.checkpointer.save_and_keep_only(
                meta=valid_stats, max_keys=["faithfulnesstop-3_fid"]
            )

        if stage == sb.Stage.TEST:
            current_fid = torch.Tensor(self.inp_fid.scores).mean()
            test_stats = {
                "loss": stage_loss,
                "acc": self.acc_metric.summarize("average"),
                "input_fidelity": current_fid,
                "AI": torch.Tensor(self.AI.scores).mean(),
                "AD": torch.Tensor(self.AD.scores).mean(),
                "AG": torch.Tensor(self.AG.scores).mean(),
                "faithfulness_mean": torch.Tensor(
                    self.faithfulness.scores
                ).mean(),
            }
            test_stats.update(extra_m)
            test_stats.update(quantus_metrics)

            # The train_logger writes a summary to stdout and to the log file
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch}, test_stats=test_stats
            )

            test_stats = {
                k: (
                    test_stats[k].item()
                    if isinstance(test_stats[k], torch.Tensor)
                    else test_stats[k]
                )
                for k in test_stats
            }
            print(test_stats)
