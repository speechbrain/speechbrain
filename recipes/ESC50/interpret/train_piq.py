#!/usr/bin/python3
"""This recipe to train PIQ to interepret audio classifiers.

Authors
    * Cem Subakan 2022, 2023
    * Francesco Paissan 2022, 2023
"""
import os
import sys
import torch
import torchaudio
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from esc50_prepare import prepare_esc50
from speechbrain.utils.metric_stats import MetricStats
from wham_prepare import WHAMDataset, combine_batches
from os import makedirs
import torch.nn.functional as F
from speechbrain.processing.NMF import spectral_phase
import matplotlib.pyplot as plt

eps = 1e-10


def tv_loss(mask, tv_weight=1, power=2, border_penalty=0.3):
    if tv_weight is None or tv_weight == 0:
        return 0.0
    # https://github.com/chongyangma/cs231n/blob/master/assignments/assignment3/style_transfer_pytorch.py
    # https://github.com/PiotrDabkowski/pytorch-saliency/blob/bfd501ec7888dbb3727494d06c71449df1530196/sal/utils/mask.py#L5
    w_variance = torch.sum(torch.pow(mask[:, :, :-1] - mask[:, :, 1:], power))
    h_variance = torch.sum(torch.pow(mask[:, :-1, :] - mask[:, 1:, :], power))

    loss = tv_weight * (h_variance + w_variance) / float(power * mask.size(0))
    return loss


class InterpreterESC50Brain(sb.core.Brain):
    """Class for sound class embedding training" """

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

        #if self.hparams.use_stft2mel:
        #    X_stft_logpower = X_stft_power
        #else:
        if not self.hparams.use_melspectra:
            X_stft_logpower = torch.log1p(X_stft_power)
        else:
            X_stft_power = self.hparams.compute_fbank(X_stft_power)
            X_stft_logpower = torch.log1p(X_stft_power)

        return X_stft_logpower, X_stft, X_stft_power

    def classifier_forward(self, X_stft_logpower):
        """The forward pass for the classifier"""
        if self.hparams.use_stft2mel:
            X_in = torch.expm1(X_stft_logpower)
            X_stft_logpower = self.hparams.compute_fbank(X_in)
            X_stft_logpower = torch.log1p(X_stft_logpower)

        if hasattr(self.hparams, 'return_reps'):
            embeddings, hs = self.hparams.embedding_model(X_stft_logpower)
            hcat = hs
        else:
            hcat = self.hparams.embedding_model(X_stft_logpower)
            embeddings = hcat.mean((-1, -2))
        predictions = self.hparams.classifier(embeddings).squeeze(1)
        class_pred = predictions.argmax(1)

        return hcat, embeddings, predictions, class_pred

    def interpret_computation_steps(self, wavs, print_probability=False):
        """Computation steps to get the interpretation spectrogram"""
        X_stft_logpower, X_stft, X_stft_power = self.preprocess(wavs)
        X_stft_phase = spectral_phase(X_stft)

        hcat, embeddings, predictions, class_pred = self.classifier_forward(
            X_stft_logpower
        )
        if print_probability:
            predictions = F.softmax(predictions, dim=1)
            class_prob = predictions[0, class_pred].item()
            print(f"classifier_prob: {class_prob}")

        if self.hparams.use_vq:
            xhat, hcat, z_q_x = self.modules.psi(hcat, class_pred)
        else:
            xhat = self.modules.psi.forward(hcat)
            z_q_x = None
        xhat = xhat.squeeze(1)

        Tmax = xhat.shape[1]
        if self.hparams.use_mask_output:
            xhat = F.sigmoid(xhat)
            X_int = xhat * X_stft_logpower[:, :Tmax, :]
        else:
            xhat = F.softplus(xhat)
            th = xhat.max() * self.hparams.mask_th
            X_int = (xhat > th) * X_stft_logpower[:, :Tmax, :]

        return X_int, X_stft_phase, class_pred, X_stft_logpower, xhat

    def interpret_sample(self, wavs, batch=None):
        """Get the interpratation for a given wav file."""

        # get the interpretation spectrogram, phase, and the predicted class
        X_int, X_stft_phase, pred_cl, _, _ = self.interpret_computation_steps(
            wavs
        )
        X_stft_phase = X_stft_phase[:, : X_int.shape[1], :]
        if not (batch is None) and (not self.hparams.use_melspectra):
            x_int_sb = self.invert_stft_with_phase(X_int, X_stft_phase)

            # save reconstructed and original spectrograms
            makedirs(
                os.path.join(
                    self.hparams.output_folder,
                    "audios_from_interpretation",
                ),
                exist_ok=True,
            )

            current_class_ind = batch.class_string_encoded.data[0].item()
            current_class_name = self.hparams.label_encoder.ind2lab[
                current_class_ind
            ]
            predicted_class_name = self.hparams.label_encoder.ind2lab[
                pred_cl.item()
            ]
            torchaudio.save(
                os.path.join(
                    self.hparams.output_folder,
                    "audios_from_interpretation",
                    f"original_tc_{current_class_name}_pc_{predicted_class_name}.wav",
                ),
                wavs[0].unsqueeze(0).cpu(),
                self.hparams.sample_rate,
            )

            torchaudio.save(
                os.path.join(
                    self.hparams.output_folder,
                    "audios_from_interpretation",
                    f"interpretation_tc_{current_class_name}_pc_{predicted_class_name}.wav",
                ),
                x_int_sb.cpu(),
                self.hparams.sample_rate,
            )

        return X_int

    def overlap_test(self, batch):
        """Interpration test with overlapped audio"""
        wavs, _ = batch.sig
        wavs = wavs.to(self.device)

        if wavs.shape[0] <= 1:
            return

        s1 = wavs[0]
        s1 = s1 / s1.max()
        s2 = wavs[1]
        s2 = s2 / s2.max()

        # create the mixture with s2 being the noise (lower gain)
        mix = (s1 * 0.8 + (s2 * 0.2)).unsqueeze(0)
        mix = mix / mix.max()

        # get the interpretation spectrogram, phase, and the predicted class
        (
            X_int,
            X_stft_phase,
            pred_cl,
            X_mix,
            mask,
        ) = self.interpret_computation_steps(mix)
        X_int = X_int[0, ...]
        X_stft_phase = X_stft_phase[0, : X_int.shape[0], ...].unsqueeze(0)
        pred_cl = pred_cl[0, ...]
        mask = mask[0, ...]

        if not self.hparams.use_melspectra:
            temp = torch.expm1(X_int).unsqueeze(0).unsqueeze(-1)
            x_int_sb = self.invert_stft_with_phase(temp, X_stft_phase)

        # save reconstructed and original spectrograms
        current_class_ind = batch.class_string_encoded.data[0].item()
        current_class_name = self.hparams.label_encoder.ind2lab[
            current_class_ind
        ]
        predicted_class_name = self.hparams.label_encoder.ind2lab[
            pred_cl.item()
        ]

        noise_class_ind = batch.class_string_encoded.data[1].item()
        noise_class_name = self.hparams.label_encoder.ind2lab[noise_class_ind]

        out_folder = os.path.join(
            self.hparams.output_folder,
            "overlap_test",
            f"tc_{current_class_name}_nc_{noise_class_name}_pc_{predicted_class_name}",
        )
        makedirs(
            out_folder,
            exist_ok=True,
        )

        torchaudio.save(
            os.path.join(out_folder, "mixture.wav"),
            mix.data.cpu(),
            self.hparams.sample_rate,
        )

        torchaudio.save(
            os.path.join(out_folder, "source.wav"),
            s1.unsqueeze(0).data.cpu(),
            self.hparams.sample_rate,
        )

        torchaudio.save(
            os.path.join(out_folder, "noise.wav"),
            s2.unsqueeze(0).data.cpu(),
            self.hparams.sample_rate,
        )

        if not self.hparams.use_melspectra:
            torchaudio.save(
                os.path.join(out_folder, "interpretation.wav"),
                x_int_sb.data.cpu(),
                self.hparams.sample_rate,
            )

        plt.figure(figsize=(12, 10), dpi=100)

        plt.subplot(311)
        X_target = X_mix[0].permute(1, 0)[:, :X_int.shape[0]].cpu()
        plt.imshow(X_target, origin='lower')
        plt.colorbar()

        plt.subplot(312)
        plt.imshow(mask.data.cpu().permute(1, 0), origin='lower')
        plt.title("Estimated Mask")
        plt.colorbar()

        plt.subplot(313)
        plt.imshow(X_int.data.cpu().permute(1, 0).data.cpu(), origin='lower')
        plt.colorbar()
        plt.title("masked")
        plt.savefig(os.path.join(out_folder, "specs.png"))
        plt.close()

    def debug_files(self, X_stft, xhat, X_stft_logpower, batch, wavs):
        """The helper function to create debugging images"""
        X_stft_phase = spectral_phase(X_stft)
        temp = xhat[0].transpose(0, 1).unsqueeze(0).unsqueeze(-1)
        Xspec_est = torch.expm1(temp.permute(0, 2, 1, 3))
        if not self.hparams.use_melspectra:
            xhat_tm = self.invert_stft_with_phase(Xspec_est, X_stft_phase)

        Tmax = Xspec_est.shape[1]
        X_masked = xhat[0] * X_stft_logpower[0, :Tmax, :]

        X_est_masked = torch.expm1(X_masked).unsqueeze(0).unsqueeze(-1)
        if not self.hparams.use_melspectra:
            xhat_tm_masked = self.invert_stft_with_phase(
                X_est_masked, X_stft_phase[0:1]
            )

        plt.figure(figsize=(11, 10), dpi=100)

        plt.subplot(311)
        X_target = X_stft_logpower[0].permute(1, 0)[:, : xhat.shape[1]].cpu()
        plt.imshow(X_target, origin="lower")
        plt.title("input")
        plt.colorbar()

        plt.subplot(312)
        mask = xhat[0]
        X_masked = mask * X_stft_logpower[0, :Tmax, :]
        plt.imshow(X_masked.permute(1, 0).data.cpu(), origin="lower")
        plt.colorbar()
        plt.title("masked")

        plt.subplot(313)
        plt.imshow(mask.permute(1, 0).data.cpu(), origin="lower")
        plt.colorbar()
        plt.title("mask")

        out_folder = os.path.join(
            self.hparams.output_folder,
            "reconstructions/" f"{batch.id[0]}",
        )
        makedirs(
            out_folder,
            exist_ok=True,
        )

        plt.savefig(
            os.path.join(out_folder, "reconstructions.png"),
            format="png",
        )
        plt.close()

        if not self.hparams.use_melspectra:
            torchaudio.save(
                os.path.join(out_folder, "interpretation.wav"),
                xhat_tm_masked.data.cpu(),
                self.hparams.sample_rate,
            )

        torchaudio.save(
            os.path.join(out_folder, "original.wav"),
            wavs[0:1].data.cpu(),
            self.hparams.sample_rate,
        )

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + sound classifier.
        Data augmentation and environmental corruption are applied to the
        input sound.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # augment batch with WHAM!
        if hasattr(self.hparams, 'add_wham_noise'):
            if self.hparams.add_wham_noise:
                wavs = combine_batches(wavs, iter(self.hparams.wham_dataset))

        X_stft_logpower, X_stft, X_stft_power = self.preprocess(wavs)

        # Embeddings + sound classifier
        hcat, embeddings, predictions, class_pred = self.classifier_forward(
            X_stft_logpower
        )

        if self.hparams.use_vq:
            xhat, hcat, z_q_x = self.modules.psi(hcat, class_pred)
        else:
            xhat = self.modules.psi.forward(hcat)
            z_q_x = None
        xhat = xhat.squeeze(1)

        if self.hparams.use_mask_output:
            xhat = F.sigmoid(xhat)
        else:
            xhat = F.softplus(xhat)

        garbage = 0

        if stage == sb.Stage.VALID:
            # save some samples
            if (
                self.hparams.epoch_counter.current
                % self.hparams.interpret_period
            ) == 0 and self.hparams.save_interpretations:
                # self.interpret_sample(wavs, batch)
                self.overlap_test(batch)
                self.debug_files(X_stft, xhat, X_stft_logpower, batch, wavs[0:1])

        return (wavs, lens), predictions, xhat, hcat, z_q_x, garbage

    def crosscor(self, spectrogram, template):
        if self.hparams.crosscortype == 'conv':
            spectrogram = spectrogram - spectrogram.mean((-1, -2), keepdim=True)
            template = template - template.mean((-1, -2), keepdim=True)
            template = template.unsqueeze(1)
            # 1 x BS x T x F
            # BS x 1 x T x F
            tmp = F.conv2d(
                spectrogram[None], template, bias=None, groups=spectrogram.shape[0]
            )

            normalization1 = F.conv2d(
                spectrogram[None] ** 2,
                torch.ones_like(template),
                groups=spectrogram.shape[0],
            )
            normalization2 = F.conv2d(
                torch.ones_like(spectrogram[None]),
                template**2,
                groups=spectrogram.shape[0],
            )

            ncc = (
                tmp / torch.sqrt(normalization1 * normalization2 + 1e-8)
            ).squeeze()

            return ncc
        elif self.hparams.crosscortype == 'dotp':
            dotp = (spectrogram * template).mean((-1, -2))
            norms_specs = spectrogram.pow(2).mean((-1, -2)).sqrt()
            norms_templates = template.pow(2).mean((-1, -2)).sqrt()
            norm_dotp = dotp / (norms_specs * norms_templates)
            return norm_dotp
        else:
            raise ValueError('unknown crosscor type!')

    def compute_objectives(self, pred, batch, stage):
        """Helper function to compute the objectives"""
        batch_sig, predictions, xhat, hcat, z_q_x, garbage = pred

        
        batch = batch.to(self.device)
        wavs_clean, lens_clean = batch.sig

        # taking them from forward because they are augmented there!
        wavs, lens = batch_sig

        uttid = batch.id
        labels, _ = batch.class_string_encoded

        (
            X_stft_logpower_clean,
            X_stft_clean,
            X_stft_power_clean,
        ) = self.preprocess(wavs_clean)
        X_stft_logpower, X_stft, X_stft_power = self.preprocess(wavs)

        Tmax = xhat.shape[1]

        # map clean to same dimensionality
        X_stft_logpower_clean = X_stft_logpower_clean[:, :Tmax, :]

        mask_in = xhat * X_stft_logpower[:, :Tmax, :]
        mask_out = (1 - xhat) * X_stft_logpower[:, :Tmax, :]

        if self.hparams.finetuning:
            crosscor = self.crosscor(X_stft_logpower_clean, mask_in)
            crosscor_mask = (crosscor >= self.hparams.crosscor_th).float()

            max_batch = (
                X_stft_logpower_clean.view(X_stft_logpower_clean.shape[0], -1)
                .max(1)
                .values.view(-1, 1, 1)
            )
            binarized_oracle = (
                X_stft_logpower_clean >= self.hparams.bin_th * max_batch
            ).float()

            # samples if we apply the binarization threshold on the spectrogram
            # this is just to debug the cross-correlation
            # with torch.no_grad():
            #     maskin_bin = binarized_oracle * X_stft_logpower_clean
            #     for idx, s in enumerate(
            #             zip(
            #                 X_stft_logpower_clean.cpu(),
            #                 mask_in.cpu(),
            #                 crosscor.cpu(),
            #                 binarized_oracle.cpu(),
            #             )
            #         ):
            #         ax = plt.subplot(141)
            #         plt.imshow(s[0].t(), origin="lower")
            #         plt.title("Oracle")

            #         plt.subplot(142, sharex=ax)
            #         plt.imshow(maskin_bin[idx].cpu().t(), origin="lower")
            #         plt.title("th * oracle")

            #         plt.subplot(143, sharex=ax)
            #         plt.imshow(s[3].t(), origin="lower")
            #         plt.title("Binarized Oracle")

            #         plt.subplot(144, sharex=ax)
            #         plt.imshow(s[1].t(), origin="lower")
            #         plt.title("Mask in") 
            #         plt.tight_layout()
            #         plt.suptitle("Cross correlation: %.2f - made the thr: %s" % (s[2].item(), bool(crosscor_mask[idx])))
            #         plt.savefig(f"batch/{idx}.png")
            #         torchaudio.save(f"batch/{idx}.wav", wavs_clean[idx][None].cpu(), sample_rate=16000)
            #         torchaudio.save(f"batch/inp{idx}.wav", wavs[idx][None].cpu(), sample_rate=16000)

            #     #plt.savefig('batch_situation.png')
            #     #torchaudio.save(f"batch/{idx}.wav", wavs_clean[idx][None].cpu(), sample_rate=16000)

            # import pdb; pdb.set_trace()
            
            if self.hparams.guidelosstype == 'binary':
                rec_loss = (
                    F.binary_cross_entropy(
                        xhat, binarized_oracle, reduce=False
                    ).mean((-1, -2))
                    * self.hparams.g_w
                    * crosscor_mask
                ).mean()
            else:
                temp = ((xhat * X_stft_logpower[:, :X_stft_logpower_clean.shape[1], :]) - X_stft_logpower_clean).pow(2).mean((-1, -2))
                rec_loss = (temp * crosscor_mask).mean() * self.hparams.g_w

        else:
            rec_loss = 0
            crosscor_mask = torch.zeros(xhat.shape[0], device=self.device)

        # if self.hparams.use_stft2mel:
        #    import pdb; pdb.set_trace()
        #    mask_in = self.hparams.compute_fbank(mask_in)
        #    mask_in = torch.log1p(mask_in)

        #    mask_out = self.hparams.compute_fbank(mask_out)
        #    mask_out = torch.log1p(mask_out)

        mask_in_preds = self.classifier_forward(mask_in)[2]

        mask_out_preds = self.classifier_forward(mask_out)[2]

        class_pred = predictions.argmax(1)
        l_in = F.nll_loss(mask_in_preds.log_softmax(1), class_pred)
        l_out = -F.nll_loss(mask_out_preds.log_softmax(1), class_pred)
        ao_loss = l_in * self.hparams.l_in_w + self.hparams.l_out_w * l_out

        r_m = (
            xhat.abs().mean((-1, -2, -3))
            * self.hparams.reg_w_l1
            * torch.logical_not(crosscor_mask)
        ).sum()
        r_m += (
            tv_loss(xhat)
            * self.hparams.reg_w_tv
            * torch.logical_not(crosscor_mask)
        ).sum()

        mask_in_preds = mask_in_preds.softmax(1)
        mask_out_preds = mask_out_preds.softmax(1)

        if stage == sb.Stage.VALID or stage == sb.Stage.TEST:
            self.inp_fid.append(
                uttid,
                mask_in_preds,
                predictions.softmax(1),
            )
            self.AD.append(
                uttid,
                mask_in_preds,
                predictions.softmax(1),
            )
            self.AI.append(
                uttid,
                mask_in_preds,
                predictions.softmax(1),
            )
            self.AG.append(
                uttid,
                mask_in_preds,
                predictions.softmax(1),
            )
            self.faithfulness.append(
                uttid,
                predictions.softmax(1),
                mask_out_preds,
            )

        self.in_masks.append(uttid, c=crosscor_mask)
        self.acc_metric.append(
            uttid,
            predict=predictions,
            target=labels,
        )

        if stage != sb.Stage.TEST:
            if hasattr(self.hparams.lr_annealing, "on_batch_end"):
                self.hparams.lr_annealing.on_batch_end(self.optimizer)

        return ao_loss + r_m + rec_loss

    @torch.no_grad()
    def accuracy_value(self, predict, target):
        """Computes Accuracy"""
        predict = predict.argmax(1)

        return (predict.unsqueeze(1) == target).float().squeeze()

    @torch.no_grad()
    def compute_fidelity(self, theta_out, predictions):
        """Computes top-`k` fidelity of interpreter."""
        # predictions = F.softmax(predictions, dim=1)
        # theta_out = F.softmax(theta_out, dim=1)

        pred_cl = torch.argmax(predictions, dim=1)
        k_top = torch.topk(theta_out, k=1, dim=1)[1]

        # 1 element for each sample in batch, is 0 if pred_cl is in top k
        temp = (k_top - pred_cl.unsqueeze(1) == 0).sum(1)

        return temp

    @torch.no_grad()
    def compute_faithfulness(self, predictions, predictions_masked):
        # get the prediction indices
        pred_cl = predictions.argmax(dim=1, keepdim=True)

        # get the corresponding output probabilities
        predictions_selected = torch.gather(predictions, dim=1, index=pred_cl)
        predictions_masked_selected = torch.gather(
            predictions_masked, dim=1, index=pred_cl
        )

        faithfulness = (
            predictions_selected - predictions_masked_selected
        ).squeeze()

        return faithfulness

    @torch.no_grad()
    def compute_AD(self, theta_out, predictions):
        """Computes top-`k` fidelity of interpreter."""
        predictions = F.softmax(predictions, dim=1)
        theta_out = F.softmax(theta_out, dim=1)

        pc = torch.gather(
            predictions, dim=1, index=predictions.argmax(1, keepdim=True)
        ).squeeze()
        oc = torch.gather(
            theta_out, dim=1, index=predictions.argmax(1, keepdim=True)
        ).squeeze()

        # 1 element for each sample in batch, is 0 if pred_cl is in top k
        temp = (F.relu(pc - oc) / (pc + eps)) * 100

        return temp

    @torch.no_grad()
    def compute_AI(self, theta_out, predictions):
        """Computes top-`k` fidelity of interpreter."""
        # predictions = F.softmax(predictions, dim=1)
        # theta_out = F.softmax(theta_out, dim=1)

        pc = torch.gather(
            predictions, dim=1, index=predictions.argmax(1, keepdim=True)
        ).squeeze()
        oc = torch.gather(
            theta_out, dim=1, index=predictions.argmax(1, keepdim=True)
        ).squeeze()

        # 1 element for each sample in batch, is 0 if pred_cl is in top k
        temp = (pc < oc).float() * 100

        return temp

    @torch.no_grad()
    def compute_AG(self, theta_out, predictions):
        """Computes top-`k` fidelity of interpreter."""
        # predictions = F.softmax(predictions, dim=1)
        # theta_out = F.softmax(theta_out, dim=1)

        pc = torch.gather(
            predictions, dim=1, index=predictions.argmax(1, keepdim=True)
        ).squeeze()
        oc = torch.gather(
            theta_out, dim=1, index=predictions.argmax(1, keepdim=True)
        ).squeeze()

        # 1 element for each sample in batch, is 0 if pred_cl is in top k
        temp = (F.relu(oc - pc) / (1 - pc + eps)) * 100

        return temp

    def on_stage_start(self, stage, epoch=None):
        """Steps taken before stage start"""
        self.inp_fid = MetricStats(metric=self.compute_fidelity)
        self.AD = MetricStats(metric=self.compute_AD)
        self.AI = MetricStats(metric=self.compute_AI)
        self.AG = MetricStats(metric=self.compute_AG)
        self.faithfulness = MetricStats(metric=self.compute_faithfulness)
        self.acc_metric = sb.utils.metric_stats.MetricStats(
            metric=self.accuracy_value, n_jobs=1
        )
        def counter(c):
            return c
        self.in_masks = MetricStats(metric=counter)

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
                "in_masks": sum(self.in_masks.scores)
            }
            # if self.hparams.use_mask_output:
            # self.train_stats["mask_ll"] = self.mask_ll.summarize("average")

        if stage == sb.Stage.VALID:
            current_fid = self.inp_fid.summarize("average")
            old_lr, new_lr = self.hparams.lr_annealing(
                [self.optimizer], epoch, -current_fid
            )
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            valid_stats = {
                "loss": stage_loss,
                "acc": self.acc_metric.summarize("average"),
                "input_fid": current_fid,
                "faithfulness_median": torch.Tensor(
                    self.faithfulness.scores
                ).median(),
                "faithfulness_mean": torch.Tensor(
                    self.faithfulness.scores
                ).mean(),
                "AD": self.AD.summarize("average"),
                "AI": self.AI.summarize("average"),
                "AG": self.AG.summarize("average"),
                "in_masks": sum(self.in_masks.scores)
            }
            # if self.hparams.use_mask_output:
            # valid_stats["mask_ll"] = self.mask_ll.summarize("average")

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
            current_fid = self.inp_fid.summarize("average")
            test_stats = {
                "loss": stage_loss,
                "acc": self.acc_metric.summarize("average"),
                "input_fid": current_fid,
                "faithfulness_median": torch.Tensor(
                    self.faithfulness.scores
                ).median(),
                "faithfulness_mean": torch.Tensor(
                    self.faithfulness.scores
                ).mean(),
                "faithfulness_std": torch.Tensor(
                    self.faithfulness.scores
                ).std(),
                "AD": self.AD.summarize("average"),
                "AI": self.AI.summarize("average"),
                "AG": self.AG.summarize("average"),
                # "freche": self.fid.compute(),
            }

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch}, test_stats=test_stats
            )


def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    data_audio_folder = hparams["audio_data_folder"]
    config_sample_rate = hparams["sample_rate"]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    hparams["resampler"] = torchaudio.transforms.Resample(
        new_freq=config_sample_rate
    )

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""

        wave_file = data_audio_folder + "/{:}".format(wav)

        sig, read_sr = torchaudio.load(wave_file)

        # If multi-channels, downmix it to a mono channel
        sig = torch.squeeze(sig)
        if len(sig.shape) > 1:
            sig = torch.mean(sig, dim=0)

        # Convert sample rate to required config_sample_rate
        if read_sr != config_sample_rate:
            # Re-initialize sampler if source file sample rate changed compared to last file
            if read_sr != hparams["resampler"].orig_freq:
                hparams["resampler"] = torchaudio.transforms.Resample(
                    orig_freq=read_sr, new_freq=config_sample_rate
                )
            # Resample audio
            sig = hparams["resampler"].forward(sig)

        sig = sig.float()
        sig = sig / sig.max()
        return sig

    # 3. Define label pipeline:
    @sb.utils.data_pipeline.takes("class_string")
    @sb.utils.data_pipeline.provides("class_string", "class_string_encoded")
    def label_pipeline(class_string):
        """the label pipeline"""
        yield class_string
        class_string_encoded = label_encoder.encode_label_torch(class_string)
        yield class_string_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "class_string_encoded"],
        )

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mappinng.
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="class_string",
    )

    return datasets, label_encoder


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
    hparams["embedding_model"].requires_grad_(False)
    hparams["classifier"].requires_grad_(False)

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

    if hparams["finetuning"]:
        if hparams["pretrained_PIQ"] is None:
            raise AssertionError(
                "You should specificy pretrained model for finetuning."
            )

    Interpreter_brain = InterpreterESC50Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    if hparams["pretrained_PIQ"] is not None and hparams["finetuning"]:
        run_on_main(hparams["load_pretrained"].collect_files)
        hparams["load_pretrained"].load_collected()

    if "pretrained_esc50" in hparams and hparams["use_pretrained"]:
        print("Loading model...")
        run_on_main(hparams["pretrained_esc50"].collect_files)
        hparams["pretrained_esc50"].load_collected()

    hparams["embedding_model"].to(hparams["device"])
    hparams["classifier"].to(hparams["device"])
    hparams["embedding_model"].eval()

    if not hparams["test_only"]:
        Interpreter_brain.fit(
            epoch_counter=Interpreter_brain.hparams.epoch_counter,
            train_set=datasets["train"],
            valid_set=datasets["valid"],
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )

    Interpreter_brain.checkpointer.recover_if_possible(
        min_key="loss",
        device=torch.device(Interpreter_brain.device),
    )

    test_stats = Interpreter_brain.evaluate(
        test_set=datasets["test"],
        min_key="loss",
        progressbar=True,
        test_loader_kwargs=hparams["dataloader_options"],
    )
