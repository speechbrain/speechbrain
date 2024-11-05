#!/usr/bin/env python3
"""Recipe for training a diffusion model on spectrogram data

To run this recipe, do the following:
> python train.py hparams/train.yaml

To read the code, first scroll to the bottom to see the "main" code.
This gives a high-level overview of what is going on, while the
Brain class definition provides the details of what happens
for each batch during training.

Authors
 * Artem Ploujnikov 2022
"""
import os
import sys
from collections import namedtuple
from enum import Enum

import torch
from audiomnist_prepare import prepare_audiomnist
from hyperpyyaml import load_hyperpyyaml
from torchaudio import functional as AF

import speechbrain as sb
from speechbrain.dataio.dataio import length_to_mask, write_audio
from speechbrain.dataio.dataset import apply_overfit_test
from speechbrain.utils import data_utils
from speechbrain.utils.data_utils import (
    dict_value_combinations,
    dist_stats,
    masked_max,
    masked_mean,
    masked_min,
    masked_std,
    match_shape,
)
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger
from speechbrain.utils.train_logger import plot_spectrogram

logger = get_logger(__name__)


class DiffusionMode(Enum):
    SIMPLE = "simple"
    LATENT = "latent"


DiffusionPredictions = namedtuple(
    "DiffusionPredictions",
    [
        "pred",
        "noise",
        "noisy_sample",
        "feats",
        "lens",
        "autoencoder_output",
        "feats_done",
        "lens_done",
        "pred_done",
    ],
)


# Brain class for speech enhancement training
class DiffusionBrain(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain.

    Arguments
    ---------
    modules : dict of str:torch.nn.Module pairs
        These modules are passed to the optimizer by default if they have
        trainable parameters, and will have ``train()``/``eval()`` called on them.
    opt_class : torch.optim class
        A torch optimizer constructor that takes only the list of
        parameters (e.g. a lambda or partial function definition). By default,
        this will be passed all modules in ``modules`` at the
        beginning of the ``fit()`` method. This behavior can be changed
        by overriding the ``configure_optimizers()`` method.
    hparams : dict
        Each key:value pair should consist of a string key and a hyperparameter
        that is used within the overridden methods. These will
        be accessible via an ``hparams`` attribute, using "dot" notation:
        e.g., self.hparams.model(x).
    run_opts : dict
        A set of options to change the runtime environment.
    checkpointer : Checkpointer
    """

    def __init__(
        self,
        modules=None,
        opt_class=None,
        hparams=None,
        run_opts=None,
        checkpointer=None,
    ):
        super().__init__(modules, opt_class, hparams, run_opts, checkpointer)
        self.diffusion_mode = DiffusionMode(self.hparams.diffusion_mode)
        self.use_done_detector = "done_detector" in self.modules

    def init_optimizers(self):
        """Initializes the diffusion model optimizer - and the
        autoencoder optimizer, if applicable"""
        self.optimizers_dict = {}
        if self.opt_class is not None:
            self.optimizer = self.opt_class(self.modules.unet.parameters())
            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer", self.optimizer)
            self.optimizers_dict["opt_class"] = self.optimizer

        if self.use_done_detector:
            self.optimizer_done = self.hparams.opt_class_done(
                self.modules.done_detector.parameters()
            )
            if self.checkpointer is not None:
                self.checkpointer.add_recoverable(
                    "optimizer_done", self.optimizer
                )
            self.optimizers_dict["opt_class_done"] = self.optimizer_done

        if self.diffusion_mode == DiffusionMode.LATENT:
            self.autoencoder_optimizer = self.hparams.opt_class_autoencoder(
                self.modules.autoencoder.parameters()
            )
            if self.checkpointer is not None:
                self.checkpointer.add_recoverable(
                    "autoencoder_optimizer", self.autoencoder_optimizer
                )
            self.optimizers_dict["opt_class_autoencoder"] = (
                self.autoencoder_optimizer
            )

    def compute_forward(self, batch, stage):
        """Runs all the computation of that transforms the input into the
        output probabilities over the N classes.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : torch.Tensor
            torch.Tensor that contains the posterior probabilities over the N classes.
        """

        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)

        # Compute features, embeddings, and predictions
        feats, lens = self.prepare_features(batch, stage)

        autoencoder_out = None
        cond_emb = None
        if self.is_conditioned:
            cond_labels = self.get_cond_labels(batch)
            cond_emb = self.compute_cond_emb(cond_labels)
        if self.diffusion_mode == DiffusionMode.LATENT:
            mask_value = self.modules.global_norm.normalize(
                self.mask_value_norm
            )
            latent_mask_value = self.get_latent_mask_value(mask_value)
            (
                train_sample_diffusion,
                autoencoder_out,
            ) = self.modules.diffusion_latent.train_sample_latent(
                feats,
                length=lens,
                out_mask_value=mask_value,
                cond_emb=cond_emb,
                latent_mask_value=latent_mask_value,
            )
            pred, noise, noisy_sample = train_sample_diffusion
        else:
            pred, noise, noisy_sample = self.modules.diffusion.train_sample(
                feats, length=lens, cond_emb=cond_emb
            )

        pred_done, feats_done, lens_done = None, None, None
        if self.use_done_detector:
            feats_done, lens_done = self.prepare_features_done(
                batch, feats, lens
            )
            pred_done = self.modules.done_detector(
                feats_done.squeeze(1), lens_done
            )

        # NOTE: lens can change because of the additional padding needed to account
        # NOTE: for downsampling
        return DiffusionPredictions(
            pred,
            noise,
            noisy_sample,
            feats,
            lens,
            autoencoder_out,
            feats_done,
            lens_done,
            pred_done,
        )

    def compute_latent_mask_value(self, mask_value):
        """Computes the value with which to mask the latent
        space. The core idea is that masked space should
        not produce any sound

        Arguments
        ---------
        mask_value: float
            the value to be used for the mask in the original space

        Returns
        -------
        latent_mask_value: float
            the value that will be used in the latent space
        """
        with torch.no_grad():
            fake_feats = (
                torch.ones(
                    1,
                    1,
                    self.hparams.spec_min_sample_size,
                    self.hparams.spec_n_mels,
                ).to(self.device)
                * mask_value
            )
            length = torch.tensor([1.0]).to(self.device)
            latent = self.modules.autoencoder.encode(fake_feats, length=length)
            latent_mask_value = (
                latent[:, :, : self.hparams.latent_mask_offset, :].mean().item()
            )
            return latent_mask_value

    def get_latent_mask_value(self, mask_value):
        """Returns the latent mask value, recomputing it if necessary

        Arguments
        ---------
        mask_value: float
            the value to be used for the mask in the original space

        Returns
        -------
        latent_mask_value: float
            the value that will be used in the latent space
        """
        if (
            not self.latent_mask_value
            or self.step < self.hparams.latent_mask_recompute_steps
        ):
            self.latent_mask_value = self.compute_latent_mask_value(mask_value)
        return self.latent_mask_value

    def compute_cond_emb(self, labels):
        """Computes conditioning embeddings for a set
        of labels

        Arguments
        ---------
        labels: dict
            A key -> label dictionary

        Returns
        -------
        emb: dict
            A key -> embedding dictionary
        """
        cond_emb = {}
        for key, emb_config in self.get_active_cond_emb().items():
            emb_module = emb_config["emb"]
            emb = emb_module(labels[key])
            cond_emb[key] = emb
        return cond_emb

    def get_cond_labels(self, batch):
        """Returns the conditioning labels for the batch provided
        based on information from the hparams file on which
        conditioning labels are enabled

        Arguments
        ---------
        batch: PaddedBatch
            a batch

        Returns
        -------
        result: dict
            the result
        """
        return {
            key: getattr(batch, emb_config["key"])
            for key, emb_config in self.hparams.cond_emb.items()
            if self.hparams.use_cond_emb[key]
        }

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        if self.reference_batch is None:
            self.reference_batch = batch

        should_step = self.step % self.grad_accumulation_factor == 0
        outputs = self.compute_forward(batch, sb.Stage.TRAIN)
        loss, loss_autoencoder, loss_done = self.compute_objectives(
            outputs, batch, sb.Stage.TRAIN
        )
        if self.train_diffusion:
            with self.no_sync(not should_step):
                (loss / self.grad_accumulation_factor).backward(
                    retain_graph=True
                )
        # Done loss - iff applicable
        if self.use_done_detector:
            with self.no_sync(not should_step):
                (loss_done / self.grad_accumulation_factor).backward(
                    retain_graph=True
                )

        if should_step:
            if self.train_diffusion:
                self.optimizer.step()
            self.optimizer.zero_grad()

            if self.use_done_detector:
                self.optimizer_done.step()
                self.optimizer_done.zero_grad()

        # Latent diffusion: Step through the autoencoder
        if (
            self.diffusion_mode == DiffusionMode.LATENT
            and loss_autoencoder is not None
        ):
            with self.no_sync(not should_step):
                (loss_autoencoder / self.grad_accumulation_factor).backward()
            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    self.modules.parameters(), self.max_grad_norm
                )
                self.autoencoder_optimizer.step()
            self.autoencoder_optimizer.zero_grad()

        self.optimizer_step += 1
        self.hparams.lr_annealing(self.optimizer, self.optimizer_step)
        if self.diffusion_mode == DiffusionMode.LATENT:
            self.hparams.lr_annealing_autoencoder(
                self.autoencoder_optimizer, self.optimizer_step
            )
        if (
            self.hparams.enable_train_metrics
            and self.hparams.use_tensorboard
            and (
                self.step == 1
                or self.step % self.hparams.train_log_interval == 0
            )
        ):
            self.log_batch(outputs)
        return loss

    def evaluate_batch(self, batch, stage):
        """Evaluate one batch, override for different procedure than train.

        The default implementation depends on two methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for evaluation. Default implementation assumes
            this batch has two elements: inputs and targets.
        stage : Stage
            The stage of the experiment: Stage.VALID, Stage.TEST

        Returns
        -------
        detached loss
        """

        out = self.compute_forward(batch, stage=stage)
        loss, _, _ = self.compute_objectives(out, batch, stage=stage)
        return loss.detach().cpu()

    def log_batch(self, predictions):
        """Saves information from a single batch to the log

        Arguments
        ---------
        predictions: DiffusionPredictions
            the predictions from compute_forward
        """
        loss_stats = self.loss_metric.summarize()
        stats = {
            "loss": loss_stats["average"],
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        stats.update(
            self.extract_dist_stats(self.data_dist_stats_metric, prefix="data")
        )
        if self.use_done_detector:
            stats["done_loss"] = self.done_loss_metric.summarize(
                field="average"
            )

        if (
            self.diffusion_mode == DiffusionMode.LATENT
            and self.train_autoencoder
        ):
            stats.update(
                self.autoencoder_loss_metric.summarize(field="average")
            )
            stats["laplacian_loss"] = (
                self.autoencoder_laplacian_loss_stats_metric.summarize(
                    field="average"
                )
            )
            stats["weighted_laplacian_loss"] = (
                self.hparams.loss_laplacian_weight * stats["laplacian_loss"]
            )
            stats["lr_autoencoder"] = self.autoencoder_optimizer.param_groups[
                0
            ]["lr"]
            stats.update(
                self.extract_dist_stats(
                    self.autoencoder_rec_dist_stats_metric,
                    prefix="autoencoder_rec",
                )
            )
            stats.update(
                self.extract_dist_stats(
                    self.autoencoder_latent_dist_stats_metric,
                    prefix="autoencoder_latent",
                )
            )

        self.hparams.tensorboard_train_logger.log_stats(
            stats_meta={"step": self.step}, train_stats=stats
        )
        if (
            self.diffusion_mode == DiffusionMode.LATENT
            and self.hparams.enable_reconstruction_sample
        ):
            self.hparams.tensorboard_train_logger.log_figure(
                "train_ref_spectrogram", predictions.feats[0]
            )
            self.hparams.tensorboard_train_logger.log_figure(
                "train_rec_spectrogram", predictions.autoencoder_output.rec[0]
            )
            latent = predictions.autoencoder_output.latent[0]
            latent = latent.view(
                latent.size(0) * latent.size(1), latent.size(2)
            )
            self.hparams.tensorboard_train_logger.log_figure(
                "train_rec_latent", latent
            )

    def extract_dist_stats(self, dist_stats_metric, prefix):
        """Extracts stats from a MultiMetricStats instance with a dist_stats metric
        into a flattened dictionary, converting the keys to <prefix>_<metric> for the average,
        <prefix>_<metric>_(min|max) for the minimum and the maximum

        Arguments
        ---------
        dist_stats_metric: speechbrain.utils.metric_stats.MultiMetricStats
            the metric for which statistics will be extracted
        prefix: str
            The string prefix.

        Returns
        -------
        Extracted stats
        """
        dist_stats = dist_stats_metric.summarize()
        return {
            self.get_stat_key(prefix, stat, metric_key): value
            for stat, stat_details in dist_stats.items()
            for metric_key, value in stat_details.items()
            if metric_key in {"average", "min_score", "max_score"}
        }

    def get_stat_key(self, prefix, stat, metric_key):
        """Returns the statistics key for the specified metric and statistics

        Arguments
        ---------
        prefix: str
            the prefix to be used
        stat: str
            the name of the statistic
        metric_key: str
            the metric key

        Returns
        -------
        key: str
            the key to be used
        """
        suffix = ""
        if metric_key != "average":
            suffix = "_" + metric_key.replace("_score", "")
        return f"{prefix}_{stat}{suffix}"

    def prepare_features(self, batch, stage):
        """Prepare the features for computation, including augmentation.

        Arguments
        ---------
        batch: PaddedData
            An input batch
        stage : sb.Stage
            The current stage of training.

        Returns
        -------
        feats: torch.Tensor
            features (normalized spectrograms)

        lens: torch.Tensor
            item lengths

        done: torch.Tensor
            a tensor indicating whether the sequence/spectrogram
            is finished

        """
        wavs, lens = batch.sig

        feats, feats_raw, lens = self.sig_to_feats(wavs, lens)

        # Compute metrics
        if self.hparams.enable_train_metrics:
            max_len = feats.size(2)
            mask = length_to_mask(lens * max_len, max_len)[
                :, None, :, None
            ].bool()
            self.data_dist_stats_metric.append(
                batch.file_name, feats_raw, mask=mask
            )

        return feats, lens

    def sig_to_feats(self, wavs, lens):
        """Performs feature extraction on the raw signal: MEL spectrogram +
        normalization + padding to fit UNets

        Arguments
        ---------
        wavs: torch.Tensor
            raw waveforms
        lens: torch.Tensor
            feature lengths

        Returns
        -------
        feats: torch.Tensor
            Global normed features
        feats_raw: torch.Tensor
            Unnormalized features
        lens: torch.Tensor
            Corresponding lengths of features
        """
        # Compute features
        feats = self.modules.compute_features(wavs)
        feats = feats.transpose(-1, -2)
        feats = feats.unsqueeze(1)

        # UNet downsamples features in multiples of 2. Reshape to ensure
        # there are no mismatched tensors due to ambiguity
        feats, lens = data_utils.pad_divisible(
            feats, lens, factor=self.hparams.downsample_factor, len_dim=2
        )

        feats, _ = data_utils.pad_divisible(
            feats, factor=self.hparams.downsample_factor, len_dim=3
        )

        # Min Level Norm
        feats_raw = self.modules.min_level_norm(feats)

        # Global Norm

        feats = self.modules.global_norm(
            feats_raw, lens, mask_value=self.mask_value_norm
        )
        return feats, feats_raw, lens

    def prepare_features_done(self, batch, feats, lens):
        """Prepares features for the done detector (a concatenation of one sample
        and a random sample)

        Arguments
        ---------
        batch: PaddedBatch
            a single batch of data
        feats: torch.Tensor
            spectrogram features
        lens: torch.Tensor
            feature lengths

        Returns
        -------
        feats_done: torch.Tensor
            features for the done detector (a concatenation)
        lens_done: torch.Tensor
            relative lengths of these features

        """
        wavs_random, lens_random = batch.sig_random
        feats_random, _, lens_random = self.sig_to_feats(
            wavs_random, lens_random
        )
        feats_done, lens_done = data_utils.concat_padded_features(
            feats=[feats, feats_random],
            lens=[lens, lens_random],
            feats_slice_start=[0.0, self.hparams.done_random_start_offset],
            feats_slice_end=[0.0, self.hparams.done_random_end_offset],
            dim=2,
        )
        return feats_done, lens_done

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        predictions : tensor
            The output tensor from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """

        (
            preds,
            noise,
            noisy_sample,
            feats,
            lens,
            autoencoder_out,
            feats_done,
            lens_done,
            pred_done,
        ) = predictions
        if self.train_diffusion:
            # NOTE: Padding of the latent space can affect the lengths
            lens_diffusion = (
                autoencoder_out.latent_length
                if self.diffusion_mode == DiffusionMode.LATENT
                else lens
            )
            loss = self.hparams.compute_cost(
                reshape_feats(preds),
                reshape_feats(noise),
                length=lens_diffusion,
            )
        else:
            loss = torch.tensor(0.0, device=self.device)

        # Append this batch of losses to the loss metric for easy
        self.loss_metric.append(
            batch.file_name, preds, noise, lens, reduction="batch"
        )

        if self.use_done_detector:
            max_len = feats.size(2)
            lens_target = (lens * max_len).int() - 1
            loss_done = self.hparams.compute_cost_done(
                pred_done.squeeze(-1), lens_target, length=lens_done
            )
            self.done_loss_metric.append(
                batch.file_name,
                pred_done.squeeze(-1),
                lens_target,
                length=lens_done,
                reduction="batch",
            )
        else:
            loss_done = None

        loss_autoencoder = None
        if (
            self.diffusion_mode == DiffusionMode.LATENT
            and self.train_autoencoder
        ):
            loss_autoencoder = self.hparams.compute_cost_autoencoder(
                autoencoder_out, feats, length=lens
            )
            self.autoencoder_loss_metric.append(
                batch.file_name,
                autoencoder_out,
                feats,
                length=lens,
                reduction="batch",
            )
            loss_laplacian = self.modules.compute_cost_laplacian(
                autoencoder_out.rec, length=lens
            )

            self.autoencoder_laplacian_loss_stats_metric.append(
                batch.file_name,
                autoencoder_out.rec,
                length=lens,
                reduction="batch",
            )

            loss_autoencoder += (
                self.hparams.loss_laplacian_weight * loss_laplacian
            )

            max_len = autoencoder_out.rec.size(2)
            rec_mask = length_to_mask(lens * max_len, max_len).unsqueeze(1)
            rec_mask = match_shape(rec_mask, autoencoder_out.rec)
            rec_denorm = self.modules.global_norm.denormalize(
                autoencoder_out.rec
            )
            self.autoencoder_rec_dist_stats_metric.append(
                batch.file_name, rec_denorm, mask=rec_mask
            )
            max_len = autoencoder_out.latent.size(2)
            latent_mask = length_to_mask(lens * max_len, max_len).unsqueeze(1)
            latent_mask = match_shape(latent_mask, autoencoder_out.latent)
            self.autoencoder_latent_dist_stats_metric.append(
                batch.file_name, autoencoder_out.latent, mask=latent_mask
            )

        return loss, loss_autoencoder, loss_done

    def generate_samples(self):
        """Generates spectrogram and (optionally) audio samples using the
        denoising diffusion model
        """
        labels, samples = self.generate_spectrograms()

        wav = None
        if self.hparams.eval_generate_audio:
            samples_denorm = self.denormalize(samples)
            wav = self.generate_audio(samples_denorm)

        return labels, samples, samples_denorm, wav

    def cut_samples(self, samples, wav):
        """Uses the done predictor to "chop" a batch of samples
        according to when it believes generation to be finished
        at a given state

        Arguments
        ---------
        samples: torch.Tensor
            a tensor of samples

        wav: torch.Tensor
            a tensor of generated audio (optional)

        Returns
        -------
        done_pred: torch.Tensor
            the raw output of the "done" predictor
        samples_cut: list
            a list of samples

        """
        done_in = samples.squeeze(1)[:, :, : self.hparams.spec_n_mels]
        done_pred = self.modules.done_detector(done_in)
        lens_pred = done_pred.squeeze().argmax(dim=-1)
        # NOTE: A poorly trained "done detector" may not cross the threshold
        # at all - in this case the sample will not be "cut"
        lens_pred[lens_pred == 0] = samples.size(2)
        samples_cut = [
            sample[:, :length, :] for sample, length in zip(samples, lens_pred)
        ]
        wav_lens_pred = (lens_pred / samples.size(2) * wav.size(-1)).int()
        wav_cut = [
            sample[:length] for sample, length in zip(wav, wav_lens_pred)
        ]
        return samples_cut, wav_cut

    def generate_spectrograms(self):
        """Generates sample spectrograms"""
        if self.is_conditioned:
            logger.info("Conditioned sampling")
            sample = self.generate_spectrograms_conditioned()
        else:
            logger.info("Unconditioned sampling")
            sample = self.generate_spectrograms_unconditioned()
        return sample

    def generate_spectrograms_unconditioned(self):
        """Generates spectrograms without conditioning"""
        sample = self.modules.diffusion_sample.sample(
            (
                self.hparams.eval_num_samples,
                self.hparams.diffusion_channels,
                self.hparams.eval_time_steps,
                self.hparams.spec_sample_size,
            )
        )
        labels = [str(idx) for idx in range(1, len(sample) + 1)]
        sample = self.modules.global_norm.denormalize(sample)
        return labels, sample

    def generate_spectrograms_conditioned(self):
        """Generates spectrograms with label conditioning"""
        sample_labels = self.sample_cond_labels()
        samples = [
            (label, idx, sample)
            for label in sample_labels
            for idx, sample in enumerate(
                self.generate_spectrograms_for_label(label)
            )
        ]
        labels = [
            self.get_sample_label(label, idx) for label, idx, _ in samples
        ]
        samples = [sample for _, _, sample in samples]
        return labels, samples

    def sample_cond_labels(self):
        """Generates a sample of conditioning labels
        based on hparams

        Returns
        -------
        result: list
            a list of dictionaries with speaker/digit
            combinations
        """
        label_samples = {}
        for key, cond_config in self.get_active_cond_emb().items():
            sample_count = cond_config["sample_count"]
            if sample_count is None:
                sample = torch.arange(cond_config["count"], device=self.device)
            else:
                sample = torch.randperm(
                    cond_config["count"], device=self.device
                )[:sample_count]
            label_samples[key] = sample

        samples = dict_value_combinations(label_samples)
        return samples

    def get_active_cond_emb(self):
        """Returns conditional embeddings that have been enabled
        in hyperparameters

        Returns
        -------
        cond_emb: dict
            all enabled conditional embedding configurations
        """
        return {
            key: value
            for key, value in self.hparams.cond_emb.items()
            if self.hparams.use_cond_emb[key]
        }

    def generate_spectrograms_for_label(self, label):
        """Generates samples for a specific label

        Arguments
        ---------
        label: dict
            a dictionary of labels with values to compute
            the embeddings

        Returns
        -------
        sample: torch.tensor
            a batch of spectrograms
        """
        label_msg = ", ".join(
            f"{key} = {value.item()}" for key, value in label.items()
        )
        logger.info("Generating samples for labels %s", label_msg)
        cond_emb = self.compute_cond_emb(label)
        sample = self.modules.diffusion_sample.sample(
            (
                self.hparams.eval_num_samples,
                self.hparams.diffusion_sample_channels,
                self.hparams.eval_time_steps,
                self.hparams.spec_sample_size,
            ),
            cond_emb=cond_emb,
        )
        return sample

    def get_sample_label(self, label, idx):
        """Gets a filename label for the specified sample

        Arguments
        ---------
        label: dict
            a dictionary similar to the following:
            {"digit": 4, "speaker": 10}
        idx: int
            the item index (will be appended)

        Returns
        -------
        result: str
            a formatted label. For the example above, it will
            be "digit_4_speaker_10"
        """
        label_str = "_".join(f"{key}_{value}" for key, value in label.items())
        return f"{label_str}_{idx}"

    def generate_rec_samples(self):
        predictions = self.compute_forward(self.reference_batch, sb.Stage.VALID)
        feats = predictions.autoencoder_output.rec
        if self.hparams.eval_generate_audio:
            wav = self.generate_audio(feats)
        return feats, wav

    def save_spectrograms(self, samples, path, folder="spec", labels=None):
        """Saves sample spectrograms to filesystem files

        Arguments
        ---------
        samples: torch.Tensor
            a tensor of sample spectrograms
        path: str
            the path to samples for a given epoch
        folder: str
            the name of the folder where the spectrograms
            will be saved
        labels: list
            a list of labels - for saving. If omitted, sequential
            samples will be used
        """
        spec_sample_path = os.path.join(path, folder)
        if not os.path.exists(spec_sample_path):
            os.makedirs(spec_sample_path)
        if labels is None:
            labels = range(len(samples))
        for label, sample in zip(labels, samples):
            spec_file_name = os.path.join(spec_sample_path, f"spec_{label}.png")
            self.save_spectrogram_sample(sample, spec_file_name, label=label)

    def save_raw(self, path=".", **kwargs):
        """Saves generated audio samples and spectrograms in
        raw form, for further analysis.

        This method accepts keywords arguments, and each argument
        becomes a key in the dictionary to be saved.

        Arguments
        ---------
        path: str
            the path
        **kwargs: dict
            The data to save
        """
        file_name = os.path.join(path, "raw.pt")
        data = {
            key: value for key, value in kwargs.items() if value is not None
        }
        torch.save(data, file_name)

    def save_spectrogram_sample(self, sample, file_name, label=None):
        """Saves a single spectrogram sample as an image

        Arguments
        ---------
        sample: torch.Tensor
            a single generated spectrogram (2D tensor)
        file_name: str
            the destination file name
        label: str
            The sample label to add to the title.
        """
        fig = plot_spectrogram(sample.transpose(-1, -2))
        if fig is not None:
            ax = fig.axes[0]
            if label:
                ax.set_title(f"Spectrogram Sample {label}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Features")
            fig.savefig(file_name)

    def denormalize(self, samples):
        """Undoes the normalization performed on spectrograms

        Arguments
        ---------
        samples: torch.Tensor
            normalized samples

        Returns
        -------
        result: torch.Tensor
            denormalized samples"""
        if not torch.is_tensor(samples):
            samples = torch.stack(samples)
        samples = samples[:, :, :, : self.hparams.spec_n_mels]
        samples = self.modules.min_level_norm.denormalize(samples)
        samples = AF.DB_to_amplitude(
            samples, ref=self.hparams.spec_ref, power=1.0
        )
        samples = self.modules.dynamic_range_compression(samples)
        return samples

    def generate_audio(self, samples):
        """Generates audio from spectrogram samples using a vocoder

        Arguments
        ---------
        samples: torch.Tensor
            a batch of generated spectrograms

        Returns
        -------
        audio: torch.Tensor
            generated audio for the samples (vocoder output)
        """
        vocoder_in = samples
        vocoder_in = vocoder_in.transpose(-1, -2)
        vocoder_in = vocoder_in.squeeze(1)
        return self.vocoder(vocoder_in)

    def save_audio(self, wav, path, folder="wav", labels=None):
        """Saves a batch of audio samples

        wav: torch.Tensor
            a batch of audio samples

        path: str
            the destination directory

        folder: str
            the subfolder within the destination directory

        labels: list
            a list of labels, for each sample. If omitted,
            sequential labels will be generated
        """
        wav_sample_path = os.path.join(path, folder)
        if not os.path.exists(wav_sample_path):
            os.makedirs(wav_sample_path)

        if labels is None:
            labels = range(len(wav))

        for label, sample in zip(labels, wav):
            wav_file_name = os.path.join(wav_sample_path, f"sample_{label}.wav")
            if self.hparams.norm_out_sample:
                max_samp, _ = sample.abs().max(1)
                sample = sample / max_samp
            self.save_audio_sample(sample.squeeze(0), wav_file_name)

    def compute_sample_metrics(self, samples):
        """Computes metrics (mean/std) on samples

        Arguments
        ---------
        samples: torch.Tensor
            a tensor of samples
        """
        sample_ids = torch.arange(1, len(samples) + 1)
        self.sample_mean_metric.append(sample_ids, samples)
        self.sample_std_metric.append(sample_ids, samples)

    def save_audio_sample(self, sample, file_name):
        """Saves a single audio sample

        Arguments
        ---------
        sample: torch.Tensor
            an audio sample
        file_name: str
            the file name to save
        """
        write_audio(
            file_name, sample, self.hparams.data_prepare_sample_rate_tgt
        )

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        self.train_diffusion = (epoch is None) or (
            epoch >= self.hparams.train_diffusion_start_epoch
        )
        self.train_autoencoder = (
            (epoch is not None)
            and (self.diffusion_mode == DiffusionMode.LATENT)
            and (epoch <= self.hparams.train_autoencoder_stop_epoch)
        )

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=self.hparams.compute_cost
        )
        if self.use_done_detector:
            self.done_loss_metric = sb.utils.metric_stats.MetricStats(
                metric=self.hparams.compute_cost_done
            )

        self.mask_value_norm = self.modules.min_level_norm(
            torch.tensor(self.hparams.pad_level_db, device=self.device)
        )

        if self.hparams.enable_train_metrics:
            self.data_dist_stats_metric = (
                sb.utils.metric_stats.MultiMetricStats(
                    metric=dist_stats, batch_eval=True
                )
            )

        if self.diffusion_mode == DiffusionMode.LATENT:
            self.autoencoder_loss_metric = (
                sb.utils.metric_stats.MultiMetricStats(
                    metric=self.hparams.compute_cost_autoencoder.details,
                    batch_eval=True,
                )
            )
            self.autoencoder_rec_dist_stats_metric = (
                sb.utils.metric_stats.MultiMetricStats(
                    metric=dist_stats, batch_eval=True
                )
            )
            self.autoencoder_latent_dist_stats_metric = (
                sb.utils.metric_stats.MultiMetricStats(
                    metric=dist_stats, batch_eval=True
                )
            )
            self.autoencoder_laplacian_loss_stats_metric = (
                sb.utils.metric_stats.MetricStats(
                    metric=self.hparams.compute_cost_laplacian, batch_eval=True
                )
            )

        self.sample_mean_metric = sb.utils.metric_stats.MetricStats(
            metric=masked_mean
        )
        self.sample_std_metric = sb.utils.metric_stats.MetricStats(
            metric=masked_std
        )
        self.sample_min_metric = sb.utils.metric_stats.MetricStats(
            metric=masked_min
        )
        self.sample_max_metric = sb.utils.metric_stats.MetricStats(
            metric=masked_max
        )
        self.sample_metrics = [
            self.sample_mean_metric,
            self.sample_std_metric,
            self.sample_min_metric,
            self.sample_max_metric,
        ]
        if stage == sb.Stage.TRAIN:
            self.modules.global_norm.unfreeze()
        else:
            self.modules.global_norm.freeze()

        if (
            self.hparams.enable_reference_samples or stage != sb.Stage.TRAIN
        ) and not hasattr(self, "vocoder"):
            self.vocoder = self.hparams.vocoder()
        if not hasattr(self, "reference_batch"):
            self.reference_batch = None
        self.reference_samples_needed = False
        self.is_conditioned = hasattr(self.hparams, "use_cond_emb") and any(
            self.hparams.use_cond_emb.values()
        )
        self.latent_mask_value = None
        self.use_done_detector = "done_detector" in self.modules

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {"loss": stage_loss}

        # At the end of validation...
        if stage == sb.Stage.VALID:
            # The train_logger writes a summary to stdout and to the logfile.
            lr = self.optimizer.param_groups[0]["lr"]
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["loss"])

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

        if stage == sb.Stage.TRAIN and self.hparams.enable_reference_samples:
            self.generate_reference_samples(self.reference_batch)

        if (
            stage != sb.Stage.TRAIN
            and epoch is not None
            and epoch % self.hparams.samples_interval == 0
        ):
            labels, samples, samples_denorm, wav = self.generate_samples()
            samples_rec, wav_rec = None, None
            data = {
                "labels": labels,
                "samples": samples,
                "samples_denorm": samples_denorm,
                "wav": wav,
            }
            if self.diffusion_mode == DiffusionMode.LATENT:
                samples_rec, wav_rec = self.generate_rec_samples()
                data["samples_rec"] = samples_rec
                data["wav_rec"] = wav_rec
            if self.use_done_detector:
                samples_cut, wav_cut = self.cut_samples(samples, wav)
                data["samples_cut"] = samples_cut
                data["wav_cut"] = wav_cut
            self.log_epoch(data, epoch, stage)

    def generate_reference_samples(self, batch):
        """Generate an audio sample from one of the spectrograms
        using the same normalization techniques

        Arguments
        ---------
        batch: speechbrain.dataio.batch.PaddedBatch
            a batch of audio
        """
        feats, lens = self.prepare_features(batch, sb.Stage.VALID)
        feats = self.modules.global_norm.denormalize(feats)
        feats_denorm = self.denormalize(feats)
        wav = self.generate_audio(feats_denorm)
        self.log_samples(
            spectrogram_samples=feats,
            wav_samples=wav,
            lens=lens,
            key_prefix="reference_",
        )
        ref_sample_path = os.path.join(self.hparams.sample_folder, "ref")
        self.save_spectrograms(feats, ref_sample_path)
        self.save_audio(wav, path=ref_sample_path)

    def log_epoch(self, data, epoch, stage):
        """Saves end-of-epoch logs

        Arguments
        ---------
        data: dict
            the data to be logged, with the following keys
            samples: generated samples
            wav: generated waveform
            samples_rec: reconstruction samples (to assess autoencoder quality)
            samples_wav: reconstruction audio (to assess autoencoder quality)
        epoch: int
            the epoch number
        stage: speechbrain.Stage
            the training stage

        """
        epoch_sample_path = os.path.join(self.hparams.sample_folder, str(epoch))
        samples, samples_denorm, wav, labels, samples_rec, wav_rec = [
            data.get(key)
            for key in [
                "samples",
                "samples_denorm",
                "wav",
                "labels",
                "samples_rec",
                "wav_rec",
            ]
        ]
        if not torch.is_tensor(samples):
            samples = torch.stack(samples)
        samples_log = data.get("samples_cut", samples)
        wav_log = data.get("wav_cut", wav)
        self.save_spectrograms(samples_log, epoch_sample_path, labels=labels)
        sample_ids = torch.arange(1, len(samples) + 1)
        for metric in self.sample_metrics:
            metric.append(sample_ids, samples)
        if wav is not None:
            self.save_audio(wav_log, epoch_sample_path, labels=labels)
        if self.diffusion_mode == DiffusionMode.LATENT:
            self.save_spectrograms(
                samples_rec, epoch_sample_path, folder="spec_rec"
            )
            if wav_rec is not None:
                self.save_audio(wav_rec, epoch_sample_path, folder="wav_rec")

        self.save_raw(
            spec=samples,
            spec_denorm=samples_denorm,
            wav=wav,
            spec_rec=samples_rec,
            wav_rec=wav_rec,
            path=epoch_sample_path,
        )
        if self.hparams.use_tensorboard:
            sample_mean_stats = self.sample_mean_metric.summarize()
            sample_std_stats = self.sample_std_metric.summarize()
            sample_min_stats = self.sample_min_metric.summarize()
            sample_max_stats = self.sample_max_metric.summarize()
            stats = {
                "sample_mean": sample_mean_stats["average"],
                "sample_mean_min": sample_mean_stats["min_score"],
                "sample_mean_max": sample_mean_stats["max_score"],
                "sample_std": sample_std_stats["average"],
                "sample_std_min": sample_std_stats["min_score"],
                "sample_std_max": sample_std_stats["max_score"],
                "sample_min": sample_min_stats["min_score"],
                "sample_max": sample_max_stats["max_score"],
            }
            stats_args = {f"{stage.name.lower()}_stats": stats}
            self.hparams.tensorboard_train_logger.log_stats(
                stats_meta={"step": self.step}, **stats_args
            )
            self.log_samples(
                spectrogram_samples=samples_log, wav_samples=wav_log
            )
            if self.diffusion_mode == DiffusionMode.LATENT:
                self.log_samples(
                    spectrogram_samples=samples_rec,
                    wav_samples=wav_rec,
                    key_prefix="rec_",
                )

    def log_samples(
        self,
        spectrogram_samples=None,
        wav_samples=None,
        key_prefix=None,
        lens=None,
    ):
        """Logs a set of audio and spectrogram samples

        Arguments
        ---------
        spectrogram_samples: torch.Tensor
            a tensor of spectrogram samples

        wav_samples: torch.Tensor
            a tensor of audio samples

        key_prefix: str
            the prefix to use for keys in Tensorboard logging (if applicable)

        lens: torch.Tensor
            relative sample lengths

        """
        if key_prefix is None:
            key_prefix = ""
        if lens is None:
            lens = torch.ones(len(spectrogram_samples), device=self.device)
        if self.hparams.use_tensorboard:
            for sample in spectrogram_samples:
                self.hparams.tensorboard_train_logger.log_figure(
                    f"{key_prefix}spectrogram", sample
                )
            if wav_samples is not None:
                max_len = max(sample.size(-1) for sample in wav_samples)
                lens_full = (lens * max_len).int()
                for wav_sample, sample_len in zip(wav_samples, lens_full):
                    self.tb_writer.add_audio(
                        f"{key_prefix}audio", wav_sample[:, :sample_len]
                    )

    @property
    def tb_writer(self):
        """Returns the raw Tensorboard logger writer"""
        return self.hparams.tensorboard_train_logger.writer


DATASET_SPLITS = ["train", "valid", "test"]


def reshape_feats(feats):
    """Reshapes tensors of shape (batch x channels x features x length)
    to (batch x length x features), suitable for standard SpeechBrain
    losses, such as `mse_loss`

    Arguments
    ---------
    feats: torch.Tensor
        a feature tensor of shape (batch x channels x features x length)

    Returns
    -------
    result: torch.Tensor
        a feature tensor of shape (batch x length x features)
    """
    return feats.squeeze(1).transpose(1, -1)


def apply_sort(hparams, dataset):
    if hparams["sort"]:
        dataset = dataset.filtered_sorted(sort_key=hparams["sort"])
    if hparams["batch_shuffle"]:
        dataset = dataset.batch_shuffle(hparams["batch_size"])
    return dataset


def load_dataset(hparams):
    dataset_splits = {}
    data_folder = hparams["data_save_folder"]
    for split_id in DATASET_SPLITS:
        split_path = hparams[f"{split_id}_json"]
        dataset_split = sb.dataio.dataset.DynamicItemDataset.from_json(
            split_path, replacements={"data_root": data_folder}
        )
        dataset_split = apply_sort(hparams, dataset_split)
        dataset_splits[split_id] = dataset_split
    return dataset_splits


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    We expect `prepare_mini_librispeech` to have been called before this,
    so that the `train.json`, `valid.json`,  and `valid.json` manifest files
    are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("file_name")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and output it"""
        return read_audio(wav, hparams)

    @sb.utils.data_pipeline.takes("digit", "speaker_id")
    @sb.utils.data_pipeline.provides("digit_label", "speaker_label")
    def labels_pipeline(digit, speaker_id):
        yield int(digit)
        yield int(speaker_id) - 1

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.

    dataset_splits = load_dataset(hparams)
    dataset_splits_values = dataset_splits.values()

    output_keys = ["file_name", "sig", "digit_label", "speaker_label"]
    if "done_detector" in hparams:
        output_keys += ["file_name_random", "sig_random"]

    sb.dataio.dataset.set_output_keys(
        dataset_splits_values,
        output_keys,
    )
    sb.dataio.dataset.add_dynamic_item(dataset_splits_values, audio_pipeline)
    sb.dataio.dataset.add_dynamic_item(dataset_splits_values, labels_pipeline)
    for dataset_split in dataset_splits_values:
        enhance_with_random(dataset_split, hparams)

    train_split = dataset_splits["train"]
    data_count = None
    train_split = apply_overfit_test(
        hparams["overfit_test"],
        hparams["overfit_test_sample_count"],
        hparams["overfit_test_epoch_data_count"],
        train_split,
    )

    if hparams["train_data_count"] is not None:
        data_count = hparams["train_data_count"]
        train_split.data_ids = train_split.data_ids[:data_count]
    dataset_splits["train"] = train_split

    return dataset_splits


def read_audio(wav, hparams):
    """Reads an audio file, applying random amplitude

    Arguments
    ---------
    wav: str
        the file name (absolute or relative)
    hparams: dict
        hyperparameters

    Returns
    -------
    sig: torch.Tensor
        The loaded audio.
    """
    sig = sb.dataio.dataio.read_audio(wav)

    # To Support random amplitude
    if hparams["rand_amplitude"]:
        rand_amp = (hparams["max_amp"] - hparams["min_amp"]) * torch.rand(
            1
        ) + hparams["min_amp"]
        sig = sig / sig.abs().max()
        sig = sig * rand_amp
    return sig


def enhance_with_random(dataset, hparams):
    """Enhances the pipeline with an additional randomly chosen sample for
    each sample - used for training the Done detector - to determine word
    boundaries

    Arguments
    ---------
    dataset: DynamicItemDataset
        the dataset to be enhanced
    hparams: dict
        the hyperparameters dictionary
    """
    item_count = len(dataset)

    @sb.utils.data_pipeline.provides("file_name_random", "sig_random")
    def extra_random_sample():
        idx = torch.randint(item_count, (1,)).item()
        data_id = dataset.data_ids[idx]
        data_item = dataset.data[data_id]
        wav_random = data_item["file_name"]
        return wav_random, read_audio(wav_random, hparams)

    dataset.add_dynamic_item(extra_random_sample)


def check_tensorboard(hparams):
    """Checks whether Tensorboard is enabled and initializes the logger if it is

    Arguments
    ---------
    hparams: dict
        the hyperparameter dictionary
    """
    if hparams["use_tensorboard"]:
        try:
            from speechbrain.utils.train_logger import TensorboardLogger

            hparams["tensorboard_train_logger"] = TensorboardLogger(
                hparams["tensorboard_logs"]
            )
        except ImportError:
            logger.warning(
                "Could not enable Tensorboard logging - Tensorboard is not available"
            )
            hparams["use_tensorboard"] = False


# Recipe begins!
if __name__ == "__main__":
    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Check whether Tensorboard is available and enabled
    check_tensorboard(hparams)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    run_on_main(
        prepare_audiomnist,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["data_save_folder"],
            "train_json": hparams["train_json"],
            "valid_json": hparams["valid_json"],
            "test_json": hparams["test_json"],
            "metadata_folder": hparams["metadata_folder"],
            "norm": hparams["data_prepare_norm"],
            "trim": hparams["data_prepare_trim"],
            "trim_threshold": hparams["data_prepare_trim_threshold"],
            "src_sample_rate": hparams["data_prepare_sample_rate_src"],
            "tgt_sample_rate": hparams["data_prepare_sample_rate_tgt"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Create dataset objects "train", "valid", and "test".
    diffusion_datasets = dataio_prep(hparams)

    # Initialize the Brain object to prepare for mask training.
    diffusion_brain = DiffusionBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    diffusion_brain.fit(
        epoch_counter=diffusion_brain.hparams.epoch_counter,
        train_set=diffusion_datasets["train"],
        valid_set=diffusion_datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    test_stats = diffusion_brain.evaluate(
        test_set=diffusion_datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["dataloader_options"],
    )
