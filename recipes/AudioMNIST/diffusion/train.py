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
import datasets
import logging
import math
import sys
import torch
import speechbrain as sb
import os
from hyperpyyaml import load_hyperpyyaml
from torchaudio import functional as AF
from torchaudio import transforms
from speechbrain.dataio.dataio import length_to_mask, write_audio
from speechbrain.utils import data_utils
from speechbrain.utils.train_logger import plot_spectrogram
from speechbrain.utils.data_utils import unsqueeze_as

logger = logging.getLogger(__name__)


# Brain class for speech enhancement training
class DiffusionBrain(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""

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
        predictions : Tensor
            Tensor that contains the posterior probabilities over the N classes.
        """

        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)

        # Compute features, embeddings, and predictions
        feats, lens = self.prepare_features(batch, stage)

        pred, noise, noisy_sample = self.modules.diffusion.train_sample(
            feats, lens=lens
        )

        # NOTE: lens can change because of the additional padding needed to account
        # NOTE: for downsampling
        return pred, noise, noisy_sample, lens

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        if self.reference_batch is None:
            self.reference_batch = batch
        loss = super().fit_batch(batch)
        self.hparams.lr_annealing(self.optimizer, self.optimizer_step)
        if (
            self.hparams.enable_train_metrics
            and self.hparams.use_tensorboard
            and (
                self.step == 1
                or self.step % self.hparams.train_log_interval == 0
            )
        ):
            self.log_batch()
        return loss

    def log_batch(self):
        loss_stats = self.loss_metric.summarize()
        data_mean_stats = self.data_mean_metric.summarize()
        data_std_stats = self.data_std_metric.summarize()
        data_min_stats = self.data_min_metric.summarize()
        data_max_stats = self.data_max_metric.summarize()
        stats = {
            "loss": loss_stats["average"],
            "lr": self.optimizer.param_groups[0]["lr"],
            "data_mean": data_mean_stats["average"],
            "data_mean_min": data_mean_stats["min_score"],
            "data_mean_max": data_mean_stats["max_score"],
            "data_std": data_std_stats["average"],
            "data_std_min": data_std_stats["min_score"],
            "data_std_max": data_std_stats["max_score"],
            "data_min": data_min_stats["min_score"],
            "data_max": data_max_stats["max_score"],
        }
        self.hparams.tensorboard_train_logger.log_stats(
            stats_meta={"step": self.step}, train_stats=stats
        )

    def prepare_features(self, batch, stage):
        """Prepare the features for computation, including augmentation.

        Arguments
        ---------
        batch: PaddedData
            An input batch
        stage : sb.Stage
            The current stage of training.
        """
        wavs, lens = batch.sig

        # Compute features
        feats = self.modules.compute_features(wavs)
        feats = feats.transpose(-1, -2)
        feats = feats.unsqueeze(1)

        # UNet downsamples features in multiples of 2. Reshape to ensure
        # there are no mismatched tensors due to ambiguity
        batch_dim, channel_dim, time_dim, feats_dim = feats.shape
        desired_time_dim = (
            math.ceil(time_dim / self.hparams.downsample_factor)
            * self.hparams.downsample_factor
        )
        desired_feats_dim = (
            math.ceil(feats_dim / self.hparams.downsample_factor)
            * self.hparams.downsample_factor
        )
        feats, _ = data_utils.pad_right_to(
            feats,
            (batch_dim, channel_dim, desired_time_dim, desired_feats_dim),
            value=self.hparams.pad_level_db,
        )
        tail = time_dim % self.hparams.downsample_factor
        if tail > 0:
            feats[:, :, -tail:, :] = self.hparams.pad_level_db

        # Adjust lengths to the new dimenson, post-padding
        lens = lens * (time_dim / desired_time_dim)

        # Min Level Norm
        feats_raw = self.modules.min_level_norm(feats)

        # Global Norm
        mask_value = self.modules.min_level_norm(
            torch.tensor(self.hparams.pad_level_db, device=feats_raw.device)
        )

        feats = self.modules.global_norm(feats_raw, lens, mask_value=mask_value)

        # Compute metrics
        if self.hparams.enable_train_metrics:
            max_len = feats.size(2)
            mask = length_to_mask(lens * max_len, max_len)[
                :, None, :, None
            ].bool()
            for metric in self.data_metrics:
                metric.append(batch.file_name, feats_raw, mask=mask)

        return feats, lens

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

        preds, noise, noisy_sample, lens = predictions

        loss = self.hparams.compute_cost(
            preds.squeeze(1), noise.squeeze(1), length=lens
        )

        # Append this batch of losses to the loss metric for easy
        self.loss_metric.append(
            batch.file_name, preds, noise, lens, reduction="batch"
        )

        return loss

    def generate_samples(self):
        """Generates spectrogram and (optionally) audio samples using the
        denoising diffusion model
        """
        samples = self.generate_spectrograms()

        wav = None
        if self.hparams.eval_generate_audio:
            wav = self.generate_audio(samples)

        return samples, wav

    def generate_spectrograms(self):
        """Generates sample spectrograms"""
        sample = self.modules.diffusion.sample(
            (
                self.hparams.eval_num_samples,
                1,
                self.hparams.spec_sample_size,
                self.hparams.spec_sample_size,
            )
        )
        sample = self.modules.global_norm.denormalize(sample)
        return sample

    def save_spectrograms(self, samples, path):
        """Saves sample spectrograms to filesystem files

        Arguments
        ---------
        samples: torch.Tensor
            a tensor of sample spectrograms
        path: str
            ths path to samples for a given epoch
        """
        spec_sample_path = os.path.join(path, "spec")
        if not os.path.exists(spec_sample_path):
            os.makedirs(spec_sample_path)
        for idx, sample in enumerate(samples):
            spec_file_name = os.path.join(spec_sample_path, f"spec_{idx}.png")
            self.save_spectrogram_sample(sample, spec_file_name)

    def save_spectrogram_sample(self, sample, file_name):
        """Saves a single spectrogram sample as an image


        Arguments
        ---------
        sample: torch.Tensor
            a single generated spectrogram (2D tensor)
        file_name: str
            the destination file name

        """
        fig = plot_spectrogram(sample)
        if fig is not None:
            fig.savefig(file_name)

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
        vocoder_in = samples[:, :, : self.hparams.spec_n_mels]
        vocoder_in = vocoder_in.transpose(-1, -2)
        vocoder_in = self.modules.min_level_norm.denormalize(vocoder_in)
        vocoder_in = AF.DB_to_amplitude(
            vocoder_in, ref=self.hparams.spec_ref, power=1.0
        )
        vocoder_in = self.modules.dynamic_range_compression(vocoder_in)
        vocoder_in = vocoder_in.squeeze(1)
        return self.vocoder(vocoder_in)

    def save_audio(self, wav, path):
        """Saves a batch of audio samples

        wav: torch.Tensor
            a batch of audio samples

        path: str
            the destination directory
        """
        wav_sample_path = os.path.join(path, "wav")
        if not os.path.exists(wav_sample_path):
            os.makedirs(wav_sample_path)
        for idx, sample in enumerate(wav):
            wav_file_name = os.path.join(wav_sample_path, f"sample_{idx}.wav")
            self.save_audio_sample(sample.squeeze(0), wav_file_name)

    def compute_sample_metrics(self, samples):
        """Computes metrics (mean/std) on samples

        Arguments
        ---------
        """
        sample_ids = torch.arange(1, len(samples) + 1)
        self.sample_mean_metric.append(sample_ids, samples)
        self.sample_std_metric.append(sample_ids, samples)

    def save_audio_sample(self, sample, file_name):
        write_audio(file_name, sample, self.hparams.sample_rate_tgt)

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

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.mse_loss
        )
        if self.hparams.enable_train_metrics:
            self.data_mean_metric = sb.utils.metric_stats.MetricStats(
                metric=masked_mean
            )
            self.data_std_metric = sb.utils.metric_stats.MetricStats(
                metric=masked_std
            )
            self.data_min_metric = sb.utils.metric_stats.MetricStats(
                metric=masked_min
            )
            self.data_max_metric = sb.utils.metric_stats.MetricStats(
                metric=masked_max
            )
            self.data_metrics = [
                self.data_mean_metric,
                self.data_std_metric,
                self.data_min_metric,
                self.data_max_metric,
            ]

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
        self.reference_batch = None
        self.reference_samples_neeed = False

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

        if stage != sb.Stage.TRAIN:
            samples, wav = self.generate_samples()
            self.log_epoch(samples, wav, epoch, stage)

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
        wav = self.generate_audio(feats)
        self.log_samples(
            spectrogram_samples=feats,
            wav_samples=wav,
            lens=lens,
            key_prefix="reference_",
        )
        ref_sample_path = os.path.join(self.hparams.sample_folder, "ref")
        self.save_spectrograms(feats, ref_sample_path)
        self.save_audio(wav, path=ref_sample_path)

    def log_epoch(self, samples, wav, epoch, stage):
        """Saves end-of-epoch logs

        Arguments
        ---------
        samples: torch.Tensor
            Generated spectrograms
        wav: torch.Tensor
            Generated audio waveforms
        epoch: int
            the epoch number
        stage: speechbrain.Stage
            the training stage

        """
        epoch_sample_path = os.path.join(self.hparams.sample_folder, str(epoch))
        self.save_spectrograms(samples, epoch_sample_path)
        sample_ids = torch.arange(1, len(samples) + 1)
        for metric in self.sample_metrics:
            metric.append(sample_ids, samples)
        if wav is not None:
            self.save_audio(wav, epoch_sample_path)
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
            self.log_samples(spectrogram_samples=samples, wav_samples=wav)

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
            lens = torch.ones(
                len(spectrogram_samples), device=spectrogram_samples.device
            )
        if self.hparams.use_tensorboard:
            for sample in spectrogram_samples:
                self.hparams.tensorboard_train_logger.log_figure(
                    f"{key_prefix}spectrogram", sample
                )
            if wav_samples is not None:
                max_len = wav_samples.size(-1)
                lens_full = (lens * max_len).int()
                for wav_sample, sample_len in zip(wav_samples, lens_full):
                    self.tb_writer.add_audio(
                        f"{key_prefix}audio", wav_sample[:, :sample_len]
                    )

    @property
    def tb_writer(self):
        """Returns the raw TensorBoard logger writer"""
        return self.hparams.tensorboard_train_logger.writer


DATASET_SPLITS = ["train", "valid", "test"]


def load_dataset(hparams):
    dataset_splits = {}
    if os.path.exists(hparams["dataset"]):
        # Use a folder:
        for split_id in DATASET_SPLITS:
            split_path = os.path.join(hparams["dataset"], f"{split_id}.json")
            dataset_splits[
                split_id
            ] = sb.dataio.dataset.DynamicItemDataset.from_json(split_path)
    else:
        dataset = datasets.load_dataset(hparams["dataset"])
        for split_id in DATASET_SPLITS:
            dataset_splits[
                split_id
            ] = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(
                dataset[split_id],
                replacements={"data_root": hparams["data_folder"]},
            )
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

    resample = transforms.Resample(
        orig_freq=hparams["sample_rate_src"],
        new_freq=hparams["sample_rate_tgt"],
    )

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("file_name")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and output it"""
        if not os.path.isabs(wav):
            wav = os.path.join(hparams["dataset"], wav)
        sig = sb.dataio.dataio.read_audio(wav)
        sig = resample(sig)
        return sig

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.

    dataset_splits = load_dataset(hparams)
    dataset_splits_values = dataset_splits.values()

    sb.dataio.dataset.set_output_keys(
        dataset_splits_values, ["file_name", "sig", "digit", "speaker_id"]
    )
    sb.dataio.dataset.add_dynamic_item(dataset_splits_values, audio_pipeline)

    train_split = dataset_splits["train"]
    data_count = None
    if hparams["overfit_test"]:
        sample_count = hparams["overfit_test_sample_count"]
        epoch_data_count = hparams["overfit_test_epoch_data_count"]
        num_repetitions = math.ceil(epoch_data_count / sample_count)
        overfit_samples = train_split.data_ids[:sample_count] * num_repetitions
        overfit_samples = overfit_samples[:epoch_data_count]
        train_split.data_ids = overfit_samples
    elif hparams["train_data_count"] is not None:
        data_count = hparams["train_data_count"]
        train_split.data_ids = train_split.data_ids[:data_count]

    return dataset_splits


def non_batch_dims(sample):
    """Returns all dimensons of the specified tensor
    except the batch dimension

    Arguments
    ---------
    sample: torch.Tensor
        an arbitrary tensor

    Returns
    -------
    dims: list
        a list of dimensions
    """
    return list(range(1, sample.dim()))


def masked_mean(sample, mask=None):
    """A metric function that computes the mean of each sample, excluding
    padding

    Arguments
    ---------
    samples: torch.Tensor
        a tensor of spectrograms

    mask: torch.Tensor
        a length mask

    Returns
    -------
    result: torch.Tensor
        a tensor fo means
    """
    if mask is None:
        mask = torch.ones_like(sample).bool()
    dims = non_batch_dims(sample)
    return (sample * mask).sum(dim=dims) / mask.expand_as(sample).sum(dim=dims)


def masked_std(sample, mask=None):
    """A metric function that computes the standard deviation of each
    sample, excluding padding

    Arguments
    ---------
    samples: torch.Tensor
        a tensor of spectrograms

    mask: torch.Tensor
        a length mask

    Returns
    -------
    result: torch.Tensor
        a tensor fo means
    """
    if mask is None:
        mask = torch.ones_like(sample).bool()
    dims = non_batch_dims(sample)
    mean = unsqueeze_as(masked_mean(sample, mask), sample)
    diff_sq = ((sample - mean) * mask) ** 2
    return (
        diff_sq.sum(dim=dims) / (mask.expand_as(diff_sq).sum(dim=dims) - 1)
    ).sqrt()


def masked_min(sample, mask=None):
    """A metric function that computes the minimum of each sample

    Arguments
    ---------
    samples: torch.Tensor
        a tensor of spectrograms

    mask: torch.Tensor
        a length mask

    Returns
    -------
    result: torch.Tensor
        a tensor fo means
    """
    if mask is None:
        mask = torch.ones_like(sample).bool()
    dims = non_batch_dims(sample)
    return sample.masked_fill(~mask, torch.inf).amin(dim=dims)


def masked_max(sample, mask=None):
    """A metric function that computes the minimum of each sample

    Arguments
    ---------
    samples: torch.Tensor
        a tensor of spectrograms

    mask: torch.Tensor
        a length mask

    Returns
    -------
    result: torch.Tensor
        a tensor fo means
    """
    if mask is None:
        mask = torch.ones_like(sample).bool()
    dims = non_batch_dims(sample)
    return sample.masked_fill(~mask, -torch.inf).amax(dim=dims)


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
                "Could not enable TensorBoard logging - TensorBoard is not available"
            )
            hparams["use_tensorboard"] = False


# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Check whether Tensorboard is available and enabled
    check_tensorboard(hparams)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
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
