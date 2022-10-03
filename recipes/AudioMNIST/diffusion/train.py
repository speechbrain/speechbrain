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
import math
import sys
import torch
import speechbrain as sb
import os
from hyperpyyaml import load_hyperpyyaml
from torchaudio import functional as AF
from speechbrain.dataio.dataio import write_audio
from speechbrain.utils import data_utils
from speechbrain.utils.train_logger import plot_spectrogram


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
        feats, lens = self.prepare_features(batch.sig, stage)

        pred, noise, noisy_sample = self.modules.diffusion.train_sample(feats, lens=lens)

        return pred, noise, noisy_sample

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        result = super().fit_batch(batch)
        self.hparams.lr_annealing(self.optimizer)
        return result

    def prepare_features(self, wavs, stage):
        """Prepare the features for computation, including augmentation.

        Arguments
        ---------
        wavs : tuple
            Input signals (tensor) and their relative lengths (tensor).
        stage : sb.Stage
            The current stage of training.
        """
        wavs, lens = wavs

        # Compute features
        feats = self.modules.compute_features(wavs)
        feats = feats.transpose(-1, -2)
        feats = feats.unsqueeze(1)

        # Randomly initialize time steps for each image
        timesteps = torch.randint(
            0, self.hparams.train_timesteps, (len(feats),), device=self.device
        )

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
            value=self.hparams.min_level_db,
        )

        # Min Level Norm
        feats = self.modules.min_level_norm(feats)

        # Global Norm
        feats = self.modules.global_norm(feats, lens)

        # Add noise
        #        noise = torch.randn_like(feats)

        # Match padding on the noise
        #        max_len = feats.size(2)
        #        mask = length_to_mask(lens * max_len, max_len).bool()
        #        mask = mask.unsqueeze(1).unsqueeze(-1)
        #        mask = mask.expand(*feats.shape)
        #        noise[:, :, :, self.hparams.spec_n_mels :] = 0.0
        #        noise[~mask] = 0.0

        #        feats_noisy = self.hparams.noise_scheduler.add_noise(
        #            feats, noise, timesteps
        #        )
        #        feats_noisy[:, :, :, self.hparams.spec_n_mels :] = 0.0
        #        feats_noisy[~mask] = -1.0
        tail = time_dim % self.hparams.downsample_factor
        if tail > 0:
            feats[:, :, -tail:, :] = 0.0

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
        _, lens = batch.sig

        preds, noise, noisy_sample = predictions

        loss = self.hparams.compute_cost(
            preds.squeeze(1), noise.squeeze(1), length=lens
        )

        # Append this batch of losses to the loss metric for easy
        self.loss_metric.append(
            batch.file_name, preds, noise, lens, reduction="batch"
        )

        return loss

    def generate_samples(self, epoch_sample_path):
        """Generates samples from diffusion
        
        epochs_sample_path: str
            the path to the samples for the current epoch
        """
        samples = self.generate_spectrograms()
        self.save_spectrograms(samples, epoch_sample_path)

        if self.hparams.eval_generate_audio:
            wav = self.generate_audio(samples)
            self.save_audio(wav, epoch_sample_path)

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

    def save_spectrograms(self, samples, epoch_sample_path):
        """Saves sample spectrograms to filesystem files
        
        Arguments
        ---------
        samples: torch.Tensor
            a tensor of sample spectrograms
        epoch_sample_path
            ths path to samples for a given epoch
        """
        spec_sample_path = os.path.join(epoch_sample_path, "spec")
        if not os.path.exists(spec_sample_path):
            os.makedirs(spec_sample_path)
        for idx, sample in enumerate(samples):
            spec_file_name = os.path.join(spec_sample_path, f"spec_{idx}.png")
            self.save_spectrogram_sample(sample, spec_file_name)

    def save_spectrogram_sample(self, sample, file_name):
        """Saves a single spectrogram sample
        
        """
        fig = plot_spectrogram(sample)
        if fig is not None:
            fig.savefig(file_name)

    def generate_audio(self, samples):
        vocoder_in = samples[:, :, : self.hparams.spec_n_mels]
        vocoder_in = vocoder_in.transpose(-1, -2)
        vocoder_in = self.modules.min_level_norm.denormalize(vocoder_in)
        vocoder_in = AF.DB_to_amplitude(
            vocoder_in, ref=self.hparams.spec_ref, power=1.0
        )
        vocoder_in = self.modules.dynamic_range_compression(vocoder_in)
        vocoder_in = vocoder_in.squeeze(1)
        return self.vocoder(vocoder_in)

    def save_audio(self, wav, epoch_sample_path):
        wav_sample_path = os.path.join(epoch_sample_path, "wav")
        if not os.path.exists(wav_sample_path):
            os.makedirs(wav_sample_path)
        for idx, sample in enumerate(wav):
            wav_file_name = os.path.join(wav_sample_path, f"sample_{idx}.wav")
            self.save_audio_sample(sample.squeeze(0), wav_file_name)

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
        if stage == sb.Stage.VALID and not hasattr(self, "vocoder"):
            self.vocoder = self.hparams.vocoder()

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

        if stage != sb.Stage.TRAIN:
            epoch_sample_path = os.path.join(
                self.hparams.sample_folder, str(epoch)
            )
            self.generate_samples(epoch_sample_path)


DATASET_SPLITS = ["train", "valid", "test"]


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
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    dataset = datasets.load_dataset(hparams["dataset"])

    datasets_splits = {}
    hparams["dataloader_options"]["shuffle"] = True
    for split_id in DATASET_SPLITS:
        datasets_splits[
            split_id
        ] = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(
            dataset[split_id],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline],
            output_keys=["file_name", "sig", "digit", "speaker_id"],
        )
#        datasets_splits[split_id].data_ids = datasets_splits[split_id].data_ids[:2000]

    return datasets_splits


# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Create dataset objects "train", "valid", and "test".
    datasets = dataio_prep(hparams)

    # Initialize the Brain object to prepare for mask training.
    spk_id_brain = DiffusionBrain(
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
    spk_id_brain.fit(
        epoch_counter=spk_id_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    test_stats = spk_id_brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["dataloader_options"],
    )

