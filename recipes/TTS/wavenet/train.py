'''
Recipe for training WaveNet, a fully-convolutional generative model for raw audio, 
with option to condition on mel spectrograms and speaker embeddings. 
When conditioned on mel spectrograms, it can be used as a vocoder in a text-to-speech (TTS) system,
taking as input the mel spectrogram from a model like Tacotron or DeepVoice3.

https://arxiv.org/abs/1609.03499

To run this recipe, do:
1. Select data path for training data within hparams/hparams.yaml
2. Run: python train.py hparams/hparams.yaml 

Authors
* Aleksandar Rachkov 2021
'''

import os
import sys
import math
import torch
import numpy as np
import speechbrain as sb
from torch.nn import functional as F
from hyperpyyaml import load_hyperpyyaml

sys.path.append("..")
import torchvision
import torchaudio
from datasets.vctk import VCTK
from common.dataio_wavenet import dataio_prep, mulaw_quantize, inv_mulaw_quantize


class WavenetBrain(sb.core.Brain):
    """
    The Brain for WaveNet implementation within SpeechBrain
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_forward(self, batch, stage):
        """
        For an input batch, it computes the predictions at each timestep, for each of the mu=256 classes

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains the relevant tensors for training
            batch.x.data: torch.Tensor, (B x T x C)
                pre-processed input audio, quantized in C=256 channels
            batch.mel.data: torch.Tensor , (B x T x C)
                pre-processed mel spectrogram with C=num_mels hyperparameter
            batch.speker_id_cat.data: torch.Tensor, (B)
                categorically-encoded speaker embeddings for the batch
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions: torch.Tensor, (B x T x C)
            Computed predictions at each timestep
        """

        batch = batch.to(self.device)

        predictions = self.hparams.model(
                x=batch.x.data,
                c=batch.mel.data,
                g=batch.speaker_id_cat.data
            )
            
        return predictions

    def compute_objectives(self, predictions, batch, stage):
        """
        Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        predictions : torch.Tensor
            The posterior probabilities from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        
        # wee need 4d inputs for spatial cross entropy loss
        # (B, T, C, 1)
        y_hat = predictions.unsqueeze(-1)
        target = batch.target.data.unsqueeze(-1)
        lengths = batch.target_length

        loss = self.hparams.compute_cost(
            y_hat[:,:-1,:,:], target[:,1:,:], lengths=lengths
        )

        # (B, T)
        y_hat = F.softmax(y_hat, dim=1).max(2)[1]

        predicted_audio = inv_mulaw_quantize(y_hat)
        target_audio = inv_mulaw_quantize(target)

        # creating progress samples
        if self.hparams.progress_samples:
            predicted_mel = torchaudio.transforms.MelSpectrogram(self.hparams.sample_rate)(predicted_audio.squeeze())
            target_mel = torchaudio.transforms.MelSpectrogram(self.hparams.sample_rate)(target_audio.squeeze())

            self.last_predicted_audio = predicted_audio.detach().cpu() 
            self.last_target_audio = target_audio.detach().cpu()
            self.last_predicted_mel = predicted_mel.detach().cpu()
            self.last_target_mel = target_mel.detach().cpu()

        return loss

    def on_fit_start(self):
        # modifies on_fit_start to create progress sample path directory
        super().on_fit_start()
        if self.hparams.progress_samples:
            if not os.path.exists(self.hparams.progress_sample_path):
                os.makedirs(self.hparams.progress_sample_path)

    def on_stage_end(self, stage, stage_loss, epoch):
        """
        Gets called at the end of an epoch.

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

            # Update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(stage_loss)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            stats={
                "loss": stage_loss,
            }

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
            )

            # Save the current checkpoint and delete previous checkpoints.
            save_checkpoint = (
                epoch % self.hparams.checkpoint_frequency == 0
                or epoch == self.hparams.number_of_epochs)
            if save_checkpoint:
                self.checkpointer.save_and_keep_only(meta=stats, min_keys=["loss"])
            
            # Output samples
            output_progress_sample =(
                self.hparams.progress_samples
                and epoch % self.hparams.progress_samples_interval == 0)
            if output_progress_sample:
                self.save_progress_sample() 

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
            }

        # At the end of validation, we can write
        if stage == sb.Stage.VALID:

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints.
            save_checkpoint = (epoch % self.hparams.checkpoint_frequency == 0
                                or epoch == self.hparams.number_of_epochs)
            if save_checkpoint:
              self.checkpointer.save_and_keep_only(meta=stats, min_keys=["loss"])
            
            output_progress_sample =(
                self.hparams.progress_samples
                and epoch % self.hparams.progress_samples_interval == 0)
            if output_progress_sample:
                self.save_progress_sample() 

    def save_progress_sample(self):
        """
        Function gets called to save progress samples, depending on chosen output interval
        """
        if len(self.last_target_mel.size())==3: # batch
            last_target_audio = self.last_target_audio[0,:,:]
            last_target_mel = self.last_target_mel[0,:,:]
            last_predicted_audio = self.last_predicted_audio[0,:,:]
            last_predicted_mel = self.last_predicted_mel[0,:,:]
        else: # overfit with 1 ex
            last_target_audio = self.last_target_audio.squeeze()
            last_target_mel = self.last_target_mel
            last_predicted_audio = self.last_predicted_audio.squeeze()
            last_predicted_mel = self.last_predicted_mel

        self.save_sample_audio(
                    'target_audio.wav', last_target_audio.unsqueeze(0))
        self.save_sample_audio(
                    'predicted_audio.wav', last_predicted_audio.unsqueeze(0))
        self.save_sample_image(
                    'target_mel.png', last_target_mel.unsqueeze(0))
        self.save_sample_image(
                    'predicted_mel.png', last_predicted_mel.unsqueeze(0))

    def save_sample_image(self, file_name, data):
        """
        Save a sample image

        Arguments
        ---------
        file_name: str
            Path to the file name
        data: torch.Tensor, (B x T x C)
            sample image to be saved
        """

        effective_file_name = os.path.join(self.hparams.progress_sample_path, file_name)
        torchvision.utils.save_image(data, effective_file_name)

    def save_sample_audio(self, file_name, data):
        """
        Save a sample image

        Arguments
        ---------
        file_name: str
            Path to the file name
        data: torch.Tensor, (B x T)
            sample audio to be saved
        """
        effective_file_name = os.path.join(self.hparams.progress_sample_path, file_name)
        torchaudio.save(effective_file_name,data, sample_rate=self.hparams.sample_rate)


def main():

    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
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
    wavenet_brain = WavenetBrain(
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
    wavenet_brain.fit(
        epoch_counter=wavenet_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
