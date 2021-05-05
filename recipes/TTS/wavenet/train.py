"""Recipe for training the Wavenet model

https://arxiv.org/abs/1710.07654

To run this recipe, do the following:
> python train.py hparams/train.yaml --train_data_path /path/to/trainset --valid_data_path /path/to/validset

Authors
* Aleksandar Rachkov 2020
"""

import torch
import sys
import speechbrain as sb
from torch.nn import functional as F
from hyperpyyaml import load_hyperpyyaml
from speechbrain.lobes.models.synthesis.wavenet.dataio import dataset_prep
import os

import torchvision
import torchaudio

sys.path.append("..")
from datasets.vctk import VCTK

from scipy.signal import firwin, lfilter
from torchaudio import transforms


class WavenetBrain(sb.core.Brain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_forward(self, batch, stage, use_targets=True):

        batch = batch.to(self.device)

        pred = self.hparams.model(
                x=batch.x.data,
                c=batch.mel.data.transpose(1,2)
            )

        return pred

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.
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
        #batch = BatchWrapper(batch).to(self.device)

        # wee need 4d inputs for spatial cross entropy loss
        # (B, T, C, 1)
        y_hat = predictions.unsqueeze(-1)
        target = batch.target.data.unsqueeze(-1)
        lengths = batch.target_length

        loss = self.hparams.compute_cost(
            y_hat[:,:-1,:,:], target[:,1:,:], lengths=lengths
        )
        # (B, T)

        y_hat = F.softmax(y_hat.squeeze(), dim=1).max(1)[1]

        predicted_audio = inv_mulaw_quantize(y_hat)
        target_audio = inv_mulaw_quantize(target.squeeze())

        predicted_mel = torchaudio.transforms.MelSpectrogram(self.hparams.sample_rate)(predicted_audio)
        target_mel = torchaudio.transforms.MelSpectrogram(self.hparams.sample_rate)(target_audio)

        (self.last_predicted_audio,
         self.last_target_audio,
         self.last_predicted_mel,
         self.last_target_mel) = [
            tensor.detach().cpu()
            for tensor in (
                predicted_audio, target_audio,
                predicted_mel, target_mel
            )]

        return loss

    def _save_progress_sample(self):
        for i,b in enumerate(self.last_target_audio):
            self._save_sample_audio(
            'target_audio_'+str(i)+'.wav', b.unsqueeze(0))
        for i,b in enumerate(self.last_predicted_audio):        
            self._save_sample_audio(
            'predicted_audio_'+str(i)+'.wav', b.unsqueeze(0))
        for i,b in enumerate(self.last_target_mel):
            self._save_sample_image(
            'target_mel_'+str(i)+'.png', b)
        for i,b in enumerate(self.last_predicted_mel):
            self._save_sample_image(
            'output_mel_'+str(i)+'.png', b)

    def _save_sample_image(self, file_name, data):
        effective_file_name = os.path.join(self.hparams.progress_sample_path, file_name)
        torchvision.utils.save_image(data, effective_file_name)

    def _save_sample_audio(self, file_name, data):
        effective_file_name = os.path.join(self.hparams.progress_sample_path, file_name)
        torchaudio.save(effective_file_name,data, sample_rate=self.hparams.sample_rate)

    def on_fit_start(self):
        super().on_fit_start()
        if self.hparams.progress_samples:
            if not os.path.exists(self.hparams.progress_sample_path):
                os.makedirs(self.hparams.progress_sample_path)

    def on_stage_end(self, stage, stage_loss, epoch):
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
            stats={
                "loss": stage_loss,
            }
            self.train_loss = stage_loss

            # Update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(stage_loss)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                #valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["loss"])
            output_progress_sample =(
                self.hparams.progress_samples
                and epoch % self.hparams.progress_samples_interval == 0)
            if output_progress_sample:
                self._save_progress_sample()
        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
            }

        # At the end of validation, we can write
        if stage == sb.Stage.VALID:
            # Update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(stage_loss)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["loss"])
            output_progress_sample =(
                self.hparams.progress_samples
                and epoch % self.hparams.progress_samples_interval == 0)
            if output_progress_sample:
                self._save_progress_sample()

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

def inv_mulaw(y, mu=256):
    return y.sign() * (1.0 / mu) * ((1.0 + mu)**y.abs() - 1.0)

def inv_mulaw_quantize(y,mu=256):
    y = 2*y.type(torch.FloatTensor)/mu -1
    return inv_mulaw(y,mu)

def dataio_prep(hparams):
    result = {}
    for name, dataset_params in hparams['datasets'].items():
        # TODO: Add support for multiple datasets by instantiating from hparams - this is temporary
        vctk = VCTK(dataset_params['path']).to_dataset()
        result[name] = dataset_prep(vctk, hparams)
    return result

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
        # TODO: Implement splitting - this is not ready yet
        #valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        #valid_loader_kwargs=hparams["dataloader_options"],
    )

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
