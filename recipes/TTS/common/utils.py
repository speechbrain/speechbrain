"""
Utilities for TTS recipes

Authors
* Artem Ploujnikov 2021
"""
import os
import torchvision

from speechbrain.utils.checkpoints import torch_save


class PretrainedModelMixin:
    """
    A brain mixin that provides a function to save models and other
    artefacts for pretrained models
    """

    def save_for_pretrained(self):
        """
        Saves the necessary files for the pretrained model
        """
        pretrainer = self.hparams.pretrainer
        for key, value in pretrainer.loadables.items():
            path = pretrainer.paths[key]
            torch_save(value, path)

class ProgressSampleImageMixin:
    """
    A brain mixin that provides a function to save progress sample
    images
    """

    def init_progress_samples(self):
        """
        Initializes the collection of progress samples
        """
        self.progress_samples = {}

    def remember_progress_sample(self, **kwargs):
        """
        Updates the internal dictionary of snapshots
        """
        self.progress_samples.update(
            {key: value.detach().cpu() for key, value in kwargs.items()}
        )

    def save_progress_sample(self, epoch):
        """
        Saves a set of spectrogram samples

        Arguments:
        ----------
        epoch: int
            The epoch number
        """
        entries = [
            (f"{key}.png", value) for key, value in self.progress_samples.items()
        ]
        for file_name, data in entries:
            self.save_sample_image(file_name, data, epoch)

    def save_sample_image(self, file_name, data, epoch):
        """
        Saves a single sample image

        Arguments
        ---------
        file_name: str
            the target file name
        data: torch.Tensor
            the image data
        epoch: int
            the epoch number (used in file path calculations)
        """
        target_path = os.path.join(
            self.hparams.progress_sample_path, str(epoch)
        )
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        effective_file_name = os.path.join(target_path, file_name)
        torchvision.utils.save_image(data, effective_file_name)
