"""
Utilities for TTS recipes

Authors
* Artem Ploujnikov 2021
"""
import os
import torch
import torchvision

from speechbrain.utils.checkpoints import torch_save


class ProgressSampleImageMixin:
    _FORMATS = {
        "raw": {"extension": "pth", "saver": torch_save},
        "image": {"extension": "png", "saver": torchvision.utils.save_image},
    }
    DEFAULT_FORMAT = "image"
    PROGRESS_SAMPLE_FORMATS = {}

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
            {key: _detach(value) for key, value in kwargs.items()}
        )

    def get_batch_sample(self, value):
        if isinstance(value, dict):
            result = {
                key: self.get_batch_sample(item_value)
                for key, item_value in value.items()
            }
        elif isinstance(value, (torch.Tensor, list)):
            result = value[: self.hparams.progress_batch_sample_size]
        else:
            result = value
        return result

    def save_progress_sample(self, epoch):
        """
        Saves a set of spectrogram samples

        Arguments:
        ----------
        epoch: int
            The epoch number
        """
        for key, data in self.progress_samples.items():
            self.save_progress_sample_item(key, data, epoch)

    def save_progress_sample_item(self, key, data, epoch):
        """
        Saves a single sample item

        Arguments
        ---------
        key: str
            the key/identifier of tthe item
        data: torch.Tensor
            the  data to save
        epoch: int
            the epoch number (used in file path calculations)
        """
        target_path = os.path.join(
            self.hparams.progress_sample_path, str(epoch)
        )
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        sample_formats = getattr(self, "PROGRESS_SAMPLE_FORMATS", {})
        format = self._FORMATS[sample_formats.get(key, self.DEFAULT_FORMAT)]
        file_name = f"{key}.{format['extension']}"
        effective_file_name = os.path.join(target_path, file_name)
        format["saver"](data, effective_file_name)


def _detach(value):
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu()
    elif isinstance(value, dict):
        result = {key: _detach(item_value) for key, item_value in value.items()}
    else:
        result = value
    return result


def scalarize(value):
    """
    Converts a namedtuple or dictionary containing tensors
    to their scalar value

    Arguments:
    ----------
    value: dict or namedtuple
        a dictionary or named tuple of tensors

    Returns
    -------
    result: dict
        a result dictionary
    """
    if hasattr(value, "_asdict"):
        value_dict = value._asdict()
    else:
        value_dict = value
    return {key: item_value.item() for key, item_value in value_dict.items()}
