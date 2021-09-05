"""A logger that helps save samples as training progresses,
useful mostly for generative tasks, such as Text-to-Speech (TTS)

Authors
* Artem Ploujnikov 2021
"""
import os
import torch
import torchvision

from speechbrain.utils.data_utils import detach


class ProgressSampleLogger:
    _DEFAULT_FORMAT_DEFS = {
        "raw": {"extension": "pth", "saver": torch.save},
        "image": {"extension": "png", "saver": torchvision.utils.save_image},
    }
    DEFAULT_FORMAT = "image"

    """A logger that outputs samples during training progress, used primarily in speech synthesis but customizable, reusable and applicable to any other generative task

    Natively, this logger supports images and raw PyTorch output.
    Other custom formats can be added as needed.

    Example:

    In hparams.yaml
    progress_sample_logger: !new:speechbrain.utils.progress_samples.ProgressSampleLogger
        output_path: output/samples
        progress_batch_sample_size: 3
        format_defs:
            extension: my
            saver: !speechbrain.dataio.mystuff.save_my_format
        formats:
            foo: image
            bar: my



    In the brain:

    Run the following to "remember" a sample (e.g. from compute_objectives)

    self.hparams.progress_sample_logger.remember(
        target=spectrogram_target,
        output=spectrogram_output,
        alignments=alignments_output,
        my_output=
        raw_batch={
            "inputs": inputs,
            "spectrogram_target": spectrogram_target,
            "spectrogram_output": spectrorgram_outputu,
            "alignments": alignments_output
        }
    )

    Run the following at the end of the epoch (e.g. from on_stage_end)
    self.progress_sample_logger.save(epoch)



    Arguments
    ---------
    output_path: str
        the filesystem path to which samples will be saved
    formats: dict
        a dictionary with format identifiers as keys and dictionaries with
        handler callables and extensions as values. The signature of the handler
        should be similar to torch.save

        Example:
        {
            "myformat": {
                "extension": "myf",
                "saver": somemodule.save_my_format
            }
        }
    batch_sample_size: int
        The number of items to retrieve when extracting a batch sample
    """

    def __init__(
        self, output_path, formats=None, format_defs=None, batch_sample_size=1
    ):
        self.progress_samples = {}
        self.formats = formats or {}
        self.format_defs = dict(self._DEFAULT_FORMAT_DEFS)
        if format_defs is not None:
            self.format_defs.update(format_defs)
        self.batch_sample_size = batch_sample_size
        self.output_path = output_path

    def reset(self):
        """Initializes the collection of progress samples"""
        self.progress_samples = {}

    def remember(self, **kwargs):
        """Updates the internal dictionary of snapshots with the provided
        values

        Arguments
        ---------
        kwargs: dict
            the parameters to be saved with
        """
        self.progress_samples.update(
            {key: detach(value) for key, value in kwargs.items()}
        )

    def get_batch_sample(self, value):
        """Obtains a sample of a batch for saving. This can be useful to
        monitor raw data (both samples and predictions) over the course
        of training

        Arguments
        ---------
        value: dict|torch.Tensor|list
            the raw values from the batch

        Returns
        -------
        result: object
            the same type of object as the provided value
        """
        if isinstance(value, dict):
            result = {
                key: self.get_batch_sample(item_value)
                for key, item_value in value.items()
            }
        elif isinstance(value, (torch.Tensor, list)):
            result = value[: self.batch_sample_size]
        else:
            result = value
        return result

    def save(self, epoch):
        """Saves all items previously saved with remember() calls

        Arguments
        ---------
        epoch: int
            The epoch number
        """
        for key, data in self.progress_samples.items():
            self.save_item(key, data, epoch)

    def save_item(self, key, data, epoch):
        """Saves a single sample item

        Arguments
        ---------
        key: str
            the key/identifier of the item
        data: torch.Tensor
            the  data to save
        epoch: int
            the epoch number (used in file path calculations)
        """
        target_path = os.path.join(self.output_path, str(epoch))
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        format = self.formats.get(key, self.DEFAULT_FORMAT)
        format_def = self.format_defs.get(format)
        if format_def is None:
            raise ValueError("Unsupported format {format}")
        file_name = f"{key}.{format_def['extension']}"
        effective_file_name = os.path.join(target_path, file_name)
        format_def["saver"](data, effective_file_name)
