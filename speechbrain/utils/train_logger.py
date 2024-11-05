"""Loggers for experiment monitoring.

Authors
 * Peter Plantinga 2020
 * Jarod Duret 2023
"""

import os

import torch

from speechbrain.utils.distributed import if_main_process, main_process_only
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class TrainLogger:
    """Abstract class defining an interface for training loggers."""

    def log_stats(
        self,
        stats_meta,
        train_stats=None,
        valid_stats=None,
        test_stats=None,
        verbose=False,
    ):
        """Log the stats for one epoch.

        Arguments
        ---------
        stats_meta : dict of str:scalar pairs
            Meta information about the stats (e.g., epoch, learning-rate, etc.).
        train_stats : dict of str:list pairs
            Each loss type is represented with a str : list pair including
            all the values for the training pass.
        valid_stats : dict of str:list pairs
            Each loss type is represented with a str : list pair including
            all the values for the validation pass.
        test_stats : dict of str:list pairs
            Each loss type is represented with a str : list pair including
            all the values for the test pass.
        verbose : bool
            Whether to also put logging information to the standard logger.
        """
        raise NotImplementedError


class FileTrainLogger(TrainLogger):
    """Text logger of training information.

    Arguments
    ---------
    save_file : str
        The file to use for logging train information.
    precision : int
        Number of decimal places to display. Default 2, example: 1.35e-5.
    """

    def __init__(self, save_file, precision=2):
        self.save_file = save_file
        self.precision = precision

    def _item_to_string(self, key, value, dataset=None):
        """Convert one item to string, handling floats"""
        if isinstance(value, float) and 1.0 < value < 100.0:
            value = f"{value:.{self.precision}f}"
        elif isinstance(value, float):
            value = f"{value:.{self.precision}e}"
        if dataset is not None:
            key = f"{dataset} {key}"
        return f"{key}: {value}"

    def _stats_to_string(self, stats, dataset=None):
        """Convert all stats to a single string summary"""
        return ", ".join(
            [self._item_to_string(k, v, dataset) for k, v in stats.items()]
        )

    @main_process_only
    def log_stats(
        self,
        stats_meta,
        train_stats=None,
        valid_stats=None,
        test_stats=None,
        verbose=True,
    ):
        """See TrainLogger.log_stats()"""
        string_summary = self._stats_to_string(stats_meta)
        for dataset, stats in [
            ("train", train_stats),
            ("valid", valid_stats),
            ("test", test_stats),
        ]:
            if stats is not None:
                string_summary += " - " + self._stats_to_string(stats, dataset)

        with open(self.save_file, "a", encoding="utf-8") as fout:
            print(string_summary, file=fout)
        if verbose:
            logger.info(string_summary)


class TensorboardLogger(TrainLogger):
    """Logs training information in the format required by Tensorboard.

    Arguments
    ---------
    save_dir : str
        A directory for storing all the relevant logs.

    Raises
    ------
    ImportError if Tensorboard is not installed.
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir

        # Raises ImportError if TensorBoard is not installed
        from torch.utils.tensorboard import SummaryWriter

        # Initialize writer only on main
        self.writer = None
        if if_main_process():
            self.writer = SummaryWriter(self.save_dir)
        self.global_step = {"train": {}, "valid": {}, "test": {}, "meta": 0}

    @main_process_only
    def log_stats(
        self,
        stats_meta,
        train_stats=None,
        valid_stats=None,
        test_stats=None,
        verbose=False,
    ):
        """See TrainLogger.log_stats()"""
        self.global_step["meta"] += 1
        for name, value in stats_meta.items():
            self.writer.add_scalar(name, value, self.global_step["meta"])

        for dataset, stats in [
            ("train", train_stats),
            ("valid", valid_stats),
            ("test", test_stats),
        ]:
            if stats is None:
                continue
            for stat, value_list in stats.items():
                if stat not in self.global_step[dataset]:
                    self.global_step[dataset][stat] = 0
                tag = f"{stat}/{dataset}"

                # Both single value (per Epoch) and list (Per batch) logging is supported
                if isinstance(value_list, list):
                    for value in value_list:
                        new_global_step = self.global_step[dataset][stat] + 1
                        self.writer.add_scalar(tag, value, new_global_step)
                        self.global_step[dataset][stat] = new_global_step
                else:
                    value = value_list
                    new_global_step = self.global_step[dataset][stat] + 1
                    self.writer.add_scalar(tag, value, new_global_step)
                    self.global_step[dataset][stat] = new_global_step

    @main_process_only
    def log_audio(self, name, value, sample_rate):
        """Add audio signal in the logs."""
        self.writer.add_audio(
            name, value, self.global_step["meta"], sample_rate=sample_rate
        )

    @main_process_only
    def log_figure(self, name, value):
        """Add a figure in the logs."""
        fig = plot_spectrogram(value)
        if fig is not None:
            self.writer.add_figure(name, fig, self.global_step["meta"])


class WandBLogger(TrainLogger):
    """
    Logger for WandB (Weights & Biases). This logger is designed to be used in the same way as TrainLogger
    and supports handling nested dictionaries as well.

    Arguments
    ---------
    initializer: callable
        A callable function that initializes the WandB run.
        For more information on the parameters that can be passed to the initializer, refer to
        the documentation: https://docs.wandb.ai/ref/python/init
    *args: tuple
        Positional arguments to be passed to the initializer function.
    **kwargs: dict
        Keyword arguments to be passed to the initializer function.

    Example
    -------
    To initialize the logger, use the following pattern in hparams.yaml:

    ```
    train_logger: !new:speechbrain.utils.train_logger.WandBLogger
        initializer: !name:wandb.init
            entity: speechbrain
            project: sb_project
            name: sb_run
            reinit: True
            resume: False
            dir: !ref <output_folder>/wandb
            id: sb_run
            resume: allow
    ```

    NOTE
    ----
    If there is an issue with the WandB Logger initialization, it raises an exception.
    """

    def __init__(self, initializer, *args, **kwargs):
        try:
            self.run = None
            if if_main_process():
                self.run = initializer(*args, **kwargs)
        except Exception as e:
            raise e("There was an issue with the WandB Logger initialization")

    @main_process_only
    def log_stats(
        self,
        stats_meta,
        train_stats=None,
        valid_stats=None,
        test_stats=None,
        verbose=False,
    ):
        """See TrainLogger.log_stats()"""
        logs = {}
        for dataset, stats in [
            ("train", train_stats),
            ("valid", valid_stats),
            ("test", test_stats),
        ]:
            if stats is None:
                continue
            logs[dataset] = stats

        step = stats_meta.get("epoch", None)
        if step is not None:  # Useful for continuing runs that crashed
            self.run.log({**logs, **stats_meta}, step=step)
        else:
            self.run.log({**logs, **stats_meta})


def _get_image_saver():
    """Returns the TorchVision image saver, if available
    or None if it is not - optional dependency
    """
    try:
        import torchvision

        return torchvision.utils.save_image
    except ImportError:
        logger.warning("torchvision is not available - cannot save figures")
        return None


class ProgressSampleLogger:
    """A logger that outputs samples during training progress, used primarily in speech synthesis but customizable, reusable and applicable to any other generative task

    Natively, this logger supports images and raw PyTorch output.
    Other custom formats can be added as needed.

    Example:

    In hparams.yaml
    progress_sample_logger: !new:speechbrain.utils.progress_samples.ProgressSampleLogger
        output_path: output/samples
        progress_batch_sample_size: 3
        format_defs:
            foo:
                extension: bar
                saver: !speechbrain.dataio.mystuff.save_my_format
                kwargs:
                    baz: qux
        formats:
            foobar: foo



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
            "spectrogram_output": spectrogram_output,
            "alignments": alignments_output
        }
    )

    Run the following at the end of the epoch (e.g. from on_stage_end)
    self.progress_sample_logger.save(epoch)



    Arguments
    ---------
    output_path: str
        the filesystem path to which samples will be saved.
    formats: dict
        A mapping from keys to formats.
    format_defs: dict
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

    _DEFAULT_FORMAT_DEFS = {
        "raw": {"extension": "pth", "saver": torch.save, "kwargs": {}},
        "image": {
            "extension": "png",
            "saver": _get_image_saver(),
            "kwargs": {},
        },
    }
    DEFAULT_FORMAT = "image"

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
        **kwargs: dict
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

    @main_process_only
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
        saver = format_def.get("saver")
        if saver is not None:
            saver(data, effective_file_name, **format_def["kwargs"])


def plot_spectrogram(spectrogram, ap=None, fig_size=(16, 10), output_fig=False):
    """Returns the matplotlib spectrogram if available
    or None if it is not - optional dependency
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    except ImportError:
        logger.warning("matplotlib is not available - cannot log figures")
        return None

    spectrogram = spectrogram.detach().cpu().numpy().squeeze()
    fig = plt.figure(figsize=fig_size)
    plt.imshow(spectrogram, aspect="auto", origin="lower")
    plt.colorbar()
    plt.tight_layout()
    if not output_fig:
        plt.close()
    return fig


def detach(value):
    """Detaches the specified object from the graph, which can be a
    single tensor or a dictionary of tensors. Dictionaries of tensors are
    converted recursively

    Arguments
    ---------
    value: torch.Tensor|dict
        a tensor or a dictionary of tensors

    Returns
    -------
    result: torch.Tensor|dict
        a tensor of dictionary of tensors
    """
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu()
    elif isinstance(value, dict):
        result = {key: detach(item_value) for key, item_value in value.items()}
    else:
        result = value
    return result
