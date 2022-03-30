"""Loggers for experiment monitoring.

Authors
 * Peter Plantinga 2020
 * Titouan Parcollet 2021
"""
import os
import logging
import torch
import ruamel.yaml
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils import checkpoints

logger = logging.getLogger(__name__)


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
    summary_fns : dict of str:function pairs
        Each summary function should take a list produced as output
        from a training/validation pass and summarize it to a single scalar.
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

        with open(self.save_file, "a") as fout:
            print(string_summary, file=fout)
        if verbose:
            logger.info(string_summary)


@checkpoints.register_checkpoint_hooks
class TensorboardLogger(TrainLogger):
    """Logs training information in the format required by Tensorboard.
    If you wish your loader to be resumable i.e., the logging does not restart
    from the global step 0 every time, you must create a SpeechBrain checkpointer
    that will automatically save the corresponding global steps.

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
        self.writer = None
        self.global_step = {"train": {}, "valid": {}, "test": {}, "meta": 0}

    def prepare_tensorboard_logger(self, purge_step=None):
        """Used to create the dir of the tensorboard logs compliant with DDP
        Indeed, the init function is call before DDP is initialised. Hence,
        the SummaryWriter would try to create the save_folder with all processes,
        leading to a R/W error. The SummaryWriter must be initialised after DDP.

        Arguments
        ---------
        purge_step : int
            When logging crashes at step T + X and restart at step T,
            any events whose global_step larger or equal to T will be purged and
            hidden from TensorBoard.
        """
        if not os.path.isdir(self.save_dir):
            run_on_main(os.makedirs, args=[self.save_dir])

        # Raises ImportError if TensorBoard is not installed
        from torch.utils.tensorboard import SummaryWriter

        self.purge_step = purge_step
        self.writer = SummaryWriter(self.save_dir, purge_step=purge_step)

    def log_stats(
        self,
        global_step,
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
                elif self.global_step[dataset][stat] > self.purge_step:
                    self.global_step[dataset][stat] = self.purge_step

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

    @checkpoints.mark_as_saver
    def save(self, path):
        data = {"global_step": self.global_step}
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch=False, device=None):
        del end_of_epoch  # Unused in this class
        del device
        data = torch.load(path)
        self.global_step = data["global_step"]

        # Remove all the logged event up until purge step
        # This is usefull in case of resuming (we restart the log from the last
        # epoch and not the last log that might be newer than the epoch)
        for dict in self.global_step:
            if dict != "meta":
                for subdict in self.global_step[dict]:
                    if self.global_step[dict][subdict] > self.purge_step:
                        self.global_step[dict][subdict] = self.purge_step
            else:
                if self.global_step["meta"] > self.purge_step:
                    self.global_step["meta"] = self.purge_step

        if self.global_step["meta"] > self.purge_step:
            self.global_step["meta"] = self.purge_step


class WandBLogger(TrainLogger):
    """Logger for wandb. To be used the same way as TrainLogger. Handles nested dicts as well.
    An example on how to use this can be found in recipes/Voicebank/MTL/CoopNet/"""

    def __init__(self, *args, **kwargs):
        try:
            yaml_file = kwargs.pop("yaml_config")
            with open(yaml_file, "r") as yaml_stream:
                # Read yaml with ruamel to ignore bangs
                config_dict = ruamel.yaml.YAML().load(yaml_stream)
            self.run = kwargs.pop("initializer", None)(
                *args, **kwargs, config=config_dict
            )
        except Exception as e:
            raise e("There was an issue with the WandB Logger initialization")

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
