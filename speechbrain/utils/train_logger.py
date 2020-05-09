import logging
from speechbrain.utils.edit_distance import wer_summary

logger = logging.getLogger(__name__)


class TrainLogger:
    """Abstract class defining an interface for training loggers."""

    def log_epoch(
        self, epoch_stats, train_stats, valid_stats=None, verbose=False,
    ):
        """Log the stats for one epoch.

        Arguments
        ---------
        epoch_stats : dict of str:scalar pairs
            Stats relevant to the epoch (e.g. count, learning-rate, etc.)
        train_stats : dict of str:list pairs
            Each loss type is represented with a str : list pair including
            all the values for the training pass.
        valid_stats : dict of str:list pairs
            Each loss type is represented with a str : list pair including
            all the values for the validation pass.
        verbose : bool
            Whether to also put logging information to the standard logger.
        """
        raise NotImplementedError


class FileTrainLogger(TrainLogger):
    """Text logger of training information

    Arguments
    ---------
    save_file : str
        The file to use for logging train information.
    summary_fns : dict of str:function pairs
        Each summary function should take a list produced as output
        from a training/validation pass and summarize it to a single scalar.
    """

    def __init__(self, save_file, summary_fns):
        self.save_file = save_file
        self.summary_fns = summary_fns

    def _item_to_string(self, key, value):
        """Convert one item to string, handling floats"""
        if isinstance(value, float):
            value = f"{value:.2f}"
        return f"{key}: {value}"

    def _stats_to_string(self, epoch_stats, train_stats, valid_stats=None):
        """Convert all stats to a single string summary"""
        log_string = " - ".join(
            [self._item_to_string(k, v) for k, v in epoch_stats.items()]
        )
        for stat, value_list in train_stats.items():
            value = self.summary_fns[stat](value_list)
            log_string += " - train " + self._item_to_string(stat, value)
        if valid_stats is not None:
            for stat, value_list in valid_stats.items():
                value = self.summary_fns[stat](value_list)
                log_string += " - valid " + self._item_to_string(stat, value)
        return log_string

    def log_epoch(
        self, epoch_stats, train_stats, valid_stats=None, verbose=True,
    ):
        """See TrainLogger.log_epoch()"""
        summary = self._stats_to_string(epoch_stats, train_stats, valid_stats)
        with open(self.save_file, "a") as fout:
            print(summary, file=fout)
        if verbose:
            logger.info(summary)


class TensorboardLogger(TrainLogger):
    """Logs training information in the format required by Tensorboard.

    Arguments
    ---------
    save_dir : str
        A directory for storing all the relevant logs

    Raises
    ------
    ImportError if Tensorboard is not installed.
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir

        # Raises ImportError if TensorBoard is not installed
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(self.save_dir)
        self.global_step = {"train": {}, "valid": {}, "epoch": 0}

    def log_epoch(
        self, epoch_stats, train_stats, valid_stats=None, verbose=False,
    ):
        """See TrainLogger.log_epoch()"""
        self.global_step["epoch"] += 1
        for name, value in epoch_stats.items():
            self.writer.add_scalar(name, value, self.global_step["epoch"])

        for dataset, stats in [("train", train_stats), ("valid", valid_stats)]:
            for stat, value_list in stats.items():
                if stat not in self.global_step[dataset]:
                    self.global_step[dataset][stat] = 0
                tag = f"{stat}/{dataset}"
                for value in value_list:
                    new_global_step = self.global_step[dataset][stat] + 1
                    self.writer.add_scalar(tag, value, new_global_step)
                    self.global_step[dataset][stat] = new_global_step


def summarize_average(stat_list):
    return float(sum(stat_list) / len(stat_list))


def summarize_error_rate(stat_list):
    summary = wer_summary(stat_list)
    return summary["WER"]
