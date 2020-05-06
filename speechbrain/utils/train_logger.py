import logging
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class TrainLogger:
    def __init__(self, save_file):
        self.save_file = save_file

    def _item_to_string(self, key, value):
        if isinstance(value, float):
            value = f"{value:.2f}"
        return f"{key}: {value}"

    def _stats_to_string(self, epoch_stats, train_stats, valid_stats=None):
        log_string = " - ".join(
            [self._item_to_string(k, v) for k, v in epoch_stats.items()]
        )
        for key, value in train_stats.items():
            log_string += " - train " + self._item_to_string(key, value)
        if valid_stats is not None:
            for key, value in valid_stats.items():
                log_string += " - valid " + self._item_to_string(key, value)
        return log_string

    def log_epoch(
        self, epoch_stats, train_stats, valid_stats=None, verbose=True,
    ):
        summary = self._stats_to_string(epoch_stats, train_stats, valid_stats)
        with open(self.save_file, "a") as fout:
            print(summary, file=fout)
        if verbose:
            logger.info(summary)


class TensorboardLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.writer = SummaryWriter(self.save_dir)
        self.global_step = {}

    def log_batch(self, batch_stats, dataset):
        if dataset not in self.global_step:
            self.global_step[dataset] = 0
        self.global_step[dataset] += 1
        for stat, value in batch_stats.items():
            tag = f"{stat}/{dataset}"
            self.writer.add_scalar(tag, value, self.global_step[dataset])
