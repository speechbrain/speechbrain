"""Implements a checkpointable epoch counter (loop), optionally integrating early stopping.

Authors
 * Aku Rouhe 2020
 * Davide Borra 2021
"""
from .checkpoints import register_checkpoint_hooks
from .checkpoints import mark_as_saver
from .checkpoints import mark_as_loader
import logging
import yaml

logger = logging.getLogger(__name__)


@register_checkpoint_hooks
class EpochCounter:
    """An epoch counter which can save and recall its state.

    Use this as the iterator for epochs.
    Note that this iterator gives you the numbers from [1 ... limit] not
    [0 ... limit-1] as range(limit) would.

    Example
    -------
    >>> from speechbrain.utils.checkpoints import Checkpointer
    >>> tmpdir = getfixture('tmpdir')
    >>> epoch_counter = EpochCounter(10)
    >>> recoverer = Checkpointer(tmpdir, {"epoch": epoch_counter})
    >>> recoverer.recover_if_possible()
    >>> # Now after recovery,
    >>> # the epoch starts from where it left off!
    >>> for epoch in epoch_counter:
    ...     # Run training...
    ...     ckpt = recoverer.save_checkpoint()
    """

    def __init__(self, limit):
        self.current = 0
        self.limit = int(limit)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.limit:
            self.current += 1
            logger.info(f"Going into epoch {self.current}")
            return self.current
        raise StopIteration

    @mark_as_saver
    def _save(self, path):
        with open(path, "w") as fo:
            fo.write(str(self.current))

    @mark_as_loader
    def _recover(self, path, end_of_epoch=True):
        # NOTE: end_of_epoch = True by default so that when
        #  loaded in parameter transfer, this starts a new epoch.
        #  However, parameter transfer to EpochCounter should
        #  probably never be used really.
        with open(path) as fi:
            saved_value = int(fi.read())
            if end_of_epoch:
                self.current = saved_value
            else:
                self.current = saved_value - 1


class EpochCounterWithStopper(EpochCounter):
    """An epoch counter which can save and recall its state, integrating an early stopper by tracking a target metric.

    Arguments
    ---------
    limit: int
        maximum number of epochs
    limit_to_stop : int
        maximum number of consecutive epochs without improvements in performance
    limit_warmup : int
        number of epochs to wait until start checking for early stopping
    direction : "max" or "min"
        direction to optimize the target metric

    Example
    -------
    >>> limit = 10
    >>> limit_to_stop = 5
    >>> limit_warmup = 2
    >>> direction = "min"
    >>> epoch_counter = EpochCounterWithStopper(limit, limit_to_stop, limit_warmup, direction)
    >>> for epoch in epoch_counter:
    ...     # Run training...
    ...     # Track a validation metric, (insert calculation here)
    ...     current_valid_metric = 0
    ...     # Update epoch counter so that we stop at the appropriate time
    ...     epoch_counter.update_metric(current_valid_metric)
    ...     print(epoch)
    1
    2
    3
    4
    5
    6
    7
    8
    """

    def __init__(self, limit, limit_to_stop, limit_warmup, direction):
        super().__init__(limit)
        self.limit_to_stop = limit_to_stop
        self.limit_warmup = limit_warmup
        self.direction = direction
        self.should_stop = False

        self.best_limit = 0
        self.min_delta = 1e-6

        if self.limit_to_stop < 0:
            raise ValueError("Stopper 'limit_to_stop' must be >= 0")
        if self.limit_warmup < 0:
            raise ValueError("Stopper 'limit_warmup' must be >= 0")
        if self.direction == "min":
            self.best_score, self.sign = float("inf"), 1
        elif self.direction == "max":
            self.best_score, self.sign = -float("inf"), -1
        else:
            raise ValueError("Stopper 'direction' must be 'min' or 'max'")

    def __next__(self):
        """Stop iteration if we've reached the condition."""
        if self.should_stop:
            raise StopIteration
        else:
            return super().__next__()

    def update_metric(self, current_metric):
        """Update the state to reflect most recent value of the relevant metric.

        NOTE: Should be called only once per validation loop.

        Arguments
        ---------
        current_metric : float
            The metric used to make a stopping decision.
        """
        if self.current > self.limit_warmup:
            if self.sign * current_metric < self.sign * (
                (1 - self.min_delta) * self.best_score
            ):
                self.best_limit = self.current
                self.best_score = current_metric

            epochs_without_improvement = self.current - self.best_limit
            self.should_stop = epochs_without_improvement >= self.limit_to_stop
            if self.should_stop:
                logger.info(
                    f"{epochs_without_improvement} epochs without improvement.\n"
                    f"Patience of {self.limit_to_stop} is exhausted, stopping."
                )

    @mark_as_saver
    def _save(self, path):
        with open(path, "w") as fo:
            yaml.dump(
                {
                    "current_epoch": self.current,
                    "best_epoch": self.best_limit,
                    "best_score": self.best_score,
                    "should_stop": self.should_stop,
                },
                fo,
            )

    @mark_as_loader
    def _recover(self, path, end_of_epoch=True, device=None):
        del device  # Not used.
        with open(path) as fi:
            saved_dict = yaml.safe_load(fi)
            if end_of_epoch:
                self.current = saved_dict["current_epoch"]
            else:
                self.current = saved_dict["current_epoch"] - 1
            self.best_limit = saved_dict["best_epoch"]
            self.best_score = saved_dict["best_score"]
            self.should_stop = saved_dict["should_stop"]
