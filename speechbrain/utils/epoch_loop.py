"""Implements a checkpointable epoch counter (loop).

Authors
 * Aku Rouhe 2020
"""
from .checkpoints import register_checkpoint_hooks
from .checkpoints import mark_as_saver
from .checkpoints import mark_as_loader
import logging

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
    def _recover(self, path, end_of_epoch=True, device=None):
        # NOTE: end_of_epoch = True by default so that when
        #  loaded in parameter transfer, this starts a new epoch.
        #  However, parameter transfer to EpochCounter should
        #  probably never be used really.
        del device  # Not used.
        with open(path) as fi:
            saved_value = int(fi.read())
            if end_of_epoch:
                self.current = saved_value
            else:
                self.current = saved_value - 1
