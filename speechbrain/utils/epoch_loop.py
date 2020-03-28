from .recovery import register_recovery_hooks
from .recovery import mark_as_saver
from .recovery import mark_as_loader

@register_recovery_hooks
class EpochCounter:
    """
    An epoch counter which can save and recall its state.
    Use this as the iterator for epochs.

    Note that this iterator gives you the numbers from [1 ... limit] not
    [0 ... limit-1] as range(limit) would.

    Example:
        from speechbrain.utils.epoch_loop import EpochCounter
        from speechbrain.utils.recovery import Recoverer
        epoch_counter = EpochCounter(10)
        recoverer = Recoverer("exp/train", {"epoch": epoch_counter})
        recoverer.recover_if_possible()
        # Now after recovery, the epoch starts from where it left off!
        for epoch in epoch_counter:
            # Run training...
            recoverer.save_checkpoint()
    Author:
        Aku Rouhe 2020
    """
    def __init__(self, limit):
        self.current = 0
        self.limit = int(limit)

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current <= self.limit:
            self.current += 1
            return self.current
        raise StopIteration

    @mark_as_saver
    def _save(self, path):
        with open(path, "w") as fo:
            fo.write(str(self.current))
    
    @mark_as_loader
    def _recover(self, path):
        with open(path) as fi:
            self.current = int(fi.read())
