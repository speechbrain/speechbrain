"""PyTorch compatible DataLoaders

Essentially we extend PyTorch DataLoader by adding the ability to save the
data loading state, so that a checkpoint may be saved in the middle of an
epoch.

Example
-------
>>> import torch
>>> from speechbrain.utils.checkpoints import Checkpointer
>>> # An example "dataset" and its loader
>>> dataset = torch.randn(10, 1)
>>> dataloader = SaveableDataLoader(dataset, num_workers = 3)
>>> # Setup the checkpointer:
>>> tmpdir = getfixture('tmpdir')
>>> checkpointer = Checkpointer(tmpdir, {"dataloader": dataloader})
>>> # Iterate:
>>> for i, data_point in enumerate(dataloader):
...     # Here you would process the data:
...     rainfall_amount_prediction = data_point * 4.
...     # Now, imagine the experiment gets killed on the fifth batch:
...     if i == 4:
...         break
...     # Luckily, you had just saved a checkpoint:
...     if i == 3:
...         _ = checkpointer.save_checkpoint(end_of_epoch = False)
>>> # So when you restart the experiment:
>>> new_dataloader = SaveableDataLoader(dataset, num_workers = 3)
>>> new_checkpointer = Checkpointer(tmpdir, {"dataloader": new_dataloader})
>>> _ = new_checkpointer.recover_if_possible()
>>> # The dataloader fast-forwards to the position where we left off:
>>> assert next(iter(new_dataloader)) == dataset[4]

Authors:
  * Aku Rouhe 2020
"""
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import _BaseDataLoaderIter
import logging
import functools
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.sampler import ReproducibleRandomSampler
from speechbrain.utils.checkpoints import (
    register_checkpoint_hooks,
    mark_as_saver,
    mark_as_loader,
)

logger = logging.getLogger(__name__)


def make_dataloader(dataset, **loader_kwargs):
    """Makes a basic DataLoader with SpeechBrain defaults.

    For DynamicItemDatasets (which return dicts), use
    PaddedBatch as the default collate_fn.

    Shuffling gets implemented by ReproducibleRandomSampler.

    If the Dataset is not an IterableDataset, the DataLoader
    is a SaveableDataLoader.

    Arguments
    ---------
    dataset : Dataset
        The dataset to make a DataLoader for.
    **loader_kwargs : dict
        Keyword args to DataLoader, see PyTorch DataLoader for
        options.

    Returns
    -------
    DataLoader
    """
    # PaddedBatch as default collation for DynamicItemDataset
    if "collate_fn" not in loader_kwargs and isinstance(
        dataset, DynamicItemDataset
    ):
        loader_kwargs["collate_fn"] = PaddedBatch
    # Reproducible random sampling
    if loader_kwargs.get("shuffle", False):
        if loader_kwargs.get("sampler") is not None:
            raise ValueError(
                "Cannot specify both shuffle=True and a "
                "sampler in loader_kwargs"
            )
        sampler = ReproducibleRandomSampler(dataset)
        loader_kwargs["sampler"] = sampler
        # Should delete shuffle because you can't set both Sampler and
        # shuffle
        # NOTE: the dict of loader options may get used elsewhere!
        # However, this del doesn't touch those because loader_kwargs comes
        # from a **kwargs dict.
        del loader_kwargs["shuffle"]
    # Create the loader
    if isinstance(dataset, IterableDataset):
        dataloader = DataLoader(dataset, **loader_kwargs)
    else:
        dataloader = SaveableDataLoader(dataset, **loader_kwargs)
    return dataloader


# We essentially want to make the DataLoader iterators able to skip ahead
# after checkpoint recovery
# This should be handled by the DataLoader iterators' base class.
# To make the implementation here a little more maintainable
# we decide to patch some PyTorch functionality


def __new_init(self, loader, *args, **kwargs):
    self.__old_init__(loader, *args, **kwargs)
    if (
        hasattr(loader, "_speechbrain_recovery_skip_to")
        and loader._speechbrain_recovery_skip_to is not None
    ):
        # Fast forward the sampler iterator since we have recovered:
        for _ in range(loader._speechbrain_recovery_skip_to):
            next(self._sampler_iter)
        self._num_yielded = loader._speechbrain_recovery_skip_to
        # Mark recovery as done:
        loader._speechbrain_recovery_skip_to = None


def __new_reset(self, loader, first_iter=False, *args, **kwargs):
    # On the first iteration, these have already normally been set by the init anyway.
    # And we don't want to overwrite them if we've recovered
    if not first_iter:
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        self._IterableDataset_len_called = loader._IterableDataset_len_called


# functools.update_wrapper is meant for decorators, but it should basically
# preserve what we want:
functools.update_wrapper(__new_init, _BaseDataLoaderIter.__init__)
_BaseDataLoaderIter.__old_init__ = _BaseDataLoaderIter.__init__
_BaseDataLoaderIter.__init__ = __new_init
if hasattr(_BaseDataLoaderIter, "_reset"):
    _BaseDataLoaderIter._reset = __new_reset


@register_checkpoint_hooks
class SaveableDataLoader(DataLoader):
    """A saveable version of the PyTorch DataLoader.

    See `torch.utils.data.DataLoader` for usage. This class should work exactly
    like the PyTorch basic DataLoader, but this can be checkpointed with
    SpeechBrain's Checkpointer.

    Note
    ----
    1. The saveability is implemented via some unfortunately slightly magical
    means.
    2. The data loader cannot recover after entering __iter__. Normally this is
    not a problem, as recovery should happen before training begins.  However,
    just before evaluation, it is also typical to recover the checkpoint at
    which performance was the best. Thus, if a checkpoint is loaded after
    entering __iter__, we just assume it is for this reason. A warning is
    logged, but that is all.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.dataset, IterableDataset):
            logging.warning(
                "SaveableDataLoader cannot save the position in an "
                "IterableDataset. Save the position on the dataset itself."
            )
        self._speechbrain_recovery_skip_to = None
        self._speechbrain_iterator = None

    def __iter__(self):
        iterator = super().__iter__()
        # Keep a reference to the iterator,
        # to be able to access the iterator._num_yielded value.
        # Keep a full reference (keeping the iterator alive)
        # rather than e.g. a weakref, as we may want to save a checkpoint
        # after the iterator has been exhausted, but before the full epoch has
        # ended (e.g. validation is still running)
        self._speechbrain_iterator = iterator
        return iterator

    @mark_as_saver
    def _speechbrain_save(self, path):
        if isinstance(self.dataset, IterableDataset):
            logging.warning(
                "Warning again: a checkpoint was requested on "
                "SaveableDataLoader, but the dataset is an IterableDataset. "
                "Cannot save the position in an IterableDataset. Not raising "
                "an error; assuming that you know what you're doing."
            )
        if self._speechbrain_iterator is None:
            to_save = None
        else:
            to_save = self._speechbrain_iterator._num_yielded
        with open(path, "w") as fo:
            fo.write(str(to_save))

    @mark_as_loader
    def _speechbrain_load(self, path, end_of_epoch, device=None):
        del device  # Unused here
        if self._speechbrain_iterator is not None:
            logging.warning(
                "SaveableDataLoader was requested to load a "
                "checkpoint, but the data loader has already been "
                "iterated. Cannot load checkpoint here. Assuming that the "
                "checkpoint was only loaded for e.g. retrieving the best "
                "model"
            )
            return
        if end_of_epoch:
            # Don't load at end of epoch, as we actually want to start a fresh
            # epoch iteration next.
            return
        with open(path) as fi:
            saved = fi.read()
            if saved == str(None):
                # Saved at a point where e.g. an iterator did not yet exist.
                return
            else:
                self._speechbrain_recovery_skip_to = int(saved)
