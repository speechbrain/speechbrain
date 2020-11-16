"""
PyTorch compatible samplers

These determine the order of iteration through a dataset.
"""
import torch
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from speechbrain.data_io.datasets import SegmentedDataset
import logging

logger = logging.getLogger(__name__)


class ReproducibleRandomSampler(RandomSampler):
    """
    A modification of RandomSampler which always returns the same values.

    Also look at `torch.utils.data.RandomSampler`. This has mostly
    the same behaviour and arguments, expect for adding 'seed' and 'epoch' and
    not supporting 'generator'.

    Note
    ----
    Call `set_epoch` before every epoch. Otherwise the sampler will produce the
    same sequence of indices every epoch.

    Arguments
    ---------
    seed : int
        The base seed to use for the random number generator. It is recommended
        to use a value which has a good mix of 0 and 1 bits.
    epoch : int
        The epoch to start at.

    Example
    -------
    >>> import torch
    >>> from speechbrain.utils.checkpoints import Checkpointer
    >>> from speechbrain.data_io.dataloader import SaveableDataLoader
    >>> # An example "dataset"
    >>> dataset = torch.arange(10).unsqueeze(1)
    >>> # Create the random sampler:
    >>> sampler = ReproducibleRandomSampler(dataset)
    >>> dataloader = SaveableDataLoader(dataset, sampler = sampler,
    ...     num_workers = 3)
    >>> # Setup the checkpointer.
    >>> # Note that the sampler doesn't need to be saved itself.
    >>> tmpdir = getfixture('tmpdir')
    >>> checkpointer = Checkpointer(tmpdir, {"dataloader": dataloader})
    >>> # Iterate:
    >>> subset = []
    >>> for i, data_point in enumerate(dataloader):
    ...     # Say you save a checkpoint on the fourth batch:
    ...     if i == 3:
    ...         _ = checkpointer.save_checkpoint(end_of_epoch = False)
    ...     # So let's save the numbers you would get if you continue
    ...     if i >= 4:
    ...         subset.append(data_point.item())
    >>> # What if instead you had to restart the experiment?
    >>> new_sampler = ReproducibleRandomSampler(dataset)
    >>> new_dataloader = SaveableDataLoader(dataset, sampler = new_sampler,
    ...        num_workers = 3)
    >>> new_checkpointer = Checkpointer(tmpdir, {"dataloader": new_dataloader})
    >>> _ = new_checkpointer.recover_if_possible()
    >>> # You'll get the same random order again:
    >>> new_subset = [data_point.item() for data_point in new_dataloader]
    >>> assert subset == new_subset

    """

    def __init__(self, data_source, seed=563375142, epoch=0, **kwargs):
        if "generator" in kwargs:
            MSG = (
                "Cannot give a separate generator when using "
                + "ReproducibleRandomSampler"
            )
            raise ValueError(MSG)
        super().__init__(data_source, **kwargs)
        self.seed = int(seed)
        self.epoch = epoch
        self.generator = torch.Generator()

    def set_epoch(self, epoch):
        """
        You can also just access self.epoch, but we maintain this interface
        to mirror torch.utils.data.distributed.DistributedSampler
        """
        self.epoch = epoch

    def __iter__(self):
        self.generator.manual_seed(self.seed + self.epoch)
        return super().__iter__()


class DynamicBatchSampler(SequentialSampler):
    def __init__(
        self,
        data_source: SegmentedDataset,
        max_seq_len: int,
        sorting="descending",
    ):

        if not isinstance(max_seq_len, int) or max_seq_len <= 0:
            raise ValueError(
                "max_batch_size should be a positive integer value, "
                "got max_batch_size={}".format(max_seq_len)
            )

        self.sorting = sorting
        self.max_bsz = max_seq_len

        # sorting data_source
        if len(data_source.examples) == 0:
            raise IndexError("Data source must not be empty.")

        examples = data_source.examples

        if not ("length" in examples[examples.keys()[0]].keys()):
            raise ValueError(
                "Each example must have a 'length' key containing the length of the example "
                "in order to be able to sort them and use Dynamic Batching"
            )

        # we always sort in descending at beginning.
        # WHY ? because we want to account also for the padding into the max_batch_len

        ex_ids = sorted(
            data_source.ex_ids,
            key=lambda x: examples[x]["length"],
            reverse=True,
        )

        tot_batches = []
        c_batch = []
        c_batch_len = 0
        first_len = None  # the first element is the one which is always longer.
        for c_id in ex_ids:
            c_len = examples[c_id]["length"]
            if first_len is None or (c_batch_len + first_len) <= max_seq_len:
                c_batch.append(c_id)
                first_len = c_len
                c_batch_len += first_len
            else:
                # put it in another batch
                tot_batches.append(c_batch)
                c_batch = [c_id]
                c_batch_len = c_len
                first_len = c_len

        if self.sorting == "ascending":
            # reverse all
            self.bucketed = [x[::-1] for x in tot_batches[::-1]]
        elif self.sorting == "descending":
            # already in descending
            self.bucketed = tot_batches
        else:
            raise ValueError("Sorting must be either ascending or descending")

        super().__init__(data_source)

    def __iter__(self):
        return iter(self.bucketed)

    def __len__(self) -> int:
        return len(self.bucketed)
