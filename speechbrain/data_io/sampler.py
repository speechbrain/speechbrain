"""
PyTorch compatible samplers

These determine the order of iteration through a dataset.
"""
import torch
import logging
from torch.utils.data.sampler import RandomSampler, Sampler
from speechbrain.data_io.datasets import SegmentedDataset

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


class OrderedSampler(Sampler[int]):
    """
        This sampler returns the index of examples according to "ascending" or "descending" order
        according to their lengths.
        This has same exact behaviour as  `torch.utils.data.SequentialSampler`
        the only difference is that we sort the examples.

        Note
        ----
        In the annotation ther must be a key for length in order to use this
        (we need to know th elength of each example in order to be able to sort them).

        Arguments
        ---------
        data_source : dict
            The dictionary containing all examples.
        sorting : str
            Desired order in which examples must be returned according to their length.

        Example
        -------
    """

    def __init__(
        self, data_source: SegmentedDataset, sorting="ascending",
    ):
        if sorting not in ["descending", "ascending"]:
            raise ValueError('Sorting must be in ["descending", "ascending"]')

            # sorting data_source
        if len(data_source.examples) == 0:
            raise IndexError("Data source must not be empty.")

        examples = data_source.examples

        if "length" not in examples[list(examples.keys())[0]].keys():
            raise ValueError(
                "Each example must have a 'length' key containing the length of the example "
                "in order to be able to sort them and use Dynamic Batching"
            )

        ex_indx_length = [
            (i, examples[k]["length"]) for i, k in enumerate(data_source.ex_ids)
        ]

        if sorting == "ascending":
            ex_indx_length = sorted(ex_indx_length, key=lambda x: x[-1])

        else:  # descending
            ex_indx_length = sorted(ex_indx_length, key=lambda x: x[-1])

        self.ex_indx = [x[0] for x in ex_indx_length]

    def __iter__(self):
        return iter(self.ex_indx)

    def __len__(self) -> int:
        return len(self.ex_indx)
