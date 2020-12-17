"""PyTorch compatible samplers

These determine the order of iteration through a dataset.

Authors:
  * Aku Rouhe 2020
"""
import torch
import logging
from torch.utils.data.sampler import RandomSampler

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
    ...     num_workers = 3, collate_fn=None)
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
    ...        num_workers = 3, collate_fn=None)
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


class SequenceByBucketSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        idx_to_num_tokens: any,  # function
        batch_size: int,
        sampler: Sampler[int] = None,
        bucket_boundaries: List[int] = [],
        min_bucket_length: int = -1,
        bucket_length_multiplier: float = 1.1,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ):
        self._dataset = dataset
        self._bucket_boundaries = np.array(
            self._get_data_boundaries(
                batch_size=batch_size,
                bucket_boundaries=bucket_boundaries,
                min_bucket_length=min_bucket_length,
                bucket_length_multiplier=bucket_length_multiplier,
            )
        )
        self._idx_to_num_tokens = idx_to_num_tokens
        self._batch_size = batch_size
        self._sampler = sampler
        self._shuffle = shuffle
        self._seed = seed
        self._drop_last = drop_last
        # Calculate bucket lengths
        self._bucket_lens = [
            max(1, math.ceil(batch_size / self._bucket_boundaries[i]))
            for i in range(len(self._bucket_boundaries))
        ] + [1]
        self._epoch = 0
        self._batches = []

    def _get_data_boundaries(
        self,
        batch_size: int,
        bucket_boundaries: List[int],
        min_bucket_length: int,
        bucket_length_multiplier: float,
    ) -> List[int]:
        if not bucket_boundaries:
            if min_bucket_length <= 0:
                raise ValueError(
                    "min_bucket_length must be >0 if no bucket_boundaries set"
                )
            if bucket_length_multiplier < 1.0:
                raise ValueError(
                    "bucket_length_multiplier must be >1.0 if no bucket_boundaries set"
                )
            bucket_boundaries = {min_bucket_length}
            bucket_boundary = float(min_bucket_length)
            while True:
                bucket_boundary *= bucket_length_multiplier
                bucket_boundaries.add(round(bucket_boundary))
                if bucket_boundary >= batch_size:
                    break
        return list(sorted(bucket_boundaries))

    def _generate_batches(self):
        logger.info("SequenceByBucketSampler:  Generating dynamic batches")
        sampler = self._sampler
        if self._sampler is None:
            if self._shuffle:
                # deterministically shuffle based on epoch and seed
                g = torch.Generator()
                g.manual_seed(self._seed + self._epoch)
                sampler = torch.randperm(len(self._dataset), generator=g).tolist()  # type: ignore
            else:
                sampler = range(len(self._dataset))  # type: ignore

        self._batches = []
        bucket_batches = [[] for i in self._bucket_lens]
        for idx in sampler:
            item_len = self._idx_to_num_tokens(idx)
            bucket_id = np.searchsorted(self._bucket_boundaries, item_len)
            bucket_batches[bucket_id].append(idx)
            if len(bucket_batches[bucket_id]) >= self._bucket_lens[bucket_id]:
                self._batches.append(bucket_batches[bucket_id])
                bucket_batches[bucket_id] = []
        # Dump remaining batches - we might even want to shuffle those
        if not self._drop_last:
            for batch in bucket_batches:
                if batch:
                    self._batches.append(batch)

    def __iter__(self):
        if not self._batches:
            self._generate_batches()
        for batch in self._batches:
            yield batch
        self._batches = []

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __len__(self):
        if not self._batches:
            self._generate_batches()
        return len(self._batches)
