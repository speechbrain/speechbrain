"""Batch collation

Authors
  * Aku Rouhe 2020
"""
import collections
import torch
import speechbrain.utils.data_utils
from torch.utils.data._utils.collate import default_convert
from torch.utils.data._utils.collate import default_collate
from torch.utils.data._utils.pin_memory import (
    pin_memory as recursive_pin_memory,
)
import collections.abc

PaddedData = collections.namedtuple("PaddedData", ["data", "lengths"])


class PaddedBatch:
    """Collate_fn when examples are dicts and have variable length sequences.

    Different elements in the examples get matched by key.
    All numpy tensors get converted to Torch (PyTorch default_convert)
    Then, by default, all torch.Tensor valued elements get padded and support
    collective pin_memory() and to() calls.
    Regular Python data types are just collected in a list.

    Arguments
    ---------
    examples : list
        List of example dicts, as produced by Dataloader.
    padded_keys : list, None, optional
        List of keys to pad on. If None, pad all torch.Tensors
    device_prep_keys : list, None, optional
        Only these keys participate in collective memory pinning and moving with
        to()
        If None, defaults to all items with torch.Tensor values.
    padding_func : callable, optional
        Called with a list of tensors to be padded together. Needs to return
        two tensors: the padded data, and another tensor for the data lengths.
    padding_kwargs : dict, optional
        Extra kwargs to pass to padding_func. E.G. mode, value
    apply_default_convert : bool
        Whether to apply PyTorch default_convert (numpy to torch recursively,
        etc.) on all data. Default:True, usually does the right thing.
    nonpadded_default_collate : bool
        Whether to apply PyTorch default_collate on values that didn't get padded.
        Default:True, usually does the right thing.

    Example
    -------
    >>> batch = PaddedBatch([
    ...     {"id": "ex1", "foo": torch.Tensor([1.])},
    ...     {"id": "ex2", "foo": torch.Tensor([2., 1.])}])
    >>> # Attribute or key-based access:
    >>> batch.id
    ['ex1', 'ex2']
    >>> batch["id"]
    ['ex1', 'ex2']
    >>> # torch.Tensors get padded
    >>> type(batch.foo)
    <class 'speechbrain.data_io.batch.PaddedData'>
    >>> batch.foo.data
    tensor([[1., 0.],
            [2., 1.]])
    >>> batch.foo.lengths
    tensor([0.5000, 1.0000])
    >>> # Batch supports collective operations:
    >>> _ = batch.to(dtype=torch.half)
    >>> batch.foo.data
    tensor([[1., 0.],
            [2., 1.]], dtype=torch.float16)
    >>> batch.foo.lengths
    tensor([0.5000, 1.0000], dtype=torch.float16)

    """

    def __init__(
        self,
        examples,
        padded_keys=None,
        device_prep_keys=None,
        padding_func=speechbrain.utils.data_utils.batch_pad_right,
        padding_kwargs={},
        apply_default_convert=True,
        nonpadded_default_collate=True,
    ):
        self.__keys = list(examples[0].keys())
        self.__padded_keys = []
        self.__device_prep_keys = []
        for key in self.__keys:
            values = [example[key] for example in examples]
            # Default convert usually does the right thing (numpy2torch etc.)
            if apply_default_convert:
                values = default_convert(values)
            if (padded_keys is not None and key in padded_keys) or (
                padded_keys is None and isinstance(values[0], torch.Tensor)
            ):
                # Padding and PaddedData
                self.__padded_keys.append(key)
                padded = PaddedData(*padding_func(values, **padding_kwargs))
                setattr(self, key, padded)
            else:
                # Default PyTorch collate usually does the right thing
                # (convert lists of equal sized tensors to batch tensors, etc.)
                if nonpadded_default_collate:
                    values = default_collate(values)
                setattr(self, key, values)
            if (device_prep_keys is not None and key in device_prep_keys) or (
                device_prep_keys is None and isinstance(values[0], torch.Tensor)
            ):
                self.__device_prep_keys.append(key)

    def __getitem__(self, key):
        if key in self.__keys:
            return getattr(self, key)
        else:
            raise KeyError(f"Batch doesn't have key: {key}")

    def __iter__(self):
        """Iterates over the different elements of the batch

        Example
        -------
        >>> batch = PaddedBatch([
        ...     {"id": "ex1", "val": torch.Tensor([1.])},
        ...     {"id": "ex2", "val": torch.Tensor([2., 1.])}])
        >>> ids, vals = batch
        >>> ids
        ['ex1', 'ex2']
        """
        return iter((getattr(self, key) for key in self.__keys))

    def pin_memory(self):
        """In-place, moves relevant elements to pinned memory."""
        for key in self.__device_prep_keys:
            value = getattr(self, key)
            pinned = recursive_pin_memory(value)
            setattr(self, key, pinned)
        return self

    def to(self, *args, **kwargs):
        """In-place move/cast relevant elements.

        Passes all arguments to torch.Tensor.to, see its documentation.
        """
        for key in self.__device_prep_keys:
            value = getattr(self, key)
            moved = recursive_to(value, *args, **kwargs)
            setattr(self, key, moved)
        return self

    def at_position(self, pos):
        """Fetch an item by its position in the batch"""
        key = self.__keys[pos]
        return getattr(self, key)


def recursive_to(data, *args, **kwargs):
    """Moves data to device, or other type, and handles containers

    Very similar to torch.utils.data._utils.pin_memory.pin_memory,
    but applies .to() instead.
    """
    if isinstance(data, torch.Tensor):
        return data.to(*args, **kwargs)
    elif isinstance(data, collections.abc.Mapping):
        return {
            k: recursive_to(sample, *args, **kwargs)
            for k, sample in data.items()
        }
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return type(data)(
            *(recursive_to(sample, *args, **kwargs) for sample in data)
        )
    elif isinstance(data, collections.abc.Sequence):
        return [recursive_to(sample, *args, **kwargs) for sample in data]
    elif hasattr(data, "to"):
        return data.to(*args, **kwargs)
    # What should be done with unknown data?
    # For now, just return as they are
    else:
        return data
