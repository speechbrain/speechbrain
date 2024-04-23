"""SpeechBrain utility functions and classes for distributed training.

This file is heavily inspired by the `huggingface accelerate` library, which is licensed under
the Apache License 2.0. The original code can be found at: https://github.com/huggingface/accelerate/tree/main

The code has been adapted to work with SpeechBrain and only for CPU and GPU training.

Authors:
 * Adel Moumen 2024
"""

import os
from typing import Mapping

import torch


class DistributedState:
    """
    Singleton class that has information about the current training environment and functions to help with process
    control.
    """

    _shared_state = dict()

    def __init__(self, device: str = "cpu", distributed_backend=None):
        self.__dict__ = self._shared_state

        if not self.initialized:
            self.device = device
            self.distributed_backend = distributed_backend

            if self.distributed_backend is None:
                self.num_processes = 1
                self.process_index = 0
                self.local_process_index = 0
            else:
                self.num_processes = torch.distributed.get_world_size()
                self.process_index = torch.distributed.get_rank()
                self.local_process_index = int(os.environ.get("LOCAL_RANK", -1))

    def __repr__(self) -> str:
        return (
            f"Distributed environment: {('  Backend: ' + self.distributed_backend) if self.distributed_backend else ''}\n"
            f"Num processes: {self.num_processes}\n"
            f"Process index: {self.process_index}\n"
            f"Local process index: {self.local_process_index}\n"
            f"Device: {self.device}\n"
        )

    @staticmethod
    def _reset_state():
        "Resets `_shared_state`, is used internally and should not be called"
        DistributedState._shared_state.clear()

    @property
    def initialized(self) -> bool:
        "Returns whether the `DistributedState` has been initialized"
        return self._shared_state != {}

    @property
    def use_distributed(self):
        """
        Whether the Trainer is configured for distributed training
        """
        return self.num_processes > 1


def is_torch_tensor(tensor):
    """Check if `tensor` is a `torch.Tensor` or not."""
    return isinstance(tensor, torch.Tensor)


def is_namedtuple(data):
    """
    Checks if `data` is a `namedtuple` or not. Can have false positives, but only if a user is trying to mimic a
    `namedtuple` perfectly.
    """
    return (
        isinstance(data, tuple)
        and hasattr(data, "_asdict")
        and hasattr(data, "_fields")
    )


def honor_type(obj, generator):
    """
    Cast a generator to the same type as obj (list, tuple, or namedtuple)
    """
    # Some objects may not be able to instantiate from a generator directly
    if is_namedtuple(obj):
        return type(obj)(*list(generator))
    else:
        return type(obj)(generator)


def recursively_apply(
    func,
    data,
    *args,
    test_type=is_torch_tensor,
    error_on_other_type=False,
    **kwargs,
):
    """
    Recursively apply a function on a data structure that is a nested list/tuple/dictionary of a given base type.

    This function is useful when you want to explore a nested data structure and apply a function to every object of a
    given type. For example, you may want to apply a function to every `torch.Tensor` in a nested data structure as
    synchronizing those tensors requires a different function than synchronizing other objects.

    In `gather_for_metrics`, we use this function to first discover the type of the data structure and then apply the
    appropriate function to gather the data (object vs tensor).

    Arguments
    ---------
    func (`callable`):
        The function to recursively apply.
    data (nested list/tuple/dictionary of `main_type`):
        The data on which to apply `func`
    *args:
        Positional arguments that will be passed to `func` when applied on the unpacked data.
    main_type (`type`, *optional*, defaults to `torch.Tensor`):
        The base type of the objects to which apply `func`.
    error_on_other_type (`bool`, *optional*, defaults to `False`):
        Whether to return an error or not if after unpacking `data`, we get on an object that is not of type
        `main_type`. If `False`, the function will leave objects of types different than `main_type` unchanged.
    **kwargs (additional keyword arguments, *optional*):
        Keyword arguments that will be passed to `func` when applied on the unpacked data.

    Returns
    -------
        The same data structure as `data` with `func` applied to every object of type `main_type`.

    Example
    -------
    >>> tensor_list = [torch.zeros(2, dtype=torch.int64) for _ in range(2)]
    >>> tensor_list
    [tensor([0, 0]), tensor([0, 0])]
    >>> recursively_apply(lambda x: x + 1, tensor_list)
    [tensor([1, 1]), tensor([1, 1])]
    >>> complex_data = {"a": [torch.zeros(2, dtype=torch.int64) for _ in range(2)], "b": {"c": torch.zeros(2, dtype=torch.int64)}}
    >>> recursively_apply(lambda x: x + 1, complex_data)
    {'a': [tensor([1, 1]), tensor([1, 1])], 'b': {'c': tensor([1, 1])}}
    >>> complex_data = {"a": 0.0, "b": {"c": torch.zeros(2, dtype=torch.int64)}}
    >>> recursively_apply(lambda x: x + 1, complex_data, test_type=lambda x: isinstance(x, float))
    {'a': 1.0, 'b': {'c': tensor([0, 0])}}
    """
    if isinstance(data, (tuple, list)):
        return honor_type(
            data,
            (
                recursively_apply(
                    func,
                    o,
                    *args,
                    test_type=test_type,
                    error_on_other_type=error_on_other_type,
                    **kwargs,
                )
                for o in data
            ),
        )
    elif isinstance(data, Mapping):
        return type(data)(
            {
                k: recursively_apply(
                    func,
                    v,
                    *args,
                    test_type=test_type,
                    error_on_other_type=error_on_other_type,
                    **kwargs,
                )
                for k, v in data.items()
            }
        )
    elif test_type(data):
        return func(data, *args, **kwargs)
    elif error_on_other_type:
        raise TypeError(
            f"Unsupported types ({type(data)}) passed to `{func.__name__}`. Only nested list/tuple/dicts of "
            f"objects that are valid for `{test_type.__name__}` should be passed."
        )
    return data
