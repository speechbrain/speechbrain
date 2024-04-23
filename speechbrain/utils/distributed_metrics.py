"""SpeechBrain utility functions for distributed metrics.

This file is heavily inspired by the `huggingface accelerate` library, which is licensed under
the Apache License 2.0. The original code can be found at: https://github.com/huggingface/accelerate/tree/main

The code has been adapted to work with SpeechBrain and only for CPU and GPU training.

Authors:
 * Adel Moumen 2024
"""

from typing import Any, Mapping

import torch
from packaging import version

from speechbrain.core import DistributedState


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


def _gpu_gather_object(object: Any):
    """Gathers an object from all processes and returns a list containing the object from each process."""
    output_objects = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(output_objects, object)
    # all_gather_object returns a list of lists, so we need to flatten it
    return [x for y in output_objects for x in y]


def _gpu_gather(tensor):
    """Gathers a tensor from all processes and returns a tensor containing the gathered tensors."""
    state = DistributedState()
    if version.parse(torch.__version__) >= version.parse("1.13"):
        gather_op = torch.distributed.all_gather_into_tensor
    else:
        gather_op = torch.distributed._all_gather_base

    def _gpu_gather_one(tensor):
        if tensor.ndim == 0:
            tensor = tensor.clone()[None]

        # Can only gather contiguous tensors
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        if (
            state.distributed_backend is not None
            and state.distributed_backend != "gloo"
        ):
            # We use `empty` as `all_gather_into_tensor` slightly
            # differs from `all_gather` for better efficiency,
            # and we rely on the number of items in the tensor
            # rather than its direct shape
            output_tensors = torch.empty(
                torch.distributed.get_world_size() * tensor.numel(),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            gather_op(output_tensors, tensor)
            return output_tensors.view(-1, *tensor.size()[1:])
        else:
            # Gloo backend does not support `all_gather_into_tensor`,
            # which will result in a larger memory overhead for the op
            output_tensors = [
                torch.empty_like(tensor) for _ in range(state.num_processes)
            ]
            torch.distributed.all_gather(output_tensors, tensor)
            return torch.cat(output_tensors, dim=0)

    return recursively_apply(_gpu_gather_one, tensor, error_on_other_type=True)


def gather(tensor):
    """Gathers a tensor from all processes and returns a tensor containing the gathered tensors."""
    if DistributedState().use_distributed:
        return _gpu_gather(tensor)
    else:
        return tensor


def gather_object(object: Any):
    """Gathers an object from all processes and returns a list containing the object from each process."""
    if DistributedState().use_distributed:
        return _gpu_gather_object(object)
    else:
        return object


def gather_for_metrics(input_data):
    """Gathers the input data from all processes and returns a list containing the data from each process.

    Arguments
    ---------
    input_data (nested list/tuple/dictionary of `torch.Tensor` or obj):
        The data to gather.

    Returns
    -------
        The same data structure as `input_data` with the data from all processes gathered.
        If this is a tuple of tensors, the return will be a tuple of gathered tensors.
        If this is a list/tuple/dict of object, the return will be a list of gathered objects.
    """
    try:
        recursively_apply(lambda x: x, input_data, error_on_other_type=True)
        all_tensors = True
    except TypeError:
        all_tensors = False

    if not all_tensors:
        data = gather_object(input_data)
    else:
        data = gather(input_data)

    return data
