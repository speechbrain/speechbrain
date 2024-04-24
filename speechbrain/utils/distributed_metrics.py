"""SpeechBrain utility functions for distributed metrics.

This file is heavily inspired by the `huggingface accelerate` library, which is licensed under
the Apache License 2.0. The original code can be found at: https://github.com/huggingface/accelerate/tree/main

The code has been adapted to work with SpeechBrain and only for CPU and GPU training.

Authors:
 * Adel Moumen 2024
"""

from typing import Any

import torch
from packaging import version

from speechbrain.utils.distributed_utils import (
    distributed_is_initialized,
    get_distributed_backend,
    recursively_apply,
)


def _gpu_gather_object(object: Any):
    """Gathers an object from all processes and returns a list containing the object from each process."""
    output_objects = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(output_objects, object)
    # `all_gather_object` returns a list of lists, so we need to flatten it
    return [x for y in output_objects for x in y]


def _gpu_gather(tensor):
    """Gathers a tensor from all processes and returns a tensor containing the gathered tensors."""
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

        if get_distributed_backend() != "gloo":
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
                torch.empty_like(tensor)
                for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(output_tensors, tensor)
            return torch.cat(output_tensors, dim=0)

    return recursively_apply(_gpu_gather_one, tensor, error_on_other_type=True)


def gather(tensor):
    """Gather the values in *tensor* across all processes and concatenate them on the first dimension. Useful to
    regroup the predictions from all processes when doing evaluation.

    Arguments
    ---------
    tensor (`torch.Tensor`, or a nested tuple/list/dictionary of `torch.Tensor`):
        The tensors to gather across all processes.

    Returns
    -------
        `torch.Tensor`, or a nested tuple/list/dictionary of `torch.Tensor`: The gathered tensor(s). Note that the
        first dimension of the result is *num_processes* multiplied by the first dimension of the input tensors.

    Example
    -------
    >>> tensor = torch.arange(2) + 1 + 2 * rank  # doctest: +SKIP
    >>> tensor  # doctest: +SKIP
    tensor([1, 2]) # Rank 0
    tensor([3, 4]) # Rank 1
    >>> sync_tensor = gather(tensor)  # doctest: +SKIP
    >>> sync_tensor  # doctest: +SKIP
    tensor([1, 2, 3, 4]) # Rank 0 and 1 combined
    """
    if distributed_is_initialized():
        return _gpu_gather(tensor)
    else:
        return tensor


def gather_object(object: Any):
    """Recursively gather object in a nested list/tuple/dictionary of objects from all devices.

    Arguments
    ---------
    object (nested list/tuple/dictionary of picklable object):
        The data to gather.

    Returns
    -------
        The same data structure as `object` with all the objects sent to every device.

    Example
    -------
    >>> obj = [{"rank": rank, "tensor": torch.tensor([rank])}]  # doctest: +SKIP
    >>> obj  # doctest: +SKIP
    [{'rank': 0, 'tensor': tensor([0])}] # Rank 0
    [{'rank': 1, 'tensor': tensor([1])}] # Rank 1
    >>> sync_obj = gather_object(obj)  # doctest: +SKIP
    >>> sync_obj  # doctest: +SKIP
    [{'rank': 0, 'tensor': tensor([0])}, {'rank': 1, 'tensor': tensor([1])}] # Rank 0 and 1 combined
    """
    if distributed_is_initialized():
        return _gpu_gather_object(object)
    else:
        return object


def gather_for_metrics(input_data):
    """Gathers the input data from all processes and returns a list containing the data from each process.
    Should be used for gathering the inputs and targets for metric calculation.

    Arguments
    ---------
    input_data (`torch.Tensor`, `object`, a nested tuple/list/dictionary of `torch.Tensor`, or a nested tuple/list/dictionary of `object`):
        The tensors or objects for calculating metrics across all processes

    Returns
    -------
        The same data structure as `input_data` with the data from all processes gathered.
        If this is a tuple of tensors, the return will be a tuple of gathered tensors.
        If this is a list/tuple/dict of object, the return will be a list of gathered objects.

    Example
    -------
    >>> tensor_list = [torch.zeros(2, dtype=torch.int64) for _ in range(2)]  # doctest: +SKIP
    >>> tensor_list  # doctest: +SKIP
    [tensor([0, 0]), tensor([0, 0])] # Rank 0 and 1
    >>> tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank  # doctest: +SKIP
    >>> tensor  # doctest: +SKIP
    tensor([1, 2]) # Rank 0
    tensor([3, 4]) # Rank 1
    >>> sync_tensor = gather_for_metrics(tensor)  # doctest: +SKIP
    >>> sync_tensor  # doctest: +SKIP
    tensor([1, 2, 3, 4]) # Rank 0 and 1 combined
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
