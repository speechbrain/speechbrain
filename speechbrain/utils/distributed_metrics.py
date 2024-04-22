from typing import Any, Mapping

import torch


def is_torch_tensor(tensor):
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

    Args:
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

    Returns:
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
    output_objects = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(output_objects, object)
    # return output_objects
    # # all_gather_object returns a list of lists, so we need to flatten it
    return [x for y in output_objects for x in y]


def _gpu_gather(tensor):
    gather_op = torch.distributed.all_gather_into_tensor

    def _gpu_gather_one(tensor):
        if tensor.ndim == 0:
            tensor = tensor.clone()[None]

        # Can only gather contiguous tensors
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # if state.backend is not None and state.backend != "gloo":
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
        # TODO: deal with CPU case
        # else:
        #     # a backend of `None` is always CPU
        #     # also gloo does not support `all_gather_into_tensor`,
        #     # which will result in a larger memory overhead for the op
        #     output_tensors = [torch.empty_like(tensor) for _ in range(state.num_processes)]
        #     torch.distributed.all_gather(output_tensors, tensor)
        #     return torch.cat(output_tensors, dim=0)

    return recursively_apply(_gpu_gather_one, tensor, error_on_other_type=True)


def gather_for_metrics(input_data):
    # found = False
    # for lst in input_data:
    #     if lst["key"] == "6930-75918-0017":
    #         import os
    #         print("found key "+ os.environ["RANK"])
    #         found = True
    # if not found:
    #     import os
    #     print("not found key "+ os.environ["RANK"])
    try:
        recursively_apply(lambda x: x, input_data, error_on_other_type=True)
        all_tensors = True
    except TypeError:
        all_tensors = False

    # print(all_tensors)

    if not all_tensors:
        data = _gpu_gather_object(input_data)
    else:
        data = _gpu_gather(input_data)
    # print(data[0]["key"])
    # for lst in data:
    #     if lst["key"] == "6930-75918-0017":
    #         import os
    #         print("found key "+ os.environ["RANK"])

    return data
