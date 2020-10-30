import torch


def data_collection_sanity_check(data_collection):
    # every key most be unique for each data object.
    # in each data_object keys must be unique (even when nested):
    # e.g. {"audio" : {"file": /path/to/audio.wav}, "alignments": {"file": /path/to/a,ign.pkl}}
    # is not allowed !!
    # e.g. {"audio" : {"file": /path/to/audio.wav}, "alignments": {"alignment_file": /path/to/a,ign.pkl}}
    # is allowed.
    # also every data object must contain same keys and have same structure.
    pass


def replace_entries(data_coll, replacements_dict):
    """

    Parameters
    ----------
    data_coll
    replacements_dict

    Returns
    -------

    """

    for k in data_coll.keys():
        if isinstance(data_coll[k], dict):
            replace_entries(data_coll[k], replacements_dict)
        elif isinstance(data_coll[k], str):
            for repl_k in replacements_dict.keys():
                if k == repl_k:
                    # we should replace
                    for repl_regex in replacements_dict[repl_k].keys():
                        data_coll[k] = data_coll[k].replace(
                            repl_regex, replacements_dict[repl_k][repl_regex]
                        )
        else:
            pass
    return


def pad_right_to(
    tensor: torch.Tensor,
    target_shape: (list, tuple),
    mode="constant",
    value=0.0,
):
    """
    This function takes a torch tensor of arbitrary shape and pads it to target
    shape by appending values on the right.

    Parameters
    ----------
    tensor: input torch tensor
        Input tensor whose dimension we need to pad.
    target_shape: (list, tuple)
        Target shape we want for the target tensor its len must be equal to tensor.ndim
    mode: str
        Pad mode, please refer to torch.nn.functional.pad documentation.
    value: float
        Pad value, please refer to torch.nn.functional.pad documentation.

    Returns
    -------
    tensor: torch.Tensor
        Padded tensor
    valid_vals: list
        List containing proportion for each dimension of original, non-padded values

    """
    assert len(target_shape) == tensor.ndim

    pads = []
    valid_vals = []
    i = len(target_shape) - 1
    j = 0
    while i >= 0:
        assert (
            target_shape[i] >= tensor.shape[i]
        ), "Target shape must be >= original shape for every dim"
        pads.extend([0, target_shape[i] - tensor.shape[i]])
        valid_vals.append(tensor.shape[j] / target_shape[j])
        i -= 1
        j += 1

    tensor = torch.nn.functional.pad(tensor, pads, mode=mode, value=value)

    return tensor, valid_vals


def batch_pad_right(tensors: list, mode="constant", value=0.0):
    """
    Given a list of torch tensors it batches them together by padding to the right
    on each dimension in order to get same length for all.

    Parameters
    ----------
    tensors: list
        List of tensor we wish to pad together.
    mode: str
        Padding mode see torch.nn.functional.pad documentation.
    value: float
        Padding value see torch.nn.functional.pad documentation.

    Returns
    -------
    tensor: torch.Tensor
        Padded tensor
    valid_vals: list
        List containing proportion for each dimension of original, non-padded values

    """

    assert len(tensors), "Tensors list must not be empty"
    if len(tensors) == 1:
        return tensors[0].unsqueeze(0), [[1.0 for x in range(tensors[0].ndim)]]
    assert any(
        [tensors[i].ndim == tensors[0].ndim for i in range(1, len(tensors))]
    ), "All tensors must have same number of dimensions"

    # we gather the max length for each dimension
    max_shape = []
    for dim in range(tensors[0].ndim):
        max_shape.append(max([x.shape[dim] for x in tensors]))

    batched = []
    valid = []
    for t in tensors:
        padded, valid_percent = pad_right_to(
            t, max_shape, mode=mode, value=value
        )
        batched.append(padded)
        valid.append(valid_percent)

    batched = torch.stack(batched)

    return batched, valid
