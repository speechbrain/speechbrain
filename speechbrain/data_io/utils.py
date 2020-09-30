import torch


def pad_right_to(tensor, target_shape, mode="constant", value=0.0):
    assert len(target_shape) == tensor.ndim

    pads = []
    valid = []
    i = len(target_shape) - 1
    j = 0
    while i >= 0:
        assert (
            target_shape[i] >= tensor.shape[i]
        ), "Target shape must be >= original shape for every dim"
        pads.extend([0, target_shape[i] - tensor.shape[i]])
        valid.append(tensor.shape[j] / target_shape[j])
        i -= 1
        j += 1

    tensor = torch.nn.functional.pad(tensor, pads, mode=mode, value=value)

    return tensor, valid


def batch_pad_right(tensors: list, mode="constant", value=0.0):

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


def replace_entries(data_coll, replacements_dict):

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
