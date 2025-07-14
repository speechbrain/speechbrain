"""Utils functions for the decoding modules.

Authors
 * Adel Moumen 2023
 * Ju-Chieh Chou 2020
 * Peter Plantinga 2020
 * Mirco Ravanelli 2020
 * Sung-Lin Yeh 2020
"""

import torch


def _update_mem(inp_tokens, memory):
    """This function is for updating the memory for transformer searches.
    it is called at each decoding step. When being called, it appends the
    predicted token of the previous step to existing memory.

    Arguments
    ---------
    inp_tokens : torch.Tensor
        Predicted token of the previous decoding step.
    memory : torch.Tensor
        Contains all the predicted tokens.

    Returns
    -------
    Updated memory
    """
    if memory is None:
        memory = torch.empty(inp_tokens.size(0), 0, device=inp_tokens.device)
    return torch.cat([memory, inp_tokens.unsqueeze(1)], dim=-1)


def inflate_tensor(tensor, times, dim):
    """This function inflates the tensor for times along dim.

    Arguments
    ---------
    tensor : torch.Tensor
        The tensor to be inflated.
    times : int
        The tensor will inflate for this number of times.
    dim : int
        The dim to be inflated.

    Returns
    -------
    torch.Tensor
        The inflated tensor.

    Example
    -------
    >>> tensor = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    >>> new_tensor = inflate_tensor(tensor, 2, dim=0)
    >>> new_tensor
    tensor([[1., 2., 3.],
            [1., 2., 3.],
            [4., 5., 6.],
            [4., 5., 6.]])
    """
    return torch.repeat_interleave(tensor, times, dim=dim)


def mask_by_condition(tensor, cond, fill_value):
    """This function will mask some element in the tensor with fill_value, if condition=False.

    Arguments
    ---------
    tensor : torch.Tensor
        The tensor to be masked.
    cond : torch.BoolTensor
        This tensor has to be the same size as tensor.
        Each element represents whether to keep the value in tensor.
    fill_value : float
        The value to fill in the masked element.

    Returns
    -------
    torch.Tensor
        The masked tensor.

    Example
    -------
    >>> tensor = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    >>> cond = torch.BoolTensor([[True, True, False], [True, False, False]])
    >>> mask_by_condition(tensor, cond, 0)
    tensor([[1., 2., 0.],
            [4., 0., 0.]])
    """
    return torch.where(cond, tensor, fill_value)


def batch_filter_seq2seq_output(prediction, eos_id=-1):
    """Calling batch_size times of filter_seq2seq_output.

    Arguments
    ---------
    prediction : list of torch.Tensor
        A list containing the output ints predicted by the seq2seq system.
    eos_id : int, string
        The id of the eos.

    Returns
    -------
    list
        The output predicted by seq2seq model.

    Example
    -------
    >>> predictions = [
    ...     torch.IntTensor([1, 2, 3, 4]),
    ...     torch.IntTensor([2, 3, 4, 5, 6]),
    ... ]
    >>> predictions = batch_filter_seq2seq_output(predictions, eos_id=4)
    >>> predictions
    [[1, 2, 3], [2, 3]]
    """
    outputs = []
    for p in prediction:
        res = filter_seq2seq_output(p.tolist(), eos_id=eos_id)
        outputs.append(res)
    return outputs


def filter_seq2seq_output(string_pred, eos_id=-1):
    """Filter the output until the first eos occurs (exclusive).

    Arguments
    ---------
    string_pred : list
        A list containing the output strings/ints predicted by the seq2seq system.
    eos_id : int, string
        The id of the eos.

    Returns
    -------
    list
        The output predicted by seq2seq model.

    Example
    -------
    >>> string_pred = ["a", "b", "c", "d", "eos", "e"]
    >>> string_out = filter_seq2seq_output(string_pred, eos_id="eos")
    >>> string_out
    ['a', 'b', 'c', 'd']
    """
    if isinstance(string_pred, list):
        try:
            eos_index = next(
                i for i, v in enumerate(string_pred) if v == eos_id
            )
        except StopIteration:
            eos_index = len(string_pred)
        string_out = string_pred[:eos_index]
    else:
        raise ValueError("The input must be a list.")
    return string_out
