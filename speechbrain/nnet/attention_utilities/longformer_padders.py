import torch


def longformer_src_padder(tens, window_padding_size, permutation=True):
    """
    Thin wrapper function to do padding for the Longformer's window sliding to work properly
    Parameters
    ----------
    tens : tensor
        the tensor to be padded
    window_padding_size : int
        size of the padding
    permutation : bool
        permute or not the tensor - to respect SpeechBrain's dimensionality

    Returns padded tensor
    -------

    """
    assert window_padding_size > 0, "you need to provide a window padding size"

    if permutation:
        tens = tens.permute((1, 0, 2))

    shape_modulo = tens.shape[0] % (2 * window_padding_size)
    input_size = (
        tens.shape[0] - shape_modulo + (2 * window_padding_size)
        if shape_modulo != 0
        else tens.shape[0]
    )

    batch_size, seq_len, hidden_size = (
        tens.shape[1],
        tens.shape[0],
        tens.shape[2],
    )
    padding_amount = input_size - seq_len
    if padding_amount > 0:
        net_tensor = torch.zeros(
            (seq_len + padding_amount, batch_size, tens.shape[-1]), device=tens.device,
        )
        net_tensor[:seq_len, :, :] = tens
        return net_tensor if not permutation else net_tensor.permute((1, 0, 2))
    else:
        return tens if not permutation else tens.permute((1, 0, 2))


def longformer_src_mask_padder(src_key_padding_mask, window_padding_size):
    """
    Thin wrapper function to do padding for the Longformer's window sliding to work properly
    Parameters
    ----------
    src_key_padding_mask : torch.tensor
        the tensor to be padded
    window_padding_size : int
        size of the padding

    Returns padded tensor
    -------
    """
    longformuler_modulo = src_key_padding_mask.shape[1] % (
        2 * window_padding_size
    )
    new_dim = (
        src_key_padding_mask.shape[1]
        - longformuler_modulo
        + (2 * window_padding_size)
        if longformuler_modulo != 0
        else src_key_padding_mask.shape[1]
    )
    net_tensor = torch.zeros(
        (src_key_padding_mask.shape[0], new_dim),
        device=src_key_padding_mask.device,
    )
    net_tensor[:, : src_key_padding_mask.shape[1]] = src_key_padding_mask
    net_tensor = net_tensor.bool()
    return net_tensor
