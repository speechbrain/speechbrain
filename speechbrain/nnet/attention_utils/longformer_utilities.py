"""
Module for utility function for the Longformer implementation. The code comes from:
https://github.com/allenai/longformer and some change were done to fit SpeechBrain's architecture.

    GitHub's repo information:
    Longformer is an open-source project developed by the Allen Institute for Artificial Intelligence (AI2).
    AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and
    engineering.
"""
from typing import Union
from functools import lru_cache
import torch
import torch.nn.functional as F


def mask_invalid_locations(
    attn_weights: torch.Tensor,
    attn_window: int,
    attn_dilatation: Union[torch.Tensor, int],
    autoregressive: bool,
) -> torch.Tensor:
    """
    This helpers function will mask invalid locations for the Longformer
    input_tensor is used for the attention weights
    w is the attention window
    d is the attention dilatation

    Parameters
    ----------
    attn_weights : torch.Tensor
    attn_window : int
    attn_dilatation : Union[torch.Tensor, int],
    autoregressive : bool

    Returns torch.Tensor
    -------

    """
    affected_seq_len, beginning_mask, ending_mask = _get_invalid_locations_mask(
        attn_window, attn_dilatation, autoregressive, attn_weights.device
    )
    seq_len = attn_weights.size(1)
    beginning_input = attn_weights[:, :affected_seq_len, :, : attn_window + 1]
    beginning_mask = beginning_mask[:, :seq_len].expand(beginning_input.size())
    beginning_input.masked_fill_(beginning_mask, -float("inf"))
    if not autoregressive:
        ending_input = attn_weights[
            :, -affected_seq_len:, :, -(attn_window + 1) :
        ]
        ending_mask = ending_mask[:, -seq_len:].expand(ending_input.size())
        ending_input.masked_fill_(ending_mask, -float("inf"))


def _get_invalid_locations_mask_fixed_dilation(
    seq_len: int, attn_window: int, attn_dilatation: int
):
    """
    Internal to get invalid locations (for fixed dilatation)

    Parameters
    ----------
    seq_len : int
    attn_window : int
    attn_dilatation : int

    Returns torch.Tensor
    -------

    """
    diagonals_list = []
    for j in range(
        -attn_dilatation * attn_window, attn_dilatation, attn_dilatation
    ):
        diagonal_mask = torch.zeros(seq_len, device="cpu", dtype=torch.uint8)
        diagonal_mask[:-j] = 1
        diagonals_list.append(diagonal_mask)
    return torch.stack(diagonals_list, dim=-1)


@lru_cache()
def _get_invalid_locations_mask(
    attn_window: int,
    attn_dilatation: Union[torch.Tensor, int],
    autoregressive: bool,
    device: str,
):
    """
    Internal helper function to mask the invalid location for the Longformer

    Parameters
    ----------
    attn_window : int
    attn_dilatation : Union[torch.Tensor, int]
    autoregressive : bool
    device : str

    Returns tuple(int, torch.Tensor, torch.Tensor)
    -------

    """
    if isinstance(attn_dilatation, int):
        affected_seq_len = attn_window * attn_dilatation
        mask = _get_invalid_locations_mask_fixed_dilation(
            affected_seq_len, attn_window, attn_dilatation
        )
        mask = mask[None, :, None, :]
    else:
        affected_seq_len = attn_window * attn_dilatation.max()
        head_masks = []
        d_list = attn_dilatation.cpu().numpy().tolist()
        for d in d_list:
            one_head_mask = _get_invalid_locations_mask_fixed_dilation(
                affected_seq_len, attn_window, d
            )
            head_masks.append(one_head_mask)
        mask = torch.stack(head_masks, dim=-2)
        mask = mask[None, :, :, :]

    ending_mask = (
        None if autoregressive else mask.flip(dims=(1, 3)).bool().to(device)
    )
    return affected_seq_len, mask.bool().to(device), ending_mask


def _skew(x: torch.Tensor, direction: tuple, padding_value: float):
    """
    Convert diagonals into columns (or columns into diagonals depending on `direction`
    Parameters
    ----------
    x : torch.Tensor
    direction : tuple
    padding_value : float

    Returns torch.Tensor
    -------
    """
    x_padded = F.pad(x, direction, value=padding_value)
    x_padded = x_padded.view(
        *x_padded.size()[:-2], x_padded.size(-1), x_padded.size(-2)
    )
    return x_padded


def _skew2(x: torch.Tensor, padding_value: float):
    """
    Shift every row 1 step to right converting columns into diagonals

    Parameters
    ----------
    x : torch.Tensor
    padding_value : float

    Returns torch.Tensor
    -------

    """
    # X = B x C x M x L
    B, C, M, L = x.size()
    x = F.pad(x, (0, M + 1), value=padding_value)  # B x C x M x (L + M + 1)
    x = x.view(B, C, -1)  # B x C x ML+ MM + M
    x = x[:, :, :-M]  # B x C x ML + MM
    x = x.view(B, C, M, M + L)  # B x C, M x L + M
    x = x[:, :, :, :-1]
    return x


def _chunk(x: torch.Tensor, w: int):
    """
    convert into overlapping chunkings. Chunk size = 2w, overlap size = w
    Parameters
    ----------
    x : torch.Tensor
    w : int

    Returns torch.Tensor
    -------
    """

    # non-overlapping chunks of size = 2w
    x = x.view(x.size(0), x.size(1) // (w * 2), w * 2, x.size(2))

    # use `as_strided` to make the chunks overlap with an overlap size = w
    chunk_size = list(x.size())
    chunk_size[1] = chunk_size[1] * 2 - 1

    chunk_stride = list(x.stride())
    chunk_stride[1] = chunk_stride[1] // 2
    return x.as_strided(size=chunk_size, stride=chunk_stride)


def sliding_chunks_matmul_qk(
    q: torch.Tensor, k: torch.Tensor, w: int, padding_value: float
):
    """
    Matrix multiplication of query x key tensors using with a sliding window attention pattern.
    This implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer)
    with an overlap of size w
    Parameters
    ----------
    q : torch.Tensor
    k : torch.Tensor
    w : int
    padding_value : float

    Returns torch.Tensor
    -------

    """
    bsz, seqlen, num_heads, head_dim = q.size()
    assert seqlen % (w * 2) == 0
    assert q.size() == k.size()

    chunks_count = seqlen // w - 1

    # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size w * 2
    q = q.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)
    k = k.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)

    chunk_q = _chunk(q, w)
    chunk_k = _chunk(k, w)

    # matrix multipication
    # bcxd: bsz*num_heads x chunks x 2w x head_dim
    # bcyd: bsz*num_heads x chunks x 2w x head_dim
    # bcxy: bsz*num_heads x chunks x 2w x 2w
    chunk_attn = torch.einsum("bcxd,bcyd->bcxy", (chunk_q, chunk_k))  # multiply

    # convert diagonals into columns
    diagonal_chunk_attn = _skew(
        chunk_attn, direction=(0, 0, 0, 1), padding_value=padding_value
    )

    # allocate space for the overall attention matrix where the chunks are compined. The last dimension
    # has (w * 2 + 1) columns. The first (w) columns are the w lower triangles (attention from a word to
    # w previous words). The following column is attention score from each word to itself, then
    # followed by w columns for the upper triangle.

    diagonal_attn = diagonal_chunk_attn.new_empty(
        (bsz * num_heads, chunks_count + 1, w, w * 2 + 1)
    )

    # copy parts from diagonal_chunk_attn into the compined matrix of attentions
    # - copying the main diagonal and the upper triangle
    diagonal_attn[:, :-1, :, w:] = diagonal_chunk_attn[:, :, :w, : w + 1]
    diagonal_attn[:, -1, :, w:] = diagonal_chunk_attn[:, -1, w:, : w + 1]
    # - copying the lower triangle
    diagonal_attn[:, 1:, :, :w] = diagonal_chunk_attn[
        :, :, -(w + 1) : -1, w + 1 :
    ]
    diagonal_attn[:, 0, 1:w, 1:w] = diagonal_chunk_attn[:, 0, : w - 1, 1 - w :]

    # separate bsz and num_heads dimensions again
    diagonal_attn = diagonal_attn.view(
        bsz, num_heads, seqlen, 2 * w + 1
    ).transpose(2, 1)

    mask_invalid_locations(diagonal_attn, w, 1, False)
    return diagonal_attn


def sliding_chunks_matmul_pv(prob: torch.Tensor, v: torch.Tensor, w: int):
    """
    Same as sliding_chunks_matmul_qk but for prob and value tensors. It is expecting the same output
    format from sliding_chunks_matmul_qk
    Parameters
    ----------
    prob : torch.Tensor
    v : torch.Tensor
    w : int

    Returns torch.Tensor
    -------

    """
    bsz, seqlen, num_heads, head_dim = v.size()
    assert seqlen % (w * 2) == 0
    assert prob.size()[:3] == v.size()[:3]
    assert prob.size(3) == 2 * w + 1
    chunks_count = seqlen // w - 1
    # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size 2w
    chunk_prob = prob.transpose(1, 2).reshape(
        bsz * num_heads, seqlen // w, w, 2 * w + 1
    )

    # group bsz and num_heads dimensions into one
    v = v.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)

    # pad seqlen with w at the beginning of the sequence and another w at the end
    padded_v = F.pad(v, (0, 0, w, w), value=-1)

    # chunk padded_v into chunks of size 3w and an overlap of size w
    chunk_v_size = (bsz * num_heads, chunks_count + 1, 3 * w, head_dim)
    chunk_v_stride = padded_v.stride()
    chunk_v_stride = (
        chunk_v_stride[0],
        w * chunk_v_stride[1],
        chunk_v_stride[1],
        chunk_v_stride[2],
    )
    chunk_v = padded_v.as_strided(size=chunk_v_size, stride=chunk_v_stride)

    skewed_prob = _skew2(chunk_prob, padding_value=0)

    context = torch.einsum("bcwd,bcdh->bcwh", (skewed_prob, chunk_v))
    return context.view(bsz, num_heads, seqlen, head_dim).transpose(1, 2)


def pad_to_window_size(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    one_sided_window_size: int,
    pad_token_id: int,
):
    """A helper function to pad tokens and mask to work with the sliding_chunks implementation of Longformer selfattention.
    Input:
        input_ids = torch.Tensor(bsz x seqlen): ids of wordpieces
        attention_mask = torch.Tensor(bsz x seqlen): attention mask
        one_sided_window_size = int: window size on one side of each token
        pad_token_id = int: tokenizer.pad_token_id
    Returns
        (input_ids, attention_mask) padded to length divisible by 2 * one_sided_window_size
    """
    w = int(2 * one_sided_window_size)
    seqlen = input_ids.size(1)
    padding_len = (w - seqlen % w) % w
    input_ids = F.pad(input_ids, (0, padding_len), value=pad_token_id)
    attention_mask = F.pad(
        attention_mask, (0, padding_len), value=False
    )  # no attention on the padding tokens
    return input_ids, attention_mask


def sliding_chunks_no_overlap_matmul_qk(
    q: torch.Tensor, k: torch.Tensor, w: int, padding_value: float
):
    """
    Matrix multiplication of Q and K
    Parameters
    ----------
    q : torch.Tensor
    k : torch.Tensor
    w : int
    padding_value : float

    Returns
    -------

    """
    bsz, seqlen, num_heads, head_dim = q.size()
    assert seqlen % w == 0
    assert q.size() == k.size()
    # chunk seqlen into non-overlapping chunks of size w
    chunk_q = q.view(bsz, seqlen // w, w, num_heads, head_dim)
    chunk_k = k.view(bsz, seqlen // w, w, num_heads, head_dim)
    chunk_k_expanded = torch.stack(
        (
            F.pad(chunk_k[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0), value=0.0),
            chunk_k,
            F.pad(chunk_k[:, 1:], (0, 0, 0, 0, 0, 0, 0, 1), value=0.0),
        ),
        dim=-1,
    )
    diagonal_attn = torch.einsum(
        "bcxhd,bcyhde->bcxhey", (chunk_q, chunk_k_expanded)
    )  # multiply
    return diagonal_attn.reshape(bsz, seqlen, num_heads, 3 * w)


def sliding_chunks_no_overlap_matmul_pv(
    prob: torch.Tensor, v: torch.Tensor, w: int
):
    """
    Matrix multiplication for sliding chunks methodology
    Parameters
    ----------
    prob : torch.Tensor
    v : torch.Tensor
    w : int

    Returns torch.Tensor
    -------

    """
    bsz, seqlen, num_heads, head_dim = v.size()
    chunk_prob = prob.view(bsz, seqlen // w, w, num_heads, 3, w)
    chunk_v = v.view(bsz, seqlen // w, w, num_heads, head_dim)
    chunk_v_extended = torch.stack(
        (
            F.pad(chunk_v[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0), value=0.0),
            chunk_v,
            F.pad(chunk_v[:, 1:], (0, 0, 0, 0, 0, 0, 0, 1), value=0.0),
        ),
        dim=-1,
    )
    context = torch.einsum(
        "bcwhpd,bcdhep->bcwhe", (chunk_prob, chunk_v_extended)
    )
    return context.reshape(bsz, seqlen, num_heads, head_dim)


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

    batch_size, seq_len, _ = (
        tens.shape[1],
        tens.shape[0],
        tens.shape[2],
    )
    padding_amount = input_size - seq_len
    if padding_amount > 0:
        net_tensor = torch.zeros(
            (seq_len + padding_amount, batch_size, tens.shape[-1]),
            device=tens.device,
        )
        net_tensor[:seq_len, :, :] = tens
        return net_tensor if not permutation else net_tensor.permute((1, 0, 2))
    else:
        return tens if not permutation else tens.permute((1, 0, 2))


def longformer_src_mask_padder(
    src_key_padding_mask: torch.Tensor, window_padding_size: int
):
    """
    Thin wrapper function to do padding for the Longformer's window sliding to work properly
    Parameters
    ----------
    src_key_padding_mask : torch.Tensor
        the tensor to be padded
    window_padding_size : int
        size of the padding

    Returns torch.Tensor
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
