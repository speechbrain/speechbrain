"""High-level interface for optimized 1D convolution kernels"""

from .common import triton_enabled, log_deselect_reason, log_perf_warning, log_picked_kernel

import torch
import torch.nn.functional as F

def _can_select_triton_conv1d_depthwise(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    k = "conv1d_depthwise"
    selected = True
    if not triton_enabled:
        log_deselect_reason(k, "Triton is not enabled")
        selected = False
    if groups != weight.shape[0]:
        log_deselect_reason(k, "Not a depthwise kernel")
        return False  # no need to log further down this point
    if stride != 1:
        log_deselect_reason(k, "Stride !=1 is not implemented")
        selected = False
    if padding != "same":  # TODO: figure out if a passed integer is eq to same
        log_deselect_reason(k, "Padding !=same is not implemented")
        selected = False
    if dilation != 1:
        log_deselect_reason(k, "Dilation !=1 is not implemented")
        selected = False
    return selected

def _fallback_conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, channels_first: bool = False, mask_chunk_size: int = 0):
    if channels_first:
        # convert input from (B, N, C) format to (B, C, N)
        input = input.transpose(-2, -1)

    if mask_chunk_size == 0:
        # simple fallback: no dynamic chunk convolution
        return F.conv1d(input, weight, bias, stride, padding, dilation, groups)

    assert (
        dilation == 1
    ), "Dynamic chunk convolutions currently do not support dilation != 1"

    # in a causal convolution, which is not the case here, an output
    # frame would never be able to depend on a input frame from any
    # point in the future.

    # but with the dynamic chunk convolution, we instead use a "normal"
    # convolution but where, for any output frame, the future beyond the
    # "current" chunk gets masked.

    batch_size = input.shape[0]
    padding = (weight.shape[-1] - 1) // 2

    # determine the amount of padding we need to insert at the right of
    # the last chunk so that all chunks end up with the same size.
    if input.shape[2] % mask_chunk_size != 0:
        final_right_padding = mask_chunk_size - (input.shape[2] % mask_chunk_size)
    else:
        final_right_padding = 0

    out = input

    # -> [batch_size, in_channels, lc+t+final_right_padding]
    out = F.pad(out, (padding, final_right_padding), value=0)

    # now, make chunks with left context.
    # as a recap to what the above padding and this unfold do, consider
    # each a/b/c letter represents a frame as part of chunks a, b, c.
    # consider a chunk size of 4 and a kernel size of 5 (padding=2):
    #
    # input seq: 00aaaabbbbcc00
    # chunk #1:  00aaaa
    # chunk #2:      aabbbb
    # chunk #3:          bbcc00
    #
    # a few remarks here:
    # - the left padding gets inserted early so that the unfold logic
    #   works trivially
    # - the right 0-padding got inserted as the number of time steps
    #   could not be evenly split in `chunk_size` chunks

    # -> [batch_size, in_channels, num_chunks, lc+chunk_size]
    out = out.unfold(2, size=mask_chunk_size + padding, step=mask_chunk_size)

    # as we manually disable padding in the convolution below, we insert
    # right 0-padding to the chunks, e.g. reusing the above example:
    #
    # chunk #1:  00aaaa00
    # chunk #2:      aabbbb00
    # chunk #3:          bbcc0000

    # -> [batch_size, in_channels, num_chunks, lc+chunk_size+rpad]
    out = F.pad(out, (0, padding), value=0)

    # the transpose+flatten effectively flattens chunks into the batch
    # dimension to be processed into the time-wise convolution. the
    # chunks will later on be unflattened.

    # -> [batch_size, num_chunks, in_channels, lc+chunk_size+rpad]
    out = out.transpose(1, 2)

    # -> [batch_size * num_chunks, in_channels, lc+chunk_size+rpad]
    out = out.flatten(start_dim=0, end_dim=1)

    # let's keep backwards compat by pointing at the weights from the
    # already declared Conv1d.
    #
    # still reusing the above example, the convolution will be applied,
    # with the padding truncated on both ends. the following example
    # shows the letter corresponding to the input frame on which the
    # convolution was centered.
    #
    # as you can see, the sum of lengths of all chunks is equal to our
    # input sequence length + `final_right_padding`.
    #
    # chunk #1:  aaaa
    # chunk #2:      bbbb
    # chunk #3:          cc00

    # -> [batch_size * num_chunks, out_channels, chunk_size]
    out = F.conv1d(
        out,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=0,
        dilation=dilation,
        groups=weight.shape[0],
    )

    # -> [batch_size * num_chunks, chunk_size, out_channels]
    out = out.transpose(1, 2)

    # -> [batch_size, num_chunks, chunk_size, out_channels]
    out = torch.unflatten(out, dim=0, sizes=(batch_size, -1))

    # -> [batch_size, t + final_right_padding, out_channels]
    out = torch.flatten(out, start_dim=1, end_dim=2)

    # -> [batch_size, t, out_channels]
    if final_right_padding > 0:
        out = out[:, :-final_right_padding, :]

    return out

def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, channels_first: bool = False, mask_chunk_size: int = 0):
    """Compatible implementation of `torch.nn.functional.conv1d` which can
    automatically select high-performance kernels.

    For Triton-based kernels, `channels_first=True` is preferred."""

    if _can_select_triton_conv1d_depthwise(input, weight, bias, stride, padding, dilation, groups):
        from .triton.conv1d_depthwise import conv1d_depthwise

        log_picked_kernel("conv1d_depthwise")

        if not channels_first:
            # convert input from (B, C, N) format to (B, N, C)
            log_perf_warning("conv1d_depthwise", "channels_first=False is slower for triton kernel")
            input = input.transpose(-2, -1)

        # FIXME: stride
        # FIXME: native bias
        out = conv1d_depthwise(input.contiguous(), weight, mask_chunk_size)
        if bias is not None:
            out = out + bias

        if not channels_first:
            out = out.transpose(-2, -1)

        return out
    else:
        if channels_first:
            # convert input from (B, N, C) format to (B, C, N)
            log_perf_warning("fallback_kernel", "channels_first=True is slower for fallback kernel (but faster for triton kernel)")
            input = input.transpose(-2, -1)

        out = _fallback_conv1d(input, weight, bias, stride, padding, dilation, groups, channels_first, mask_chunk_size)

        if channels_first:
            out = out.transpose(-2, -1)

        return out
