import triton
import triton.language as tl
import torch

@triton.autotune(configs=[
    triton.Config(kwargs={"BLOCK_SIZE_W": 2}, num_warps=2),
  ],
  key=['STEP_SIZE', 'KERNEL_SIZE']
)
@triton.jit
def conv1d_depthwise_nwc_chunked_kernel_0pad(
    # TODO: bias
    in_ptr,
    kernel_ptr,  # [kernel_size, n_channels]
    out_ptr,
    batch_stride,  # element stride from a batch to the next (w*c)
    step_stride,  # element stride from a step (2nd axis) to the next (c)
    kernel_batch_stride,  # stride in kernel for the batch, 0 if kernel isn't batched
    output_batch_stride,
    KERNEL_SIZE: "tl.constexpr",
    STEP_SIZE,
    OUTPUT_STEP_SIZE,
    CHANNEL_SIZE: "tl.constexpr",
    mask_chunk_size: int,
    transpose: "tl.constexpr",
    pad_count: "tl.constexpr",
    BLOCK_SIZE_W: "tl.constexpr",
):
    # indices within the grid
    batch_index = tl.program_id(axis=0)
    step_indices = (
        (tl.program_id(axis=1) * BLOCK_SIZE_W)
        + tl.arange(0, BLOCK_SIZE_W)
    )

    # (expressed as offsets in element counts, for pointer calculations)
    batch_offset = batch_index * batch_stride
    step_offsets = step_indices * step_stride

    # TODO: fuse bias
    # TODO: configurable accumulator type?
    accumulator = tl.zeros((BLOCK_SIZE_W, CHANNEL_SIZE), dtype=tl.float32)

    # NOTE: it would be better if this loop was unrolled into native tensor
    # manipulation and the result accumulated with a simple dot product, but
    # this would be a bit of a complicated refactor
    # chances are it would perform way better on nv though
    for k in tl.range(KERNEL_SIZE):
        # grab a tensor this block has to care about for a given kernel offset

        # offset the block's steps offsets for the current k value
        cur_window_indices = (step_indices - pad_count + k)
        in_offsets = (
            (batch_offset + cur_window_indices * step_stride)[:, None]
            + tl.arange(0, CHANNEL_SIZE)[None, :]
        )
        in_mask = (
            (cur_window_indices >= 0)[:, None]
            & (cur_window_indices < STEP_SIZE)[:, None]
        )

        if mask_chunk_size > 0:
            # for each output index (0..T), get their "chunk ID"
            # e.g. for a chunk size of 4 the first 12 input frames would look like:
            # 000011112222
            #        ^ if we consider this frame, with e.g. k=3
            #       112
            #       <-> these frames would be considered
            #         2
            #         ^ but as this frame is in a future chunk; mask it as 0
            #
            # Triton has stricter scoping rules than Python, this cannot be
            # hoisted out of the loop manually (while still gated by the
            # mask_chunk_size), but the compiler should easily optimize it out
            chunk_ids = step_indices // mask_chunk_size
            step_chunk_ids = cur_window_indices // mask_chunk_size

            # mask away accesses where the frame we'd look at has a bigger chunk
            # ID than the output index
            if not transpose:
                in_mask &= (step_chunk_ids <= chunk_ids)[:, None]
            else:
                in_mask &= (step_chunk_ids >= chunk_ids)[:, None]

        # grab value of shape [BLOCK_SIZE_W, CHANNEL_SIZE]
        in_value = tl.load(in_ptr + in_offsets, mask=in_mask, other=0.0)

        # load k-th value of kernel, i.e. a tensor of shape (CHANNEL_SIZE,)
        base_kernel_index = k
        if transpose:
            base_kernel_index = (KERNEL_SIZE - (k + 1))

        in_kernel_offsets = (
            (batch_index * kernel_batch_stride)
            + (base_kernel_index * CHANNEL_SIZE)
            + tl.arange(0, CHANNEL_SIZE)
        )
        in_kernel = tl.load(kernel_ptr + in_kernel_offsets)

        accumulator += in_value * in_kernel

    out_offsets = (
        (batch_index * output_batch_stride + step_offsets)[:, None]
        + tl.arange(0, CHANNEL_SIZE)[None, :] 
    )
    out_offsets = tl.broadcast_to(out_offsets, (BLOCK_SIZE_W, CHANNEL_SIZE))
    out_mask = (step_indices < OUTPUT_STEP_SIZE)[:, None]
    tl.store(out_ptr + out_offsets, accumulator, mask=out_mask)

def conv1d_depthwise_forward(x, kernel, mask_chunk_size=0, force_padding=None, transpose=False, batch_kernel_format=False):
    output = torch.empty_like(x)

    B, N, C = x.shape

    if batch_kernel_format:
        # that kernel would be in form of B, K, C
        assert kernel.shape[0] == B
        _, K, C = kernel.shape
        kernel_batch_stride = K * C
    else:
        assert kernel.shape[0] == C
        assert kernel.shape[1] == 1
        C, _, K = kernel.shape
        kernel_batch_stride = 0

        kernel = kernel.squeeze(1).permute(1, 0).contiguous() # C, 1, K => K, C

    assert x.is_contiguous()
    assert kernel.is_contiguous()

    batch_stride = N*C
    step_stride = C

    if force_padding is not None:
        pad_count = force_padding
        max_steps = N - (K - 1) + pad_count * 2
        # print(f"N={N} K={K} pads={pad_count} => steps={max_steps}")
    else:
        pad_count = (K - 1) // 2
        max_steps = N

    # we use BTC
    output = torch.empty((B, max_steps, C), device=x.device, dtype=x.dtype)

    grid = lambda meta: (B, triton.cdiv(max_steps, meta["BLOCK_SIZE_W"]))
    conv1d_depthwise_nwc_chunked_kernel_0pad[grid](
        in_ptr=x,
        kernel_ptr=kernel,
        out_ptr=output,
        batch_stride=batch_stride,
        step_stride=step_stride,
        kernel_batch_stride=kernel_batch_stride,
        output_batch_stride=max_steps * C,
        KERNEL_SIZE=K,
        STEP_SIZE=N,
        OUTPUT_STEP_SIZE=max_steps,
        CHANNEL_SIZE=C,
        mask_chunk_size=mask_chunk_size,
        transpose=transpose,
        pad_count=pad_count,
    )

    return output

class TritonDepthwiseConv1DFn(torch.autograd.Function):
    @staticmethod
    def setup_context(ctx, inputs, output):
        x, kernel, mask_chunk_size = inputs
        ctx.save_for_backward(x, kernel)
        ctx.mask_chunk_size = mask_chunk_size

    @staticmethod
    def forward(x, kernel, mask_chunk_size=0):
        output = conv1d_depthwise_forward(x, kernel, mask_chunk_size=mask_chunk_size)
        return output
    
    @staticmethod
    def backward(ctx, grad_out):
        x, kernel = ctx.saved_tensors

        c, _, k = kernel.shape
        padding = (k-1)//2

        grad_inp = grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_inp = conv1d_depthwise_forward(
                grad_out.contiguous(),
                kernel,
                ctx.mask_chunk_size,
                transpose=True,
            )

        if ctx.needs_input_grad[1]:
            grad_weight = conv1d_depthwise_forward(
                x.contiguous(),
                grad_out.contiguous(),
                ctx.mask_chunk_size,
                transpose=False,
                batch_kernel_format=True,
                force_padding=padding,
            )
            grad_weight = grad_weight.permute(0, 2, 1).sum(dim=0).unsqueeze(1)

        return grad_inp, grad_weight, None, None

conv1d_depthwise = TritonDepthwiseConv1DFn.apply
