"""
Optimizers for neural network training.
Expanded from fairseq implementation to allow for Conv1D
link: https://github.com/pytorch/fairseq/blob/5e82514d687289a73a6dec33b555217acd97cb0d/fairseq/modules/quant_noise.py

Authors
 * Samuele Cornell 2020
"""

import torch
import torch.nn as nn


def quant_noise(module, p, block_size):
    """
    Module Wrapper that applies quantization noise to the weights to allow for
    better performance once quantization is applied after it is trained [1]

    Refs:
    [1] Training with Quantization Noise for Extreme Model Compression
    Args:
        - module: nn.Module
        - p: amount of quant noise
        - block_size: blocks size when iPQ is used. Note if used
                        Input features must be a multiple of block sizes
    """

    # skip if no noise
    if p <= 0:
        return module

    # supported modules
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d, nn.Conv1d))

    # 2D matrix
    if isinstance(module, (nn.Linear, nn.Embedding)):
        assert (
            module.weight.size(1) % block_size == 0
        ), "Input features must be a multiple of block sizes"

    elif isinstance(module, nn.Conv1d):
        assert (
            module.in_channels % block_size == 0
        ), "Input features must be a multiple of block sizes"

    # 4D matrix 2D Convolution
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            assert (
                module.in_channels % block_size == 0
            ), "Input channels must be a multiple of block sizes"
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            assert (
                k % block_size == 0
            ), "Kernel size must be a multiple of block size"

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        if mod.training:
            if isinstance(mod, (nn.Linear, nn.Embedding)):
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks
                mask = torch.zeros(
                    in_features // block_size * out_features,
                    device=weight.device,
                )
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(
                    -1, in_features
                )
            elif isinstance(mod, (nn.Conv1d)):

                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                if mod.kernel_size == (1,):
                    mask = torch.zeros(
                        int(in_channels // block_size * out_channels),
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(
                        -1, in_channels
                    )
                else:
                    mask = torch.zeros(
                        weight.size(0), weight.size(1), device=weight.device
                    )
                    mask.bernoulli_(p)
                mask = mask.unsqueeze(-1).repeat(1, 1, mod.kernel_size[0])

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(
                        int(in_channels // block_size * out_channels),
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(
                        -1, in_channels
                    )
                else:
                    mask = torch.zeros(
                        weight.size(0), weight.size(1), device=weight.device
                    )
                    mask.bernoulli_(p)

                mask = (
                    mask.unsqueeze(2)
                    .unsqueeze(3)
                    .repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
                )
            # scale weights and apply mask
            mask = mask.to(
                torch.bool
            )  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)

            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module
