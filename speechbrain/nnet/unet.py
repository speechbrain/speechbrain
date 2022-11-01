"""A UNet model implementation for use with diffusion models

Adapted from OpenAI guided diffusion, with slight modifications
and additional features
https://github.com/openai/guided-diffusion

MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Authors
 * Artem Ploujnikov 2022
"""

from abc import abstractmethod

from speechbrain.utils.data_utils import pad_divisible
from .autoencoder import VariationalAutoencoder

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.

    Arguments
    ---------
    dims: int
        The number of dimensions

    Any remaining arguments are passed to the constructor
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    Arguments
    ---------
    timesteps: torch.Tensor
        a 1-D Tensor of N indices, one per batch element. These may be fractional.
    dim: int
        the dimension of the output.
    max_period: int
        controls the minimum frequency of the embeddings.

    Returns
    -------
    result: torch.Tensor
         an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


class AttentionPool2d(nn.Module):
    """Two-dimensional attentional pooling

    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py

    Arguments
    ---------
    spatial_dim: int
        the number of spatial dimensions
    embed_dim: int
        the embedding dimension
    num_heads_channels: int
        the number of attention heads
    output_dim: int
        the output dimension
    """

    def __init__(
        self,
        spatial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(embed_dim, spatial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        """Computes the attention forward pass

        Arguments
        ---------
        x: torch.Tensor
            the tensor to be attended to

        Returns
        -------
        result: torch.Tensor
            the attention output
        """
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb=None):
        """
        Apply the module to `x` given `emb` timestep embeddings.

        Arguments
        ---------
        x: torch.Tensor
            the data tensor
        emb: torch.Tensor
            the embedding tensor
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb=None):
        """Computes a sequential pass with sequential embeddings where applicable

        Arguments
        ---------
        x: torch.Tensor
            the data tensor
        emb: torch.Tensor
            timestep embeddings"""
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    Arguments
    ---------
    channels: torch.Tensor
        channels in the inputs and outputs.
    use_conv: bool
        a bool determining if a convolution is applied.
    dims: int
        determines if the signal is 1D, 2D, or 3D. If 3D, then
        upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(
                dims, self.channels, self.out_channels, 3, padding=1
            )

    def forward(self, x):
        """Computes the upsampling pass

        Arguments
        ---------
        x: torch.Tensor
            layer inputs

        Results
        -------
        result: torch.Tensor
            upsampled outputs"""
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    Arguments
    ---------
    channels: int
        channels in the inputs and outputs.
    use_conv: bool
         a bool determining if a convolution is applied.
    dims: int
        determines if the signal is 1D, 2D, or 3D. If 3D, then
        downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=1,
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        """Computes the downsampling pass

        Arguments
        ---------
        x: torch.Tensor
            layer inputs

        Returns
        -------
        result: torch.Tensor
            downsampled outputs
        """
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    Arguments
    ---------
    channels: int
        the number of input channels.
    emb_channels: int
        the number of timestep embedding channels.
    dropout: float
        the rate of dropout.
    out_channels: int
        if specified, the number of out channels.
    use_conv: bool
        if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    dims: int
        determines if the signal is 1D, 2D, or 3D.
    up: bool
        if True, use this block for upsampling.
    down: bool
        if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
        norm_num_groups=32,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.GroupNorm(norm_num_groups, channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        if emb_channels is not None:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    emb_channels,
                    2 * self.out_channels
                    if use_scale_shift_norm
                    else self.out_channels,
                ),
            )
        else:
            self.emb_layers = None
        self.out_layers = nn.Sequential(
            nn.GroupNorm(norm_num_groups, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(
                    dims, self.out_channels, self.out_channels, 3, padding=1
                )
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        Arguments
        ---------
        x: torch.Tensor
            an [N x C x ...] Tensor of features.
        emb: torch.Tensor
            an [N x emb_channels] Tensor of timestep embeddings.

        Returns
        -------
        result: torch.Tensor
            an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        if emb is not None:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
        else:
            emb_out = torch.zeros_like(h)
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self, channels, num_heads=1, num_head_channels=-1,
        norm_num_groups=32
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = nn.GroupNorm(norm_num_groups, channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        """Completes the forward pass

        Arguments
        ---------
        x: torch.Tensor
            the data to be attended to

        Returns
        -------
        result: torch.Tensor
            The data, with attention applied
        """
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """Apply QKV attention.

        Arguments
        ---------
        qkv: torch.Tensor
            an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.

        Results
        -------
        result: torch.Tensor
            an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum(
            "bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length)
        )
        return a.reshape(bs, -1, length)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    Arguments
    ---------
    in_channels: int
        channels in the input Tensor.
    model_channels: int
        base channel count for the model.
    out_channels: int
        channels in the output Tensor.
    num_res_blocks: int
        number of residual blocks per downsample.
    attention_resolutions: int
        a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    dropout: float
        the dropout probability.
    channel_mult: int
        channel multiplier for each level of the UNet.
    conv_resample: bool
        if True, use learned convolutions for upsampling and
        downsampling.
    dims: int
        determines if the signal is 1D, 2D, or 3D.
    num_classes: int
        if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    num_heads: int
        the number of attention heads in each attention layer.
    num_heads_channels: int
        if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    num_heads_upsample: int
        works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    use_scale_shift_norm: bool
        use a FiLM-like conditioning mechanism.

    resblock_updown: bool
        use residual blocks for up/downsampling.

    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        norm_num_groups=32,
        use_scale_shift_norm=False,
        resblock_updown=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.dtype = torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, ch, 3, padding=1)
                )
            ]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        norm_num_groups=norm_num_groups
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            norm_num_groups=norm_num_groups
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            norm_num_groups=norm_num_groups
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
                norm_num_groups=norm_num_groups
            ),
            AttentionBlock(
                ch, num_heads=num_heads, num_head_channels=num_head_channels,
                norm_num_groups=norm_num_groups
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
                norm_num_groups=norm_num_groups
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        norm_num_groups=norm_num_groups
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            norm_num_groups=norm_num_groups
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            norm_num_groups=norm_num_groups                            
                        )
                        if resblock_updown
                        else Upsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            nn.GroupNorm(norm_num_groups, ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, y=None):
        """Apply the model to an input batch.

        Arguments
        ---------
        x: torch.Tensor
            an [N x C x ...] Tensor of inputs.
        timesteps: torch.Tensor
            a 1-D batch of timesteps.
        y: torch.Tensor
            an [N] Tensor of labels, if class-conditional.

        Returns
        -------
        result: torch.Tensor
            an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(
            timestep_embedding(timesteps, self.model_channels)
        )

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


class EncoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.
    For usage, see UNet.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        norm_num_groups=32,
        use_scale_shift_norm=False,
        resblock_updown=False,
        pool=None,
        attention_pool_dim=None,
        out_kernel_size=3,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dtype = torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.out_kernel_size = out_kernel_size

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, ch, 3, padding=1)
                )
            ]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        norm_num_groups=norm_num_groups
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            norm_num_groups=norm_num_groups
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            norm_num_groups=norm_num_groups
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch, num_heads=num_heads, num_head_channels=num_head_channels,
                norm_num_groups=norm_num_groups
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self.pool = pool
        self.spatial_pooling = False
        if pool is None:
            self.out = nn.Sequential(
                nn.GroupNorm(num_channels=ch, num_groups=norm_num_groups, eps=1e-6),
                nn.SiLU(),
                conv_nd(
                    dims,
                    ch,
                    out_channels,
                    kernel_size=out_kernel_size,
                    padding="same"
                )
            )
        elif pool == "adaptive":
            self.out = nn.Sequential(
                nn.GroupNorm(norm_num_groups, ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                nn.GroupNorm(norm_num_groups, ch),
                nn.SiLU(),
                AttentionPool2d(
                    attention_pool_dim // ds,
                    ch,
                    num_head_channels,
                    out_channels,
                ),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
            self.spatial_pooling = True
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.GroupNorm(norm_num_groups, 2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
            self.spatial_pooling = True
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def forward(self, x, timesteps=None):
        """
        Apply the model to an input batch.

        Arguments
        ---------
        x:  torch.Tensor
            an [N x C x ...] Tensor of inputs.
        timesteps: torch.Tensor
            a 1-D batch of timesteps.
        Returns
        --------
        result: torch.Tensor
            an [N x K] Tensor of outputs.
        """
        emb = None
        if timesteps is not None:
            emb = self.time_embed(
                timestep_embedding(timesteps, self.model_channels)
            )

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.spatial_pooling:
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)
        if self.spatial_pooling:
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = torch.cat(results, axis=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)

class DecoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.
    For usage, see UNet.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        norm_num_groups=32,
        out_kernel_size=3
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dtype = torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        ch = int(channel_mult[0] * model_channels)

        self.input_block =  TimestepEmbedSequential(
            conv_nd(dims, in_channels, ch, 3, padding=1)
        )

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
                norm_num_groups=norm_num_groups
            ),
            AttentionBlock(
                ch, num_heads=num_heads, num_head_channels=num_head_channels,
                norm_num_groups=norm_num_groups
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
                norm_num_groups=norm_num_groups
            ),
        )

        self.upsample_blocks = nn.ModuleList()
        self._feature_size = ch
        ds = 1

        for level, mult in enumerate(reversed(channel_mult)):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        norm_num_groups=norm_num_groups
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            norm_num_groups=norm_num_groups
                        )
                    )
                self.upsample_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.upsample_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            norm_num_groups=norm_num_groups
                        )
                        if resblock_updown
                        else Upsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                ds *= 2
                self._feature_size += ch

        self.out = nn.Sequential(
            nn.GroupNorm(num_channels=ch, num_groups=norm_num_groups, eps=1e-6),
            nn.SiLU(),
            conv_nd(
                dims,
                ch,
                out_channels,
                kernel_size=out_kernel_size,
                padding="same"
            )
        )
        self._feature_size += ch

    def forward(self, x, timesteps=None):
        """
        Apply the model to an input batch.

        Arguments
        ---------
        x:  torch.Tensor
            an [N x C x ...] Tensor of inputs.
        timesteps: torch.Tensor
            a 1-D batch of timesteps.
        Returns
        --------
        result: torch.Tensor
            an [N x K] Tensor of outputs.
        """
        emb = None
        if timesteps is not None:
            emb = self.time_embed(
                timestep_embedding(timesteps, self.model_channels)
            )

        h = x.type(self.dtype)
        h = self.input_block(h, emb)
        h = self.middle_block(h, emb)
        for module in self.upsample_blocks:
            h = module(h, emb)
        h = self.out(h)
        return h


DEFAULT_PADDING_DIMS = [2, 3]
class DownsamplingPadding(nn.Module):
    """A wrapper module that applies the necessary padding for
    the downsampling factor
    
    Arguments
    ---------
    factor: int
        the downsampling / divisibility factor
    len_dim: int
        the index of the dimensions in which the length will vary
    dims: list
        the list of dimensions to be included in padding
    """

    def __init__(self, factor, len_dim=2, dims=None):
        super().__init__()
        self.factor = factor
        self.len_dim = len_dim
        if dims is None:
            dims = DEFAULT_PADDING_DIMS
        self.dims = dims
    
    def forward(self, x, lens=None):
        """Applies the padding
        
        Arguments
        ---------
        x: torch.Tensor
            the sample
        lens: torch.Tensor
            the length tensor

        Returns
        -------
        x_pad: torch.Tensor
            the padded tensor
        lens: torch.Tensor
            the new, adjusted lengths, if applicable
        """
        for dim in self.dims:
            #TODO: Consider expanding pad_divisible to support multiple dimensions
            x, lens_pad = pad_divisible(x, lens, self.factor, len_dim=self.len_dim)
            if dim == self.len_dim:
                lens = lens_pad
        return x, lens_pad


#TODO: Get rid of all hard-coded constants
class UNetVariationalAutencoder(VariationalAutoencoder):
    """A convenience class for a UNet-based Variational Autoencoder (VAE) -
    useful in constructing Latent Diffusion models
    
    Arguments
    ---------
    in_channels: int
        the number of input channels
    model_channels: int
        the number of channels in the convolutional layers of the
        UNet encoder and decoder
    encoder_out_channels: int
        the number of channels the encoder will output
    latent_channels: int
        the number of channels in the latent space
    encoder_num_res_blocks: int
        the number of residual blocks in the encoder
    encoder_attention_resolutions: list
        the resolutions at which to apply attention layers in the encoder
    decoder_num_res_blocks: int
        the number of residual blocks in the decoder
    decoder_attention_resolutions: list
        the resolutions at which to apply attention layers in the encoder
    dropout: float
        the dropout probability
    channel_mult: tuple
        channel multipliers for each layer
    dims: int
        the convolution dimension to use (1, 2 or 3)
    num_heads: int
        the number of attention heads
    num_head_channels: int
        the number of channels in attention heads
    num_heads_upsample: int
        the number of upsampling heads
    use_scale_shift_norm: bool
        whether to use scale shift normalization
    resblock_updown: bool
        whether to use residual blocks for upsampling and downsampling
    out_kernel_size: int
        the kernel size for output convolution layers (if applicable)
    """
    def __init__(
        self,
        in_channels,
        model_channels,
        encoder_out_channels,
        latent_channels,
        encoder_num_res_blocks,
        encoder_attention_resolutions,
        decoder_num_res_blocks,
        decoder_attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        dims=2,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        norm_num_groups=32,
        use_scale_shift_norm=False,
        resblock_updown=False,
        out_kernel_size=3,
        len_dim=2,
        out_mask_value=0.,
        latent_mask_value=0.
    ):
        encoder_unet = EncoderUNetModel(
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=encoder_out_channels,
            num_res_blocks=encoder_num_res_blocks,
            attention_resolutions=encoder_attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            dims=dims,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            norm_num_groups=norm_num_groups,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            out_kernel_size=out_kernel_size
        )

        encoder_pad = DownsamplingPadding(
            2 ** len(channel_mult)
        )

        encoder = nn.Sequential(
            encoder_unet,
            encoder_pad
        )

        decoder = DecoderUNetModel(
            in_channels=latent_channels,
            out_channels=in_channels,
            model_channels=model_channels,
            num_res_blocks=decoder_num_res_blocks,
            attention_resolutions=decoder_attention_resolutions,
            dropout=dropout,
            channel_mult=list(channel_mult),
            dims=dims,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            norm_num_groups=norm_num_groups,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            out_kernel_size=out_kernel_size
        )
        mean, log_var = [
            conv_nd(
                dims=dims, in_channels=encoder_out_channels, out_channels=latent_channels,
                kernel_size=1
            )
            for _ in range(2)
        ]
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            mean=mean,
            log_var=log_var,
            len_dim=len_dim,
            out_mask_value=out_mask_value,
            latent_mask_value=latent_mask_value
        )