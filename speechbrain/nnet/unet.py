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

import math
from abc import abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from speechbrain.utils.data_utils import pad_divisible

from .autoencoders import NormalizingAutoencoder


def fixup(module, use_fixup_init=True):
    """
    Zero out the parameters of a module and return it.

    Arguments
    ---------
    module: torch.nn.Module
        a module
    use_fixup_init: bool
        whether to zero out the parameters. If set to
        false, the function is a no-op

    Returns
    -------
    The fixed module
    """
    if use_fixup_init:
        for p in module.parameters():
            p.detach().zero_()
    return module


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.

    Arguments
    ---------
    dims: int
        The number of dimensions
    *args: tuple
    **kwargs: dict
        Any remaining arguments are passed to the constructor

    Returns
    -------
    The constructed Conv layer
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


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
        the size of the spatial dimension
    embed_dim: int
        the embedding dimension
    num_heads_channels: int
        the number of attention heads
    output_dim: int
        the output dimension

    Example
    -------
    >>> attn_pool = AttentionPool2d(
    ...     spatial_dim=64, embed_dim=16, num_heads_channels=2, output_dim=4
    ... )
    >>> x = torch.randn(4, 1, 64, 64)
    >>> x_pool = attn_pool(x)
    >>> x_pool.shape
    torch.Size([4, 4])
    """

    def __init__(
        self,
        spatial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(embed_dim, spatial_dim**2 + 1) / embed_dim**0.5
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

    Example
    -------
    >>> from speechbrain.nnet.linear import Linear
    >>> class MyBlock(TimestepBlock):
    ...     def __init__(self, input_size, output_size, emb_size):
    ...         super().__init__()
    ...         self.lin = Linear(n_neurons=output_size, input_size=input_size)
    ...         self.emb_proj = Linear(
    ...             n_neurons=output_size,
    ...             input_size=emb_size,
    ...         )
    ...
    ...     def forward(self, x, emb):
    ...         return self.lin(x) + self.emb_proj(emb)
    >>> tes = TimestepEmbedSequential(
    ...     MyBlock(128, 64, 16), Linear(n_neurons=32, input_size=64)
    ... )
    >>> x = torch.randn(4, 10, 128)
    >>> emb = torch.randn(4, 10, 16)
    >>> out = tes(x, emb)
    >>> out.shape
    torch.Size([4, 10, 32])
    """

    def forward(self, x, emb=None):
        """Computes a sequential pass with sequential embeddings where applicable

        Arguments
        ---------
        x: torch.Tensor
            the data tensor
        emb: torch.Tensor
            timestep embeddings

        Returns
        -------
        The processed input
        """
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
    out_channels: int
        Number of output channels. If None, same as input channels.

    Example
    -------
    >>> ups = Upsample(channels=4, use_conv=True, dims=2, out_channels=8)
    >>> x = torch.randn(8, 4, 32, 32)
    >>> x_up = ups(x)
    >>> x_up.shape
    torch.Size([8, 8, 64, 64])
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

        Returns
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
    out_channels: int
        Number of output channels. If None, same as input channels.

    Example
    -------
    >>> ups = Downsample(channels=4, use_conv=True, dims=2, out_channels=8)
    >>> x = torch.randn(8, 4, 32, 32)
    >>> x_up = ups(x)
    >>> x_up.shape
    torch.Size([8, 8, 16, 16])
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
    norm_num_groups: int
        the number of groups for group normalization
    use_fixup_init: bool
        whether to use FixUp initialization

    Example
    -------
    >>> res = ResBlock(
    ...     channels=4,
    ...     emb_channels=8,
    ...     dropout=0.1,
    ...     norm_num_groups=2,
    ...     use_conv=True,
    ... )
    >>> x = torch.randn(2, 4, 32, 32)
    >>> emb = torch.randn(2, 8)
    >>> res_out = res(x, emb)
    >>> res_out.shape
    torch.Size([2, 4, 32, 32])
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        dims=2,
        up=False,
        down=False,
        norm_num_groups=32,
        use_fixup_init=True,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

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
                    self.out_channels,
                ),
            )
        else:
            self.emb_layers = None
        self.out_layers = nn.Sequential(
            nn.GroupNorm(norm_num_groups, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            fixup(
                conv_nd(
                    dims, self.out_channels, self.out_channels, 3, padding=1
                ),
                use_fixup_init=use_fixup_init,
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
        Apply the block to a torch.Tensor, conditioned on a timestep embedding.

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

        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.


    Arguments
    ---------
    channels: int
        the number of channels
    num_heads: int
        the number of attention heads
    num_head_channels: int
        the number of channels in each attention head
    norm_num_groups: int
        the number of groups used for group normalization
    use_fixup_init: bool
        whether to use FixUp initialization

    Example
    -------
    >>> attn = AttentionBlock(
    ...     channels=8, num_heads=4, num_head_channels=4, norm_num_groups=2
    ... )
    >>> x = torch.randn(4, 8, 16, 16)
    >>> out = attn(x)
    >>> out.shape
    torch.Size([4, 8, 16, 16])
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        norm_num_groups=32,
        use_fixup_init=True,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, (
                f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            )
            self.num_heads = channels // num_head_channels
        self.norm = nn.GroupNorm(norm_num_groups, channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)

        self.proj_out = fixup(conv_nd(1, channels, channels, 1), use_fixup_init)

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

    Arguments
    ---------
    n_heads : int
        Number of attention heads.

    Example
    -------
    >>> attn = QKVAttention(4)
    >>> n = 4
    >>> c = 8
    >>> h = 64
    >>> w = 16
    >>> qkv = torch.randn(4, (3 * h * c), w)
    >>> out = attn(qkv)
    >>> out.shape
    torch.Size([4, 512, 16])
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

        Returns
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


def build_emb_proj(emb_config, proj_dim=None, use_emb=None):
    """Builds a dictionary of embedding modules for embedding
    projections

    Arguments
    ---------
    emb_config: dict
        a configuration dictionary
    proj_dim: int
        the target projection dimension
    use_emb: dict
        an optional dictionary of "switches" to turn
        embeddings on and off

    Returns
    -------
    result: torch.nn.ModuleDict
        a ModuleDict with a module for each embedding
    """
    emb_proj = {}
    if emb_config is not None:
        for key, item_config in emb_config.items():
            if use_emb is None or use_emb.get(key):
                if "emb_proj" in item_config:
                    emb_proj[key] = emb_proj
                else:
                    emb_proj[key] = EmbeddingProjection(
                        emb_dim=item_config["emb_dim"], proj_dim=proj_dim
                    )
    return nn.ModuleDict(emb_proj)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    Arguments
    ---------
    in_channels: int
        channels in the input torch.Tensor.
    model_channels: int
        base channel count for the model.
    out_channels: int
        channels in the output torch.Tensor.
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
        downsampling
    dims: int
        determines if the signal is 1D, 2D, or 3D.
    emb_dim: int
        time embedding dimension (defaults to model_channels * 4)
    cond_emb: dict
        embeddings on which the model will be conditioned

        Example:
        {
            "speaker": {
                "emb_dim": 256
            },
            "label": {
                "emb_dim": 12
            }
        }
    use_cond_emb: dict
        a dictionary with keys corresponding to keys in cond_emb
        and values corresponding to Booleans that turn embeddings
        on and off. This is useful in combination with hparams files
        to turn embeddings on and off with simple switches

        Example:
        {"speaker": False, "label": True}
    num_heads: int
        the number of attention heads in each attention layer.
    num_head_channels: int
        if specified, ignore num_heads and instead use
        a fixed channel width per attention head.
    num_heads_upsample: int
        works with num_heads to set a different number
        of heads for upsampling. Deprecated.
    norm_num_groups: int
        Number of groups in the norm, default 32
    resblock_updown: bool
        use residual blocks for up/downsampling.
    use_fixup_init: bool
        whether to use FixUp initialization

    Example
    -------
    >>> model = UNetModel(
    ...     in_channels=3,
    ...     model_channels=32,
    ...     out_channels=1,
    ...     num_res_blocks=1,
    ...     attention_resolutions=[1],
    ... )
    >>> x = torch.randn(4, 3, 16, 32)
    >>> ts = torch.tensor([10, 100, 50, 25])
    >>> out = model(x, ts)
    >>> out.shape
    torch.Size([4, 1, 16, 32])
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
        emb_dim=None,
        cond_emb=None,
        use_cond_emb=None,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        norm_num_groups=32,
        resblock_updown=False,
        use_fixup_init=True,
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
        self.cond_emb = cond_emb
        self.use_cond_emb = use_cond_emb

        if emb_dim is None:
            emb_dim = model_channels * 4
        self.time_embed = EmbeddingProjection(model_channels, emb_dim)

        self.cond_emb_proj = build_emb_proj(
            emb_config=cond_emb, proj_dim=emb_dim, use_emb=use_cond_emb
        )

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
                        emb_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        norm_num_groups=norm_num_groups,
                        use_fixup_init=use_fixup_init,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            norm_num_groups=norm_num_groups,
                            use_fixup_init=use_fixup_init,
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
                            emb_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            down=True,
                            norm_num_groups=norm_num_groups,
                            use_fixup_init=use_fixup_init,
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
                emb_dim,
                dropout,
                dims=dims,
                norm_num_groups=norm_num_groups,
                use_fixup_init=use_fixup_init,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                norm_num_groups=norm_num_groups,
                use_fixup_init=use_fixup_init,
            ),
            ResBlock(
                ch,
                emb_dim,
                dropout,
                dims=dims,
                norm_num_groups=norm_num_groups,
                use_fixup_init=use_fixup_init,
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
                        emb_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        norm_num_groups=norm_num_groups,
                        use_fixup_init=use_fixup_init,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            norm_num_groups=norm_num_groups,
                            use_fixup_init=use_fixup_init,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            emb_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            up=True,
                            norm_num_groups=norm_num_groups,
                            use_fixup_init=use_fixup_init,
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
            fixup(
                conv_nd(dims, input_ch, out_channels, 3, padding=1),
                use_fixup_init=use_fixup_init,
            ),
        )

    def forward(self, x, timesteps, cond_emb=None):
        """Apply the model to an input batch.

        Arguments
        ---------
        x: torch.Tensor
            an [N x C x ...] Tensor of inputs.
        timesteps: torch.Tensor
            a 1-D batch of timesteps.
        cond_emb: dict
            a string -> tensor dictionary of conditional
            embeddings (multiple embeddings are supported)

        Returns
        -------
        result: torch.Tensor
            an [N x C x ...] Tensor of outputs.
        """

        hs = []
        emb = self.time_embed(
            timestep_embedding(timesteps, self.model_channels)
        )

        if cond_emb is not None:
            for key, value in cond_emb.items():
                emb_proj = self.cond_emb_proj[key](value)
                emb += emb_proj

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

    def diffusion_forward(
        self,
        x,
        timesteps,
        cond_emb=None,
        length=None,  # unused for unet
        out_mask_value=None,  # unused for unet
        latent_mask_value=None,  # unused for unet
    ):
        """Forward function suitable for wrapping by diffusion.
        For this model, `length`/`out_mask_value`/`latent_mask_value` are unused
        and discarded.
        See :meth:`~UNetModel.forward` for details."""

        return self(x, timesteps, cond_emb=cond_emb)


class EncoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.
    For usage, see UNetModel.

    Arguments
    ---------
    in_channels: int
        channels in the input torch.Tensor.
    model_channels: int
        base channel count for the model.
    out_channels: int
        channels in the output torch.Tensor.
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
        downsampling
    dims: int
        determines if the signal is 1D, 2D, or 3D.
    num_heads: int
        the number of attention heads in each attention layer.
    num_head_channels: int
        if specified, ignore num_heads and instead use
        a fixed channel width per attention head.
    num_heads_upsample: int
        works with num_heads to set a different number
        of heads for upsampling. Deprecated.
    norm_num_groups: int
        Number of groups in the norm, default 32.
    resblock_updown: bool
        use residual blocks for up/downsampling.
    pool: str
        Type of pooling to use, one of:
        ["adaptive", "attention", "spatial", "spatial_v2"].
    attention_pool_dim: int
        The dimension on which to apply attention pooling.
    out_kernel_size: int
        the kernel size of the output convolution
    use_fixup_init: bool
        whether to use FixUp initialization


    Example
    -------
    >>> model = EncoderUNetModel(
    ...     in_channels=3,
    ...     model_channels=32,
    ...     out_channels=1,
    ...     num_res_blocks=1,
    ...     attention_resolutions=[1],
    ... )
    >>> x = torch.randn(4, 3, 16, 32)
    >>> ts = torch.tensor([10, 100, 50, 25])
    >>> out = model(x, ts)
    >>> out.shape
    torch.Size([4, 1, 2, 4])

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
        resblock_updown=False,
        pool=None,
        attention_pool_dim=None,
        out_kernel_size=3,
        use_fixup_init=True,
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

        emb_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
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
                        emb_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        norm_num_groups=norm_num_groups,
                        use_fixup_init=use_fixup_init,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            norm_num_groups=norm_num_groups,
                            use_fixup_init=use_fixup_init,
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
                            emb_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            down=True,
                            norm_num_groups=norm_num_groups,
                            use_fixup_init=use_fixup_init,
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
                emb_dim,
                dropout,
                dims=dims,
                use_fixup_init=use_fixup_init,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                norm_num_groups=norm_num_groups,
                use_fixup_init=use_fixup_init,
            ),
            ResBlock(
                ch,
                emb_dim,
                dropout,
                dims=dims,
                use_fixup_init=use_fixup_init,
            ),
        )
        self._feature_size += ch
        self.pool = pool
        self.spatial_pooling = False
        if pool is None:
            self.out = nn.Sequential(
                nn.GroupNorm(
                    num_channels=ch, num_groups=norm_num_groups, eps=1e-6
                ),
                nn.SiLU(),
                conv_nd(
                    dims,
                    ch,
                    out_channels,
                    kernel_size=out_kernel_size,
                    padding="same",
                ),
            )
        elif pool == "adaptive":
            self.out = nn.Sequential(
                nn.GroupNorm(norm_num_groups, ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                fixup(
                    conv_nd(dims, ch, out_channels, 1),
                    use_fixup_init=use_fixup_init,
                ),
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
        -------
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
            h = torch.cat(results, dim=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)


class EmbeddingProjection(nn.Module):
    """A simple module that computes the projection of an
    embedding vector onto the specified number of dimensions

    Arguments
    ---------
    emb_dim: int
        the original embedding dimensionality

    proj_dim: int
        the dimensionality of the target projection
        space

    Example
    -------
    >>> mod_emb_proj = EmbeddingProjection(emb_dim=16, proj_dim=64)
    >>> emb = torch.randn(4, 16)
    >>> emb_proj = mod_emb_proj(emb)
    >>> emb_proj.shape
    torch.Size([4, 64])
    """

    def __init__(self, emb_dim, proj_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        self.input = nn.Linear(emb_dim, proj_dim)
        self.act = nn.SiLU()
        self.output = nn.Linear(proj_dim, proj_dim)

    def forward(self, emb):
        """Computes the forward pass

        Arguments
        ---------
        emb: torch.Tensor
            the original embedding tensor

        Returns
        -------
        result: torch.Tensor
            the target embedding space
        """
        x = self.input(emb)
        x = self.act(x)
        x = self.output(x)
        return x


class DecoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.
    For usage, see UNet.

    Arguments
    ---------
    in_channels: int
        channels in the input torch.Tensor.
    model_channels: int
        base channel count for the model.
    out_channels: int
        channels in the output torch.Tensor.
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
        downsampling
    dims: int
        determines if the signal is 1D, 2D, or 3D.
    num_heads: int
        the number of attention heads in each attention layer.
    num_head_channels: int
        if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    num_heads_upsample: int
        works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    resblock_updown: bool
        use residual blocks for up/downsampling.
    norm_num_groups: int
        Number of groups to use in norm, default 32
    out_kernel_size: int
        Output kernel size, default 3
    use_fixup_init: bool
        whether to use FixUp initialization

    Example
    -------
    >>> model = DecoderUNetModel(
    ...     in_channels=1,
    ...     model_channels=32,
    ...     out_channels=3,
    ...     num_res_blocks=1,
    ...     attention_resolutions=[1],
    ... )
    >>> x = torch.randn(4, 1, 2, 4)
    >>> ts = torch.tensor([10, 100, 50, 25])
    >>> out = model(x, ts)
    >>> out.shape
    torch.Size([4, 3, 16, 32])
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
        resblock_updown=False,
        norm_num_groups=32,
        out_kernel_size=3,
        use_fixup_init=True,
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

        emb_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        ch = int(channel_mult[0] * model_channels)

        self.input_block = TimestepEmbedSequential(
            conv_nd(dims, in_channels, ch, 3, padding=1)
        )

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                emb_dim,
                dropout,
                dims=dims,
                norm_num_groups=norm_num_groups,
                use_fixup_init=use_fixup_init,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                norm_num_groups=norm_num_groups,
                use_fixup_init=use_fixup_init,
            ),
            ResBlock(
                ch,
                emb_dim,
                dropout,
                dims=dims,
                norm_num_groups=norm_num_groups,
                use_fixup_init=use_fixup_init,
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
                        emb_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        norm_num_groups=norm_num_groups,
                        use_fixup_init=use_fixup_init,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            norm_num_groups=norm_num_groups,
                            use_fixup_init=use_fixup_init,
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
                            emb_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            up=True,
                            norm_num_groups=norm_num_groups,
                            use_fixup_init=use_fixup_init,
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
                padding="same",
            ),
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
        -------
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
        the index of the dimension in which the length will vary
    dims: list
        the list of dimensions to be included in padding

    Example
    -------
    >>> padding = DownsamplingPadding(factor=4, dims=[1, 2], len_dim=1)
    >>> x = torch.randn(4, 7, 14)
    >>> length = torch.tensor([1.0, 0.8, 1.0, 0.7])
    >>> x, length_new = padding(x, length)
    >>> x.shape
    torch.Size([4, 8, 16])
    >>> length_new
    tensor([0.8750, 0.7000, 0.8750, 0.6125])
    """

    def __init__(self, factor, len_dim=2, dims=None):
        super().__init__()
        self.factor = factor
        self.len_dim = len_dim
        if dims is None:
            dims = DEFAULT_PADDING_DIMS
        self.dims = dims

    def forward(self, x, length=None):
        """Applies the padding

        Arguments
        ---------
        x: torch.Tensor
            the sample
        length: torch.Tensor
            the length tensor

        Returns
        -------
        x_pad: torch.Tensor
            the padded tensor
        lens: torch.Tensor
            the new, adjusted lengths, if applicable
        """
        updated_length = length
        for dim in self.dims:
            # TODO: Consider expanding pad_divisible to support multiple dimensions
            x, length_pad = pad_divisible(x, length, self.factor, len_dim=dim)
            if dim == self.len_dim:
                updated_length = length_pad
        return x, updated_length


class UNetNormalizingAutoencoder(NormalizingAutoencoder):
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
    norm_num_groups: int
        Number of norm groups, default 32
    resblock_updown: bool
        whether to use residual blocks for upsampling and downsampling
    out_kernel_size: int
        the kernel size for output convolution layers (if applicable)
    len_dim: int
        Size of the output.
    out_mask_value: float
        Value to fill when masking the output.
    latent_mask_value: float
        Value to fill when masking the latent variable.
    use_fixup_norm: bool
        whether to use FixUp normalization
    downsampling_padding: int
        Amount of padding to apply in downsampling, default 2 ** len(channel_mult)

    Example
    -------
    >>> unet_ae = UNetNormalizingAutoencoder(
    ...     in_channels=1,
    ...     model_channels=4,
    ...     encoder_out_channels=16,
    ...     latent_channels=3,
    ...     encoder_num_res_blocks=1,
    ...     encoder_attention_resolutions=[],
    ...     decoder_num_res_blocks=1,
    ...     decoder_attention_resolutions=[],
    ...     norm_num_groups=2,
    ... )
    >>> x = torch.randn(4, 1, 32, 32)
    >>> x_enc = unet_ae.encode(x)
    >>> x_enc.shape
    torch.Size([4, 3, 4, 4])
    >>> x_dec = unet_ae.decode(x_enc)
    >>> x_dec.shape
    torch.Size([4, 1, 32, 32])
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
        resblock_updown=False,
        out_kernel_size=3,
        len_dim=2,
        out_mask_value=0.0,
        latent_mask_value=0.0,
        use_fixup_norm=False,
        downsampling_padding=None,
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
            resblock_updown=resblock_updown,
            out_kernel_size=out_kernel_size,
            use_fixup_init=use_fixup_norm,
        )

        encoder = nn.Sequential(
            encoder_unet,
            conv_nd(
                dims=dims,
                in_channels=encoder_out_channels,
                out_channels=latent_channels,
                kernel_size=1,
            ),
        )
        if downsampling_padding is None:
            downsampling_padding = 2 ** len(channel_mult)

        encoder_pad = DownsamplingPadding(downsampling_padding)

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
            resblock_updown=resblock_updown,
            out_kernel_size=out_kernel_size,
            use_fixup_init=use_fixup_norm,
        )
        super().__init__(
            encoder=encoder,
            latent_padding=encoder_pad,
            decoder=decoder,
            len_dim=len_dim,
            out_mask_value=out_mask_value,
            latent_mask_value=latent_mask_value,
        )
