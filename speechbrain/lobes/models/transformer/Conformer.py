"""Conformer implementation.

Authors
* Jianyuan Zhong 2020
* Samuele Cornell 2021
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import speechbrain as sb
import math
import warnings


from speechbrain.nnet.attention import (
    RelPosMHAXL,
    MultiheadAttention,
    PositionalwiseFeedForward,
)
from speechbrain.lobes.models.transformer.hypermixing import HyperMixing
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.nnet.activations import Swish


@dataclass
class ConformerEncoderLayerStreamingContext:
    """Streaming metadata and state for a `ConformerEncoderLayer`.

    The multi-head attention and Dynamic Chunk Convolution require to save some
    left context that gets inserted as left padding."""

    mha_left_context_size: int
    """For this layer, specifies how many frames of inputs should be saved.
    Usually, the same value is used across all layers, but this can be modified.
    """

    mha_left_context: Optional[torch.Tensor] = None
    """Left context to insert at the left of the current chunk as inputs to the
    multi-head attention. It can be `None` (if we're dealing with the first
    chunk) or `<= mha_left_context_size` because for the first few chunks, not
    enough left context may be available to pad.
    """

    dcconv_left_context: Optional[torch.Tensor] = None
    """Left context to insert at the left of the convolution according to the
    Dynamic Chunk Convolution method.
    
    Unlike `mha_left_context`, here the amount of frames to keep is fixed and
    inferred from the kernel size of the convolution module.
    """


@dataclass
class ConformerEncoderStreamingContext:
    """Streaming metadata and state for a `ConformerEncoder`."""

    layers: List[ConformerEncoderLayerStreamingContext]
    """Streaming metadata and state for each layer of the encoder."""


class ConvolutionModule(nn.Module):
    """This is an implementation of convolution module in Conformer.

    Arguments
    ----------
    input_size : int
        The expected size of the input embedding dimension.
    kernel_size: int, optional
        Kernel size of non-bottleneck convolutional layer.
    bias: bool, optional
        Whether to use bias in the non-bottleneck conv layer.
    activation: torch.nn.Module
         Activation function used after non-bottleneck conv layer.
    dropout: float, optional
         Dropout rate.
    causal: bool, optional
         Whether the convolution should be causal or not.
    dilation: int, optional
         Dilation factor for the non bottleneck conv layer.

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = ConvolutionModule(512, 3)
    >>> output = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        input_size,
        kernel_size=31,
        bias=True,
        activation=Swish,
        dropout=0.0,
        causal=False,
        dilation=1,
    ):
        super().__init__()

        self.causal = causal

        if self.causal:
            self.padding = (kernel_size - 1) * 2 ** (dilation - 1)
        else:
            self.padding = (kernel_size - 1) * 2 ** (dilation - 1) // 2

        self.layer_norm = nn.LayerNorm(input_size)
        self.bottleneck = nn.Sequential(
            # pointwise
            nn.Conv1d(
                input_size, 2 * input_size, kernel_size=1, stride=1, bias=bias
            ),
            nn.GLU(dim=1),
        )
        # depthwise
        self.conv = nn.Conv1d(
            input_size,
            input_size,
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding,
            dilation=dilation,
            groups=input_size,
            bias=bias,
        )

        # NOTE: there appears to be a mismatch compared to the Conformer paper:
        # I believe the first LayerNorm below is supposed to be a BatchNorm.

        self.after_conv = nn.Sequential(
            nn.LayerNorm(input_size),
            activation(),
            # pointwise
            nn.Linear(input_size, input_size, bias=bias),
            nn.Dropout(dropout),
        )

    def _do_conv(self, x, inhibit_padding: bool):
        out = self.layer_norm(x)
        out = out.transpose(1, 2)
        out = self.bottleneck(out)

        if not inhibit_padding:
            out = self.conv(out)
        else:
            # let's keep backwards compat by pointing at the weights from the
            # already declared Conv1d.

            # we do not need to edit the bottleneck as it is pointwise (i.e.
            # time step by time step), thus, it doesn't need padding along the
            # time dimension
            out = F.conv1d(
                out,
                weight=self.conv.weight,
                bias=self.conv.bias,
                stride=self.conv.stride,
                padding=0,
                dilation=self.conv.dilation,
                groups=out.shape[-2],
            )

        if self.causal:
            # chomp
            out = out[..., : -self.padding]

        out = out.transpose(1, 2)
        out = self.after_conv(out)
        return out

    def forward(self, x, mask=None, chunk_size=-1):
        """ Processes the input tensor x and returns the output an output tensor"""

        # ref: Dynamic chunk convolution for unified streaming and non-streaming
        # conformer ASR
        # https://www.amazon.science/publications/dynamic-chunk-convolution-for-unified-streaming-and-non-streaming-conformer-asr
        # split the input into chunks of size `chunk_size`, but for each chunk
        # provide a left context for left chunk dependencies to be possible.

        if chunk_size >= 1:
            # chances are chunking+causal is unintended; i don't know where it
            # may make sense, but if it does to you, feel free to implement it.
            assert (
                not self.causal
            ), "Chunked convolution not supported with causal padding"

            batch_size = x.shape[0]
            chunk_left_context = self.padding

            chunk_count = int(math.ceil(x.shape[1] / chunk_size))

            if x.shape[1] % chunk_size != 0:
                final_right_padding = chunk_size - (x.shape[1] % chunk_size)
            else:
                final_right_padding = 0

            # compute the left context that can and should be added, for each
            # chunk. for the first few chunks, we will need to add extra padding
            applied_left_context = [
                min(chunk_left_context, i * chunk_size,)
                for i in range(chunk_count)
            ]

            # build views of chunks with left context (but no 0-padding yet)
            # the left context effectively becomes "left padding", we do not
            # want to keep any convolution results centered on the left context
            out = [
                x[
                    :,
                    i * chunk_size
                    - applied_left_context[i] : (i + 1) * chunk_size,
                    ...,
                ]
                for i in range(chunk_count)
            ]

            # TODO: experiment around reflect padding, which is difficult
            # because small chunks have too little time steps to reflect from
            out = [
                F.pad(
                    out[i],
                    (
                        # channel dims, we do not to pad these
                        0,
                        0,
                        # add missing left 0-padding if we lacked left context
                        chunk_left_context - applied_left_context[i],
                        # add missing right 0-padding as we disable default padding
                        # also add missing frames of the rightmost chunk
                        self.padding
                        + (final_right_padding if i == len(out) - 1 else 0),
                    ),
                )
                for i in range(len(out))
            ]

            # we pack together chunks in a single tensor so that we can feed it
            # to the convolution directly. this is much more performant than
            # doing the same with lists.

            # -> [batch_size, num_chunks, chunk_size + lc + rpad, in_channels]
            out = torch.stack(out, dim=1)

            # -> [batch_size * num_chunks, chunk_size + lc + rpad, in_channels]
            out = torch.flatten(out, end_dim=1)

            # -> [batch_size * num_chunks, chunk_size, out_channels]
            out = self._do_conv(out, inhibit_padding=True)

            # -> [batch_size, num_chunks, chunk_size, out_channels]
            out = torch.unflatten(out, dim=0, sizes=(batch_size, -1))

            # -> [batch_size, time_steps + extra right padding, out_channels]
            out = torch.flatten(out, start_dim=1, end_dim=2)

            # -> [batch_size, time_steps, out_channels]
            if final_right_padding > 0:
                out = out[:, :-final_right_padding, :]
        else:
            out = self._do_conv(x, inhibit_padding=False)

        if mask is not None:
            out.masked_fill_(mask, 0.0)

        return out


class ConformerEncoderLayer(nn.Module):
    """This is an implementation of Conformer encoder layer.

    Arguments
    ----------
    d_model : int
        The expected size of the input embedding.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    nhead : int
        Number of attention heads.
    kernel_size : int, optional
        Kernel size of convolution model.
    kdim : int, optional
        Dimension of the key.
    vdim : int, optional
        Dimension of the value.
    activation: torch.nn.Module
         Activation function used in each Conformer layer.
    bias : bool, optional
        Whether  convolution module.
    dropout : int, optional
        Dropout for the encoder.
    causal: bool, optional
        Whether the convolutions should be causal or not.
    attention_type: str, optional
        type of attention layer, e.g. regulaMHA for regular MultiHeadAttention.

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> pos_embs = torch.rand((1, 2*60-1, 512))
    >>> net = ConformerEncoderLayer(d_ffn=512, nhead=8, d_model=512, kernel_size=3)
    >>> output = net(x, pos_embs=pos_embs)
    >>> output[0].shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_model,
        d_ffn,
        nhead,
        kernel_size=31,
        kdim=None,
        vdim=None,
        activation=Swish,
        bias=True,
        dropout=0.0,
        causal=False,
        attention_type="RelPosMHAXL",
    ):
        super().__init__()

        if attention_type == "regularMHA":
            self.mha_layer = MultiheadAttention(
                nhead=nhead,
                d_model=d_model,
                dropout=dropout,
                kdim=kdim,
                vdim=vdim,
            )
        elif attention_type == "RelPosMHAXL":
            # transformerXL style positional encoding
            self.mha_layer = RelPosMHAXL(
                num_heads=nhead,
                embed_dim=d_model,
                dropout=dropout,
                mask_pos_future=causal,
            )
        elif attention_type == "hypermixing":
            self.mha_layer = HyperMixing(
                input_output_dim=d_model,
                hypernet_size=d_ffn,
                tied=False,
                num_heads=nhead,
                fix_tm_hidden_size=False,
            )

        self.convolution_module = ConvolutionModule(
            d_model, kernel_size, bias, activation, dropout, causal=causal
        )

        self.ffn_module1 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(
                d_ffn=d_ffn,
                input_size=d_model,
                dropout=dropout,
                activation=activation,
            ),
            nn.Dropout(dropout),
        )

        self.ffn_module2 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(
                d_ffn=d_ffn,
                input_size=d_model,
                dropout=dropout,
                activation=activation,
            ),
            nn.Dropout(dropout),
        )

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: torch.Tensor = None,
        chunk_size: Optional[int] = None,
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder layer.
        src_mask : torch.Tensor, optional
            The mask for the src sequence.
        src_key_padding_mask : torch.Tensor, optional
            The mask for the src keys per batch.
        pos_embs: torch.Tensor, torch.nn.Module, optional
            Module or tensor containing the input sequence positional embeddings
        chunk_size: int, optional
            Whether to preform convolution chunking to hide future context,
            useful for chunked conformers in a dynamic chunk training setting
        """
        # TODO: cite paper for chunk size
        # TODO: document left frames

        conv_mask: Optional[torch.Tensor] = None
        if src_key_padding_mask is not None:
            conv_mask = src_key_padding_mask.unsqueeze(-1)
        # ffn module
        x = x + 0.5 * self.ffn_module1(x)
        # muti-head attention module
        skip = x
        x = self.norm1(x)

        x, self_attn = self.mha_layer(
            x,
            x,
            x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs,
        )
        x = x + skip
        # convolution module
        x = x + self.convolution_module(x, conv_mask, chunk_size=chunk_size)
        # ffn module
        x = self.norm2(x + 0.5 * self.ffn_module2(x))
        return x, self_attn

    def forward_streaming(
        self,
        x,
        context: ConformerEncoderLayerStreamingContext,
        pos_embs: torch.Tensor = None,
    ):
        orig_len = x.shape[-2]
        # ffn module
        x = x + 0.5 * self.ffn_module1(x)

        # TODO: make the approach for MHA left context more efficient.
        # currently, this saves the inputs to the MHA.
        # the naive approach is suboptimal in a few ways, namely that the
        # outputs for this left padding is being re-computed even though we
        # discard them immediately after.

        # left pad `x` with our MHA left context
        if context.mha_left_context is not None:
            x = torch.cat((context.mha_left_context, x), dim=1)

        # compute new MHA left context for the next call to our function
        if context.mha_left_context_size > 0:
            context.mha_left_context = x[
                ..., -context.mha_left_context_size :, :
            ]

        # multi-head attention module
        skip = x
        x = self.norm1(x)

        x, self_attn = self.mha_layer(
            x, x, x, attn_mask=None, key_padding_mask=None, pos_embs=pos_embs,
        )
        x = x + skip

        # truncate outputs corresponding to the MHA left context (we only care
        # about our chunk's outputs); see above to-do
        x = x[..., -orig_len:, :]

        # TODO: this is slightly suboptimal as this will add left padding inside
        # the convolution code that we do not need. it would be better to
        # manually add the right-padding ourselves and disable padding inside
        # the convolution module for this usecase, but it would need some
        # refactoring.

        if context.dcconv_left_context is not None:
            x = torch.cat((context.dcconv_left_context, x), dim=1)

        # compute new DCConv left context for the next call to our function
        context.dcconv_left_context = x[
            ..., -self.convolution_module.padding :, :
        ]

        # convolution module
        x = x + self.convolution_module(x)

        # truncate outputs corresponding to the DCConv left context
        x = x[..., -orig_len:, :]

        # ffn module
        x = self.norm2(x + 0.5 * self.ffn_module2(x))
        return x, self_attn

    def make_streaming_context(self, mha_left_context_size: int):
        """Creates a blank streaming context for this encoding layer.

        Arguments
        ---------
        mha_left_context_size : int
            How many left frames should be saved and used as left context to the
            current chunk when streaming
        """
        return ConformerEncoderLayerStreamingContext(
            mha_left_context_size=mha_left_context_size
        )


class ConformerEncoder(nn.Module):
    """This class implements the Conformer encoder.

    Arguments
    ---------
    num_layers : int
        Number of layers.
    d_model : int
        Embedding dimension size.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    nhead : int
        Number of attention heads.
    kernel_size : int, optional
        Kernel size of convolution model.
    kdim : int, optional
        Dimension of the key.
    vdim : int, optional
        Dimension of the value.
    activation: torch.nn.Module
         Activation function used in each Confomer layer.
    bias : bool, optional
        Whether  convolution module.
    dropout : int, optional
        Dropout for the encoder.
    causal: bool, optional
        Whether the convolutions should be causal or not.
    attention_type: str, optional
        type of attention layer, e.g. regulaMHA for regular MultiHeadAttention.


    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> pos_emb = torch.rand((1, 2*60-1, 512))
    >>> net = ConformerEncoder(1, 512, 512, 8)
    >>> output, _ = net(x, pos_embs=pos_emb)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        num_layers,
        d_model,
        d_ffn,
        nhead,
        kernel_size=31,
        kdim=None,
        vdim=None,
        activation=Swish,
        bias=True,
        dropout=0.0,
        causal=False,
        attention_type="RelPosMHAXL",
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    kernel_size=kernel_size,
                    bias=bias,
                    causal=causal,
                    attention_type=attention_type,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = LayerNorm(d_model, eps=1e-6)
        self.attention_type = attention_type

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder layer.
        src_mask : torch.Tensor, optional
            The mask for the src sequence.
        src_key_padding_mask : torch.Tensor, optional
            The mask for the src keys per batch.
        pos_embs: torch.Tensor, torch.nn.Module,
            Module or tensor containing the input sequence positional embeddings
            If custom pos_embs are given it needs to have the shape (1, 2*S-1, E)
            where S is the sequence length, and E is the embedding dimension.
        chunk_size: int, optional
            Whether to preform convolution chunking to hide future context,
            useful for chunked conformers in a dynamic chunk training setting
        """
        if self.attention_type == "RelPosMHAXL":
            if pos_embs is None:
                raise ValueError(
                    "The chosen attention type for the Conformer is RelPosMHAXL. For this attention type, the positional embeddings are mandatory"
                )

        output = src
        attention_lst = []
        for enc_layer in self.layers:
            output, attention = enc_layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                pos_embs=pos_embs,
                chunk_size=chunk_size,
            )
            attention_lst.append(attention)
        output = self.norm(output)

        return output, attention_lst

    def forward_streaming(
        self,
        src,
        context: ConformerEncoderStreamingContext,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        if self.attention_type == "RelPosMHAXL":
            if pos_embs is None:
                raise ValueError(
                    "The chosen attention type for the Conformer is RelPosMHAXL. For this attention type, the positional embeddings are mandatory"
                )

        output = src
        attention_lst = []
        for i, enc_layer in enumerate(self.layers):
            output, attention = enc_layer.forward_streaming(
                output, pos_embs=pos_embs, context=context.layers[i]
            )
            attention_lst.append(attention)
        output = self.norm(output)

        return output, attention_lst

    def make_streaming_context(self, mha_left_context_size: int):
        """Creates a blank streaming context for the encoder.

        Arguments
        ---------
        mha_left_context_size : int
            How many left frames should be saved and used as left context to the
            current chunk when streaming. This value is replicated across all
            layers.
        """
        return ConformerEncoderStreamingContext(
            layers=[
                layer.make_streaming_context(
                    mha_left_context_size=mha_left_context_size
                )
                for layer in self.layers
            ]
        )


class ConformerDecoderLayer(nn.Module):
    """This is an implementation of Conformer encoder layer.

    Arguments
    ----------
    d_model : int
        The expected size of the input embedding.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    nhead : int
        Number of attention heads.
    kernel_size : int, optional
        Kernel size of convolution model.
    kdim : int, optional
        Dimension of the key.
    vdim : int, optional
        Dimension of the value.
    activation: torch.nn.Module, optional
         Activation function used in each Conformer layer.
    bias : bool, optional
        Whether  convolution module.
    dropout : int, optional
        Dropout for the encoder.
    causal: bool, optional
        Whether the convolutions should be causal or not.
    attention_type: str, optional
        type of attention layer, e.g. regulaMHA for regular MultiHeadAttention.

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> pos_embs = torch.rand((1, 2*60-1, 512))
    >>> net = ConformerEncoderLayer(d_ffn=512, nhead=8, d_model=512, kernel_size=3)
    >>> output = net(x, pos_embs=pos_embs)
    >>> output[0].shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_model,
        d_ffn,
        nhead,
        kernel_size,
        kdim=None,
        vdim=None,
        activation=Swish,
        bias=True,
        dropout=0.0,
        causal=True,
        attention_type="RelPosMHAXL",
    ):
        super().__init__()

        if not causal:
            warnings.warn(
                "Decoder is not causal, in most applications it should be causal, you have been warned !"
            )

        if attention_type == "regularMHA":
            self.mha_layer = MultiheadAttention(
                nhead=nhead,
                d_model=d_model,
                dropout=dropout,
                kdim=kdim,
                vdim=vdim,
            )
        elif attention_type == "RelPosMHAXL":
            # transformerXL style positional encoding
            self.mha_layer = RelPosMHAXL(
                num_heads=nhead,
                embed_dim=d_model,
                dropout=dropout,
                mask_pos_future=causal,
            )

        self.convolution_module = ConvolutionModule(
            d_model, kernel_size, bias, activation, dropout, causal=causal
        )

        self.ffn_module1 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(
                d_ffn=d_ffn,
                input_size=d_model,
                dropout=dropout,
                activation=activation,
            ),
            nn.Dropout(dropout),
        )

        self.ffn_module2 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(
                d_ffn=d_ffn,
                input_size=d_model,
                dropout=dropout,
                activation=activation,
            ),
            nn.Dropout(dropout),
        )

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos_embs_tgt=None,
        pos_embs_src=None,
    ):
        """
        Arguments
        ----------
            tgt: torch.Tensor
                The sequence to the decoder layer.
            memory: torch.Tensor
                The sequence from the last layer of the encoder.
            tgt_mask: torch.Tensor, optional, optional
                The mask for the tgt sequence.
            memory_mask: torch.Tensor, optional
                The mask for the memory sequence.
            tgt_key_padding_mask : torch.Tensor, optional
                The mask for the tgt keys per batch.
            memory_key_padding_mask : torch.Tensor, optional
                The mask for the memory keys per batch.
            pos_emb_tgt: torch.Tensor, torch.nn.Module, optional
                Module or tensor containing the target sequence positional embeddings for each attention layer.
            pos_embs_src: torch.Tensor, torch.nn.Module, optional
                Module or tensor containing the source sequence positional embeddings for each attention layer.
        """
        # ffn module
        tgt = tgt + 0.5 * self.ffn_module1(tgt)
        # muti-head attention module
        skip = tgt
        x = self.norm1(tgt)
        x, self_attn = self.mha_layer(
            x,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            pos_embs=pos_embs_src,
        )
        x = x + skip
        # convolution module
        x = x + self.convolution_module(x)
        # ffn module
        x = self.norm2(x + 0.5 * self.ffn_module2(x))
        return x, self_attn, self_attn


class ConformerDecoder(nn.Module):
    """This class implements the Transformer decoder.

    Arguments
    ----------
    num_layers: int
        Number of layers.
    nhead: int
        Number of attention heads.
    d_ffn: int
        Hidden size of self-attention Feed Forward layer.
    d_model: int
        Embedding dimension size.
    kdim: int, optional
        Dimension for key.
    vdim: int, optional
        Dimension for value.
    dropout: float, optional
        Dropout rate.
    activation: torch.nn.Module, optional
         Activation function used after non-bottleneck conv layer.
    kernel_size : int, optional
        Kernel size of convolutional layer.
    bias : bool, optional
        Whether  convolution module.
    causal: bool, optional
        Whether the convolutions should be causal or not.
    attention_type: str, optional
        type of attention layer, e.g. regulaMHA for regular MultiHeadAttention.


    Example
    -------
    >>> src = torch.rand((8, 60, 512))
    >>> tgt = torch.rand((8, 60, 512))
    >>> net = ConformerDecoder(1, 8, 1024, 512, attention_type="regularMHA")
    >>> output, _, _ = net(tgt, src)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=Swish,
        kernel_size=3,
        bias=True,
        causal=True,
        attention_type="RelPosMHAXL",
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                ConformerDecoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    kernel_size=kernel_size,
                    bias=bias,
                    causal=causal,
                    attention_type=attention_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos_embs_tgt=None,
        pos_embs_src=None,
    ):
        """
        Arguments
        ----------
        tgt: torch.Tensor
            The sequence to the decoder layer.
        memory: torch.Tensor
            The sequence from the last layer of the encoder.
        tgt_mask: torch.Tensor, optional, optional
            The mask for the tgt sequence.
        memory_mask: torch.Tensor, optional
            The mask for the memory sequence.
        tgt_key_padding_mask : torch.Tensor, optional
            The mask for the tgt keys per batch.
        memory_key_padding_mask : torch.Tensor, optional
            The mask for the memory keys per batch.
        pos_emb_tgt: torch.Tensor, torch.nn.Module, optional
            Module or tensor containing the target sequence positional embeddings for each attention layer.
        pos_embs_src: torch.Tensor, torch.nn.Module, optional
            Module or tensor containing the source sequence positional embeddings for each attention layer.

        """
        output = tgt
        self_attns, multihead_attns = [], []
        for dec_layer in self.layers:
            output, self_attn, multihead_attn = dec_layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos_embs_tgt=pos_embs_tgt,
                pos_embs_src=pos_embs_src,
            )
            self_attns.append(self_attn)
            multihead_attns.append(multihead_attn)
        output = self.norm(output)

        return output, self_attns, multihead_attns
