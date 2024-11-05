"""Conformer implementation.

Authors
-------
* Jianyuan Zhong 2020
* Samuele Cornell 2021
* Sylvain de Langen 2023
"""

import warnings
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import speechbrain as sb
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.attention import (
    MultiheadAttention,
    PositionalwiseFeedForward,
    RelPosMHAXL,
)
from speechbrain.nnet.hypermixing import HyperMixing
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
from speechbrain.kernels.conv1d import conv1d


@dataclass
class ConformerEncoderLayerStreamingContext:
    """Streaming metadata and state for a `ConformerEncoderLayer`.

    The multi-head attention and Dynamic Chunk Convolution require to save some
    left context that gets inserted as left padding.

    See :class:`.ConvolutionModule` documentation for further details.
    """

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

    dynchunktrain_config: DynChunkTrainConfig
    """Dynamic Chunk Training configuration holding chunk size and context size
    information."""

    layers: List[ConformerEncoderLayerStreamingContext]
    """Streaming metadata and state for each layer of the encoder."""


class ConvolutionModule(nn.Module):
    """This is an implementation of convolution module in Conformer.

    Arguments
    ---------
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

        self.kernel_size = kernel_size
        self.causal = causal
        self.dilation = dilation

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

        # BatchNorm in the original Conformer replaced with a LayerNorm due to
        # https://github.com/speechbrain/speechbrain/pull/1329
        # see discussion
        # https://github.com/speechbrain/speechbrain/pull/933#issuecomment-1033367884

        self.after_conv = nn.Sequential(
            nn.LayerNorm(input_size),
            activation(),
            # pointwise
            nn.Linear(input_size, input_size, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dynchunktrain_config: Optional[DynChunkTrainConfig] = None,
    ):
        """Applies the convolution to an input tensor `x`.

        Arguments
        ---------
        x: torch.Tensor
            Input tensor to the convolution module.
        mask: torch.Tensor, optional
            Mask to be applied over the output of the convolution using
            `masked_fill_`, if specified.
        dynchunktrain_config: DynChunkTrainConfig, optional
            If specified, makes the module support Dynamic Chunk Convolution
            (DCConv) as implemented by
            `Dynamic Chunk Convolution for Unified Streaming and Non-Streaming Conformer ASR <https://www.amazon.science/publications/dynamic-chunk-convolution-for-unified-streaming-and-non-streaming-conformer-asr>`_.
            This allows masking future frames while preserving better accuracy
            than a fully causal convolution, at a small speed cost.
            This should only be used for training (or, if you know what you're
            doing, for masked evaluation at inference time), as the forward
            streaming function should be used at inference time.

        Returns
        -------
        out: torch.Tensor
            The output tensor.
        """

        chunk_size = 0

        if dynchunktrain_config is not None:
            # chances are chunking+causal is unintended; i don't know where it
            # may make sense, but if it does to you, feel free to implement it.
            assert (
                not self.causal
            ), "Chunked convolution not supported with causal padding"

            assert (
                self.dilation == 1
            ), "Current DynChunkTrain logic does not support dilation != 1"

            # in a causal convolution, which is not the case here, an output
            # frame would never be able to depend on a input frame from any
            # point in the future.

            # but with the dynamic chunk convolution, we instead use a "normal"
            # convolution but where, for any output frame, the future beyond the
            # "current" chunk gets masked.
            # see the paper linked in the documentation for details.

            chunk_size = dynchunktrain_config.chunk_size

        out = self.layer_norm(x)
        out = out.transpose(1, 2)
        # TODO: make bottleneck in channels_first=True by implementing it as a
        # matmul to avoid a transposition on fast Triton kernels
        out = self.bottleneck(out)
        out = conv1d(
            out,
            weight=self.conv.weight,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding="same",
            dilation=self.conv.dilation,
            groups=self.conv.groups,
            mask_chunk_size=chunk_size,
            channels_first=False,
        )

        if self.causal:
            # chomp
            out = out[..., : -self.padding]

        out = out.transpose(1, 2)
        out = self.after_conv(out)

        if mask is not None:
            out.masked_fill_(mask, 0.0)

        return out


class ConformerEncoderLayer(nn.Module):
    """This is an implementation of Conformer encoder layer.

    Arguments
    ---------
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
    causal : bool, optional
        Whether the convolutions should be causal or not.
    attention_type : str, optional
        type of attention layer, e.g. regularMHA for regular MultiHeadAttention.

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
        dynchunktrain_config: Optional[DynChunkTrainConfig] = None,
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
        dynchunktrain_config: Optional[DynChunkTrainConfig]
            Dynamic Chunk Training configuration object for streaming,
            specifically involved here to apply Dynamic Chunk Convolution to
            the convolution module.
        """
        conv_mask: Optional[torch.Tensor] = None
        if src_key_padding_mask is not None:
            conv_mask = src_key_padding_mask.unsqueeze(-1)
        # ffn module
        x = x + 0.5 * self.ffn_module1(x)
        # multi-head attention module
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
        x = x + self.convolution_module(
            x, conv_mask, dynchunktrain_config=dynchunktrain_config
        )
        # ffn module
        x = self.norm2(x + 0.5 * self.ffn_module2(x))
        return x, self_attn

    def forward_streaming(
        self,
        x,
        context: ConformerEncoderLayerStreamingContext,
        pos_embs: torch.Tensor = None,
    ):
        """Conformer layer streaming forward (typically for
        DynamicChunkTraining-trained models), which is to be used at inference
        time. Relies on a mutable context object as initialized by
        `make_streaming_context` that should be used across chunks.
        Invoked by `ConformerEncoder.forward_streaming`.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor for this layer. Batching is supported as long as you
            keep the context consistent.
        context : ConformerEncoderStreamingContext
            Mutable streaming context; the same object should be passed across
            calls.
        pos_embs : torch.Tensor, optional
            Positional embeddings, if used.

        Returns
        -------
        x : torch.Tensor
            Output tensor.
        self_attn : list
            List of self attention values.
        """

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
            x,
            x,
            x,
            attn_mask=None,
            key_padding_mask=None,
            pos_embs=pos_embs,
        )
        x = x + skip

        # truncate outputs corresponding to the MHA left context (we only care
        # about our chunk's outputs); see above to-do
        x = x[..., -orig_len:, :]

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

        Returns
        -------
        ConformerEncoderLayerStreamingContext
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
    output_hidden_states: bool, optional
        Whether the model should output the hidden states as a list of tensor.
    layerdrop_prob: float
        The probability to drop an entire layer.

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> pos_emb = torch.rand((1, 2*60-1, 512))
    >>> net = ConformerEncoder(1, 512, 512, 8)
    >>> output, _ = net(x, pos_embs=pos_emb)
    >>> output.shape
    torch.Size([8, 60, 512])

    >>> import torch
    >>> from speechbrain.lobes.models.transformer.Conformer import ConformerEncoder
    >>> x = torch.rand((8, 60, 512)); pos_emb = torch.rand((1, 2*60-1, 512));
    >>> net = ConformerEncoder(4, 512, 512, 8, output_hidden_states=True)
    >>> output, _, hs = net(x, pos_embs=pos_emb)
    >>> hs[0].shape
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
        output_hidden_states=False,
        layerdrop_prob=0.0,
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
        self.layerdrop_prob = layerdrop_prob
        self.attention_type = attention_type
        self.output_hidden_states = output_hidden_states

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
        dynchunktrain_config: Optional[DynChunkTrainConfig] = None,
    ):
        """
        Arguments
        ---------
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
        dynchunktrain_config: Optional[DynChunkTrainConfig]
            Dynamic Chunk Training configuration object for streaming,
            specifically involved here to apply Dynamic Chunk Convolution to the
            convolution module.

        Returns
        -------
        output : torch.Tensor
            The output of the Conformer.
        attention_lst : list
            The attention values.
        hidden_state_lst : list, optional
            The output of the hidden layers of the encoder.
            Only works if output_hidden_states is set to true.
        """
        if self.attention_type == "RelPosMHAXL":
            if pos_embs is None:
                raise ValueError(
                    "The chosen attention type for the Conformer is RelPosMHAXL. For this attention type, the positional embeddings are mandatory"
                )

        output = src

        if self.layerdrop_prob > 0.0:
            keep_probs = torch.rand(len(self.layers))

        attention_lst = []
        if self.output_hidden_states:
            hidden_state_lst = [output]

        for i, enc_layer in enumerate(self.layers):
            if (
                not self.training
                or self.layerdrop_prob == 0.0
                or keep_probs[i] > self.layerdrop_prob
            ):
                output, attention = enc_layer(
                    output,
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    pos_embs=pos_embs,
                    dynchunktrain_config=dynchunktrain_config,
                )
                attention_lst.append(attention)

                if self.output_hidden_states:
                    hidden_state_lst.append(output)

        output = self.norm(output)

        if self.output_hidden_states:
            return output, attention_lst, hidden_state_lst
        return output, attention_lst

    def forward_streaming(
        self,
        src: torch.Tensor,
        context: ConformerEncoderStreamingContext,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """Conformer streaming forward (typically for
        DynamicChunkTraining-trained models), which is to be used at inference
        time. Relies on a mutable context object as initialized by
        `make_streaming_context` that should be used across chunks.

        Arguments
        ---------
        src : torch.Tensor
            Input tensor. Batching is supported as long as you keep the context
            consistent.
        context : ConformerEncoderStreamingContext
            Mutable streaming context; the same object should be passed across
            calls.
        pos_embs : torch.Tensor, optional
            Positional embeddings, if used.

        Returns
        -------
        output : torch.Tensor
            The output of the streaming conformer.
        attention_lst : list
            The attention values.
        """

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

    def make_streaming_context(self, dynchunktrain_config: DynChunkTrainConfig):
        """Creates a blank streaming context for the encoder.

        Arguments
        ---------
        dynchunktrain_config: Optional[DynChunkTrainConfig]
            Dynamic Chunk Training configuration object for streaming

        Returns
        -------
        ConformerEncoderStreamingContext
        """
        return ConformerEncoderStreamingContext(
            dynchunktrain_config=dynchunktrain_config,
            layers=[
                layer.make_streaming_context(
                    mha_left_context_size=dynchunktrain_config.left_context_size_frames()
                )
                for layer in self.layers
            ],
        )


class ConformerDecoderLayer(nn.Module):
    """This is an implementation of Conformer encoder layer.

    Arguments
    ---------
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
    activation : torch.nn.Module, optional
         Activation function used in each Conformer layer.
    bias : bool, optional
        Whether  convolution module.
    dropout : int, optional
        Dropout for the encoder.
    causal : bool, optional
        Whether the convolutions should be causal or not.
    attention_type : str, optional
        type of attention layer, e.g. regularMHA for regular MultiHeadAttention.

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
        ---------
        tgt: torch.Tensor
            The sequence to the decoder layer.
        memory: torch.Tensor
            The sequence from the last layer of the encoder.
        tgt_mask: torch.Tensor, optional, optional
            The mask for the tgt sequence.
        memory_mask: torch.Tensor, optional
            The mask for the memory sequence.
        tgt_key_padding_mask: torch.Tensor, optional
            The mask for the tgt keys per batch.
        memory_key_padding_mask: torch.Tensor, optional
            The mask for the memory keys per batch.
        pos_embs_tgt: torch.Tensor, torch.nn.Module, optional
            Module or tensor containing the target sequence positional embeddings for each attention layer.
        pos_embs_src: torch.Tensor, torch.nn.Module, optional
            Module or tensor containing the source sequence positional embeddings for each attention layer.

        Returns
        -------
        x: torch.Tensor
            The output tensor
        self_attn : torch.Tensor
        self_attn : torch.Tensor
            The self attention tensor
        """
        # ffn module
        tgt = tgt + 0.5 * self.ffn_module1(tgt)
        # multi-head attention module
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
    ---------
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
        type of attention layer, e.g. regularMHA for regular MultiHeadAttention.


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
        ---------
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
        pos_embs_tgt: torch.Tensor, torch.nn.Module, optional
            Module or tensor containing the target sequence positional embeddings for each attention layer.
        pos_embs_src: torch.Tensor, torch.nn.Module, optional
            Module or tensor containing the source sequence positional embeddings for each attention layer.

        Returns
        -------
        output: torch.Tensor
            Conformer decoder output.
        self_attns : list
            Location of self attentions.
        multihead_attns : list
            Location of multihead attentions.
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
