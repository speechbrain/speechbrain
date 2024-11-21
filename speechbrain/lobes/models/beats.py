# --------------------------------------------------------
# BEATs: Audio Pre-Training with Acoustic Tokenizers (https://arxiv.org/abs/2212.09058)
# Github source: https://github.com/microsoft/unilm/tree/master/beats
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------


import logging
import math
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as ta_kaldi
from torch import Tensor, nn
from torch.nn import LayerNorm, Parameter

from speechbrain.dataio.dataio import length_to_mask

logger = logging.getLogger(__name__)


def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
        0.5
        * x
        * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )


def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)


def get_activation_fn(activation: str):
    """Returns the activation function corresponding to `activation`"""

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_fast":
        logger.warn(
            "--activation-fn=gelu_fast has been renamed to gelu_accurate"
        )
        return gelu_accurate
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "glu":
        return lambda x: x
    else:
        raise RuntimeError(
            "--activation-fn {} not supported".format(activation)
        )


class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        return x * self.act(x)


class GLU_Linear(nn.Module):
    def __init__(
        self, input_dim, output_dim, glu_type="sigmoid", bias_in_glu=True
    ):
        super(GLU_Linear, self).__init__()

        self.glu_type = glu_type
        self.output_dim = output_dim

        if glu_type == "sigmoid":
            self.glu_act = torch.nn.Sigmoid()
        elif glu_type == "swish":
            self.glu_act = Swish()
        elif glu_type == "relu":
            self.glu_act = torch.nn.ReLU()
        elif glu_type == "gelu":
            self.glu_act = torch.nn.GELU()

        if bias_in_glu:
            self.linear = nn.Linear(input_dim, output_dim * 2, True)
        else:
            self.linear = nn.Linear(input_dim, output_dim * 2, False)


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        assert (
            module.weight.size(1) % block_size == 0
        ), "Input features must be a multiple of block sizes"

    # 4D matrix
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


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt(
            (4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim)
        )
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(
            self.pos_conv, name="weight", dim=2
        )
        self.pos_conv = nn.Sequential(
            self.pos_conv, SamePad(args.conv_pos), nn.GELU()
        )

        if hasattr(args, "relative_position_embedding"):
            self.relative_position_embedding = args.relative_position_embedding
            self.num_buckets = args.num_buckets
            self.max_distance = args.max_distance
        else:
            self.relative_position_embedding = False
            self.num_buckets = 0
            self.max_distance = 0

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                    deep_norm=args.deep_norm,
                    has_relative_attention_bias=self.relative_position_embedding,
                    num_buckets=self.num_buckets,
                    max_distance=self.max_distance,
                    gru_rel_pos=args.gru_rel_pos,
                    encoder_layers=args.encoder_layers,
                )
                for i in range(args.encoder_layers)
            ]
        )
        if self.relative_position_embedding:
            for i in range(1, args.encoder_layers):
                del self.layers[i].self_attn.relative_attention_bias
                self.layers[i].self_attn.relative_attention_bias = self.layers[
                    0
                ].self_attn.relative_attention_bias

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

        if args.deep_norm:
            deep_norm_beta = math.pow(8 * args.encoder_layers, -1 / 4)
            for i in range(args.encoder_layers):
                nn.init.xavier_normal_(
                    self.layers[i].self_attn.k_proj.weight, gain=1
                )
                nn.init.xavier_normal_(
                    self.layers[i].self_attn.v_proj.weight, gain=deep_norm_beta
                )
                nn.init.xavier_normal_(
                    self.layers[i].self_attn.q_proj.weight, gain=1
                )
                nn.init.xavier_normal_(
                    self.layers[i].self_attn.out_proj.weight,
                    gain=deep_norm_beta,
                )
                nn.init.xavier_normal_(
                    self.layers[i].fc1.weight, gain=deep_norm_beta
                )
                nn.init.xavier_normal_(
                    self.layers[i].fc2.weight, gain=deep_norm_beta
                )

        self.layer_wise_gradient_decay_ratio = getattr(
            args, "layer_wise_gradient_decay_ratio", 1
        )

    def forward(self, x, padding_mask=None, output_all_hiddens=None):
        x, layer_results = self.extract_features(
            x, padding_mask, output_all_hiddens
        )

        if self.layer_norm_first and output_all_hiddens:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(self, x, padding_mask=None, output_all_hiddens=None):

        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        z = None
        if output_all_hiddens:
            layer_results.append(x)
        r = None
        pos_bias = None
        for i, layer in enumerate(self.layers):
            if self.layer_wise_gradient_decay_ratio != 1.0:
                x = GradMultiply.apply(x, self.layer_wise_gradient_decay_ratio)
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z, pos_bias = layer(
                    x,
                    self_attn_padding_mask=padding_mask,
                    need_weights=False,
                    pos_bias=pos_bias,
                )
            # if tgt_layer is not None:
            layer_results.append(x)

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results


class TransformerSentenceEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        deep_norm: bool = False,
        has_relative_attention_bias: bool = False,
        num_buckets: int = 0,
        max_distance: int = 0,
        rescale_init: bool = False,
        gru_rel_pos: bool = False,
        encoder_layers: int = 0,
    ) -> None:

        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        self.activation_name = activation_fn
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            has_relative_attention_bias=has_relative_attention_bias,
            num_buckets=num_buckets,
            max_distance=max_distance,
            rescale_init=rescale_init,
            gru_rel_pos=gru_rel_pos,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)

        if self.activation_name == "glu":
            self.fc1 = GLU_Linear(
                self.embedding_dim, ffn_embedding_dim, "swish"
            )
        else:
            self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        self.final_layer_norm = LayerNorm(self.embedding_dim)

        self.deep_norm = deep_norm
        if self.deep_norm:
            self.deep_norm_alpha = math.pow(2 * encoder_layers, 1 / 4)
        else:
            self.deep_norm_alpha = 1

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        pos_bias=None,
    ):
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
                position_bias=pos_bias,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                attn_mask=self_attn_mask,
                position_bias=pos_bias,
            )

            x = self.dropout1(x)
            x = residual * self.deep_norm_alpha + x

            x = self.self_attn_layer_norm(x)

            residual = x
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual * self.deep_norm_alpha + x
            x = self.final_layer_norm(x)

        return x, attn, pos_bias


class MultiheadAttention(nn.Module):
    """
    Implements multi-headed attention with support for advanced features like relative position 
    embeddings and gated relative position embedding (GRU-based).

    The design is inspired by "Attention Is All You Need" with extensions for flexible initialization 
    and enhanced embedding techniques.

    Parameters
    ----------
    embed_dim : int
        Total number of dimensions for input embeddings.
    num_heads : int
        Number of attention heads.
    kdim : int, optional
        Dimensionality of key embeddings. Defaults to `embed_dim`.
    vdim : int, optional
        Dimensionality of value embeddings. Defaults to `embed_dim`.
    dropout : float, optional
        Dropout probability for attention weights. Defaults to 0.0.
    bias : bool, optional
        Whether to include a bias term in projections. Defaults to True.
    add_bias_kv : bool, optional
        Whether to include bias for key and value projections. Defaults to False.
    add_zero_attn : bool, optional
        Whether to include zero attention vectors. Defaults to False.
    self_attention : bool, optional
        Whether the layer is for self-attention. Defaults to False.
    encoder_decoder_attention : bool, optional
        Whether the layer is for encoder-decoder attention. Defaults to False.
    q_noise : float, optional
        Noise level for quantization. Defaults to 0.0.
    qn_block_size : int, optional
        Block size for quantization. Defaults to 8.
    has_relative_attention_bias : bool, optional
        Whether to use relative position embeddings. Defaults to False.
    num_buckets : int, optional
        Number of buckets for relative position embeddings. Defaults to 32.
    max_distance : int, optional
        Maximum distance for relative position embeddings. Defaults to 128.
    gru_rel_pos : bool, optional
        Whether to use gated relative position embeddings. Defaults to False.
    rescale_init : bool, optional
        Whether to rescale the initialization of weights. Defaults to False.

    Attributes
    ----------
    scaling : float
        Scaling factor applied to query projections.
    dropout_module : nn.Dropout
        Dropout layer applied to attention weights.
    relative_attention_bias : nn.Embedding, optional
        Embedding layer for relative position biases, if enabled.
    """

    # Initialization method
    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        has_relative_attention_bias=False,
        num_buckets=32,
        max_distance=128,
        gru_rel_pos=False,
        rescale_init=False,
    ):
        super().__init__()

        # Attribute initialization
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)
        self.has_relative_attention_bias = has_relative_attention_bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        # Relative position bias setup
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

        self.head_dim = embed_dim // num_heads
        self.q_head_dim = self.head_dim
        self.k_head_dim = self.head_dim

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        # Self-attention and encoder-decoder attention flags
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key, and value to be of the same size."
        )

        # Initialize projection layers with optional quantization noise
        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=(not rescale_init)), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.q_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.out_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)

        # Bias terms for key and value, if applicable
        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        # Additional settings
        self.add_zero_attn = add_zero_attn
        self.gru_rel_pos = gru_rel_pos
        if self.gru_rel_pos:
            self.grep_linear = nn.Linear(self.q_head_dim, 8)
            self.grep_a = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        # Reset parameters
        self.reset_parameters()


    def reset_parameters(self):
        """
        Initializes the weights for the projection layers and relative position embeddings.

        This method applies Xavier initialization to the query, key, value, and output
        projections. Biases are initialized to zero, and relative position embedding weights
        are also initialized if applicable.

        Notes
        -----
        - For models where query, key, and value projections share dimensions (`qkv_same_dim`),
        the initialization gains are scaled to improve convergence.
        - Relative position biases (`bias_k` and `bias_v`) and embeddings are initialized
        with a normal distribution if present.
        """
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

        if self.has_relative_attention_bias:
            nn.init.xavier_normal_(self.relative_attention_bias.weight)


    def _relative_positions_bucket(self, relative_positions, bidirectional=True):
        """
        Computes bucket indices for relative positions for relative attention bias.

        This method categorizes relative positions into buckets to efficiently encode
        relative position information for the attention mechanism. For positions beyond
        a predefined range, logarithmic scaling is applied.

        Parameters
        ----------
        relative_positions : torch.Tensor
            A tensor of relative positions, where negative values indicate positions to the
            left and positive values indicate positions to the right.
        bidirectional : bool, optional
            If True, separate buckets are used for positive and negative positions.
            Defaults to True.

        Returns
        -------
        torch.Tensor
            A tensor of the same shape as `relative_positions`, where each value is the
            bucket index corresponding to the relative position.
        """
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        relative_buckets = 0

        if bidirectional:
            # Halve buckets for bidirectional attention
            num_buckets = num_buckets // 2
            relative_buckets += (relative_positions > 0).to(torch.long) * num_buckets
            relative_positions = torch.abs(relative_positions)
        else:
            relative_positions = -torch.min(relative_positions, torch.zeros_like(relative_positions))

        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact

        relative_position_if_large = max_exact + (
            torch.log(relative_positions.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(is_small, relative_positions, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int) -> torch.Tensor:
        """
        Computes relative position bias for attention scores.

        This method calculates the bias values based on the relative positions between
        query and key tokens, utilizing the relative position bucket indices and the
        associated embeddings.

        Parameters
        ----------
        query_length : int
            The length of the query sequence.
        key_length : int
            The length of the key sequence.

        Returns
        -------
        torch.Tensor
            A tensor of shape `(num_heads, query_length, key_length)` containing the
            relative position bias values for each attention head.

        Notes
        -----
        - The relative position buckets are computed using the `_relative_positions_bucket` method.
        - The embeddings for the relative positions are obtained from the
        `self.relative_attention_bias` embedding layer.
        """
        # Compute the relative position between each query and key token
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position

        # Map relative positions to bucket indices
        relative_position_bucket = self._relative_positions_bucket(
            relative_position, bidirectional=True
        )

        # Move bucket indices to the device of the bias embeddings
        relative_position_bucket = relative_position_bucket.to(
            self.relative_attention_bias.weight.device
        )

        # Fetch bias values from the relative position embedding layer
        values = self.relative_attention_bias(relative_position_bucket)

        # Rearrange dimensions to match expected output shape
        values = values.permute([2, 0, 1])  # Shape: (num_heads, query_length, key_length)

        return values


    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        position_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Forward pass for multi-head attention with support for relative position embeddings,
        caching, and optional dropout.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor of shape `(target_length, batch_size, embed_dim)`.
        key : torch.Tensor, optional
            Key tensor of shape `(source_length, batch_size, embed_dim)`.
        value : torch.Tensor, optional
            Value tensor of shape `(source_length, batch_size, embed_dim)`.
        key_padding_mask : torch.Tensor, optional
            Mask to exclude padding keys, of shape `(batch_size, source_length)`,
            where padding elements are indicated by 1s.
        incremental_state : dict, optional
            Stores cached key and value tensors for incremental decoding. Defaults to None.
        need_weights : bool, optional
            If True, returns the attention weights. Defaults to True.
        static_kv : bool, optional
            If True, the key and value tensors remain static for incremental decoding.
            Defaults to False.
        attn_mask : torch.Tensor, optional
            Attention mask to prevent certain positions from attending, typically for
            causal attention. Shape: `(target_length, source_length)`. Defaults to None.
        before_softmax : bool, optional
            If True, returns raw attention scores before softmax. Defaults to False.
        need_head_weights : bool, optional
            If True, returns attention weights for each head. Implies `need_weights=True`.
            Defaults to False.
        position_bias : torch.Tensor, optional
            Precomputed position bias tensor. If None, it is computed during the forward pass.

        Returns
        -------
        attn : torch.Tensor
            Attention output of shape `(target_length, batch_size, embed_dim)`.
        attn_weights : torch.Tensor, optional
            Attention weights of shape `(batch_size, num_heads, target_length, source_length)`,
            averaged across heads if `need_head_weights=False`.
        position_bias : torch.Tensor, optional
            Computed or passed relative position bias of shape `(num_heads, target_length, source_length)`.

        Notes
        -----
        - Supports self-attention, encoder-decoder attention, and caching for efficient
        incremental decoding.
        - Implements relative position embeddings if `self.has_relative_attention_bias` is True.
        - Attention probabilities are scaled using a normalization factor and optionally
        include dropout.
        """
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        if self.has_relative_attention_bias and position_bias is None:
            position_bias = self.compute_bias(tgt_len, src_len)
            position_bias = (
                position_bias.unsqueeze(0)
                .repeat(bsz, 1, 1, 1)
                .view(bsz * self.num_heads, tgt_len, src_len)
            )

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert (
                        self.encoder_decoder_attention
                        and not self.self_attention
                    )
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling
        alpha = 32
        q *= 1 / alpha

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)],
                    dim=1,
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.q_head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.k_head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            saved_state["prev_value"] = v.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(
                incremental_state, saved_state
            )
        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat(
                [k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1
            )
            v = torch.cat(
                [v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1
            )
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)],
                    dim=1,
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = (
            attn_weights - attn_weights.max(dim=-1, keepdim=True)[0]
        ) * alpha
        attn_weights = self.apply_sparse_mask(
            attn_weights, tgt_len, src_len, bsz
        )

        assert list(attn_weights.size()) == [
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask, float("-inf")
                )
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(
                bsz * self.num_heads, tgt_len, src_len
            )

        if before_softmax:
            return attn_weights, v, position_bias

        if position_bias is not None:
            attn_mask_rel_pos = position_bias
            if self.gru_rel_pos == 1:
                query_layer = (
                    q.view(bsz, self.num_heads, tgt_len, self.q_head_dim)
                    * alpha
                    / self.scaling
                )
                _B, _H, _L, __ = query_layer.size()
                gate_a, gate_b = torch.sigmoid(
                    self.grep_linear(query_layer)
                    .view(_B, _H, _L, 2, 4)
                    .sum(-1, keepdim=False)
                ).chunk(2, dim=-1)
                gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                attn_mask_rel_pos = (
                    gate_a_1.view(bsz * self.num_heads, tgt_len, 1)
                    * position_bias
                )

            attn_mask_rel_pos = attn_mask_rel_pos.view(attn_weights.size())

            attn_weights = attn_weights + attn_mask_rel_pos

        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights, position_bias

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - prev_key_padding_mask.size(1)),
                    device=prev_key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [prev_key_padding_mask.float(), filler.float()], dim=1
                )
            else:
                new_key_padding_mask = prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - key_padding_mask.size(1)),
                    device=key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [filler.float(), key_padding_mask.float()], dim=1
                )
            else:
                new_key_padding_mask = key_padding_mask.float()
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def _get_input_buffer(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(
            incremental_state, "attn_state", buffer
        )

    def apply_sparse_mask(
        self, attn_weights, tgt_len: int, src_len: int, bsz: int
    ):
        return attn_weights


def init_bert_params(module: nn.Module) -> None:
    """
    Initializes weights and biases for modules in the BERT model.

    This method applies specific initialization schemes based on the type of the module:
    - Linear layers: Weights are initialized using a normal distribution with mean 0 and standard deviation 0.02, and biases are set to 0.
    - Embedding layers: Weights are initialized using a normal distribution with mean 0 and standard deviation 0.02, and padding indices (if applicable) are zeroed out.
    - MultiHeadAttention: Query, key, and value projection weights are initialized using the same normal distribution.

    Parameters
    ----------
    module : nn.Module
        The module to initialize. Can be one of `nn.Linear`, `nn.Embedding`, or `MultiheadAttention`.

    Notes
    -----
    - The initialization scheme aligns with the settings typically used in BERT models.
    - For modules wrapped with FSDP (Fully Sharded Data Parallel), parameters are temporarily moved to the CPU for consistent random number generation.
    """

    def normal_(data: torch.Tensor) -> None:
        """
        Initializes a tensor with values drawn from a normal distribution.

        Parameters
        ----------
        data : torch.Tensor
            The tensor to initialize.
        """
        # Handle FSDP initialization
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        # Initialize weights and biases for linear layers
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()

    elif isinstance(module, nn.Embedding):
        # Initialize weights for embedding layers
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

    elif isinstance(module, MultiheadAttention):
        # Initialize weights for multi-head attention projections
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)



class BEATsConfig:
    """
    Configuration class for the BEATs model.

    This class defines the configuration for the BEATs model. It provides a default
    configuration that can be updated with custom settings via the `update` method.

    Parameters
    ----------
    cfg : dict, optional
        A dictionary containing custom configuration values. If provided, it will override
        the default settings.
    """

    def __init__(self, cfg=None):
        self.input_patch_size: int = 16  # path size of patch embedding
        self.embed_dim: int = 512  # patch embedding dimension
        self.conv_bias: bool = False  # include bias in conv encoder

        self.encoder_layers: int = 12  # num encoder layers in the transformer
        self.encoder_embed_dim: int = 768  # encoder embedding dimension
        self.encoder_ffn_embed_dim: int = (
            3072  # encoder embedding dimension for FFN
        )
        self.encoder_attention_heads: int = 12  # num encoder attention heads
        self.activation_fn: str = "gelu"  # activation function to use

        self.layer_wise_gradient_decay_ratio: float = (
            1.0  # ratio for layer-wise gradient decay
        )
        self.layer_norm_first: bool = (
            False  # apply layernorm first in the transformer
        )
        self.deep_norm: bool = False  # apply deep_norm first in the transformer

        # dropouts
        self.dropout: float = 0.1  # dropout probability for the transformer
        self.attention_dropout: float = (
            0.1  # dropout probability for attention weights
        )
        self.activation_dropout: float = (
            0.0  # dropout probability after activation in FFN
        )
        self.encoder_layerdrop: float = (
            0.0  # probability of dropping a tarnsformer layer
        )
        self.dropout_input: float = (
            0.0  # dropout to apply to the input (after feat extr)
        )

        # positional embeddings
        self.conv_pos: int = (
            128  # number of filters for convolutional positional embeddings
        )
        self.conv_pos_groups: int = (
            16  # number of groups for convolutional positional embedding
        )

        # relative position embedding
        self.relative_position_embedding: bool = (
            False  # apply relative position embedding
        )
        self.num_buckets: int = (
            320  # number of buckets for relative position embedding
        )
        self.max_distance: int = (
            1280  # maximum distance for relative position embedding
        )
        self.gru_rel_pos: bool = (
            False  # apply gated relative position embedding
        )

        # label predictor
        self.finetuned_model: bool = (
            False  # whether the model is a fine-tuned model.
        )
        self.predictor_dropout: float = (
            0.1  # dropout probability for the predictor
        )
        self.predictor_class: int = 527  # target class number for the predictor

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)


class BEATs(nn.Module):
    """
    BEATs: Bidirectional Encoder Representations from Audio Transformers.

    This class implements the BEATs model, which processes audio signals for feature extraction
    or downstream tasks. The model supports loading from a checkpoint, applying normalization,
    and optionally freezing parameters.

    Parameters
    ----------
    ckp_path : str, optional
        Path to the checkpoint file. If None, the model initializes without pre-trained weights.
        You could download the checkpoints from : https://github.com/microsoft/unilm/tree/master/beats
    freeze : bool, optional (default: False)
        If True, the model parameters are frozen and the model is set to evaluation mode.
    output_all_hiddens : bool, optional (default: False)
        If True, the forward function outputs hidden states from all transformer layers.
        For example BEATs_iter3 has 12 transformer layers and the output is of shape (13, B, T, C),
        where a projection of the CNN output is added to the beginning.
        If False, the forward function outputs the hidden states only from the last transformer layer.

    Example
    -------
    >>> audio = torch.randn(4, 10000)  # Batch of 4 audio signals
    >>> length = torch.tensor([1.0, 0.5, 0.75, 1.0])
    >>> model = BEATs()
    >>> outputs = model.extract_features(audio, length)
    >>> outputs.shape
    torch.Size([4, 24, 768])
    """

    def __init__(
        self,
        ckp_path: str = None,
        freeze: bool = True,
        output_all_hiddens: bool = False,
    ) -> None:
        super().__init__()

        # Load configuration and checkpoint
        cfg, checkpoint = None, None
        if ckp_path:
            if not os.path.exists(ckp_path):
                raise FileNotFoundError(
                    f"Checkpoint file '{ckp_path}' does not exist."
                )
            checkpoint = torch.load(ckp_path)
            cfg = checkpoint.get("cfg", None)

        # Initialize model configuration
        self.cfg = BEATsConfig(cfg)
        logger.info(f"BEATs Config: {self.cfg.__dict__}")

        # Model attributes
        self.freeze = freeze
        self.output_all_hiddens = output_all_hiddens
        self.embed = self.cfg.embed_dim

        # Define layers and modules
        self.post_extract_proj = (
            nn.Linear(self.embed, self.cfg.encoder_embed_dim)
            if self.embed != self.cfg.encoder_embed_dim
            else None
        )
        self.input_patch_size = self.cfg.input_patch_size
        self.patch_embedding = nn.Conv2d(
            1,
            self.embed,
            kernel_size=self.input_patch_size,
            stride=self.input_patch_size,
            bias=self.cfg.conv_bias,
        )
        self.dropout_input = nn.Dropout(self.cfg.dropout_input)

        # Configuration checks
        assert not (
            self.cfg.deep_norm and self.cfg.layer_norm_first
        ), "Configuration error: 'deep_norm' and 'layer_norm_first' cannot both be True."

        # Initialize encoder and layer normalization
        self.encoder = TransformerEncoder(self.cfg)
        self.layer_norm = LayerNorm(self.embed)

        # Define predictor for fine-tuned models
        if self.cfg.finetuned_model:
            self.predictor_dropout = nn.Dropout(self.cfg.predictor_dropout)
            self.predictor = nn.Linear(
                self.cfg.encoder_embed_dim, self.cfg.predictor_class
            )
        else:
            self.predictor = None

        # Load weights from the checkpoint if available
        if checkpoint:
            self.load_state_dict(checkpoint["model"])

        # Set the model to evaluation mode if frozen
        if self.freeze:
            self.eval()

    def forward_padding_mask(
        self, features: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Adjusts the padding mask for the given features.

        Parameters
        ----------
        features : torch.Tensor
            Input features after patch embedding.
        padding_mask : torch.Tensor
            Original padding mask for input signals.

        Returns
        -------
        torch.Tensor
            Adjusted padding mask.
        """
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(
            padding_mask.size(0), features.size(1), -1
        )
        return padding_mask.all(-1)

    def preprocess(
        self,
        source: torch.Tensor,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
    ) -> torch.Tensor:
        """
        Preprocesses the input waveform by extracting filter banks and applying normalization.

        Parameters
        ----------
        source : torch.Tensor
            Input waveform signals.
        fbank_mean : float, optional
            Mean value for filter bank normalization (default: 15.41663).
        fbank_std : float, optional
            Standard deviation for filter bank normalization (default: 6.55582).

        Returns
        -------
        torch.Tensor
            Normalized filter banks.
        """
        fbanks = []
        for waveform in source:
            waveform = waveform.unsqueeze(0) * 2**15
            fbank = ta_kaldi.fbank(
                waveform,
                num_mel_bins=128,
                sample_frequency=16000,
                frame_length=25,
                frame_shift=10,
            )
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)
        return (fbank - fbank_mean) / (2 * fbank_std)

    def extract_features(
        self,
        wav: torch.Tensor,
        wav_lens: Optional[torch.Tensor] = None,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
    ) -> torch.Tensor:
        """
        Extracts features from the input waveform.

        Parameters
        ----------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        wav_lens : torch.Tensor
            The relative length of the wav given in SpeechBrain format.

        fbank_mean : float, optional
            Mean value for filter bank normalization (default: 15.41663).
        fbank_std : float, optional
            Standard deviation for filter bank normalization (default: 6.55582).

        Returns
        -------
        torch.Tensor
            Extracted features from the BEATs model.
        """
        fbank = self.preprocess(wav, fbank_mean, fbank_std)

        if wav_lens is not None:
            max_len = wav.size(-1)
            padding_mask = ~length_to_mask(
                wav_lens * max_len, max_len, device=wav.device
            ).bool()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(fbank, padding_mask)

        fbank = fbank.unsqueeze(1)
        features = self.patch_embedding(fbank)
        features = features.reshape(
            features.shape[0], features.shape[1], -1
        ).transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)

        x, layer_results = self.encoder(
            features,
            padding_mask=padding_mask,
            output_all_hiddens=self.output_all_hiddens,
        )

        if self.predictor is not None:
            x = self.predictor_dropout(x)
            logits = self.predictor(x)

            if padding_mask is not None and padding_mask.any():
                logits[padding_mask] = 0
                logits = logits.sum(dim=1)
                logits = logits / (~padding_mask).sum(dim=1).unsqueeze(
                    -1
                ).expand_as(logits)
            else:
                logits = logits.mean(dim=1)

            lprobs = torch.sigmoid(logits)
            return lprobs, padding_mask, layer_results

        if self.output_all_hiddens:
            x = torch.stack(layer_results, dim=0)

        return x
