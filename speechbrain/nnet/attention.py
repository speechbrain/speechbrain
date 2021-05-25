"""Library implementing attention modules.

Authors
 * Ju-Chieh Chou 2020
 * Jianyuan Zhong 2020
 * Loren Lugosch 2020
"""

import torch
from torch.nn.parameter import Parameter
from torch.nn import Linear
import logging
from functools import partial
import warnings
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.functional import linear, softmax, dropout
import numpy as np
from typing import Optional
from speechbrain.dataio.dataio import length_to_mask

from speechbrain.nnet.attention_utils.longformer_utilities import (
    mask_invalid_locations,
    sliding_chunks_matmul_qk,
    sliding_chunks_matmul_pv,
    sliding_chunks_no_overlap_matmul_qk,
    sliding_chunks_no_overlap_matmul_pv,
)
from speechbrain.nnet.attention_utils.linformer_utilities import get_EF
from speechbrain.nnet.attention_utils.reformer_utilities import *

logger = logging.getLogger(__name__)


class ContentBasedAttention(nn.Module):
    """ This class implements content-based attention module for seq2seq
    learning.

    Reference: NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN
    AND TRANSLATE, Bahdanau et.al. https://arxiv.org/pdf/1409.0473.pdf

    Arguments
    ---------
    attn_dim : int
        Size of the attention feature.
    output_dim : int
        Size of the output context vector.
    scaling : float
        The factor controls the sharpening degree (default: 1.0).

    Example
    -------
    >>> enc_tensor = torch.rand([4, 10, 20])
    >>> enc_len = torch.ones([4]) * 10
    >>> dec_tensor = torch.rand([4, 25])
    >>> net = ContentBasedAttention(enc_dim=20, dec_dim=25, attn_dim=30, output_dim=5)
    >>> out_tensor, out_weight = net(enc_tensor, enc_len, dec_tensor)
    >>> out_tensor.shape
    torch.Size([4, 5])
    """

    def __init__(self, enc_dim, dec_dim, attn_dim, output_dim, scaling=1.0):
        super(ContentBasedAttention, self).__init__()

        self.mlp_enc = nn.Linear(enc_dim, attn_dim)
        self.mlp_dec = nn.Linear(dec_dim, attn_dim)
        self.mlp_attn = nn.Linear(attn_dim, 1, bias=False)
        self.mlp_out = nn.Linear(enc_dim, output_dim)

        self.scaling = scaling

        self.softmax = nn.Softmax(dim=-1)

        # reset the encoder states, lengths and masks
        self.reset()

    def reset(self):
        """Reset the memory in the attention module.
        """
        self.enc_len = None
        self.precomputed_enc_h = None
        self.mask = None

    def forward(self, enc_states, enc_len, dec_states):
        """Returns the output of the attention module.

        Arguments
        ---------
        enc_states : torch.Tensor
            The tensor to be attended.
        enc_len : torch.Tensor
            The real length (without padding) of enc_states for each sentence.
        dec_states : torch.Tensor
            The query tensor.

        """

        if self.precomputed_enc_h is None:
            self.precomputed_enc_h = self.mlp_enc(enc_states)
            self.mask = length_to_mask(
                enc_len, max_len=enc_states.size(1), device=enc_states.device
            )

        dec_h = self.mlp_dec(dec_states.unsqueeze(1))
        attn = self.mlp_attn(
            torch.tanh(self.precomputed_enc_h + dec_h)
        ).squeeze(-1)

        # mask the padded frames
        attn = attn.masked_fill(self.mask == 0, -np.inf)
        attn = self.softmax(attn * self.scaling)

        # compute context vectors
        # [B, 1, L] X [B, L, F]
        context = torch.bmm(attn.unsqueeze(1), enc_states).squeeze(1)
        context = self.mlp_out(context)

        return context, attn


class LocationAwareAttention(nn.Module):
    """This class implements location-aware attention module for seq2seq learning.

    Reference: Attention-Based Models for Speech Recognition, Chorowski et.al.
    https://arxiv.org/pdf/1506.07503.pdf

    Arguments
    ---------
    attn_dim : int
        Size of the attention feature.
    output_dim : int
        Size of the output context vector.
    conv_channels : int
        Number of channel for location feature.
    kernel_size : int
        Kernel size of convolutional layer for location feature.
    scaling : float
        The factor controls the sharpening degree (default: 1.0).

    Example
    -------
    >>> enc_tensor = torch.rand([4, 10, 20])
    >>> enc_len = torch.ones([4]) * 10
    >>> dec_tensor = torch.rand([4, 25])
    >>> net = LocationAwareAttention(
    ...     enc_dim=20,
    ...     dec_dim=25,
    ...     attn_dim=30,
    ...     output_dim=5,
    ...     conv_channels=10,
    ...     kernel_size=100)
    >>> out_tensor, out_weight = net(enc_tensor, enc_len, dec_tensor)
    >>> out_tensor.shape
    torch.Size([4, 5])
    """

    precomputed_enc_h: Optional[torch.Tensor]

    def __init__(
        self,
        enc_dim,
        dec_dim,
        attn_dim,
        output_dim,
        conv_channels,
        kernel_size,
        scaling=1.0,
    ):
        super(LocationAwareAttention, self).__init__()

        self.mlp_enc = nn.Linear(enc_dim, attn_dim)
        self.mlp_dec = nn.Linear(dec_dim, attn_dim)
        self.mlp_attn = nn.Linear(attn_dim, 1, bias=False)
        self.conv_loc = nn.Conv1d(
            1,
            conv_channels,
            kernel_size=2 * kernel_size + 1,
            padding=kernel_size,
            bias=False,
        )
        self.mlp_loc = nn.Linear(conv_channels, attn_dim)
        self.mlp_attn = nn.Linear(attn_dim, 1, bias=False)
        self.mlp_out = nn.Linear(enc_dim, output_dim)

        self.scaling = scaling

        self.softmax = nn.Softmax(dim=-1)

        # reset the encoder states, lengths and masks
        self.reset()

    def reset(self):
        """Reset the memory in attention module.
        """
        self.enc_len = None
        self.precomputed_enc_h = None
        self.mask = None
        self.prev_attn = None

    def forward(self, enc_states, enc_len, dec_states):
        """Returns the output of the attention module.

        Arguments
        ---------
        enc_states : torch.Tensor
            The tensor to be attended.
        enc_len : torch.Tensor
            The real length (without padding) of enc_states for each sentence.
        dec_states : torch.Tensor
            The query tensor.
        """
        if self.precomputed_enc_h is None:
            self.precomputed_enc_h = self.mlp_enc(enc_states)
            self.mask = length_to_mask(
                enc_len, max_len=enc_states.size(1), device=enc_states.device
            )

            # multiply mask by 1/Ln for each row
            self.prev_attn = self.mask * (1 / enc_len.float()).unsqueeze(1)

        # compute location-aware features
        # [B, 1, L] -> [B, C, L]
        attn_conv = self.conv_loc(self.prev_attn.unsqueeze(1))
        # [B, C, L] -> [B, L, C] -> [B, L, F]
        attn_conv = self.mlp_loc(attn_conv.transpose(1, 2))

        dec_h = self.mlp_dec(dec_states.unsqueeze(1))
        attn = self.mlp_attn(
            torch.tanh(self.precomputed_enc_h + dec_h + attn_conv)
        ).squeeze(-1)

        # mask the padded frames
        attn = attn.masked_fill(self.mask == 0, -np.inf)
        attn = self.softmax(attn * self.scaling)

        # set prev_attn to current attn for the next timestep
        self.prev_attn = attn.detach()

        # compute context vectors
        # [B, 1, L] X [B, L, F]
        context = torch.bmm(attn.unsqueeze(1), enc_states).squeeze(1)
        context = self.mlp_out(context)

        return context, attn


class KeyValueAttention(nn.Module):
    """ This class implements a single-headed key-value attention module for seq2seq
    learning.

    Reference: "Attention Is All You Need" by Vaswani et al., sec. 3.2.1

    Arguments
    ---------
    enc_dim : int
        Size of the encoder feature vectors from which keys and values are computed.
    dec_dim : int
        Size of the decoder feature vectors from which queries are computed.
    attn_dim : int
        Size of the attention feature.
    output_dim : int
        Size of the output context vector.

    Example
    -------
    >>> enc_tensor = torch.rand([4, 10, 20])
    >>> enc_len = torch.ones([4]) * 10
    >>> dec_tensor = torch.rand([4, 25])
    >>> net = KeyValueAttention(enc_dim=20, dec_dim=25, attn_dim=30, output_dim=5)
    >>> out_tensor, out_weight = net(enc_tensor, enc_len, dec_tensor)
    >>> out_tensor.shape
    torch.Size([4, 5])
    """

    def __init__(self, enc_dim, dec_dim, attn_dim, output_dim):
        super(KeyValueAttention, self).__init__()

        self.key_linear = nn.Linear(enc_dim, attn_dim)
        self.query_linear = nn.Linear(dec_dim, attn_dim)
        self.value_linear = nn.Linear(enc_dim, output_dim)
        self.scaling = torch.sqrt(torch.tensor(attn_dim).float())

        # reset the encoder states, lengths and masks
        self.reset()

    def reset(self):
        """Reset the memory in the attention module.
        """
        self.values = None
        self.keys = None
        self.mask = None

    def forward(self, enc_states, enc_len, dec_states):
        """Returns the output of the attention module.

        Arguments
        ---------
        enc_states : torch.Tensor
            The tensor to be attended.
        enc_len : torch.Tensor
            The real length (without padding) of enc_states for each sentence.
        dec_states : torch.Tensor
            The query tensor.
        """

        if self.keys is None:
            self.keys = self.key_linear(enc_states)
            self.values = self.value_linear(enc_states)
            self.mask = length_to_mask(
                enc_len, max_len=enc_states.size(1), device=enc_states.device
            ).unsqueeze(2)

        query = self.query_linear(dec_states).unsqueeze(2)
        scores = torch.matmul(self.keys, query) / self.scaling
        scores = scores.masked_fill(self.mask == 0, -np.inf)
        normalized_scores = scores.softmax(1).transpose(1, 2)
        out = torch.matmul(normalized_scores, self.values).squeeze(1)
        return out, normalized_scores


class MultiheadAttention(nn.Module):
    """ The class is a wrapper of MultiHead Attention for torch.nn.MultiHeadAttention.

    Reference: https://pytorch.org/docs/stable/nn.html

    Arguments
    ----------
    num_heads : int
        parallel attention heads.
    dropout : float
        a Dropout layer on attn_output_weights (default: 0.0).
    bias : bool
        add bias as module parameter (default: True).
    add_bias_kv : bool
        add bias to the key and value sequences at dim=0.
    add_zero_attn : bool
        add a new batch of zeros to the key and value sequences at dim=1.
    kdim : int
        total number of features in key (default: None).
    vdim : int
        total number of features in value (default: None).

    Example
    -------
    >>> inputs = torch.rand([8, 60, 512])
    >>> net = MultiheadAttention(nhead=8, d_model=inputs.shape[-1])
    >>> outputs, attn = net(inputs, inputs, inputs)
    >>> outputs.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        nhead,
        d_model,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
    ):
        super().__init__()

        self.att = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
        )

    def forward(
        self,
        query,
        key,
        value,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ----------
        query : tensor
            (N, L, E) where L is the target sequence length,
            N is the batch size, E is the embedding dimension.
        key : tensor
            (N, S, E) where S is the source sequence length,
            N is the batch size, E is the embedding dimension.
        value : tensor
            (N, S, E) where S is the source sequence length,
            N is the batch size, E is the embedding dimension.
        key_padding_mask : tensor
            (N, S) where N is the batch size, S is the source sequence
            length. If a ByteTensor is provided, the non-zero positions will
            be ignored while the position with the zero positions will be
            unchanged. If a BoolTensor is provided, the positions with the
            value of True will be ignored while the position with the value
            of False will be unchanged.
        attn_mask : tensor
            2D mask (L, S) where L is the target sequence length, S is
            the source sequence length.
            3D mask (N*num_heads, L, S) where N is the batch
            size, L is the target sequence length, S is the source sequence
            length. attn_mask ensure that position i is allowed to attend the
            unmasked positions. If a ByteTensor is provided, the non-zero
            positions are not allowed to attend while the zero positions will
            be unchanged. If a BoolTensor is provided, positions with True is
            not allowed to attend while False values will be unchanged. If a
            FloatTensor is provided, it will be added to the attention weight.

        Outputs
        -------
        attn_output : tensor
            (L, N, E) where L is the target sequence length, N is the
            batch size, E is the embedding dimension.
        attn_output_weights : tensor
            (N, L, S) where N is the batch size, L is the target
            sequence length, S is the source sequence length.
        """
        # give tensors of shape (time, batch, fea)
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        output, attention = self.att(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )

        # reshape the output back to (batch, time, fea)
        output = output.permute(1, 0, 2)

        return output, attention


class PositionalwiseFeedForward(nn.Module):
    """The class implements the positional-wise feed forward module in
    “Attention Is All You Need”.

    Arguments
    ----------
    d_ffn: int
        Dimension of representation space of this positional-wise feed
        forward module.
    input_shape : tuple
        Expected shape of the input. Alternatively use ``input_size``.
    input_size : int
        Expected size of the input. Alternatively use ``input_shape``.
    dropout: float
        Fraction of outputs to drop.
    activation: torch class
        activation functions to be applied (Recommendation: ReLU, GELU).

    Example
    -------
    >>> inputs = torch.rand([8, 60, 512])
    >>> net = PositionalwiseFeedForward(256, input_size=inputs.shape[-1])
    >>> outputs = net(inputs)
    >>> outputs.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_ffn,
        input_shape=None,
        input_size=None,
        dropout=0.1,
        activation=nn.ReLU,
    ):
        super().__init__()

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]

        self.ffn = nn.Sequential(
            nn.Linear(input_size, d_ffn),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, input_size),
        )

    def forward(self, x):
        # give a tensor of shap (time, batch, fea)
        x = x.permute(1, 0, 2)
        x = self.ffn(x)

        # reshape the output back to (batch, time, fea)
        x = x.permute(1, 0, 2)

        return x


class LongformerSelfAttention(nn.Module):
    """
    This class comes from: https://github.com/allenai/longformer
    Longformer is an open-source project developed by the Allen Institute for Artificial Intelligence (AI2).
    AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and
    engineering.

    Parts of the code found herein were modified by: Jonathan Tremblay in order to fit SpeechBrain's interface.
    """

    def __init__(
        self,
        layer_id,
        num_attention_heads,
        hidden_size,
        attention_probs_dropout_prob,
        attention_window,
        attention_mode,
        attention_dilation,
    ):
        super(LongformerSelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )
        self.num_heads = num_attention_heads
        self.head_dim = int(hidden_size / num_attention_heads)
        self.embed_dim = hidden_size

        self.attention_dilation = attention_dilation  # Not implemented yet

        self.query = nn.Linear(hidden_size, self.embed_dim)
        self.key = nn.Linear(hidden_size, self.embed_dim)
        self.value = nn.Linear(hidden_size, self.embed_dim)

        self.query_global = nn.Linear(hidden_size, self.embed_dim)
        self.key_global = nn.Linear(hidden_size, self.embed_dim)
        self.value_global = nn.Linear(hidden_size, self.embed_dim)

        self.dropout = attention_probs_dropout_prob

        self.layer_id = layer_id
        self.attention_window = attention_window[self.layer_id]
        self.attention_dilation = self.attention_dilation[self.layer_id]
        self.attention_mode = attention_mode
        assert self.attention_window > 0
        assert self.attention_dilation > 0
        assert self.attention_mode in [
            "sliding_chunks",
            "sliding_chunks_no_overlap",
        ]
        if self.attention_mode in [
            "sliding_chunks",
            "sliding_chunks_no_overlap",
        ]:
            assert (
                self.attention_dilation == 1
            ), "dilatation is not implemented yet"

    def forward(
        self, hidden_states, output_attentions=False,
    ):
        hidden_states = hidden_states.transpose(0, 1)
        seq_len, bsz, embed_dim = hidden_states.size()
        assert embed_dim == self.embed_dim
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        q /= math.sqrt(self.head_dim)

        q = q.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)

        if self.attention_mode == "sliding_chunks":
            attn_weights = sliding_chunks_matmul_qk(
                q, k, self.attention_window, padding_value=0
            )
        elif self.attention_mode == "sliding_chunks_no_overlap":
            attn_weights = sliding_chunks_no_overlap_matmul_qk(
                q, k, self.attention_window, padding_value=0
            )
        else:
            raise False
        mask_invalid_locations(
            attn_weights, self.attention_window, self.attention_dilation, False
        )

        assert list(attn_weights.size())[:3] == [bsz, seq_len, self.num_heads]
        assert attn_weights.size(dim=3) in [
            self.attention_window * 2 + 1,
            self.attention_window * 3,
        ]

        attn_weights_float = F.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        )  # use fp32 for numerical stability
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(
            attn_weights_float.type_as(attn_weights),
            p=self.dropout,
            training=self.training,
        )
        v = v.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        attn = 0

        if self.attention_mode == "sliding_chunks":
            attn += sliding_chunks_matmul_pv(
                attn_probs, v, self.attention_window
            )
        elif self.attention_mode == "sliding_chunks_no_overlap":
            attn += sliding_chunks_no_overlap_matmul_pv(
                attn_probs, v, self.attention_window
            )
        else:
            raise False

        attn = attn.type_as(hidden_states)
        assert list(attn.size()) == [
            bsz,
            seq_len,
            self.num_heads,
            self.head_dim,
        ]
        attn = (
            attn.transpose(0, 1).reshape(seq_len, bsz, embed_dim).contiguous()
        )

        context_layer = attn.transpose(0, 1)
        if output_attentions:
            # without global attention, return local attention probabilities
            # batch_size x num_heads x sequence_length x window_size
            # which is the attention weights of every token attending to its neighbours
            attn_weights = attn_weights.permute(0, 2, 1, 3)
        outputs = (
            (context_layer, (attn_weights.sum(dim=1) / self.num_heads))
            if output_attentions
            else (context_layer,)
        )
        return outputs


class LinearMultiheadAttention(nn.Module):
    """
    This class comes from https://github.com/kuixu/Linear-Multihead-Attention
    It was adjusted to fit SpeechBrain's architecture.
    """

    __annotations__ = {
        "bias_k": torch._jit_internal.Optional[torch.Tensor],
        "bias_v": torch._jit_internal.Optional[torch.Tensor],
    }
    __constants__ = [
        "q_proj_weight",
        "k_proj_weight",
        "v_proj_weight",
        "in_proj_weight",
        "e_proj_weight",
        "f_proj_weight",
    ]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.1,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        seq_len=512,
        proj_k=128,
        param_sharing="none",
        method="convolution",
        layerwise_proj=None,
    ):
        super(LinearMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = (
            self.kdim == embed_dim and self.vdim == embed_dim
        )

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = Parameter(
                torch.empty(3 * embed_dim, embed_dim)
            )
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)
        self.method = method
        if param_sharing != "layerwise":
            self.e_proj = get_EF(
                seq_len,
                proj_k,
                method=self.method,
                head_dim=self.embed_dim,
                bias=True,
            )
        if param_sharing == "key_value":
            self.f_proj = self.e_proj
        elif param_sharing == "layerwise":
            self.layerwise_proj = layerwise_proj
            self.f_proj = self.e_proj = self.layerwise_proj
        else:
            self.f_proj = get_EF(
                seq_len,
                proj_k,
                method=self.method,
                head_dim=self.embed_dim,
                bias=True,
            )

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.method == "learnable":
            xavier_uniform_(self.e_proj.weight)
            xavier_uniform_(self.f_proj.weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super(LinearMultiheadAttention, self).__setstate__(state)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: Optional[bool] = True,
    ):
        """
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. This is an additive mask
                (i.e. the values will be added to the attention layer). A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        Shape:
            - Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
            3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
            S is the source sequence length.
            - Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
            L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return self.linear_multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                e_proj=self.e_proj,
                f_proj=self.f_proj,
                method=self.method,
            )
        else:
            return self.linear_multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                e_proj=self.e_proj,
                f_proj=self.f_proj,
                method=self.method,
            )

    # TODO: possible refactor
    # flake8: noqa: C901
    @staticmethod
    def linear_multi_head_attention_forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: torch.Tensor,
        in_proj_bias: torch.Tensor,
        bias_k: Optional[torch.Tensor],
        bias_v: Optional[torch.Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor,
        training: bool = True,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[torch.Tensor] = None,
        k_proj_weight: Optional[torch.Tensor] = None,
        v_proj_weight: Optional[torch.Tensor] = None,
        e_proj: Optional[torch.Tensor] = None,
        f_proj: Optional[torch.Tensor] = None,
        method: str = "learnable",
        static_k: Optional[torch.Tensor] = None,
        static_v: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            embed_dim_to_check: total dimension of the model.
            num_heads: parallel attention heads.
            in_proj_weight, in_proj_bias: input projection weight and bias.
            bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
            bias_e, bias_f: bias of the two linear projection to be added at dim=0.
            add_zero_attn: add a new batch of zeros to the key and
                           value sequences at dim=1.
            dropout_p: probability of an element to be zeroed.
            out_proj_weight, out_proj_bias: the output projection weight and bias.
            training: apply dropout if is ``True``.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.
            use_separate_proj_weight: the function accept the proj. weights for query, key,
                and value in different forms. If false, in_proj_weight will be used, which is
                a combination of q_proj_weight, k_proj_weight, v_proj_weight.
            q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
            e_proj_weight, f_proj_weight: linear projection weight.
            static_k, static_v: static key and value used for attention operators.
        Shape:
            Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
              the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
              If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
              will be unchanged. If a BoolTensor is provided, the positions with the
              value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
              3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
              S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
              positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
              while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
              are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
              is provided, it will be added to the attention weight.
            - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
              N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
            - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
              N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
            Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
              L is the target sequence length, S is the source sequence length.
        """

        # give tensors of shape (time, batch, feature)
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        tgt_len, bsz, embed_dim = query.size()
        if method == "learnable":
            proj_k, seq_len = e_proj.weight.size()

        assert embed_dim == embed_dim_to_check
        # allow MHA to have different sizes for the feature dimension
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = embed_dim // num_heads
        assert (
            head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        if not use_separate_proj_weight:
            if torch.equal(query, key) and torch.equal(key, value):
                # self-attention
                q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(
                    3, dim=-1
                )

            elif torch.equal(key, value):
                # encoder-decoder attention
                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = 0
                _end = embed_dim
                _w = in_proj_weight[_start:_end, :]
                if _b is not None:
                    _b = _b[_start:_end]
                q = linear(query, _w, _b)

                if key is None:
                    assert value is None
                    k = None
                    v = None
                else:

                    # This is inline in_proj function with in_proj_weight and in_proj_bias
                    _b = in_proj_bias
                    _start = embed_dim
                    _end = None
                    _w = in_proj_weight[_start:, :]
                    if _b is not None:
                        _b = _b[_start:]
                    k, v = linear(key, _w, _b).chunk(2, dim=-1)

            else:
                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = 0
                _end = embed_dim
                _w = in_proj_weight[_start:_end, :]
                if _b is not None:
                    _b = _b[_start:_end]
                q = linear(query, _w, _b)

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = embed_dim * 2
                _w = in_proj_weight[_start:_end, :]
                if _b is not None:
                    _b = _b[_start:_end]
                k = linear(key, _w, _b)

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim * 2
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                v = linear(value, _w, _b)

        else:
            q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
            len1, len2 = q_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == query.size(-1)

            k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
            len1, len2 = k_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == key.size(-1)

            v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
            len1, len2 = v_proj_weight_non_opt.size()
            assert len1 == embed_dim and len2 == value.size(-1)
            if in_proj_bias is not None:
                q = linear(
                    query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim]
                )
                k = linear(
                    key,
                    k_proj_weight_non_opt,
                    in_proj_bias[embed_dim : (embed_dim * 2)],
                )
                v = linear(
                    value,
                    v_proj_weight_non_opt,
                    in_proj_bias[(embed_dim * 2) :],
                )
            else:
                q = linear(query, q_proj_weight_non_opt, in_proj_bias)
                k = linear(key, k_proj_weight_non_opt, in_proj_bias)
                v = linear(value, v_proj_weight_non_opt, in_proj_bias)
        q = q * scaling

        if attn_mask is not None:
            assert (
                attn_mask.dtype == torch.float32
                or attn_mask.dtype == torch.float64
                or attn_mask.dtype == torch.float16
                or attn_mask.dtype == torch.uint8
                or attn_mask.dtype == torch.bool
            ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(
                attn_mask.dtype
            )
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
                )
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError(
                        "The size of the 2D attn_mask is not correct."
                    )
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                    bsz * num_heads,
                    query.size(0),
                    key.size(0),
                ]:
                    raise RuntimeError(
                        "The size of the 3D attn_mask is not correct."
                    )
            else:
                raise RuntimeError(
                    "attn_mask's dimension {} is not supported".format(
                        attn_mask.dim()
                    )
                )
            # attn_mask's dim is 3 now.

        # convert ByteTensor key_padding_mask to bool
        if (
            key_padding_mask is not None
            and key_padding_mask.dtype == torch.uint8
        ):
            warnings.warn(
                "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
            )
            key_padding_mask = key_padding_mask.to(torch.bool)

        if bias_k is not None and bias_v is not None:
            if static_k is None and static_v is None:
                k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
                v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
                if attn_mask is not None:
                    attn_mask = pad(attn_mask, (0, 1))
                if key_padding_mask is not None:
                    key_padding_mask = pad(key_padding_mask, (0, 1))
            else:
                assert static_k is None, "bias cannot be added to static key."
                assert static_v is None, "bias cannot be added to static value."
        else:
            assert bias_k is None
            assert bias_v is None
        if method == "learnable":
            k = k.permute(1, 2, 0)
            k = linear(k, e_proj.weight[:, 0:tgt_len], e_proj.bias)
            v = v.permute(1, 2, 0)
            v = linear(v, f_proj.weight[:, 0:tgt_len], f_proj.bias)
        elif method == "convolution":
            k = k.permute(1, 2, 0)
            v = v.permute(1, 2, 0)
            k = e_proj(k)
            v = f_proj(v)

        q = (
            q.contiguous()
            .view(tgt_len, bsz * num_heads, head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * num_heads, head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * num_heads, head_dim)
                .transpose(0, 1)
            )

        if static_k is not None:
            assert static_k.size(0) == bsz * num_heads
            assert static_k.size(2) == head_dim
            k = static_k

        if static_v is not None:
            assert static_v.size(0) == bsz * num_heads
            assert static_v.size(2) == head_dim
            v = static_v

        src_len = k.size(1)

        if add_zero_attn:
            src_len += 1
            k = torch.cat(
                [
                    k,
                    torch.zeros(
                        (k.size(0), 1) + k.size()[2:],
                        dtype=k.dtype,
                        device=k.device,
                    ),
                ],
                dim=1,
            )
            v = torch.cat(
                [
                    v,
                    torch.zeros(
                        (v.size(0), 1) + v.size()[2:],
                        dtype=v.dtype,
                        device=v.device,
                    ),
                ],
                dim=1,
            )
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [
            bsz * num_heads,
            tgt_len,
            src_len,
        ]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        attn_output_weights = softmax(attn_output_weights, dim=-1)
        attn_output_weights = dropout(
            attn_output_weights, p=dropout_p, training=training
        )

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
        attn_output = (
            attn_output.transpose(0, 1)
            .contiguous()
            .view(tgt_len, bsz, embed_dim)
        )
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

        # reshape the output back to (batch, time, feature)
        attn_output = attn_output.permute(1, 0, 2)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            return attn_output, (attn_output_weights.sum(dim=1) / num_heads)
        else:
            return attn_output, None


class LSHAttention(nn.Module):
    """
    This class comes from https://github.com/lucidrains/reformer-pytorch
    It was adjusted to fit SpeechBrain's architecture.
    """

    def __init__(
        self,
        dropout=0.0,
        bucket_size=64,
        n_hashes=8,
        causal=False,
        allow_duplicate_attention=True,
        attend_across_buckets=True,
        rehash_each_round=True,
        drop_for_hash_rate=0.0,
        random_rotations_per_head=False,
        return_attn=False,
    ):
        super().__init__()
        if dropout >= 1.0:
            raise ValueError("Dropout rates must be lower than 1.")

        self.dropout = nn.Dropout(dropout)
        self.dropout_for_hash = nn.Dropout(drop_for_hash_rate)

        assert rehash_each_round or allow_duplicate_attention, (
            "The setting {allow_duplicate_attention=False, rehash_each_round=False}"
            " is not implemented."
        )

        self.causal = causal
        self.bucket_size = bucket_size

        self.n_hashes = n_hashes

        self._allow_duplicate_attention = allow_duplicate_attention
        self._attend_across_buckets = attend_across_buckets
        self._rehash_each_round = rehash_each_round
        self._random_rotations_per_head = random_rotations_per_head

        # will expend extra computation to return attention matrix
        self._return_attn = return_attn

        # cache buckets for reversible network, reported by authors to make Reformer work at depth
        self._cache = {}

    @cache_method_decorator("_cache", "buckets", reexecute=True)
    def hash_vectors(self, n_buckets, vecs):
        batch_size = vecs.shape[0]
        device = vecs.device

        # See https://arxiv.org/pdf/1509.02897.pdf
        # We sample a different random rotation for each round of hashing to
        # decrease the probability of hash misses.
        assert n_buckets % 2 == 0

        rot_size = n_buckets

        rotations_shape = (
            batch_size if self._random_rotations_per_head else 1,
            vecs.shape[-1],
            self.n_hashes if self._rehash_each_round else 1,
            rot_size // 2,
        )

        random_rotations = torch.randn(
            rotations_shape, dtype=vecs.dtype, device=device
        ).expand(batch_size, -1, -1, -1)

        dropped_vecs = self.dropout_for_hash(vecs)
        rotated_vecs = torch.einsum(
            "btf,bfhi->bhti", dropped_vecs, random_rotations
        )

        if self._rehash_each_round:
            # rotated_vectors size [batch,n_hash,seq_len,buckets]
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            buckets = torch.argmax(rotated_vecs, dim=-1)
        else:
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            # In this configuration, we map each item to the top self.n_hashes buckets
            rotated_vecs = torch.squeeze(rotated_vecs, 1)
            bucket_range = torch.arange(rotated_vecs.shape[-1], device=device)
            bucket_range = torch.reshape(bucket_range, (1, -1))
            bucket_range = bucket_range.expand_as(rotated_vecs)

            _, buckets = sort_key_val(rotated_vecs, bucket_range, dim=-1)
            # buckets size [batch size, seq_len, buckets]
            buckets = buckets[..., -self.n_hashes :].transpose(1, 2)

        # buckets is now (self.n_hashes, seq_len). Next we add offsets so that
        # bucket numbers from different hashing rounds don't overlap.
        offsets = torch.arange(self.n_hashes, device=device)
        offsets = torch.reshape(offsets * n_buckets, (1, -1, 1))
        buckets = torch.reshape(buckets + offsets, (batch_size, -1,))
        return buckets

    def forward(
        self,
        qk,
        v,
        query_len=None,
        input_mask=None,
        input_attn_mask=None,
        **kwargs,
    ):
        batch_size, seqlen, dim, device = *qk.shape, qk.device
        query_len = default(query_len, seqlen)
        is_reverse = kwargs.pop("_reverse", False)
        depth = kwargs.pop("_depth", None)

        assert (
            seqlen % (self.bucket_size * 2) == 0
        ), f"Sequence length ({seqlen}) needs to be divisible by target bucket size  x 2 - {self.bucket_size * 2}"

        n_buckets = seqlen // self.bucket_size
        buckets = self.hash_vectors(
            n_buckets,
            qk,
            key_namespace=depth,
            fetch=is_reverse,
            set_cache=self.training,
        )

        # We use the same vector as both a query and a key.
        assert int(buckets.shape[1]) == self.n_hashes * seqlen

        total_hashes = self.n_hashes

        ticker = (
            torch.arange(total_hashes * seqlen, device=device)
            .unsqueeze(0)
            .expand_as(buckets)
        )
        buckets_and_t = seqlen * buckets + (ticker % seqlen)
        buckets_and_t = buckets_and_t.detach()

        # Hash-based sort ("s" at the start of variable names means "sorted")
        sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, dim=-1)
        _, undo_sort = sticker.sort(dim=-1)
        del ticker

        sbuckets_and_t = sbuckets_and_t.detach()
        sticker = sticker.detach()
        undo_sort = undo_sort.detach()

        st = sticker % seqlen
        sqk = batched_index_select(qk, st)
        sv = batched_index_select(v, st)

        # Split off a "bin" axis so that attention only occurs within chunks.
        chunk_size = total_hashes * n_buckets
        bq_t = bkv_t = torch.reshape(st, (batch_size, chunk_size, -1))
        bqk = torch.reshape(sqk, (batch_size, chunk_size, -1, dim))
        bv = torch.reshape(sv, (batch_size, chunk_size, -1, dim))

        # Hashing operates on unit-length vectors. Unnormalized query vectors are
        # fine because they effectively provide a learnable temperature for the
        # attention softmax, but normalizing keys is needed so that similarity for
        # the purposes of attention correctly corresponds to hash locality.
        bq = bqk
        bk = F.normalize(bqk, p=2, dim=-1).type_as(bq)

        # Allow each chunk to attend within itself, and also one chunk back. Chunk
        # boundaries might occur in the middle of a sequence of items from the
        # same bucket, so this increases the chances of attending to relevant items.
        def look_one_back(x):
            x_extra = torch.cat([x[:, -1:, ...], x[:, :-1, ...]], dim=1)
            return torch.cat([x, x_extra], dim=2)

        bk = look_one_back(bk)
        bv = look_one_back(bv)
        bkv_t = look_one_back(bkv_t)

        # Dot-product attention.
        dots = torch.einsum("bhie,bhje->bhij", bq, bk) * (dim ** -0.5)
        masked_value = max_neg_value(dots)

        # Mask for post qk attention logits of the input sequence
        if input_attn_mask is not None:
            input_attn_mask = F.pad(
                input_attn_mask,
                (
                    0,
                    seqlen - input_attn_mask.shape[-1],
                    0,
                    seqlen - input_attn_mask.shape[-2],
                ),
                value=True,
            )
            dot_attn_indices = (bq_t * seqlen)[:, :, :, None] + bkv_t[
                :, :, None, :
            ]
            input_attn_mask = input_attn_mask.reshape(batch_size, -1)
            dot_attn_indices = dot_attn_indices.reshape(batch_size, -1)
            mask = input_attn_mask.gather(1, dot_attn_indices).reshape_as(dots)
            dots.masked_fill_(~mask, masked_value)
            del mask

        # Input mask for padding in variable lengthed sequences
        if input_mask is not None:
            input_mask = F.pad(
                input_mask, (0, seqlen - input_mask.shape[1]), value=True
            )
            mq = input_mask.gather(1, st).reshape((batch_size, chunk_size, -1))
            mkv = look_one_back(mq)
            mask = mq[:, :, :, None] * mkv[:, :, None, :]
            dots.masked_fill_(~mask, masked_value)
            del mask

        # Causal masking
        if self.causal:
            mask = bq_t[:, :, :, None] < bkv_t[:, :, None, :]
            if seqlen > query_len:
                mask = mask & (bkv_t[:, :, None, :] < query_len)
            dots.masked_fill_(mask, masked_value)
            del mask

        # Mask out attention to self except when no other targets are available.
        self_mask = bq_t[:, :, :, None] == bkv_t[:, :, None, :]
        dots.masked_fill_(self_mask, TOKEN_SELF_ATTN_VALUE)
        del self_mask

        # Mask out attention to other hash buckets.
        if not self._attend_across_buckets:
            bq_buckets = bkv_buckets = torch.reshape(
                sbuckets_and_t // seqlen, (batch_size, chunk_size, -1)
            )
            bkv_buckets = look_one_back(bkv_buckets)
            bucket_mask = (
                bq_buckets[:, :, :, None] != bkv_buckets[:, :, None, :]
            )
            dots.masked_fill_(bucket_mask, masked_value)
            del bucket_mask

        # Don't double-count query-key pairs across multiple rounds of hashing.
        # There are two possible strategies here. (1) The default is to count how
        # many times a query-key pair is repeated, and to lower its log-prob
        # correspondingly at each repetition. (2) When hard_k is set, the code
        # instead masks all but the first occurence of each query-key pair.
        if not self._allow_duplicate_attention:
            locs1 = undo_sort // bq_t.shape[-1]
            locs2 = (locs1 + 1) % chunk_size
            if not self._attend_across_buckets:
                locs1 = buckets * chunk_size + locs1
                locs2 = buckets * chunk_size + locs2
            locs = torch.cat(
                [
                    torch.reshape(locs1, (batch_size, total_hashes, seqlen)),
                    torch.reshape(locs2, (batch_size, total_hashes, seqlen)),
                ],
                1,
            ).permute((0, 2, 1))

            slocs = batched_index_select(locs, st)
            b_locs = torch.reshape(
                slocs, (batch_size, chunk_size, -1, 2 * total_hashes)
            )

            b_locs1 = b_locs[:, :, :, None, :total_hashes]

            bq_locs = b_locs1.expand(b_locs.shape[:3] + (2, total_hashes))
            bq_locs = torch.reshape(bq_locs, b_locs.shape)
            bkv_locs = look_one_back(b_locs)

            dup_counts = bq_locs[:, :, :, None, :] == bkv_locs[:, :, None, :, :]
            # for memory considerations, chunk summation of last dimension for counting duplicates
            dup_counts = chunked_sum(
                dup_counts, chunks=(total_hashes * batch_size)
            )
            dup_counts = dup_counts.detach()
            assert dup_counts.shape == dots.shape
            dots = dots - torch.log(dup_counts + 1e-9)
            del dup_counts

        # Softmax.
        dots_logsumexp = torch.logsumexp(dots, dim=-1, keepdim=True)
        dots = torch.exp(dots - dots_logsumexp).type_as(dots)
        dropped_dots = self.dropout(dots)

        bo = torch.einsum("buij,buje->buie", dropped_dots, bv)
        so = torch.reshape(bo, (batch_size, -1, dim))
        slogits = torch.reshape(dots_logsumexp, (batch_size, -1,))

        # unsort logits
        o = batched_index_select(so, undo_sort)
        logits = slogits.gather(1, undo_sort)

        o = torch.reshape(o, (batch_size, total_hashes, seqlen, dim))
        logits = torch.reshape(logits, (batch_size, total_hashes, seqlen, 1))

        if query_len != seqlen:
            query_slice = (slice(None), slice(None), slice(0, query_len))
            o, logits = o[query_slice], logits[query_slice]

        probs = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))
        out = torch.sum(o * probs, dim=1)

        attn = torch.empty(0, device=device)

        # return unsorted attention weights
        if self._return_attn:
            attn_unsort = (bq_t * seqlen)[:, :, :, None] + bkv_t[:, :, None, :]
            attn_unsort = attn_unsort.view(batch_size * total_hashes, -1).long()
            unsorted_dots = torch.zeros(
                batch_size * total_hashes, seqlen * seqlen, device=device
            )
            unsorted_dots.scatter_add_(
                1, attn_unsort, dots.view_as(attn_unsort)
            )
            del attn_unsort
            unsorted_dots = unsorted_dots.reshape(
                batch_size, total_hashes, seqlen, seqlen
            )
            attn = torch.sum(unsorted_dots[:, :, 0:query_len, :] * probs, dim=1)

        # return output, attention matrix, and bucket distribution
        return out, attn, buckets


class LSHSelfAttention(nn.Module):
    """
    This class comes from https://github.com/lucidrains/reformer-pytorch
    It was adjusted to fit SpeechBrain's architecture.
    """

    def __init__(
        self,
        dim,
        heads=8,
        bucket_size=64,
        n_hashes=8,
        causal=False,
        dim_head=None,
        attn_chunks=1,
        random_rotations_per_head=False,
        attend_across_buckets=True,
        allow_duplicate_attention=True,
        num_mem_kv=0,
        one_value_head=False,
        use_full_attn=False,
        full_attn_thres=None,
        return_attn=False,
        post_attn_dropout=0.0,
        dropout=0.0,
        n_local_attn_heads=0,
        **kwargs,
    ):
        super().__init__()
        assert (
            dim_head or (dim % heads) == 0
        ), "dimensions must be divisible by number of heads"
        assert (
            n_local_attn_heads < heads
        ), "local attention heads must be less than number of heads"

        dim_head = default(dim_head, dim // heads)
        dim_heads = dim_head * heads

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.attn_chunks = default(attn_chunks, 1)

        self.v_head_repeats = heads if one_value_head else 1
        v_dim = dim_heads // self.v_head_repeats

        self.toqk = nn.Linear(dim, dim_heads, bias=False)
        self.tov = nn.Linear(dim, v_dim, bias=False)
        self.to_out = nn.Linear(dim_heads, dim)

        self.bucket_size = bucket_size
        self.lsh_attn = LSHAttention(
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal,
            random_rotations_per_head=random_rotations_per_head,
            attend_across_buckets=attend_across_buckets,
            allow_duplicate_attention=allow_duplicate_attention,
            return_attn=return_attn,
            dropout=dropout,
            **kwargs,
        )
        self.full_attn = FullQKAttention(causal=causal, dropout=dropout)
        self.post_attn_dropout = nn.Dropout(post_attn_dropout)

        self.use_full_attn = use_full_attn
        self.full_attn_thres = default(full_attn_thres, bucket_size)

        self.num_mem_kv = num_mem_kv
        self.mem_kv = (
            nn.Parameter(torch.randn(1, num_mem_kv, dim, requires_grad=True))
            if num_mem_kv > 0
            else None
        )

        self.n_local_attn_heads = n_local_attn_heads
        self.local_attn = LocalAttention(
            window_size=bucket_size * 2,
            causal=causal,
            dropout=dropout,
            shared_qk=True,
            look_forward=(1 if not causal else 0),
        )

        self.callback = None

    def forward(
        self,
        x,
        keys=None,
        input_mask=None,
        input_attn_mask=None,
        context_mask=None,
        **kwargs,
    ):
        device, dtype = x.device, x.dtype
        b, t, e, h, dh, m, l_h = (
            *x.shape,
            self.heads,
            self.dim_head,
            self.num_mem_kv,
            self.n_local_attn_heads,
        )

        mem_kv = default(
            self.mem_kv, torch.empty(b, 0, e, dtype=dtype, device=device)
        )
        mem = mem_kv.expand(b, m, -1)

        keys = default(keys, torch.empty(b, 0, e, dtype=dtype, device=device))
        c = keys.shape[1]

        kv_len = t + m + c
        use_full_attn = self.use_full_attn or kv_len <= self.full_attn_thres

        x = torch.cat((x, mem, keys), dim=1)
        qk = self.toqk(x)
        v = self.tov(x)
        v = v.repeat(1, 1, self.v_head_repeats)

        def merge_heads(v):
            return v.view(b, kv_len, h, -1).transpose(1, 2)

        def split_heads(v):
            return v.view(b, h, t, -1).transpose(1, 2).contiguous()

        merge_batch_and_heads = partial(merge_dims, 0, 1)

        qk, v = map(merge_heads, (qk, v))

        has_local = l_h > 0
        lsh_h = h - l_h

        split_index_fn = partial(split_at_index, 1, l_h)
        (lqk, qk), (lv, v) = map(split_index_fn, (qk, v))
        lqk, qk, lv, v = map(merge_batch_and_heads, (lqk, qk, lv, v))

        masks = {}
        if input_mask is not None or context_mask is not None:
            default_mask = torch.tensor([True], device=device)
            i_mask = default(input_mask, default_mask.expand(b, t))
            m_mask = default_mask.expand(b, m)
            c_mask = default(context_mask, default_mask.expand(b, c))
            mask = torch.cat((i_mask, m_mask, c_mask), dim=1)
            mask = merge_batch_and_heads(expand_dim(1, lsh_h, mask))
            masks["input_mask"] = mask

        if input_attn_mask is not None:
            input_attn_mask = merge_batch_and_heads(
                expand_dim(1, lsh_h, input_attn_mask)
            )
            masks["input_attn_mask"] = input_attn_mask

        attn_fn = self.lsh_attn if not use_full_attn else self.full_attn
        partial_attn_fn = partial(attn_fn, query_len=t, **kwargs)
        attn_fn_in_chunks = process_inputs_chunk(
            partial_attn_fn, chunks=self.attn_chunks
        )

        out, attn, buckets = attn_fn_in_chunks(qk, v, **masks)

        if self.callback is not None:
            self.callback(
                attn.reshape(b, lsh_h, t, -1), buckets.reshape(b, lsh_h, -1)
            )

        if has_local:
            lqk, lv = lqk[:, :t], lv[:, :t]
            local_out = self.local_attn(lqk, lqk, lv, input_mask=input_mask)
            local_out = local_out.reshape(b, l_h, t, -1)
            out = out.reshape(b, lsh_h, t, -1)
            out = torch.cat((local_out, out), dim=1)

        out = split_heads(out).view(b, t, -1)
        out = self.to_out(out)
        return self.post_attn_dropout(out), attn
