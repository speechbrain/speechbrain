"""Library implementing attention modules.

Authors
 * Ju-Chieh Chou 2020
 * Jianyuan Zhong 2020
 * Loren Lugosch 2020
 * Samuele Cornell 2020
 * Shucong Zhang 2024

"""

import math
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from speechbrain.dataio.dataio import length_to_mask
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class ContentBasedAttention(nn.Module):
    """This class implements content-based attention module for seq2seq
    learning.

    Reference: NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN
    AND TRANSLATE, Bahdanau et.al. https://arxiv.org/pdf/1409.0473.pdf

    Arguments
    ---------
    enc_dim : int
        Size of encoder layer.
    dec_dim : int
        Size of decoder layer.
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
    >>> net = ContentBasedAttention(
    ...     enc_dim=20, dec_dim=25, attn_dim=30, output_dim=5
    ... )
    >>> out_tensor, out_weight = net(enc_tensor, enc_len, dec_tensor)
    >>> out_tensor.shape
    torch.Size([4, 5])
    """

    def __init__(self, enc_dim, dec_dim, attn_dim, output_dim, scaling=1.0):
        super().__init__()

        self.mlp_enc = nn.Linear(enc_dim, attn_dim)
        self.mlp_dec = nn.Linear(dec_dim, attn_dim)
        self.mlp_attn = nn.Linear(attn_dim, 1, bias=False)
        self.mlp_out = nn.Linear(enc_dim, output_dim)

        self.scaling = scaling

        self.softmax = nn.Softmax(dim=-1)

        # reset the encoder states, lengths and masks
        self.reset()

    def reset(self):
        """Reset the memory in the attention module."""
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

        Returns
        -------
        The output of the attention module.
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
    enc_dim : int
        Size of encoder.
    dec_dim : int
        Size of decoder.
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
    ...     kernel_size=100,
    ... )
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
        super().__init__()

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
        """Reset the memory in attention module."""
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

        Returns
        -------
        The output of the attention module.
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
    """This class implements a single-headed key-value attention module for seq2seq
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
    >>> net = KeyValueAttention(
    ...     enc_dim=20, dec_dim=25, attn_dim=30, output_dim=5
    ... )
    >>> out_tensor, out_weight = net(enc_tensor, enc_len, dec_tensor)
    >>> out_tensor.shape
    torch.Size([4, 5])
    """

    def __init__(self, enc_dim, dec_dim, attn_dim, output_dim):
        super().__init__()

        self.key_linear = nn.Linear(enc_dim, attn_dim)
        self.query_linear = nn.Linear(dec_dim, attn_dim)
        self.value_linear = nn.Linear(enc_dim, output_dim)
        self.scaling = torch.sqrt(torch.tensor(attn_dim).float())

        # reset the encoder states, lengths and masks
        self.reset()

    def reset(self):
        """Reset the memory in the attention module."""
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

        Returns
        -------
        The output of the attention module.
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


class RelPosEncXL(nn.Module):
    """Relative positional encoding for the :class:`~RelPosMHAXL`.

    Arguments
    ---------
    emb_dim : int
        Size of the embedding, which controls the size of the last dimension
        of the positional embedding as well
    dtype : torch.dtype, optional
        If unspecified, defaults to `torch.float32`. Controls the data type of
        the output embedding (but does not affect the precision of the
        computations, which remain `torch.float32`).
    """

    def __init__(self, emb_dim: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.emb_dim = emb_dim

        inv_freq = torch.exp(
            torch.arange(0, self.emb_dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.emb_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        self.emb_dtype = dtype

    @torch.no_grad()
    def make_pe(self, seq_len: int):
        """
        Builds the positional embedding tensor for a given sequence length.

        Arguments
        ---------
        seq_len : int
            The length of the sequence to create the position embedding for.

        Returns
        -------
        torch.Tensor
            Positional embedding tensor of shape `[1, 2*seq_len-1, embed_dim]`
        """

        emb_dtype = self.emb_dtype
        device = self.inv_freq.device

        with torch.no_grad():
            # perform initialization with the same type as `inv_freq`, to enable
            # migrating the embeddings to fp16 by calling
            # `posenc.to(torch.float16)`

            tot_pe = torch.empty(
                (2, seq_len, self.emb_dim),
                dtype=torch.float32,
                device=device,
            )
            pe_past = tot_pe[0]
            pe_future = tot_pe[1]
            positions = torch.arange(
                0,
                seq_len,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(-1)

            sinusoids = torch.sin(positions * self.inv_freq)
            pe_past[:, 0::2] = sinusoids
            pe_past[:, 1::2] = torch.cos(positions * self.inv_freq)
            pe_future[:, 0::2] = sinusoids  # same for past and future
            pe_future[:, 1::2] = torch.cos(-positions * self.inv_freq)

            pe_past = torch.flip(pe_past, (0,)).unsqueeze(0)
            pe_future = pe_future[1:].unsqueeze(0)
            pe = torch.cat([pe_past, pe_future], dim=1)
            pe = pe.to(emb_dtype)  # convert to type of module

        return pe

    def forward(self, x: torch.Tensor):
        """
        Builds the positional embedding tensor. Similar to
        :meth:`~RelPosEncXL.make_pe` but uses the shape information from the
        provided tensor.

        Arguments
        ---------
        x : torch.Tensor
            input tensor with shape batch_size, seq_len, embed_dim

        Returns
        -------
        pos_emb : torch.Tensor
            Positional embedding tensor of shape `[1, 2*seq_len-1, embed_dim]`
        """

        return self.make_pe(seq_len=x.size(1))


class RelPosMHAXL(nn.Module):
    """This class implements the relative multihead implementation similar to that in Transformer XL
    https://arxiv.org/pdf/1901.02860.pdf

    Arguments
    ---------
    embed_dim : int
        Size of the encoder feature vectors from which keys and values are computed.
    num_heads: int
        Number of attention heads.
    dropout : float, optional
        Dropout rate.
    vbias: bool, optional
        Whether to use bias for computing value.
    vdim: int, optional
        Size for value. Default is embed_dim (Note each head is embed_dim // num_heads).
    mask_pos_future: bool, optional
        Whether to mask future positional encodings values.
        Must be true for causal applications e.g. decoder.

    Example
    -------
    >>> inputs = torch.rand([6, 60, 512])
    >>> pos_emb = torch.rand([1, 2 * 60 - 1, 512])
    >>> net = RelPosMHAXL(num_heads=8, embed_dim=inputs.shape[-1])
    >>> outputs, attn = net(inputs, inputs, inputs, pos_emb)
    >>> outputs.shape
    torch.Size([6, 60, 512])
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        vbias=False,
        vdim=None,
        mask_pos_future=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.vdim == embed_dim
        self.mask_pos_future = mask_pos_future
        self.vbias = vbias

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.vhead_dim = self.vdim // num_heads

        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )
        assert self.vhead_dim * num_heads == self.vdim, (
            "vdim must be divisible by num_heads"
        )

        if self._qkv_same_embed_dim is False:
            self.qk_proj_weight = nn.Parameter(
                torch.empty(2 * embed_dim, embed_dim)
            )
            self.v_proj_weight = nn.Parameter(torch.empty(self.vdim, embed_dim))
        else:
            self.in_proj_weight = nn.Parameter(
                torch.empty(3 * embed_dim, embed_dim)
            )

        if vbias:
            self.value_bias_weight = nn.Parameter(torch.empty(self.vdim))
        else:
            self.vbias = None

        self.dropout_att = nn.Dropout(dropout)
        self.out_proj = nn.Linear(self.vdim, embed_dim)

        self.linear_pos = nn.Linear(embed_dim, embed_dim, bias=False)

        self.pos_bias_u = nn.Parameter(
            torch.empty(self.head_dim, self.num_heads)
        )
        self.pos_bias_v = nn.Parameter(
            torch.empty(self.head_dim, self.num_heads)
        )

        if next(self.parameters()).dtype == torch.float16:
            self.attn_fill_value = -65000
        else:
            self.attn_fill_value = -float("inf")

        self._reset_parameters()
        self.scale = 1 / math.sqrt(self.embed_dim)

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            torch.nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            torch.nn.init.xavier_uniform_(self.qk_proj_weight)
            torch.nn.init.xavier_uniform_(self.v_proj_weight)

        if self.vbias is not None:
            torch.nn.init.constant_(self.value_bias_weight, 0.0)

        # positional biases
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Relative shift implementation."""
        # batch, head, time1, 2*time1-1.

        b, h, qlen, pos_len = x.size()  # (b, h, t1, t2)
        # need to add a column of zeros on the left side of last dimension to perform the relative shifting
        x = torch.nn.functional.pad(x, pad=(1, 0))  # (b, h, t1, t2+1)
        x = x.view(b, h, -1, qlen)  # (b, h, t2+1, t1)
        # need to drop the first row
        x = x[:, :, 1:].view(b, h, qlen, pos_len)  # (b, h, t1, t2)

        # cspell:ignore tril
        if self.mask_pos_future:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x[..., : pos_len // 2 + 1]

    def forward(
        self,
        query,
        key,
        value,
        pos_embs,
        key_padding_mask=None,
        attn_mask=None,
        return_attn_weights=True,
    ):
        """Compute attention.

        Arguments
        ---------
        query : torch.Tensor
            (B, L, E) where L is the target sequence length,
            B is the batch size, E is the embedding dimension.
        key : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        value : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        pos_embs : torch.Tensor
            bidirectional sinusoidal positional embedding tensor (1, 2*S-1, E) where S is the max length between source and target sequence lengths,
            and E is the embedding dimension.
        key_padding_mask : torch.Tensor
            (B, S) where B is the batch size, S is the source sequence
            length. If a ByteTensor is provided, the non-zero positions will
            be ignored while the position with the zero positions will be
            unchanged. If a BoolTensor is provided, the positions with the
            value of True will be ignored while the position with the value
            of False will be unchanged.
        attn_mask : torch.Tensor
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
        return_attn_weights : bool
            Whether to additionally return the attention weights.

        Returns
        -------
        out : torch.Tensor
            (B, L, E) where L is the target sequence length, B is the
            batch size, E is the embedding dimension.
        attn_score : torch.Tensor
            (B, L, S) where B is the batch size, L is the target
            sequence length, S is the source sequence length.
        """

        # query, key and value are of shape batch, time, embed_dim
        bsz = query.shape[0]
        klen = key.shape[1]
        qlen = query.shape[1]

        if self._qkv_same_embed_dim:
            # self-attention
            if (query is key or torch.equal(query, key)) and (
                key is value or torch.equal(key, value)
            ):
                query, key, value = (
                    nn.functional.linear(query, self.in_proj_weight)
                    .view(bsz, -1, self.num_heads, self.head_dim * 3)
                    .chunk(3, dim=-1)
                )
            else:
                qweight, kweight, vweight = self.in_proj_weight.chunk(3, dim=0)
                query = nn.functional.linear(query, qweight).view(
                    bsz, -1, self.num_heads, self.head_dim
                )
                key = nn.functional.linear(key, kweight).view(
                    bsz, -1, self.num_heads, self.head_dim
                )
                value = nn.functional.linear(value, vweight).view(
                    bsz, -1, self.num_heads, self.head_dim
                )
        else:
            raise NotImplementedError
            query, key = (
                nn.functional.linear(query, self.qk_proj_weight)
                .view(bsz, -1, self.num_heads, self.head_dim * 2)
                .chunk(2, dim=-1)
            )
            value = nn.functional.linear(value, self.v_proj_weight).view(
                bsz, -1, self.num_heads, self.vhead_dim
            )

        if self.vbias is not None:
            value = value + self.value_bias_weight.view(
                1, 1, self.num_heads, self.vhead_dim
            )

        p_k = self.linear_pos(pos_embs).view(
            1, -1, self.num_heads, self.head_dim
        )
        # (batch, head, klen, d_k)

        q_with_bias_u = (
            query + self.pos_bias_u.view(1, 1, self.num_heads, self.head_dim)
        ).transpose(1, 2)
        # (batch, head, qlen, d_k)
        q_with_bias_v = (
            query + self.pos_bias_v.view(1, 1, self.num_heads, self.head_dim)
        ).transpose(1, 2)

        # Moved the `* self.scale` mul from after the `attn_score` sum to prior
        # to the matmul in order to lower overflow risks on fp16.
        # This change is inspired by the following paper, but no other changes
        # were ported from there so far.
        # ref: E.T.: Re-Thinking Self-Attention for Transformer Models on GPUs
        # https://asherliu.github.io/docs/sc21a.pdf

        # (batch, head, qlen, klen)
        matrix_ac = torch.matmul(
            q_with_bias_u * self.scale, key.permute(0, 2, 3, 1)
        )
        # (batch, num_heads, klen, 2*klen-1)
        matrix_bd = torch.matmul(
            q_with_bias_v * self.scale, p_k.permute(0, 2, 3, 1)
        )
        matrix_bd = self.rel_shift(matrix_bd)  # shifting trick

        # if klen != qlen:
        #   import ipdb
        #  ipdb.set_trace(

        attn_score = matrix_ac + matrix_bd  # already scaled above

        # compute attention probability
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.view(1, 1, qlen, klen)
            else:
                attn_mask = attn_mask.view(-1, self.num_heads, qlen, klen)

            if attn_mask.dtype == torch.bool:
                attn_score = attn_score.masked_fill(
                    attn_mask, self.attn_fill_value
                )
            else:
                attn_score += attn_mask

        if key_padding_mask is not None:
            attn_score = attn_score.masked_fill(
                key_padding_mask.view(bsz, 1, 1, klen),
                self.attn_fill_value,
            )

        attn_score = F.softmax(attn_score, dim=-1, dtype=torch.float32)
        attn_score = self.dropout_att(attn_score)

        # it is possible for us to hit full NaN when using chunked training
        # so reapply masks, except with 0.0 instead as we are after the softmax
        # because -inf would output 0.0 regardless anyway
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_score = attn_score.masked_fill(attn_mask, 0.0)
            else:
                # NOTE: the above fix is not implemented for this case as
                # summing the mask with NaN would still result in NaN
                pass

        if key_padding_mask is not None:
            attn_score = attn_score.masked_fill(
                key_padding_mask.view(bsz, 1, 1, klen),
                0.0,
            )

        x = torch.matmul(
            attn_score, value.transpose(1, 2)
        )  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(bsz, -1, self.vhead_dim * self.num_heads)
        )  # (batch, time1, d_model)

        out = self.out_proj(x)
        if return_attn_weights:
            return out, attn_score
        return out


class MultiheadAttention(nn.Module):
    """The class is a wrapper of MultiHead Attention for torch.nn.MultiHeadAttention.

    Reference: https://pytorch.org/docs/stable/nn.html

    Arguments
    ---------
    nhead : int
        parallel attention heads.
    d_model : int
        The size of the model layers.
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
        return_attn_weights: bool = True,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """Compute attention.

        Arguments
        ---------
        query : torch.Tensor
            (B, L, E) where L is the target sequence length,
            B is the batch size, E is the embedding dimension.
        key : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        value : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        attn_mask : torch.Tensor, optional
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
        key_padding_mask : torch.Tensor, optional
            (B, S) where B is the batch size, S is the source sequence
            length. If a ByteTensor is provided, the non-zero positions will
            be ignored while the position with the zero positions will be
            unchanged. If a BoolTensor is provided, the positions with the
            value of True will be ignored while the position with the value
            of False will be unchanged.
        return_attn_weights : bool, optional
            True to additionally return the attention weights, False otherwise.
        pos_embs : torch.Tensor, optional
            Positional embeddings added to the attention map of shape (L, S, E) or (L, S, 1).

        Returns
        -------
        attn_output : torch.Tensor
            (B, L, E) where L is the target sequence length, B is the
            batch size, E is the embedding dimension.
        attn_output_weights : torch.Tensor
            (B, L, S) where B is the batch size, L is the target
            sequence length, S is the source sequence length.
            This is returned only if `return_attn_weights=True` (True by default).
        """
        # give tensors of shape (time, batch, fea)
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        # this will be legit because of https://github.com/pytorch/pytorch/blob/5288d05cfdda85c46c4df84617fa7f37c21b10b3/torch/nn/functional.py#L4946
        # we can inject relative learnable pos embeddings directly in MHA via the attn_mask
        if pos_embs is not None:
            if attn_mask is not None:
                attn_mask += pos_embs
            else:
                attn_mask = pos_embs

        output, attention_weights = self.att(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn_weights,
        )

        # reshape the output back to (batch, time, fea)
        output = output.permute(1, 0, 2)

        if return_attn_weights:
            return output, attention_weights

        return output


class PositionalwiseFeedForward(nn.Module):
    """The class implements the positional-wise feed forward module in
    “Attention Is All You Need”.

    Arguments
    ---------
    d_ffn: int
        Hidden layer size.
    input_shape : tuple, optional
        Expected shape of the input. Alternatively use ``input_size``.
    input_size : int, optional
        Expected size of the input. Alternatively use ``input_shape``.
    dropout: float, optional
        Dropout rate.
    activation: torch.nn.Module, optional
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
        dropout=0.0,
        activation: type = nn.ReLU,
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
        """Applies PositionalwiseFeedForward to the input tensor x."""
        # give a tensor of shape (time, batch, fea)
        x = x.permute(1, 0, 2)
        x = self.ffn(x)

        # reshape the output back to (batch, time, fea)
        x = x.permute(1, 0, 2)

        return x


class PrecomputedRoPESinusoids(nn.Module):
    """
    A cache for the sines and cosines needed to rotate the vectors for rotary
    position embeddings (RoPE).
    This stores the nonzero entries from eq(15) from
    https://arxiv.org/pdf/2104.09864

    Arguments
    ---------
    max_length : int
        The allowed max length of the input sequence.
        For a fixed setting of the other arguments, the computation takes
        O(max_length) time.
    input_size : int
        Size of each vector in the input sequence, i.e. the dimension of each
        attention head.
    dtype : torch.dtype
        The dtype of the tensors.
    device : torch.device
        The Torch device to put the tensors on.

    Example
    -------
    >>> precomputed = PrecomputedRoPESinusoids(
    ...     3, 8, torch.float32, torch.device("cpu")
    ... )
    >>> precomputed.cosines.shape
    torch.Size([3, 8])
    >>> precomputed.sines.shape == precomputed.cosines.shape
    True
    >>> precomputed.cosines
    tensor([[ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000],
            [ 0.5403,  0.5403,  0.9950,  0.9950,  0.9999,  0.9999,  1.0000,  1.0000],
            [-0.4161, -0.4161,  0.9801,  0.9801,  0.9998,  0.9998,  1.0000,  1.0000]])
    >>> precomputed.sines
    tensor([[-0.0000,  0.0000, -0.0000,  0.0000, -0.0000,  0.0000, -0.0000,  0.0000],
            [-0.8415,  0.8415, -0.0998,  0.0998, -0.0100,  0.0100, -0.0010,  0.0010],
            [-0.9093,  0.9093, -0.1987,  0.1987, -0.0200,  0.0200, -0.0020,  0.0020]])
    >>> precomputed.index_swap
    tensor([1, 0, 3, 2, 5, 4, 7, 6])
    """

    def __init__(
        self,
        max_length: int,
        input_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()

        # To precompute the values, use at least float32, because
        # otherwise final accuracy is unnecessarily dreadful.
        internal_dtype = (
            torch.float64 if dtype == torch.float64 else torch.float32
        )

        assert (input_size % 2) == 0

        self.max_length = max_length

        # 10000**(-2(i-1)/d) for i in [1,2,...,d/2]
        angles = torch.exp(
            torch.arange(0, input_size, 2, dtype=internal_dtype, device=device)
            * -(math.log(10000.0) / input_size)
        )

        dimensions = torch.arange(input_size, device=device)

        times = torch.arange(0, max_length, dtype=internal_dtype, device=device)

        # equation (15) without zeros in the matrix
        times_angles = torch.outer(times, angles)

        # Construct
        #     [cos(theta_0), cos(theta_0), cos(theta_1), cos(theta_1), ... ]
        # for equation (34)
        cosines = torch.cos(times_angles)
        cosines = torch.stack([cosines, cosines], dim=-1).reshape(
            max_length, input_size
        )

        # Construct
        #     [sin(theta_0), -sin(theta_0), sin(theta_1), -sin(theta_1), ... ]
        # for equation (34)
        unsigned_sines = torch.sin(times_angles)
        unsigned_repeated_sines = torch.stack(
            [unsigned_sines, unsigned_sines], dim=-1
        ).reshape(max_length, input_size)

        sines = (
            (-1)
            ** torch.arange(input_size, dtype=internal_dtype, device=device)
        ) * -unsigned_repeated_sines

        # To perform a 2-d rotation of every pair of dimensions, a vector will
        # need to be created with every pair swapped with each other.
        # To make this easy, swap every pair of indices:
        # [1, 0, 3, 2, 5, 4, 7, 6, ...]
        index_swap = torch.stack(
            [dimensions[1::2], dimensions[::2]], dim=-1
        ).reshape(-1)

        self.register_buffer("cosines", cosines.to(dtype))
        self.register_buffer("sines", sines.to(dtype))
        self.register_buffer("index_swap", index_swap)


class MemoiseAtLeastSize:
    """
    Memoises a function which has as its first argument a value that indicates a
    minimum value to call the underlying function with.

    Arguments
    ---------
    function: Callable
        The function to call.
    round_up: Callable[[Any], Any]
        A function that rounds up.
        The fewer values this rounds up to, the less likely it is that the
        function will be called repeatedly.
    """

    def __init__(self, function: Callable, round_up: Callable[[Any], Any]):
        self.function = function
        self.round_up = round_up
        # A memo from (parameters 2, 3, ...) to (parameter_1_rounded, result)
        # that stores the result of the call to
        # function(parameter_1_rounded, parameters 2, 3, ...).
        self.memo: Dict[tuple, Tuple[Any, Any]] = {}

    def __call__(self, size: Any, *args):
        if args not in self.memo or self.memo[args][0] < size:
            rounded_size = self.round_up(size)
            assert not (rounded_size < size)
            self.memo[args] = rounded_size, self.function(rounded_size, *args)
        return self.memo[args][1]


def memoise_at_least(
    round_up: Callable[[Any], Any],
) -> Callable[[Callable], MemoiseAtLeastSize]:
    """
    Decorator that memoises a function which has as its first argument a value
    that indicates a minimum value to call the underlying function with.
    If the memo has stored the result from a matching previous function call,
    The stored result will be returned instead of calling the function again.

    Arguments
    ---------
    round_up: Callable[[Any], Any]
        A function that rounds up.
        This will be called with the first argument passed in.
        The underlying function will receive, instead of this first argument,
        the rounded-up version.
        The fewer values this rounds up to, the less likely it is that the
        function will be called repeatedly.

    Returns
    -------
    The passed function but with MemoiseAtLeastSize capability.
    """

    def with_function(function: Callable) -> MemoiseAtLeastSize:
        """
        Set the function to be memoised.
        """
        return MemoiseAtLeastSize(function, round_up)

    return with_function


@memoise_at_least(lambda length: 2 ** int(math.ceil(math.log2(length))))
def _get_precomputed_values(
    length: int, input_size: int, dtype: torch.dtype, device: torch.device
) -> PrecomputedRoPESinusoids:
    """
    Return an object of type PrecomputedRoPESinusoids that is valid for the
    length, input_size, dtype and device.
    Consider a single (input_size, dtype, device), which are usually fixed for
    one model.
    The sinusoids will be recomputed only if they are not yet available for such
    a long length (because of the decorator applied to the function).
    Each time they are precomputed, the length is rounded up to the next power
    of two.

    As a consequence, the total number of calls during one program run is
    upper-bounded by ceil(log2(max_length)) where max_length is the highest
    length that is seen in the program run.
    On realistic lengths, the total number of calls is likely only a few.
    The total number of time steps for which sinusoids are precomputed during
    the program run is O(max_length).

    Arguments
    ---------
    length : int
        The length of the input sequence.
    input_size : int
        Size of each vector in the input sequence, i.e. the dimension of each
        attention head.
    dtype : torch.dtype
        The dtype of the tensors.
    device : torch.device
        The Torch device to put the tensors on.

    Return
    ------
    An object of type PrecomputedRoPESinusoids that is valid for the length,
    input_size, dtype and device.
    """
    # length should have been rounded up to the nearest power of two by the
    # decorator.
    length_power = int(round(math.log2(length)))
    assert length == 2**length_power
    return PrecomputedRoPESinusoids(length, input_size, dtype, device)


def _rope_rotate(x):
    """
    Perform the rotation for RoPE on each of the vectors in x.
    Details about RoPE: https://arxiv.org/pdf/2104.09864.
    """
    _batch_size, length, _num_heads, head_dim = x.shape

    assert (head_dim % 2) == 0

    precomputed = _get_precomputed_values(length, head_dim, x.dtype, x.device)

    # Cut the sinusoids down to the correct length.
    cosines = precomputed.cosines[:length]
    sines = precomputed.sines[:length]

    # The fast implementation for pair-wise rotation requires a version of x
    # with the elements of each pair swapped.
    # (34) in https://arxiv.org/pdf/2104.09864.
    swapped_pairs = torch.index_select(x, dim=-1, index=precomputed.index_swap)

    # (batch_size, L, num_heads, head_dim) * (L, 1, hdead_dim)
    return x * cosines.unsqueeze(1) + swapped_pairs * sines.unsqueeze(1)


class RoPEMHA(nn.Module):
    """This is an implementation of multihead self-attention with RoPE positional embeddings. As it relies on Torch for self-attention, it is
    significantly faster than RelPosMHAXL while offering the same or better levels of accuracy.

    Details about RoPE: https://arxiv.org/pdf/2104.09864.


    Arguments
    ---------
    embed_dim : int
        Size of the encoder feature vectors from which keys and values are computed.
    num_heads: int
        Number of attention heads.
    dropout : float, optional
        Dropout rate.
    vbias: bool, optional
        Whether to use bias for computing value.
    vdim: int, optional
        Size for value. Default is embed_dim (Note each head is embed_dim // num_heads).

    Example
    -------
    >>> max_len = 64
    >>> inputs = torch.rand([6, 60, 512])
    >>> num_heads = 8
    >>> net = RoPEMHA(num_heads=num_heads, embed_dim=inputs.shape[-1])
    >>> outputs, attn = net(inputs, inputs, inputs)
    >>> outputs.shape
    torch.Size([6, 60, 512])
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        vbias=False,
        vdim=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.vdim == embed_dim
        self.vbias = vbias

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.vhead_dim = self.vdim // num_heads

        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )
        assert self.vhead_dim * num_heads == self.vdim, (
            "vdim must be divisible by num_heads"
        )

        if self._qkv_same_embed_dim is False:
            self.qk_proj_weight = nn.Parameter(
                torch.empty(2 * embed_dim, embed_dim)
            )
            self.v_proj_weight = nn.Parameter(torch.empty(self.vdim, embed_dim))
        else:
            self.in_proj_weight = nn.Parameter(
                torch.empty(3 * embed_dim, embed_dim)
            )

        if vbias:
            self.value_bias_weight = nn.Parameter(torch.empty(self.vdim))
        else:
            self.vbias = None

        self.out_proj = nn.Linear(self.vdim, embed_dim)

        if next(self.parameters()).dtype == torch.float16:
            self.attn_fill_value = -65000
        else:
            self.attn_fill_value = -float("inf")

        self._reset_parameters()

        self.scale = 1 / math.sqrt(self.embed_dim)

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            torch.nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            torch.nn.init.xavier_uniform_(self.qk_proj_weight)
            torch.nn.init.xavier_uniform_(self.v_proj_weight)

        if self.vbias is not None:
            torch.nn.init.constant_(self.value_bias_weight, 0.0)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        attn_mask=None,
        pos_embs=None,
        return_attn_weights=True,
    ):
        """Compute attention through Pytorch attention.

        Arguments
        ---------
        query : torch.Tensor
            (B, L, E) where L is the target sequence length,
            B is the batch size, E is the embedding dimension.
        key : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        value : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        key_padding_mask : torch.Tensor
            (B, S) where B is the batch size, S is the source sequence
            length. If a ByteTensor is provided, the non-zero positions will
            be ignored while the position with the zero positions will be
            unchanged. If a BoolTensor is provided, the positions with the
            value of True will be ignored while the position with the value
            of False will be unchanged.
        attn_mask : torch.BoolTensor
            2D mask (L, S) where L is the target sequence length, S is
            the source sequence length. The positions with the value of True will be ignored while the position with the value of False will be unchanged.
        pos_embs : torch.Tensor
            Not used by this class. It is kept for compliance.
        return_attn_weights : bool
            Whether to additionally return the attention weights.

        Returns
        -------
        out : torch.Tensor
            (B, L, E) where L is the target sequence length, B is the
            batch size, E is the embedding dimension.
        attn_score : torch.Tensor
            (B, L, S) where B is the batch size, L is the target
            sequence length, S is the source sequence length.
        """

        assert pos_embs is None, "pos_embs is not supported"

        # query, key and value are of shape batch, time, embed_dim
        bsz = query.shape[0]
        klen = key.shape[1]

        if self._qkv_same_embed_dim:
            # self-attention
            if (query is key or torch.equal(query, key)) and (
                key is value or torch.equal(key, value)
            ):
                query, key, value = (
                    nn.functional.linear(query, self.in_proj_weight)
                    .view(bsz, -1, self.num_heads, self.head_dim * 3)
                    .chunk(3, dim=-1)
                )
            else:
                qweight, kweight, vweight = self.in_proj_weight.chunk(3, dim=0)
                query = nn.functional.linear(query, qweight).view(
                    bsz, -1, self.num_heads, self.head_dim
                )
                key = nn.functional.linear(key, kweight).view(
                    bsz, -1, self.num_heads, self.head_dim
                )
                value = nn.functional.linear(value, vweight).view(
                    bsz, -1, self.num_heads, self.head_dim
                )
        else:
            raise NotImplementedError

        if self.vbias is not None:
            value = value + self.value_bias_weight.view(
                1, 1, self.num_heads, self.vhead_dim
            )

        q_rotated = _rope_rotate(query)
        k_rotated = _rope_rotate(key)

        final_masks = masks_union(
            bsz, klen, self.num_heads, attn_mask, key_padding_mask
        )

        x = F.scaled_dot_product_attention(
            query=q_rotated.permute(0, 2, 1, 3),
            key=k_rotated.permute(0, 2, 1, 3),
            value=value.permute(0, 2, 1, 3),
            attn_mask=final_masks,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(bsz, -1, self.vhead_dim * self.num_heads)
        )  # (batch, time1, d_model)

        out = self.out_proj(x)
        if return_attn_weights:
            return out, None  # out, attn_score
        return out


def masks_union(bsz, klen, num_heads, attn_mask, key_padding_mask):
    """This is an utility function combining standard key_padding_mask and
    attn_mask from SpeechBrain into a single one for scaled_dot_product_attention. This function does not support weighting of the attn_score. Hence, if one wish to use float values as masks, they should not use this function.

    Arguments
    ---------
    bsz : int
        Batch size dimension.
    klen : int
        Time dimension of the key tensor. (Sequence length).
    num_heads : int
        Number of heads of the attention module using these masks.
    attn_mask : torch.BoolTensor
        2D mask (L, S) where L is the target sequence length, S is
        the source sequence length. The positions with the value of True will be ignored while the position with the value of False will be unchanged.
    key_padding_mask : torch.BoolTensor
        (B, S) where B is the batch size, S is the source sequence
        length. The positions with the value of True will be ignored while the position with the value of False will be unchanged.

    Returns
    -------
    out : torch.BoolTensor
        (bsz, num_heads, klen, klen) where False values are masked and True are unmasked (opposite of the input tensors).

    """
    final_mask = None

    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, klen).expand(
            bsz, num_heads, klen, klen
        )
        final_mask = key_padding_mask

    if attn_mask is not None:
        attn_mask = attn_mask.view(1, 1, klen, klen).expand(
            bsz, num_heads, klen, klen
        )
        final_mask = attn_mask

    if attn_mask is not None and key_padding_mask is not None:
        final_mask = torch.logical_or(attn_mask, key_padding_mask)

    if final_mask is not None:
        final_mask = torch.logical_not(final_mask)

    return final_mask
