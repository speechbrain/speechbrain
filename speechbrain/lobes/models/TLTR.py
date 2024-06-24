"""
Time- and Layer-wise Transformer (TLTR) on top of the Whisper encoder representations for the audio tagging task.

Authors
* Yingzhi Wang 2024
"""

# This code uses a significant portion of the below implementation, even though it
# has been modified and enhanced.
# https://github.com/YuanGongND/ltu/blob/main/src/ltu_as/hf-dev/transformers-main/src/transformers/models/llama/modeling_llama.py
# *****************************************************************************


import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LayerNorm(nn.LayerNorm):
    """A wrapper of torch layer normalisation"""

    def forward(self, x):
        """Computes the forward pass
        Arguments
        ---------
        x: torch.Tensor
            input tensor
        Returns
        -------
        output: torch.Tensor
            the output after layer norm
        """
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    """A wrapper of torch linear layer"""

    def forward(self, x):
        """Computes the forward pass
        Arguments
        ---------
        x: torch.Tensor
            input tensor
        Returns
        -------
        output: torch.Tensor
            the output after linear layer
        """
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class MultiHeadAttention(nn.Module):
    """The class is a wrapper of MultiHead Attention.

    Arguments
    ---------
    n_state : int
        The size of the model layers.
    n_head : int
        parallel attention heads.
    """

    def __init__(self, n_state, n_head):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x,
        xa=None,
        mask=None,
        kv_cache=None,
    ):
        """forward function.
        Hooks, if kv_cache is not None, will prepend the cached kv tensors,
        otherwise, perform key/value projections for self-attention or
        cross-attention as usual.

        Arguments
        ---------
        x : torch.Tensor
        xa : torch.Tensor
        mask : torch.Tensor
            mask : torch.Tensor, optional
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
        kv_cache : torch.Tensor

        Returns
        -------
        attn_output : torch.Tensor
        attn_output_weights : torch.Tensor
        """
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(self, q, k, v, mask=None):
        """Compute attention.

        Arguments
        ---------
        q : torch.Tensor
            (B, L, E) where L is the target sequence length,
            B is the batch size, E is the embedding dimension.
        k : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        v : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        mask : torch.Tensor, optional
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

        Returns
        -------
        attn_output : torch.Tensor
        attn_output_weights : torch.Tensor
        """
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    """Transformer with residual attention block.

    Arguments
    ---------
    n_state : int
        The size of the model layers.
    n_head : int
        parallel attention heads.
    cross_attention: bool (default: False)
        Whether to do cross-attention.
    """

    def __init__(self, n_state, n_head, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x,
        xa=None,
        mask=None,
        kv_cache=None,
    ):
        """forward function.

        Arguments
        ---------
        x : torch.Tensor
        xa : torch.Tensor
        mask : torch.Tensor
            mask : torch.Tensor, optional
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
        kv_cache : torch.Tensor

        Returns
        -------
        attn_output : torch.Tensor
        """
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = (
                x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[
                    0
                ]
            )
        x = x + self.mlp(self.mlp_ln(x))
        return x


class TLTR(nn.Module):
    """Time- and Layer-wise Transformer (TLTR).
    Refer to the below paper for more details:
    https://www.isca-archive.org/interspeech_2023/gong23d_interspeech.pdf

    Arguments
    ---------
    n_layer: int
        number of layers of the audio representation.
    rep_dim: int
        feature size of the audio representation.
    """

    def __init__(self, n_layer=32, rep_dim=1280):
        super().__init__()
        self.n_layer = n_layer
        self.rep_dim = rep_dim

        self.num_tatt_head = 1
        self.num_latt_head = 8
        self.time_tr = ResidualAttentionBlock(self.rep_dim, self.num_tatt_head)
        self.layer_tr = ResidualAttentionBlock(self.rep_dim, self.num_latt_head)
        self.mlp_layer = nn.Sequential(
            nn.LayerNorm(self.rep_dim), nn.Linear(self.rep_dim, 527)
        )

    def forward(self, audio_rep):
        """Compute the TLTR representation.

        Arguments
        ---------
        audio_rep: torch.Tensor
            Input audio representation tensor

        Returns
        -------
        audio_rep : torch.Tensor
            Audio representation after tltr
        """
        # input audio_rep in shape (B, #layer, #time steps, rep_dim), e.g., (B, 32, 25, 1280) # for 10 seconds, 25 = 500 / 20 (downsampling)
        B, num_layer, audio_len, rep_dim = (
            audio_rep.shape[0],
            audio_rep.shape[1],
            audio_rep.shape[2],
            audio_rep.shape[3],
        )
        assert (
            num_layer==self.n_layer
        ), "Please verify the layer_num of the audio representation."
        
        audio_rep = audio_rep.reshape(
            [B * num_layer, audio_len, rep_dim]
        )  # [B*32, 25, 1280]
        audio_rep = self.time_tr(audio_rep)  # [B*32, 25, 1280]
        audio_rep = audio_rep.reshape(
            [B, num_layer, audio_len, rep_dim]
        )  # [B, 32, 25, 1280]
        audio_rep = audio_rep.permute([0, 2, 1, 3])  # [B, 25, 32, 1280]
        audio_rep = audio_rep.reshape(
            [B * audio_len, num_layer, rep_dim]
        )  # [B*25, 32, 1280]
        audio_rep = self.layer_tr(audio_rep)  # [B*25, 32, 1280]
        audio_rep = torch.mean(audio_rep, dim=1)  # [B*25, 1280]
        audio_rep = audio_rep.reshape([B, audio_len, rep_dim])
        return audio_rep


class AT_MODEL(nn.Module):
    """A wrapper of the TLTR class, in order to match the dict keys to load the pretrained model weights.

    Arguments
    ---------
    n_layer: int
        number of layers of the audio representation.
    rep_dim: int
        feature size of the audio representation.
    freeze: bool (default: False)
        whether to freeze the TLTR model.
    """

    def __init__(
        self,
        n_layer,
        rep_dim,
        freeze=False,
    ):
        super().__init__()
        self.at_model = TLTR(n_layer=n_layer, rep_dim=rep_dim)
        self.freeze = freeze

        for param in self.at_model.mlp_layer.parameters():
            param.requires_grad = False

        if self.freeze:
            for param in self.at_model.parameters():
                param.requires_grad = False
            logger.warning(
                "speechbrain.lobes.models.huggingface_transformers.TLTR - TLTR is frozen."
            )

    def forward(self, audio_rep):
        """Compute the TLTR representation.

        Arguments
        ---------
        audio_rep: torch.Tensor
            Input audio representation tensor

        Returns
        -------
        output : torch.Tensor
            Audio representation after tltr
        """

        with torch.set_grad_enabled(not self.freeze):
            output = self.at_model.forward(audio_rep)
        return output


class AudioProjection(nn.Module):
    """Project an audio ambedding to another dimension.

    Arguments
    ---------
    input_size: int
        feature size of the audio representation.
    hidden_size: int
        target feature size, in order to meet the text embedding size for LLMs.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.ln = nn.LayerNorm(input_size, elementwise_affine=False)
        self.proj = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        """Compute projected audio embedding.

        Arguments
        ---------
        x: torch.Tensor
            Input audio embedding

        Returns
        -------
        output : torch.Tensor
            Projected audio embedding
        """

        return self.proj(self.ln(x))
