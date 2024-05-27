import difflib
import torch
from torch import Tensor, nn
from typing import Dict, Iterable, Optional
import numpy as np
from flash_attn import flash_attn_varlen_func, flash_attn_func
from flash_attn.bert_padding import pad_input, unpad_input
import torch.nn.functional as F

class LayerNorm(nn.LayerNorm):
    # from https://github.com/openai/whisper/blob/main/whisper/model.py
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    # from https://github.com/openai/whisper/blob/main/whisper/model.py
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class FlashPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=10000):
        super(FlashPositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        assert d_model % 2 == 0
        log_timescale_increment = np.log(10000) / (d_model // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment * torch.arange(d_model // 2)
        )
        scaled_time = (
            torch.arange(max_len)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        )
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        self.register_buffer("pe", pe)

    def forward(self, x, *args, offset=None, **kwargs):
        # scale pe
        if offset is not None:

            pos_emb = self.pe[offset : offset + x.size(1)].unsqueeze(0) / (
                x.shape[-1] ** 0.5
            )
        else:
            pos_emb = self.pe[: x.size(1), ...].unsqueeze(0) / (x.shape[-1] ** 0.5)

        x = x + pos_emb
        return self.dropout(x)


class MultiHeadFlashAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int, dropout_att: float = 0.0):
        super().__init__()
        self.n_head = n_head
        self.dropout = dropout_att
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask_x: Optional[Tensor] = None,
        mask_xa: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        causal=False,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # NotImplementedError
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        n_batch, n_ctx, n_state = q.shape

        q = q.view(*q.shape[:2], self.n_head, -1)
        k = k.view(*k.shape[:2], self.n_head, -1)
        v = v.view(*v.shape[:2], self.n_head, -1)

        if xa is None and (mask_x is None or torch.all(mask_x)):
            # all same length
            out = flash_attn_func(q, k, v, dropout_p=self.dropout, causal=causal)
        else:
            if xa is None:
                q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, mask_x)
                k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(k, mask_x)
                v_unpad, indices_v, cu_seqlens_v, max_seqlen_v = unpad_input(v, mask_x)
            else:
                q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, mask_x)
                k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(k, mask_xa)
                v_unpad, indices_v, cu_seqlens_v, max_seqlen_v = unpad_input(v, mask_xa)

            out_unpad = flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                self.dropout,
                return_attn_probs=False,
                causal=causal,
            )

            out = pad_input(out_unpad, indices_q, n_batch, n_ctx)

        return out.flatten(start_dim=2), None


class FlashResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        cross_attention: bool = False,
        ff_dim: int = 1024,
        dropout: float = 0.0,
        dropout_att: float = 0.0,
        activation="gelu",
        use_flash=True,
    ):
        super().__init__()

        if use_flash:
            self.attn = MultiHeadFlashAttention(
                n_state, n_head, dropout_att=dropout_att
            )
        else:
            raise NotImplementedError
        self.attn_ln = LayerNorm(n_state)
        self.dropout = torch.nn.Dropout(dropout)

        if not use_flash:
            raise NotImplementedError
        else:
            self.cross_attn = (
                MultiHeadFlashAttention(n_state, n_head, dropout_att=dropout_att)
                if cross_attention
                else None
            )

        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None
        self.mlp = nn.Sequential(
            Linear(n_state, ff_dim),
            get_layer(activation)(),
            self.dropout,
            Linear(ff_dim, n_state),
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask_sa: Optional[Tensor] = None,
        mask_xa: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        causal: Optional[bool] = False,
    ):
        x = x + self.dropout(
            self.attn(
                self.attn_ln(x), mask_x=mask_sa, kv_cache=kv_cache, causal=causal
            )[0]
        )
        if self.cross_attn:
            x = x + self.dropout(
                self.cross_attn(
                    self.cross_attn_ln(x),
                    xa,
                    mask_x=mask_sa,
                    mask_xa=mask_xa,
                    kv_cache=kv_cache,
                    causal=False,
                )[0]
            )
        x = x + self.dropout(self.mlp(self.mlp_ln(x)))
        return x




def get_layer(l_name, library=torch.nn):
    """Return layer object handler from library e.g. from torch.nn

    E.g. if l_name=="elu", returns torch.nn.ELU.

    Args:
        l_name (string): Case insensitive name for layer in library (e.g. .'elu').
        library (module): Name of library/module where to search for object handler
        with l_name e.g. "torch.nn".

    Returns:
        layer_handler (object): handler for the requested layer e.g. (torch.nn.ELU)

    """

    all_torch_layers = [x for x in dir(torch.nn)]
    match = [x for x in all_torch_layers if l_name.lower() == x.lower()]
    if len(match) == 0:
        close_matches = difflib.get_close_matches(
            l_name, [x.lower() for x in all_torch_layers]
        )
        raise NotImplementedError(
            "Layer with name {} not found in {}.\n Closest matches: {}".format(
                l_name, str(library), close_matches
            )
        )
    elif len(match) > 1:
        close_matches = difflib.get_close_matches(
            l_name, [x.lower() for x in all_torch_layers]
        )
        raise NotImplementedError(
            "Multiple matchs for layer with name {} not found in {}.\n "
            "All matches: {}".format(l_name, str(library), close_matches)
        )
    else:
        # valid
        layer_handler = getattr(library, match[0])
        return layer_handler