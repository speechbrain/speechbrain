# import warnings

import math
import torch
import torch.nn as nn

# from torch.nn import Linear
# from torch.nn.parameter import Parameter
# from torch.nn.init import xavier_uniform_
# from torch.nn.init import constant_
# from torch.nn.init import xavier_normal_
# from torch.nn.functional import linear, softmax, dropout, pad
from torch.nn.functional import pad

# import speechbrain as sb
# from speechbrain.nnet.normalization import LayerNorm
# from speechbrain.nnet.activations import Swish

from typing import Optional


import torch.nn.functional as F
from torch.autograd import Function
from functools import partial, reduce, wraps
from operator import mul


def pad_to_multiple(tensor, seqlen, multiple, dim=-1):
    m = seqlen / multiple
    if m.is_integer():
        return tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return pad(tensor, (*pad_offset, 0, remainder), value=0)


class attention_padder(nn.Module):
    # Reformer comes with a slight drawback that the sequence must be neatly divisible by the bucket size * 2.
    # I have provided a small helper tool that can help you auto-round the sequence length to the next best multiple.
    # https://github.com/lucidrains/reformer-pytorch/blob/365d5c8918dc643a4a606da021253936d82577fc/reformer_pytorch/autopadder.py
    def __init__(self, net, attention_mechanism, window_padding_size, pad_dim):
        super().__init__()
        self.net = net
        self.pad_dim = pad_dim
        self.attention_mechanism = attention_mechanism
        self.window_padding_size = window_padding_size
        self.num_mem_kv = 0
        self.full_attn_thres = (
            self.net.full_attn_thres
            if self.attention_mechanism is not "LongformerSelfAttention"
            else 0
        )

    def forward(
        self,
        query,
        key,
        value,
        random=True,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: Optional[bool] = True,
    ):
        batch_size, seqlen, device = *query.shape[:2], query.device

        input_mask = key_padding_mask
        input_attn_mask = attn_mask
        new_key_padding_mask = key_padding_mask
        new_attn_mask = attn_mask

        if seqlen > self.full_attn_thres:
            if key_padding_mask is None:
                key_padding_mask = torch.full(
                    (batch_size, seqlen),
                    True,
                    device=query.device,
                    dtype=torch.bool,
                )

            query, key, value = map(
                lambda t: pad_to_multiple(
                    t, seqlen, self.window_padding_size, dim=self.pad_dim
                ),
                (query, key, value),
            )

            if input_mask is not None:
                new_key_padding_mask = pad(
                    input_mask,
                    (0, query.shape[1] - input_mask.shape[1]),
                    value=False,
                )

            if input_attn_mask is not None:
                offset = query.shape[1] - input_attn_mask.shape[1]
                new_attn_mask = pad(
                    input_attn_mask, (0, offset, 0, offset), value=False
                )

        if self.attention_mechanism == "LongformerSelfAttention":
            output, self_attn = self.net(
                hidden_states=query, output_attentions=True
            )
        elif self.attention_mechanism == "LSHSelfAttention":
            output, self_attn = self.net(query)
            return output[:, 0:seqlen], self_attn
        else:
            output, self_attn = self.net(
                query, key, value, attn_mask=attn_mask,
            )
        return output[:, 0:seqlen], self_attn


class MultiheadWrapper(
    nn.Module
):  # just for testing different attention mechanism
    # https://github.com/kowaalczyk/reformer-tts/blob/master/reformer_tts/model/reformer.py
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
        attention_mechanism="MultiheadAttention",
        bucket_length=64,
        chunk=8,
        rounds=4,
        layer_id=None,
        attention_window=[],
        attention_mode="sliding_chunks",
        attention_dilation=[],
        full_attn_thres=None,
    ):
        super().__init__()

        assert attention_mechanism in [
            "MultiheadAttention",
            "LinearMultiheadAttention",
            "LSHSelfAttention2",
            "LSHSelfAttention",
            "ReformerAttention",
            "ReformerAttention_fastai",
            "LongformerSelfAttention",
        ]
        self.attention_mechanism = attention_mechanism
        # print(
        #     f"{self.attention_mechanism}: embed_dim-->{embed_dim:5d}, num_heads-->{num_heads:5d}, dropout-->{dropout:8.2f},\
        # seq_len-->{seq_len:5d} , proj_k-->{proj_k:5d}, param_sharing-->{param_sharing}, \
        # method-->{method} bucket_length-->{bucket_length}, chunk-->{chunk},  rounds-->{rounds}, \
        # full_attn_thres-->{full_attn_thres},  attention_window-->{attention_window}, attention_mode-->{attention_mode}, \
        # attention_dilation-->{attention_dilation}"
        # )

        if self.attention_mechanism == "MultiheadAttention":
            from speechbrain.nnet.attention import MultiheadAttention

            self.layer = MultiheadAttention(
                nhead=num_heads,
                d_model=embed_dim,
                dropout=dropout,
                kdim=kdim,
                vdim=vdim,
            )

        elif self.attention_mechanism == "LinearMultiheadAttention":
            from speechbrain.nnet.attention import LinearMultiheadAttention

            self.layer = LinearMultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                bias=True,
                add_bias_kv=False,
                add_zero_attn=False,
                kdim=kdim,
                vdim=vdim,
                seq_len=seq_len,
                proj_k=proj_k,
                param_sharing=param_sharing,
                method=method,
                layerwise_proj=layerwise_proj,
            )

        elif self.attention_mechanism == "LSHSelfAttention":
            from speechbrain.nnet.attention import LSHSelfAttention

            self.layer = attention_padder(
                LSHSelfAttention(
                    dim=embed_dim,
                    heads=num_heads,
                    bucket_size=bucket_length,
                    n_hashes=rounds,
                    causal=False,
                    full_attn_thres=full_attn_thres,
                ),
                "LSHSelfAttention",
                bucket_length * 2,
                pad_dim=-2,
            )

        elif self.attention_mechanism == "LongformerSelfAttention":
            from speechbrain.nnet.attention import LongformerSelfAttention

            self.layer = attention_padder(
                LongformerSelfAttention(
                    layer_id=layer_id,
                    num_attention_heads=num_heads,
                    hidden_size=embed_dim,
                    attention_probs_dropout_prob=dropout,
                    attention_window=attention_window,
                    attention_mode=attention_mode,
                    attention_dilation=attention_dilation,
                ),
                "LongformerSelfAttention",
                attention_window[0] * 2,
                pad_dim=-2,
            )

    def forward(
        self,
        query,
        key,
        value,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: Optional[bool] = True,
    ):
        return self.layer.forward(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )


def get_EF(input_size, dim, method="convolution", head_dim=None, bias=True):
    # inpired from https://github.com/tatp22/linformer-pytorch/blob/master/linformer_pytorch/linformer_pytorch.py#L26
    """
    Retuns the E or F matrix, initialized via xavier initialization.
    This is the recommended way to do it according to the authors of the paper.
    Includes a method for convolution, as well as a method for no additional params.
    """
    assert method in [
        "learnable",
        "convolution",
        "maxpool",
        "avgpool",
        "no_params",
    ], "The method flag needs to be either 'learnable', 'convolution', or 'no_params'!"
    if method == "convolution":
        conv = nn.Conv1d(
            head_dim,
            head_dim,
            kernel_size=int(input_size / dim),
            stride=int(input_size / dim),
        )
        return conv
    if method == "maxpool":
        pool = nn.MaxPool1d(
            kernel_size=int(input_size / dim), stride=int(input_size / dim)
        )
        return pool
    if method == "avgpool":
        pool = nn.MaxPool1d(
            kernel_size=int(input_size / dim), stride=int(input_size / dim)
        )
        return pool
    if method == "no_params":
        mat = torch.zeros((input_size, dim))
        torch.nn.init.normal_(mat, mean=0.0, std=1 / dim)
        return mat
    lin = nn.Linear(input_size, dim, bias)
    torch.nn.init.xavier_normal_(lin.weight)
    return lin


# ***************************************************************************************************************
# LSH attention as described in https://openreview.net/pdf?id=rkgNKkHtvB
# adapted from trax, stripped to what paper said needed to work
# namely that buckets need to be at least 64 with 8 rounds of hashing
# https://github.com/google/trax/blob/master/trax/layers/research/efficient_attention.py#L442
# ***************************************************************************************************************
import torch.nn.functional as F
from torch.autograd import Function
from functools import partial, reduce, wraps
from operator import mul


# constants

TOKEN_SELF_ATTN_VALUE = -5e4  # carefully set for half precision to work

# helper fns


def sort_key_val(t1, t2, dim=-1):
    values, indices = t1.sort(dim=dim)
    t2 = t2.expand_as(t1)
    return values, t2.gather(dim, indices)


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))


def process_inputs_chunk(fn, chunks=1, dim=0):
    def inner_fn(*args, **kwargs):
        keys, values, len_args = kwargs.keys(), kwargs.values(), len(args)
        chunked_args = list(
            zip(
                *map(
                    lambda x: x.chunk(chunks, dim=dim),
                    list(args) + list(values),
                )
            )
        )
        all_args = map(
            lambda x: (x[:len_args], dict(zip(keys, x[len_args:]))),
            chunked_args,
        )
        outputs = [fn(*c_args, **c_kwargs) for c_args, c_kwargs in all_args]
        return tuple(map(lambda x: torch.cat(x, dim=dim), zip(*outputs)))

    return inner_fn


def chunked_sum(tensor, chunks=1):
    *orig_size, last_dim = tensor.shape
    tensor = tensor.reshape(-1, last_dim)
    summed_tensors = [c.sum(dim=-1) for c in tensor.chunk(chunks, dim=0)]
    return torch.cat(summed_tensors, dim=0).reshape(orig_size)


def default(val, default_val):
    return default_val if val is None else val


def cast_tuple(x):
    return x if isinstance(x, tuple) else (x,)


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, **kwargs):
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


def cache_method_decorator(cache_attr, cache_namespace, reexecute=False):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(
            self,
            *args,
            key_namespace=None,
            fetch=False,
            set_cache=True,
            **kwargs,
        ):
            namespace_str = str(default(key_namespace, ""))
            _cache = getattr(self, cache_attr)
            _keyname = f"{cache_namespace}:{namespace_str}"

            if fetch:
                val = _cache[_keyname]
                if reexecute:
                    fn(self, *args, **kwargs)
            else:
                val = fn(self, *args, **kwargs)
                if set_cache:
                    setattr(self, cache_attr, {**_cache, **{_keyname: val}})
            return val

        return wrapper

    return inner_fn


def expand_dim(dim, k, t):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)


def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]


def look_around(x, backward=1, forward=0, pad_value=-1, dim=2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value=pad_value)
    tensors = [
        padded_x[:, ind : (ind + t), ...]
        for ind in range(forward + backward + 1)
    ]
    return torch.cat(tensors, dim=dim)


# helper classes


class MatrixMultiply(nn.Module):
    def __init__(self, tensor, transpose=False, normalize=False):
        super().__init__()
        self.tensor = tensor
        self.transpose = transpose
        self.normalize = normalize

    def forward(self, x):
        tensor = self.tensor
        if self.normalize:
            tensor = F.normalize(tensor, dim=-1)
        if self.transpose:
            tensor = tensor.t()
        return x @ tensor


class ReZero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.g = nn.Parameter(torch.zeros(1))
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.g


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x / n * self.g


class PreNorm(nn.Module):
    def __init__(self, norm_class, dim, fn):
        super().__init__()
        self.norm = norm_class(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim=-1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim=self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim=self.dim)


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        n = x.shape[-2]
        t = torch.arange(n, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, :]


def rotate_half(x):
    x = x.reshape((x.shape[0], -1, 2, x.shape[-1] // 2))
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, freqs):
    q, k = map(
        lambda t: (t * freqs.cos()) + (rotate_half(t) * freqs.sin()), (q, k)
    )
    return q, k


def exists(val):
    return val is not None


def default(value, d):
    return d if not exists(value) else value


def to(t):
    return {"device": t.device, "dtype": t.dtype}


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)


def expand_dim(t, dim, k, unsqueeze=True):
    if unsqueeze:
        t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value=value)


def look_around(x, backward=1, forward=0, pad_value=-1, dim=2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value=pad_value)
    tensors = [
        padded_x[:, ind : (ind + t), ...]
        for ind in range(forward + backward + 1)
    ]
    return torch.cat(tensors, dim=dim)


# main class


class LocalAttention(nn.Module):
    def __init__(
        self,
        window_size,
        causal=False,
        look_backward=1,
        look_forward=None,
        dropout=0.0,
        shared_qk=False,
        rel_pos_emb_config=None,
        dim=None,
        autopad=False,
        exact_windowsize=False,
    ):
        super().__init__()
        look_forward = default(look_forward, 0 if causal else 1)
        assert not (
            causal and look_forward > 0
        ), "you cannot look forward if causal"

        self.window_size = window_size
        self.causal = causal
        self.look_backward = look_backward
        self.look_forward = look_forward
        self.exact_windowsize = exact_windowsize
        self.autopad = autopad

        self.dropout = nn.Dropout(dropout)

        self.shared_qk = shared_qk

        self.rel_pos = None
        if exists(rel_pos_emb_config) or exists(
            dim
        ):  # backwards compatible with old `rel_pos_emb_config` deprecated argument
            if exists(rel_pos_emb_config):
                dim = rel_pos_emb_config[0]
            self.rel_pos = SinusoidalEmbeddings(dim)

    def forward(self, q, k, v, input_mask=None):
        shape = q.shape

        merge_into_batch = lambda t: t.reshape(-1, *t.shape[-2:])
        q, k, v = map(merge_into_batch, (q, k, v))

        if exists(self.rel_pos):
            pos_emb = self.rel_pos(q)
            q, k = apply_rotary_pos_emb(q, k, pos_emb)

        if self.autopad:
            orig_t = q.shape[1]
            q, k, v = map(
                lambda t: pad_to_multiple(t, self.window_size, dim=-2),
                (q, k, v),
            )

        window_size, causal, look_backward, look_forward, shared_qk = (
            self.window_size,
            self.causal,
            self.look_backward,
            self.look_forward,
            self.shared_qk,
        )
        b, t, e, device, dtype = *q.shape, q.device, q.dtype
        assert (
            t % window_size
        ) == 0, f"sequence length {t} must be divisible by window size {window_size} for local attention"

        windows = t // window_size

        if shared_qk:
            k = F.normalize(k, 2, dim=-1).type_as(q)

        ticker = torch.arange(t, device=device, dtype=dtype)[None, :]
        b_t = ticker.reshape(1, windows, window_size)

        bucket_fn = lambda t: t.reshape(b, windows, window_size, -1)
        bq, bk, bv = map(bucket_fn, (q, k, v))

        look_around_kwargs = {
            "backward": look_backward,
            "forward": look_forward,
        }
        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)

        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)

        dots = torch.einsum("bhie,bhje->bhij", bq, bk) * (e ** -0.5)

        mask_value = max_neg_value(dots)

        if shared_qk:
            mask = bq_t[:, :, :, None] == bq_k[:, :, None, :]
            dots.masked_fill_(mask, TOKEN_SELF_ATTN_VALUE)
            del mask

        if causal:
            mask = bq_t[:, :, :, None] < bq_k[:, :, None, :]

            if self.exact_windowsize:
                max_causal_window_size = self.window_size * self.look_backward
                mask = mask | (
                    bq_t[:, :, :, None]
                    > (bq_k[:, :, None, :] + max_causal_window_size)
                )

            dots.masked_fill_(mask, mask_value)
            del mask

        mask = bq_k[:, :, None, :] == -1
        dots.masked_fill_(mask, mask_value)
        del mask

        if input_mask is not None:
            h = b // input_mask.shape[0]
            if self.autopad:
                input_mask = pad_to_multiple(
                    input_mask, window_size, dim=-1, value=False
                )
            input_mask = input_mask.reshape(-1, windows, window_size)
            mq = mk = input_mask
            mk = look_around(mk, pad_value=False, **look_around_kwargs)
            mask = mq[:, :, :, None] * mk[:, :, None, :]
            mask = merge_dims(0, 1, expand_dim(mask, 1, h))
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("bhij,bhje->bhie", attn, bv)
        out = out.reshape(-1, t, e)

        if self.autopad:
            out = out[:, :orig_t, :]

        return out.reshape(*shape)
