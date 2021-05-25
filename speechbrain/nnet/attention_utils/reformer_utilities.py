"""
Utilities for the Reformer
Taken from: https://github.com/lucidrains/reformer-pytorch
"""

# TODO: Finalize docstring and where the code was taken

import math
import torch
import torch.nn as nn
from torch.nn.functional import pad
import torch.nn.functional as F
from functools import reduce, wraps
from operator import mul

TOKEN_SELF_ATTN_VALUE = -5e4  # carefully set for half precision to work


def pad_to_multiple(tensor, seqlen, multiple, dim=-1):
    m = seqlen / multiple
    if m.is_integer():
        return tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return pad(tensor, (*pad_offset, 0, remainder), value=0)


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
    l_ = (*pre_slices, slice(None, index))
    r_ = (*pre_slices, slice(index, None))
    return t[l_], t[r_]


def look_around(x, backward=1, forward=0, pad_value=-1, dim=2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value=pad_value)
    tensors = [
        padded_x[:, ind : (ind + t), ...]
        for ind in range(forward + backward + 1)
    ]
    return torch.cat(tensors, dim=dim)


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


def to(t):
    return {"device": t.device, "dtype": t.dtype}


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

        def merge_into_batch(t):
            return t.reshape(-1, *t.shape[-2:])

        q, k, v = map(merge_into_batch, (q, k, v))

        if exists(self.rel_pos):
            pos_emb = self.rel_pos(q)
            q, k = apply_rotary_pos_emb(q, k, pos_emb)

        if self.autopad:

            def merge_into_batch_auto(t):
                return pad_to_multiple(t, self.window_size, dim=-2)

            orig_t = q.shape[1]
            q, k, v = map(merge_into_batch_auto, (q, k, v),)

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

        def bucket_fn(t):
            return t.reshape(b, windows, window_size, -1)

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


class FullQKAttention(nn.Module):
    """
    This class comes from https://github.com/lucidrains/reformer-pytorch
    It was adjusted to fit SpeechBrain's architecture.
    """

    def __init__(self, causal=False, dropout=0.0):
        super().__init__()
        self.causal = causal
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        qk,
        v,
        query_len=None,
        input_mask=None,
        input_attn_mask=None,
        **kwargs,
    ):
        b, seq_len, dim = qk.shape
        query_len = default(query_len, seq_len)
        t = query_len

        q = qk[:, 0:query_len]
        qk = F.normalize(qk, 2, dim=-1).type_as(q)

        dot = torch.einsum("bie,bje->bij", q, qk) * (dim ** -0.5)

        # qk attention requires tokens not attend to self
        i = torch.arange(t)
        dot[:, i, i] = TOKEN_SELF_ATTN_VALUE
        masked_value = max_neg_value(dot)

        # Input mask for padding in variable lengthed sequences
        if input_mask is not None:
            mask = input_mask[:, 0:query_len, None] * input_mask[:, None, :]
            mask = F.pad(mask, (0, seq_len - mask.shape[-1]), value=True)
            dot.masked_fill_(~mask, masked_value)

        # Mask for post qk attention logits of the input sequence
        if input_attn_mask is not None:
            input_attn_mask = F.pad(
                input_attn_mask,
                (0, seq_len - input_attn_mask.shape[-1]),
                value=True,
            )
            dot.masked_fill_(~input_attn_mask, masked_value)

        if self.causal:
            i, j = torch.triu_indices(t, t, 1)
            dot[:, i, j] = masked_value

        dot = dot.softmax(dim=-1)
        dot = self.dropout(dot)

        out = torch.einsum("bij,bje->bie", dot, v)

        return out, dot, torch.empty(0)
