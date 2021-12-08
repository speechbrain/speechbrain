import math
import logging
from more_itertools import sample
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import numpy as np
from contextlib import contextmanager

from speechbrain.lobes.models.transformer.Transformer import (
    TransformerEncoder,
    PositionalEncoding,
)
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.lobes.models.convolution import ConvolutionFrontEnd
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.nnet.quantisers import GumbelVectorQuantizer

logger = logging.getLogger()


class W2VLatentExtractor(nn.Module):
    """Convolution based feature extractor from raw audio.

    Channel numbers increasing is based on https://arxiv.org/abs/2109.06870
    """

    def __init__(
        self,
        out_channels=[512, 512, 512, 512, 512, 512, 512],
        kernel_sizes=[11, 3, 3, 3, 3, 3, 3],
        strides=[5, 2, 2, 2, 2, 2, 2],
        dropout=0.0,
        init="kaiming",
    ):
        super().__init__()

        assert len(out_channels) == len(kernel_sizes) == len(strides)

        num_blocks = len(out_channels)
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.out_dim = out_channels[-1]
        # ! Note this does conv, norm, gelu, dropout. while fairseq does conv, dropout, norm, gelu
        # Also fairseq layernorm is forced to fp32
        self.extractor = ConvolutionFrontEnd(
            (None, 16000, 1,),
            num_blocks=num_blocks,
            num_layers_per_block=1,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            dilations=[1] * num_blocks,
            residuals=[False] * num_blocks,
            conv_module=Conv1d,
            activation=nn.GELU,
            norm=LayerNorm,
            dropout=dropout,
            conv_bias=False,
            padding="valid",
            conv_init=init,
        )
        self.norm = nn.LayerNorm(out_channels[-1])

    def forward(self, x, normalize_signal=True):
        if normalize_signal:
            x = F.layer_norm(x, x.shape[1:])
        x = x.unsqueeze(2)
        return self.norm(self.extractor(x))

    def get_output_lengths(self, input_lengths: torch.LongTensor):
        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        for kernel_size, stride in zip(self.kernel_sizes, self.strides):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        return input_lengths.to(torch.long)


class Wav2VecPositionalConvEmbedding(nn.Module):
    """Positional encoding as implemented in fairseq.
    """

    def __init__(self, embedding_dim, pos_conv_kernel=128, pos_conv_groups=16):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=pos_conv_kernel,
            padding=pos_conv_kernel // 2,
            groups=pos_conv_groups,
        )
        std = math.sqrt((4 * (1.0)) / (pos_conv_kernel * embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(
            self.pos_conv, name="weight", dim=2
        )
        self.pos_conv = nn.Sequential(
            self.pos_conv, SamePadLayer(pos_conv_kernel), nn.GELU()
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pos_conv(x)
        x = x.transpose(1, 2)
        return x


class W2VLatentEncoder(nn.Module):
    """Wraps transformer.
    """

    def __init__(
        self,
        embedding_dim=768,
        num_layers=12,
        d_ffn=3072,
        nhead=8,
        dropout=0.1,
        attention_dropout=0.1,
        layerdrop_prob=0.05,
        normalize_before=True,
    ):
        # TODO: attention_dropout not used
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            nhead=nhead,
            d_ffn=d_ffn,
            d_model=embedding_dim,
            dropout=dropout,
            activation=nn.GELU,
            normalize_before=normalize_before,
            layerdrop_prob=layerdrop_prob,
        )
        for module in self.encoder.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x_pos, padding_mask=None, output_hidden_states=False):
        return self.encoder(
            x_pos,
            src_key_padding_mask=padding_mask,
            output_hidden_states=output_hidden_states,
        )


class W2VTargetQuantiser(nn.Module):
    def __init__(
        self, in_dim=512, out_dim=256, quantiser=GumbelVectorQuantizer
    ):
        super().__init__()
        self.quantiser = quantiser(
            in_dim, 320, (2.0, 0.25, 0.999995,), 2, out_dim
        )
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        x = self.quantiser(x)
        targets = self.proj(x["x"])
        code_perplex = x["code_perplexity"]
        prob_perplex = x["prob_perplex"]
        num_vars = x["num_vars"]
        temp = x["temp"]
        diversity_loss = (num_vars - prob_perplex) / num_vars
        meta = {
            "diversity_loss": diversity_loss,
            "code_perplex": code_perplex,
            "prob_perplex": prob_perplex,
            "num_vars": num_vars,
            "temp": temp,
        }
        return targets, meta


class EncoderWrapper(nn.Module):
    """Wrapper for wav2vec module that contains feature extractor, encoder and
    connecting elements.
    """

    def __init__(
        self,
        in_dim,
        embedding_dim,
        latent_encoder=W2VLatentEncoder(),
        positional_encoding=PositionalEncoding,
        dropout_encoder_input=0.05,
    ):
        super().__init__()
        self.input_projector = nn.Linear(in_dim, embedding_dim)
        self.latent_encoder = latent_encoder
        self.positional_encoding = positional_encoding(embedding_dim)
        self.dropout_encoder_input = nn.Dropout(dropout_encoder_input)
        self.mask_emb = nn.Parameter(
            torch.FloatTensor(embedding_dim).uniform_(), requires_grad=True
        )

    def forward(self, latents, wav_lens=None, padding_mask=None, mask=None):
        """
        Arguments
        ----------
        x : torch.Tensor
            The raw audio with shape (B,T,).
        x_lens : torch.Tensor
            The lengths for each sample of the batch (0<x_lens<1).
        """
        results = {}
        T = latents.size(1)
        latents = self.input_projector(latents)
        latents = self.dropout_encoder_input(latents)

        if mask is not None:
            latents[mask] = self.mask_emb.to(latents.dtype)
            num_masked = mask.sum()
            results["num_masked"] = num_masked
            results["ratio_masked"] = num_masked / mask.numel()

        if wav_lens is not None:
            wav_lens = torch.round(wav_lens * T)
            padding_mask = ~length_to_mask(wav_lens, dtype=bool)

        latents = latents + self.positional_encoding(latents)
        feats, _ = self.latent_encoder(latents, padding_mask)

        results["embeddings"] = feats
        return results


def compute_sample_mask(length, mask_prob, mask_length):
    num_mask = int(
        mask_prob * length / float(mask_length) + 1 + random.random()
    )
    mask_indices = np.random.choice(
        length - mask_length, num_mask, replace=False
    )
    mask_indices = np.asarray(
        [
            mask_indices[j] + offset
            for j in range(len(mask_indices))
            for offset in range(mask_length)
        ]
    )
    mask_indices = np.unique(mask_indices[mask_indices < length])
    num_masked_target = num_mask * mask_length
    num_mask_missing = num_masked_target - mask_indices.size
    # Randomly place some masks if unique removed some
    if num_mask_missing > 0:
        arange = np.arange(length)
        arange = np.delete(arange, mask_indices)
        extra_indcs = np.random.choice(arange, num_mask_missing, replace=False)
        mask_indices = np.append(mask_indices, extra_indcs)
        mask_indices.sort()
    return mask_indices


def compute_mask(shape, sample_lens, mask_prob, mask_length):
    bs, padded_sample_len = shape
    
    min_sample_len = min(sample_lens)
    # So we dont have ragged tensors number of masks is the same for each sample.
    num_mask = int(
        mask_prob * min_sample_len / float(mask_length) + random.random() + 1
    )

    mask_idcs = []
    for i in range(bs):
        sample_len = sample_lens[i]
        mask_indices = np.random.choice(
            sample_len - mask_length, num_mask, replace=False
        )

        mask_indices = np.asarray(
            [
                mask_indices[j] + offset
                for j in range(len(mask_indices))
                for offset in range(mask_length)
            ]
        )
        mask_idcs.append(np.unique(mask_indices[mask_indices < sample_len]))

    mask = np.full((bs, padded_sample_len), False)
    num_mask_total = num_mask * mask_length
    for i, mask_idc in enumerate(mask_idcs):
        # Unique could have caused number to go below target count,
        # this randomly adds some unused indices.
        if len(mask_idc) < num_mask_total:
            num_mask_missing = num_mask_total - len(mask_idc)
            arange = np.arange(sample_lens[i])
            arange = np.delete(arange, mask_idc)
            extra_indcs = np.random.choice(
                arange, num_mask_missing, replace=False
            )
            mask[i, extra_indcs] = True
        mask[i, mask_idc] = True
    return mask


class SamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states
