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

    Arguments
    ---------
    out_channels : list of ints
        Out channels of convolutional layers.
    kernel_sizes : list of ints
        Kernels of convolutional layers.
    strides : list of ints
        Strides of convolutional layers.
    dropout : float
        Dropout of CNN.
    """

    def __init__(
        self,
        out_channels=[512, 512, 512, 512, 512, 512, 512],
        kernel_sizes=[11, 3, 3, 3, 3, 3, 3],
        strides=[5, 2, 2, 2, 2, 2, 2],
        dropout=0.0,
        conv_init=None,  # will remove
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
        )
        self.norm = nn.LayerNorm(out_channels[-1])

    def forward(self, x, normalize_signal=True):
        """ Calculates latents from audio input.
        """
        if normalize_signal:
            x = F.layer_norm(x, x.shape[1:])
        x = x.unsqueeze(2)
        latents = self.extractor(x)
        return self.norm(latents)

    def get_output_lengths(self, input_lengths: torch.LongTensor):
        """ Calculates output lengths for given input lengths. """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        for kernel_size, stride in zip(self.kernel_sizes, self.strides):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        return input_lengths.to(torch.long)


class W2VLatentEncoder(nn.Module):
    """NOTE: Will remove and instead directly use ``TransformerEncoder``
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
        # for module in self.encoder.modules():
        #     if isinstance(module, nn.Linear):
        #         module.weight.data.normal_(mean=0.0, std=0.02)
        #         if module.bias is not None:
        #             module.bias.data.zero_()

    def forward(self, x_pos, padding_mask=None):
        x_pos, attn_lst = self.encoder(
            x_pos, src_key_padding_mask=padding_mask,
        )
        return x_pos, attn_lst


class W2VTargetQuantiser(nn.Module):
    """ Wraps ``GumbelVectorQuantizer``, see for documentation on
    arguments.
    """

    def __init__(
        self,
        in_dim=512,
        out_dim=256,
        quantiser=GumbelVectorQuantizer,
        num_vars=320,
        temperature_decay=(2.0, 0.25, 0.999995,),
    ):
        super().__init__()
        self.quantiser = quantiser(
            in_dim, num_vars, temperature_decay, 2, out_dim
        )
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        """ Returns quantised targets plus meta information. """
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
    """A wrapper that adds positional information,
    masks the input and then runs the latent encoder.
    Arguments
    ---------
    in_dim : int
        Last dimension of input tensor.
    embedding_dim : int
        Dimension to project input to and that the latent encoder will use.
    latent_encoder : torch.nn.module
        Initialized latent encoder object.
    positional_encoding : torch.nn.module
        Uninitialized nn.module for adding positional information, will use ``embedding_dim``.
    dropout_encoder_input : float
        Dropout on encoder input.
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

    def forward(
        self,
        latents,
        wav_lens=None,
        padding_mask=None,
        mask=None,
        disable_halfprec=False,
    ):
        """
        Arguments
        ---------
        latents : torch.Tensor, shape (B, T, C)
            Batch of latent representations (AKA frames) output from latent extractor.
        wav_lens : torch.Tensor, shape (B,)
            The actual (unpadded) relative lengths for each sample of the batch (0<wav_lens<1).
        padding_mask : Torch.Tensor, shape (B, T,)
            Can be provided instead of wav_lens.
        mask : torch.Tensor, shape (B, T)
            Boolean mask which decides which latent frames will be masked.
        """
        results = {}
        T = latents.size(1)
        with torch.cuda.amp.autocast(enabled=not disable_halfprec):
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


def compute_mask(shape, sample_lens, mask_prob, mask_length):
    """ This creates the boolean mask for a target shape which respects
    the sample lengths and will half roughly ``mask_prob`` entries set to
    ``True``.
    
    Arguments
    ---------
    shape : list of ints, like (N, M)
        Shape of boolean mask to return.
    sample_lens: list of ints
        Absolute lengths of per sample lengths.
    mask_prob : float
        Percentage to mask.
    mask_length: int
        Length of contiguous subsequence to mask.
    Returns
    -------
    mask : numpy.ndarray
        Boolean mask with shape of input argument ``shape``.
    """
    bs, padded_sample_len = shape

    min_sample_len = min(sample_lens)
    # So we dont have ragged tensors number of masks is the same for each sample.
    num_mask = int(
        mask_prob * min_sample_len / float(mask_length) + random.random() + 1
    )
    # Now loop through and for each sample select indices so that no indices land
    # in the padded part of the signal.
    mask_idcs = []
    for i in range(bs):
        sample_len = sample_lens[i]
        # This are the starting indices.
        mask_indices = np.random.choice(
            sample_len - mask_length, num_mask, replace=False
        )

        # Now using the starting indices create contiguous masks.
        mask_indices = np.asarray(
            [
                mask_indices[j] + offset
                for j in range(len(mask_indices))
                for offset in range(mask_length)
            ]
        )

        # Last step might have created overlapping masks, remove overlapping part.
        mask_idcs.append(np.unique(mask_indices[mask_indices < sample_len]))

    mask = np.full((bs, padded_sample_len), False)
    num_mask_total = num_mask * mask_length
    # Unique could have caused number to go below target count,
    # this randomly adds some unused indices.
    for i, mask_idc in enumerate(mask_idcs):
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
