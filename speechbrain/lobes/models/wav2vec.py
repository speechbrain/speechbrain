"""Components necessary to build a wav2vec 2.0 architecture following the
original paper: https://arxiv.org/abs/2006.11477.

Authors
* Rudolf A Braun 2022
* Guillermo Cambara 2022
* Titouan Parcollet 2022
"""

import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import numpy as np

from speechbrain.lobes.models.transformer.Transformer import PositionalEncoding
from speechbrain.utils.data_utils import batch_pad_right
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

    Example
    -------
    >>> extractor = W2VLatentExtractor()
    >>> inputs = torch.rand(10, 5000)
    >>> outputs = extractor(inputs)
    >>> outputs.shape
    torch.Size([10, 14, 512])
    """

    def __init__(
        self,
        out_channels=[512, 512, 512, 512, 512, 512, 512],
        kernel_sizes=[11, 3, 3, 3, 3, 3, 3],
        strides=[5, 2, 2, 2, 2, 2, 2],
        dropout=0.0,
        conv_init="kaiming",
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
            conv_init=conv_init,
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


class W2VTargetQuantiser(nn.Module):
    """ Wraps ``nnet.quantiser.GumbelVectorQuantizer``, see for documentation on
    arguments.

    Example
    -------
    >>> quantiser = W2VTargetQuantiser()
    >>> inputs = torch.rand(10, 12, 512)
    >>> output, meta = quantiser(inputs)
    >>> output.shape
    torch.Size([10, 12, 256])
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

    Example
    -------
    >>> from speechbrain.lobes.models.transformer.Transformer import TransformerEncoder
    >>> encoder = TransformerEncoder(d_model=768, num_layers=4, nhead=4, d_ffn=1024)
    >>> wrapper = EncoderWrapper(1024, 768, encoder)
    >>> inputs = torch.rand(10, 12, 1024)
    >>> outputs = wrapper(inputs)
    >>> outputs["embeddings"].shape
    torch.Size([10, 12, 768])
    """

    def __init__(
        self,
        in_dim,
        embedding_dim,
        latent_encoder,
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
        self, latents, wav_lens=None, padding_mask=None, mask=None,
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
        feats, _ = self.latent_encoder(
            latents, src_key_padding_mask=padding_mask
        )

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


def sample_negatives(y, num_neg):
    """ Samples negatives from target tensor y.
    Arguments
    ---------
    y : torch.Tensor
        Tensor of shape (B, T, C)
    num_neg : int
        Number of negatives to sample.
    Returns
    -------
    negs : torch.Tensor
        Negatives in shape (N, B, T, C)
    """
    B, T, C = y.shape
    high = T - 1
    with torch.no_grad():
        targets = torch.arange(T).unsqueeze(-1).expand(-1, num_neg).flatten()
        neg_indcs = torch.randint(low=0, high=high, size=(B, T * num_neg))
        # negative should not be target and to make distribution uniform shift all >
        neg_indcs[neg_indcs >= targets] += 1

    neg_indcs = neg_indcs + torch.arange(B).unsqueeze(1) * high
    y = y.view(-1, C)
    negs = y[neg_indcs.view(-1)]
    negs = negs.view(B, T, num_neg, C).permute(2, 0, 1, 3)  # to N, B, T, C
    return negs


def w2v_mask_collate_fn(samples_lst, get_out_len_fn, mask_prob, mask_length):
    """ This creates a batch from a list of samples and also creates
    the boolean mask that will be used to mask the inputs of the latent
    encoder. To create the mask we need to know the output shape after the
    latent extractor, therefore the argument `get_out_len_fn`.
    One could also create masks per sample (when loading the audio file) and
    then collate them but at that time one doesn't know the length of the
    shortest sample in the batch (which determines the number of masked frames)
    so it's better this way.

    Arguments
    ---------
    samples_lst : list
        List of samples returned by the audio_pipeline.
    get_out_len_fn : function
        Function that calculates length of sample after it passes through feature extractor.
    mask_prob : float
        Approximate percentage of frames to mask.
    mask_length : int
        Number of contiguous frames that will be masked.

    Returns
    -------
    wavs_padded : torch.Tensor, shape (B, T)
        Audio arrays with right-sided padding.
    wav_lens : torch.Tensor, shape (B,)
        For each sample the percentage of the array that is not padding.
    mask : torch.Tensor, shape (B, T)
        Boolean mask to mask frames.
    """
    wav_lst, latent_length_lst = [], []
    ids = []
    for sample in samples_lst:
        ids.append(sample["id"])
        sig = sample["sig"]
        wav_lst.append(sig)
        latent_length = get_out_len_fn(torch.as_tensor(sig.size(-1)))
        latent_length_lst.append(latent_length.item())
    bs = len(wav_lst)
    wavs_padded, wav_lens = batch_pad_right(wav_lst)

    batch_time_len = max(latent_length_lst)
    mask = compute_mask(
        (bs, batch_time_len,), latent_length_lst, mask_prob, mask_length
    )
    return (
        torch.as_tensor(wavs_padded),
        torch.as_tensor(wav_lens),
        torch.as_tensor(mask, dtype=torch.bool),
    )
