"""Transformer for ASR in the SpeechBrain sytle.

Authors
* Jianyuan Zhong 2020
"""

import torch  # noqa 42
from torch import nn
from typing import Optional

from speechbrain.nnet.linear import Linear
from speechbrain.nnet.containers import ModuleList
from speechbrain.lobes.models.transformer.Transformer import (
    TransformerInterface,
    get_lookahead_mask,
    get_key_padding_mask,
    NormalizedEmbedding,
)
from speechbrain.nnet.activations import Swish

from speechbrain.dataio.dataio import length_to_mask


class TransformerASR(TransformerInterface):
    """This is an implementation of transformer model for ASR.

    The architecture is based on the paper "Attention Is All You Need":
    https://arxiv.org/pdf/1706.03762.pdf

    Arguments
    ----------
    d_model : int
        The number of expected features in the encoder/decoder inputs
        (default=512).
    nhead : int
        The number of heads in the multi-head attention models (default=8).
    num_encoder_layers : int
        The number of sub-encoder-layers in the encoder (default=6).
    num_decoder_layers : int
        The number of sub-decoder-layers in the decoder (default=6).
    dim_ffn : int
        The dimension of the feedforward network model (default=2048).
    dropout : int
        The dropout value (default=0.1).
    activation : torch class
        The activation function of encoder/decoder intermediate layer.
        Recommended: relu or gelu (default=relu).

    Example
    -------
    >>> src = torch.rand([8, 120, 512])
    >>> tgt = torch.randint(0, 720, [8, 120])
    >>> net = TransformerASR(
    ...     720, 512, 512, 8, 1, 1, 1024, activation=torch.nn.GELU
    ... )
    >>> enc_out, dec_out = net.forward(src, tgt)
    >>> enc_out.shape
    torch.Size([8, 120, 512])
    >>> dec_out.shape
    torch.Size([8, 120, 512])
    """

    def __init__(
        self,
        tgt_vocab,
        input_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        positional_encoding=True,
        normalize_before=False,
        kernel_size: Optional[int] = 31,
        bias: Optional[bool] = True,
        encoder_module: Optional[str] = "transformer",
        conformer_activation: Optional[nn.Module] = Swish,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            positional_encoding=positional_encoding,
            normalize_before=normalize_before,
            kernel_size=kernel_size,
            bias=bias,
            encoder_module=encoder_module,
            conformer_activation=conformer_activation,
        )

        self.custom_src_module = ModuleList(
            Linear(
                input_size=input_size,
                n_neurons=d_model,
                bias=True,
                combine_dims=False,
            ),
            torch.nn.Dropout(dropout),
        )
        self.custom_tgt_module = ModuleList(
            NormalizedEmbedding(d_model, tgt_vocab)
        )

        # reset parameters using xavier_normal_
        self._init_params()

    def forward(
        self, src, tgt, wav_len=None, pad_idx=0,
    ):
        """
        Arguments
        ----------
        src : tensor
            The sequence to the encoder (required).
        tgt : tensor
            The sequence to the decoder (required).
        pad_idx : int
            The index for <pad> token (default=0).
        """

        # reshpae the src vector to [Batch, Time, Fea] is a 4d vector is given
        if src.dim() == 4:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        (
            src_key_padding_mask,
            tgt_key_padding_mask,
            src_mask,
            tgt_mask,
        ) = self.make_masks(src, tgt, wav_len, pad_idx=pad_idx)

        src = self.custom_src_module(src)
        src = src + self.positional_encoding(src)
        encoder_out, _ = self.encoder(
            src=src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        tgt = self.custom_tgt_module(tgt)
        tgt = tgt + self.positional_encoding(tgt)
        decoder_out, _, _ = self.decoder(
            tgt=tgt,
            memory=encoder_out,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        return encoder_out, decoder_out

    def make_masks(self, src, tgt, wav_len=None, pad_idx=0):
        """This method generates the masks for training the transformer model.

        Arguments
        ---------
        src : tensor
            The sequence to the encoder (required).
        tgt : tensor
            The sequence to the decoder (required).
        pad_idx : int
            The index for <pad> token (default=0).
        """
        src_key_padding_mask = None
        if wav_len is not None and self.training:
            abs_len = torch.round(wav_len * src.shape[1])
            src_key_padding_mask = (1 - length_to_mask(abs_len)).bool()
        tgt_key_padding_mask = get_key_padding_mask(tgt, pad_idx=pad_idx)

        src_mask = None
        tgt_mask = get_lookahead_mask(tgt)
        return src_key_padding_mask, tgt_key_padding_mask, src_mask, tgt_mask

    def decode(self, tgt, encoder_out):
        """This method implements a decoding step for the transformer model.

        Arguments
        ---------
        tgt : tensor
            The sequence to the decoder (required).
        encoder_out : tensor
            Hidden output of the encoder (required).
        """
        tgt_mask = get_lookahead_mask(tgt)
        tgt = self.custom_tgt_module(tgt)
        tgt = tgt + self.positional_encoding(tgt)
        prediction, self_attns, multihead_attns = self.decoder(
            tgt, encoder_out, tgt_mask=tgt_mask
        )
        return prediction, multihead_attns[-1]

    def encode(
        self, src, wav_len=None,
    ):
        """
        forward the encoder with source input

        Arguments
        ----------
        src : tensor
            The sequence to the encoder (required).
        """
        # reshape the src vector to [Batch, Time, Fea] if a 4d vector is given
        if src.dim() == 4:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        src_key_padding_mask = None
        if wav_len is not None and self.training:
            abs_len = torch.round(wav_len * src.shape[1])
            src_key_padding_mask = (1 - length_to_mask(abs_len)).bool()

        src = self.custom_src_module(src)
        src = src + self.positional_encoding(src)
        encoder_out, _ = self.encoder(
            src=src, src_key_padding_mask=src_key_padding_mask
        )
        return encoder_out

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)
