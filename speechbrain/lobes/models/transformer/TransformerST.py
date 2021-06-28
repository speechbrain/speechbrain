"""Transformer for ST in the SpeechBrain sytle.

Authors
* YAO FEI, CHENG 2021
"""

import logging
import torch  # noqa 42
from torch import nn
from typing import Optional

from speechbrain.nnet.containers import ModuleList
from speechbrain.lobes.models.transformer.Transformer import (
    get_lookahead_mask,
    get_key_padding_mask,
    NormalizedEmbedding,
    TransformerDecoder,
    TransformerEncoder,
)
from speechbrain.lobes.models.transformer.TransformerASR import TransformerASR
from speechbrain.nnet.activations import Swish

logger = logging.getLogger(__name__)


class TransformerST(TransformerASR):
    """This is an implementation of transformer model for ST.

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
        ctc_weight: float = 0.0,
        asr_weight: float = 0.0,
        mt_weight: float = 0.0,
        asr_tgt_vocab: int = 0,
        mt_src_vocab: int = 0,
    ):
        super().__init__(
            tgt_vocab=tgt_vocab,
            input_size=input_size,
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

        if ctc_weight < 1 and asr_weight > 0:
            self.asr_decoder = TransformerDecoder(
                num_layers=num_decoder_layers,
                nhead=nhead,
                d_ffn=d_ffn,
                d_model=d_model,
                dropout=dropout,
                activation=activation,
                normalize_before=normalize_before,
            )
            self.custom_asr_tgt_module = ModuleList(
                NormalizedEmbedding(d_model, asr_tgt_vocab)
            )

        if mt_weight > 0:
            self.custom_mt_src_module = ModuleList(
                NormalizedEmbedding(d_model, mt_src_vocab)
            )
            self.mt_encoder = TransformerEncoder(
                nhead=nhead,
                num_layers=num_encoder_layers,
                d_ffn=d_ffn,
                d_model=d_model,
                dropout=dropout,
                activation=activation,
                normalize_before=normalize_before,
            )

        # reset parameters using xavier_normal_
        self._init_params()

    def forward_asr(self, encoder_out, tgt, pad_idx):
        """This method implements a decoding step for asr task

        Arguments
        ----------
        encoder_out : tensor
            The representation of the encoder (required).
        tgt (transcription): tensor
            The sequence to the decoder (required).
        pad_idx : int
            The index for <pad> token (default=0).
        """
        tgt_key_padding_mask = get_key_padding_mask(tgt, pad_idx=pad_idx)
        tgt_mask = get_lookahead_mask(tgt)

        transcription = self.custom_asr_tgt_module(tgt)
        transcription = transcription + self.positional_encoding(transcription)
        asr_decoder_out, _, _ = self.asr_decoder(
            tgt=transcription,
            memory=encoder_out,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        return asr_decoder_out

    def forward_mt(self, src, tgt, pad_idx=0):
        """This method implements a forward step for mt task

        Arguments
        ----------
        src (transcription): tensor
            The sequence to the encoder (required).
        tgt (translation): tensor
            The sequence to the decoder (required).
        pad_idx : int
            The index for <pad> token (default=0).
        """

        (
            src_key_padding_mask,
            tgt_key_padding_mask,
            src_mask,
            tgt_mask,
        ) = self.make_masks_for_mt(src, tgt, pad_idx=pad_idx)

        src = self.custom_mt_src_module(src)
        src = src + self.positional_encoding(src)
        encoder_out, _ = self.mt_encoder(
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

    def decode_asr(self, tgt, encoder_out):
        """This method implements a decoding step for the transformer model.

        Arguments
        ---------
        tgt : tensor
            The sequence to the decoder (required).
        encoder_out : tensor
            Hidden output of the encoder (required).
        """
        tgt_mask = get_lookahead_mask(tgt)
        tgt = self.custom_asr_tgt_module(tgt)
        tgt = tgt + self.positional_encoding(tgt)
        prediction, self_attns, multihead_attns = self.decoder(
            tgt, encoder_out, tgt_mask=tgt_mask
        )
        return prediction, multihead_attns[-1]

    def make_masks_for_mt(self, src, tgt, pad_idx=0):
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
        if self.training:
            src_key_padding_mask = get_key_padding_mask(src, pad_idx=pad_idx)
        tgt_key_padding_mask = get_key_padding_mask(tgt, pad_idx=pad_idx)

        src_mask = None
        tgt_mask = get_lookahead_mask(tgt)
        return src_key_padding_mask, tgt_key_padding_mask, src_mask, tgt_mask
