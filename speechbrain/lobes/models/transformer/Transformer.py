"""Transformer implementation in the SpeechBrain sytle.

Authors
* Jianyuan Zhong 2020
"""
import torch.nn as nn
from typing import Optional
from speechbrain.nnet.activations import Swish
from speechbrain.lobes.models.transformer.TransformerUtilities import PositionalEncoding, TransformerDecoder
from speechbrain.lobes.models.transformer.EncoderManager import EncoderManager


class TransformerInterface(nn.Module):
    """This is an interface for transformer model.

    Users can modify the attributes and define the forward function as
    needed according to their own tasks.

    The architecture is based on the paper "Attention Is All You Need":
    https://arxiv.org/pdf/1706.03762.pdf

    Arguments
    ----------
    d_model : int
        The number of expected features in the encoder/decoder inputs (default=512).
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
        The activation function of encoder/decoder intermediate layer,
        e.g., relu or gelu (default=relu)
    custom_src_module : torch class
        Module that processes the src features to expected feature dim.
    custom_tgt_module : torch class
        Module that processes the src features to expected feature dim.
    longf_attention_window : int
        Size of the attention window size for the Longformer
    longf_attention_mode : str
        Type of attention for the Longformer

    """

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        custom_src_module=None,
        custom_tgt_module=None,
        positional_encoding=True,
        normalize_before=False,
        encoder_module: Optional[str] = "transformer",
        **encoder_arguments
    ):
        super().__init__()

        assert (
            num_encoder_layers + num_decoder_layers > 0
        ), "number of encoder layers and number of decoder layers cannot both be 0!"

        if positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model)

        # initialize the encoder
        if num_encoder_layers > 0:
            if custom_src_module is not None:
                self.custom_src_module = custom_src_module(d_model)
            encoder_manager = EncoderManager(encoder_module)
            self.encoder = encoder_manager.build(
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                d_ffn=d_ffn,
                d_model=d_model,
                dropout=dropout,
                activation=activation,
                normalize_before=normalize_before,
                **encoder_arguments.get('encoder_arguments'))


        # initialize the decoder
        if num_decoder_layers > 0:
            if custom_tgt_module is not None:
                self.custom_tgt_module = custom_tgt_module(d_model)

            self.decoder = TransformerDecoder(
                num_layers=num_decoder_layers,
                nhead=nhead,
                d_ffn=d_ffn,
                d_model=d_model,
                dropout=dropout,
                activation=activation,
                normalize_before=normalize_before,
            )

    def forward(self, **kwags):
        """Users should modify this function according to their own tasks.
        """
        raise NotImplementedError
