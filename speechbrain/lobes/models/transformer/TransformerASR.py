"""Transformer for ASR in the SpeechBrain sytle

Authors
* Jianyuan Zhong 2020
"""

import math
import torch  # noqa 42
from torch import nn

from speechbrain.nnet.linear import Linear
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.nnet.containers import Sequential
from speechbrain.nnet.embedding import Embedding
from speechbrain.lobes.models.transformer.Transformer import (
    TransformerInterface,
    get_lookahead_mask,
    PositionalEncoding,
)
from torch.nn import Transformer


class TransformerASR(TransformerInterface):
    """This is an implementation of transformer model for ASR

    The architecture is based on the paper "Attention Is All You Need": https://arxiv.org/pdf/1706.03762.pdf

    Arguements
    ----------
    d_model: int
        the number of expected features in the encoder/decoder inputs (default=512).
    nhead: int
        the number of heads in the multiheadattention models (default=8).
    num_encoder_layers: int
        the number of sub-encoder-layers in the encoder (default=6).
    num_decoder_layers: int
        the number of sub-decoder-layers in the decoder (default=6).
    dim_ffn: int
        the dimension of the feedforward network model (default=2048).
    dropout: int
        the dropout value (default=0.1).
    activation: torch class
        the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu)

    Example
    -------
    >>> src = torch.rand([8, 120, 60])
    >>> tgt = torch.randint(0, 720, [8, 120])
    >>> net = TransformerASR(720, 512, 8, 1, 1, 1024, activation=torch.nn.GELU)
    >>> enc_out, dec_out = net.forward(src, tgt, init_params=True)
    >>> print(enc_out.shape)
    torch.Size([8, 120, 512])
    >>> print(dec_out.shape)
    torch.Size([8, 120, 512])
    """

    def __init__(
        self,
        tgt_vocab,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        return_attention=False,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            return_attention=False,
        )

        positional_encoding = PositionalEncoding(dropout)
        transformer = Transformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            d_ffn,
            dropout,
            "gelu",
        )
        self.encoder = Encoder(positional_encoding, transformer.encoder)
        self.decoder = Decoder(positional_encoding, transformer.decoder)

        self.custom_src_module = Sequential(
            Linear(d_model, bias=True, combine_dims=False),
            LayerNorm(),
            # positional_encoding,
            # torch.nn.Dropout(dropout),
        )
        self.custom_tgt_module = Sequential(
            NormalizedEmbedding(d_model, tgt_vocab),
            # positional_encoding
        )

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        init_params=False,
    ):
        if src.dim() == 4:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        src = self.custom_src_module(src, init_params)
        encoder_out = self.encoder(
            src=src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            init_params=init_params,
        )

        tgt = self.custom_tgt_module(tgt, init_params)
        decoder_out = self.decoder(
            tgt=tgt,
            memory=encoder_out,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            init_params=init_params,
        )

        return encoder_out, decoder_out

    def decode(self, tgt, encoder_out):
        tgt_mask = get_lookahead_mask(tgt)
        tgt = self.custom_tgt_module(tgt)
        prediction = self.decoder(tgt, encoder_out, tgt_mask=tgt_mask)
        return prediction


class Encoder(nn.Module):
    def __init__(self, positional_encoding, encoder):
        super().__init__()
        self.posisitonal_encoding = positional_encoding
        self.encoder = encoder

    def init_params(self, x):
        self.posisitonal_encoding = self.posisitonal_encoding.to(x.device)
        self.encoder = self.encoder.to(x.device)

    def forward(
        self, src, src_mask=None, src_key_padding_mask=None, init_params=False
    ):

        if init_params:
            self.init_params(src)

        src = src + self.posisitonal_encoding(src, init_params)
        src = src.permute(1, 0, 2)
        src = self.encoder(
            src=src, mask=src_mask, src_key_padding_mask=src_key_padding_mask,
        )
        return src.permute(1, 0, 2)


class Decoder(nn.Module):
    def __init__(self, positional_encoding, decoder):
        super().__init__()
        self.posisitonal_encoding = positional_encoding
        self.decoder = decoder

    def init_params(self, x):
        self.posisitonal_encoding = self.posisitonal_encoding.to(x.device)
        self.decoder = self.decoder.to(x.device)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        init_params=False,
    ):

        if init_params:
            self.init_params(tgt)

        tgt = tgt + self.posisitonal_encoding(tgt, init_params)
        tgt = tgt.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)

        output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return output.permute(1, 0, 2)


class NormalizedEmbedding(nn.Module):
    """This class implements the normalized embedding layer for transformer.
    Since the dot product of the self-attention is always normalized by sqrt(d_model)
    and the final linear projection for prediction shares weight with the embedding layer,
    we multiply the output of the embedding by sqrt(d_model)

    Arguments
    ---------
    d_model: int
        the number of expected features in the encoder/decoder inputs (default=512).
    vocab: int
        the vocab size
    """

    def __init__(self, d_model, vocab):
        super().__init__()
        self.emb = Embedding(
            num_embeddings=vocab, embedding_dim=d_model, blank_id=0
        )
        self.d_model = d_model

    def forward(self, x, init_params=False):
        return self.emb(x, init_params) * math.sqrt(self.d_model)
