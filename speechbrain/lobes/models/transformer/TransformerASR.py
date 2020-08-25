"""Transformer for ASR in the SpeechBrain sytle

Authors
* Jianyuan Zhong 2020
"""

import math
import torch  # noqa 42
from torch import nn
import speechbrain as sb

from speechbrain.lobes.models.transformer.Transformer import (
    TransformerInterface,
    get_lookahead_mask,
)


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
    >>> enc_out, dec_out = net.forward(src, tgt)
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
        positional_encoding=True,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            return_attention=return_attention,
            positional_encoding=positional_encoding,
        )

        self.custom_src_module = sb.nnet.Sequential(
            sb.nnet.Linear(d_model, bias=True),
            sb.nnet.LayerNorm(),
            torch.nn.Dropout(dropout),
        )
        self.custom_tgt_module = sb.nnet.Sequential(
            NormalizedEmbedding(d_model, tgt_vocab),
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
        """
        Arguements
        ----------
        src: tensor
            the sequence to the encoder (required).
        tgt: tensor
            the sequence to the decoder (required).
        src_mask: tensor
            the additive mask for the src sequence (optional).
        tgt_mask: tensor
            the additive mask for the tgt sequence (optional).
        src_key_padding_mask: tensor
            the ByteTensor mask for src keys per batch (optional).
        tgt_key_padding_mask: tensor
            the ByteTensor mask for tgt keys per batch (optional).
        memory_key_padding_mask: tensor
            the ByteTensor mask for memory keys per batch (optional).
        """
        if src.dim() == 4:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        src = self.custom_src_module(src, init_params)
        src = src + self.positional_encoding(src, init_params)
        encoder_out = self.encoder(
            src=src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            init_params=init_params,
        )

        tgt = self.custom_tgt_module(tgt, init_params)
        tgt = tgt + self.positional_encoding(tgt, init_params)
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
        tgt = tgt + self.positional_encoding(tgt)
        prediction = self.decoder(tgt, encoder_out, tgt_mask=tgt_mask)
        return prediction


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

    Example
    -------
    >>> emb = NormalizedEmbedding(512, 1000)
    >>> trg = torch.randint(0, 999, (8, 50))
    >>> emb_fea = emb(trg)
    """

    def __init__(self, d_model, vocab):
        super().__init__()
        self.emb = sb.nnet.Embedding(
            num_embeddings=vocab, embedding_dim=d_model, blank_id=0
        )
        self.d_model = d_model

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)
