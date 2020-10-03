"""An implementation of Transformer Language model

Authors
* Jianyuan Zhong
"""

import torch  # noqa 42
from torch import nn

from speechbrain.nnet.linear import Linear
from speechbrain.nnet.containers import Sequential
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.lobes.models.transformer.Transformer import (
    TransformerInterface,
    get_lookahead_mask,
    get_key_padding_mask,
    NormalizedEmbedding,
)


class TransformerLM(TransformerInterface):
    """This is an implementation of transformer language model

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
    >>> src = torch.randint(0, 720, [8, 120])
    >>> net = TransformerLM(None, 720, 512, 8, 1, 0, 1024, activation=torch.nn.GELU)
    >>> enc_out = net.forward(src, init_params=True)
    >>> print(enc_out.shape)
    torch.Size([8, 120, 720])
    """

    def __init__(
        self,
        masking_func,
        vocab,
        d_model=512,
        nhead=8,
        num_encoder_layers=12,
        num_decoder_layers=0,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        return_attention=False,
        positional_encoding=True,
        normalize_before=False,
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
            normalize_before=normalize_before,
        )

        self.custom_src_module = NormalizedEmbedding(d_model, vocab)
        self.output_proj = Sequential(
            Linear(d_model), LayerNorm(eps=1e-6), Linear(vocab)
        )

        self.masking_func = masking_func
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

    def forward(
        self, src, init_params=False,
    ):
        """
        Arguements
        ----------
        src: tensor
            the sequence to the encoder (required).
        tgt: tensor
            the sequence to the decoder (required).
        """
        src_mask, src_key_padding_mask = None, None
        if self.masking_func is not None:
            src_mask, src_key_padding_mask = self.masking_func(src)

        src = self.custom_src_module(src, init_params)
        src = src + self.positional_encoding(src, init_params)
        if self.num_encoder_layers > 0:
            encoder_out = self.encoder(
                src=src,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                init_params=init_params,
            )

        if self.num_decoder_layers > 0:
            encoder_out = self.decoder(
                src,
                src,
                tgt_mask=src_mask,
                tgt_key_padding_mask=src_key_padding_mask,
                init_params=init_params,
            )

        pred = self.output_proj(encoder_out, init_params)

        if init_params:
            self.reset_params()
#            self.output_proj.weight = (
#                self.custom_src_module.emb.Embedding.weight
#            )

        return pred

    def reset_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)


def make_masks(src, pad_idx=0, look_ahead_mask=True, padding_mask=True):
    src_mask = None
    if look_ahead_mask:
        src_mask = get_lookahead_mask(src)

    src_key_padding_mask = None
    if padding_mask:
        src_key_padding_mask = get_key_padding_mask(src, pad_idx)

    return src_mask, src_key_padding_mask
