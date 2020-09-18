"""An implementation of Transformer Language model

Authors
* Jianyuan Zhong
"""
import random
import torch  # noqa 42
from torch import nn
from transformers import BertModel

from speechbrain.nnet.linear import Linear
from speechbrain.nnet.containers import Sequential
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.lobes.models.transformer.Transformer import NormalizedEmbedding, PositionalEncoding, get_lookahead_mask, get_key_padding_mask


class Bert(nn.Module):
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
    >>> a = torch.LongTensor([[1,1,0], [2,3,0], [4,5,0]])
    >>> bert = Bert(make_masks, 100)
    >>> out = bert(a, True)
    >>> print(out.shape)
    torch.Size([3, 3, 100])
    """

    def __init__(
        self,
        vocab,
        masking_func,
        d_model=768,
        dropout=0.1
    ):
        super().__init__()

        self.custom_src_module = BertEmbedding(d_model, vocab, dropout)
        pretrain = BertModel.from_pretrained(
            'bert-base-uncased',
            return_dict=False
        )
        self.bert = pretrain.encoder
        self.output_proj = Sequential(
            Linear(d_model),
            LayerNorm(eps=1e-12),
            Linear(vocab)
        )

        self.masking_func = masking_func

    def init_params(self, first_input):
        self.bert = self.bert.to(first_input.device)

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
        if init_params:
            self.init_params(src)

        src_mask = None
        if self.masking_func is not None:
            src, src_mask = self.masking_func(src, self.training)

        src = self.custom_src_module(src, init_params)
        encoder_out = self.bert(
            src,
            attention_mask=src_mask,
        )

        pred = self.output_proj(encoder_out[0], init_params)
        return pred


class BertEmbedding(nn.Module):

    def __init__(self, d_model, vocab, dropout=0.1):
        super().__init__()
        self.token_emb = NormalizedEmbedding(d_model, vocab)
        self.pos_emb = PositionalEncoding(512)
        self.norm = LayerNorm(eps=1e-12)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, init_params):
        x = self.token_emb(x, init_params)
        x = x + self.pos_emb(x, init_params)
        x = self.norm(x, init_params)
        return self.drop(x)


def make_masks(src, istraining, pad_idx=0, padding_mask=True, token_masking=True, p=0.15):
    src_key_padding_mask = None
    if padding_mask:
        src_key_padding_mask = get_key_padding_mask(src, pad_idx).int()
        src_key_padding_mask = 1 - src_key_padding_mask
        # extended_attention_mask = src_key_padding_mask[:, None, None, :]

        batch_size, seq_length = src.shape
        seq_ids = torch.arange(seq_length, device=src.device)
        causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        # causal and attention masks must have same type with pytorch version < 1.3
        causal_mask = causal_mask.to(src_key_padding_mask.dtype)
        extended_attention_mask = causal_mask[:, None, :, :] * src_key_padding_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    # if istraining:
    #     to_mask = (torch.rand(src.shape, device=src.device) > 0.15).int()
    #     src = src * to_mask

    return src, extended_attention_mask
