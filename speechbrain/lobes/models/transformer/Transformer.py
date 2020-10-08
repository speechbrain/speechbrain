"""Transformer implementaion in the SpeechBrain sytle

Authors
* Jianyuan Zhong 2020
"""

import torch
import math
import torch.nn as nn
import speechbrain as sb
from typing import Optional


class TransformerInterface(nn.Module):
    """This is an interface for transformer model.

    Users can modify the attributes and define the forward function as
    needed according to their own tasks.

    The architecture is based on the paper "Attention Is All You Need":
    https://arxiv.org/pdf/1706.03762.pdf

    Arguements
    ----------
    input_size : int
        Expected size of input features.
    d_model : int
        the number of expected features in the encoder/decoder inputs (default=512).
    nhead : int
        the number of heads in the multiheadattention models (default=8).
    num_encoder_layers : int
        the number of sub-encoder-layers in the encoder (default=6).
    num_decoder_layers : int
        the number of sub-decoder-layers in the decoder (default=6).
    dim_ffn : int
        the dimension of the feedforward network model (default=2048).
    dropout : int
        the dropout value (default=0.1).
    activation : torch class
        the activation function of encoder/decoder intermediate layer,
        e.g. relu or gelu (default=relu)
    custom_src_module : torch class
        module that process the src features to expected feature dim
    custom_tgt_module : torch class
        module that process the src features to expected feature dim
    """

    def __init__(
        self,
        input_size,
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
    ):
        super().__init__()

        assert (
            num_encoder_layers + num_decoder_layers > 0
        ), "number of encoder layers and number of decoder layers cannot both be 0!"

        if positional_encoding:
            self.positional_encoding = PositionalEncoding(input_size)

        # initialize the encoder
        if num_encoder_layers > 0:
            if custom_src_module is not None:
                self.custom_src_module = custom_src_module(d_model)

            self.encoder = TransformerEncoder(
                nhead=nhead,
                num_layers=num_encoder_layers,
                d_ffn=d_ffn,
                embed_dim=input_size,
                dropout=dropout,
                activation=activation,
            )

        # initialize the decoder
        if num_decoder_layers > 0:
            if custom_tgt_module is not None:
                self.custom_tgt_module = custom_tgt_module(d_model)

            self.decoder = TransformerDecoder(
                num_layers=num_decoder_layers,
                nhead=nhead,
                d_ffn=d_ffn,
                embed_dim=input_size,
                dropout=dropout,
                activation=activation,
            )

    def forward(self, **kwags):
        """Users should modify this function according to their own tasks
        """
        raise NotImplementedError


class PositionalEncoding(nn.Module):
    """This class implements the positional encoding function

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))

    Arguements
    ----------
    max_len :
        max length of the input sequences (default 2500)

    Example
    -------
    >>> a = torch.rand((8, 120, 512))
    >>> enc = PositionalEncoding(input_size=a.shape[-1])
    >>> b = enc(a)
    >>> b.shape
    torch.Size([1, 120, 512])
    """

    def __init__(self, input_size, max_len=2500):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(self.max_len, input_size, requires_grad=False)
        positions = torch.arange(0, self.max_len).unsqueeze(1).float()
        denominator = torch.exp(
            torch.arange(0, input_size, 2).float()
            * -(math.log(10000.0) / input_size)
        )

        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguements
        ----------
        x : Tensor
            input feature (batch, time, fea)
        """
        return self.pe[:, : x.size(1)].clone().detach()


class TransformerEncoderLayer(nn.Module):
    """ This is an implementation of self-attention encoder layer

    Arguements
    ----------
    d_ffn : int
        Hidden size of self-attention Feed Forward layer
    nhead : int
        number of attention heads
    embed_dim : int
        The expected size of the input embedding
    reshape : bool
        Whether to automatically shape 4-d input to 3-d
    kdim : int
        dimension for key (Optional)
    vdim : int
        dimension for value (Optional)
    dropout : int
        dropout for the encoder (Optional)

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = TransformerEncoderLayer(512, 8, embed_dim=512)
    >>> output = net(x)
    >>> output[0].shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_ffn,
        nhead,
        embed_dim=None,
        reshape=False,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.reshape = reshape

        self.self_att = sb.nnet.MultiheadAttention(
            nhead=nhead,
            embed_dim=embed_dim,
            dropout=dropout,
            kdim=kdim,
            vdim=vdim,
        )
        self.pos_ffn = sb.nnet.PositionalwiseFeedForward(
            d_ffn=d_ffn,
            input_size=embed_dim,
            dropout=dropout,
            activation=activation,
        )

        self.norm1 = sb.nnet.LayerNorm([None, None, embed_dim], eps=1e-6)
        self.norm2 = sb.nnet.LayerNorm([None, None, embed_dim], eps=1e-6)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Arguements
        ----------
        src: tensor
            the sequence to the encoder layer (required).
        src_mask: tensor
            the mask for the src sequence (optional).
        src_key_padding_mask: tensor
            the mask for the src keys per batch (optional).
        """
        in_shape = src.shape
        if self.reshape:
            src = src.reshape(
                in_shape[0], in_shape[1], in_shape[2] * in_shape[3]
            )

        output, self_attn = self.self_att(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )

        # add & norm
        src = src + self.dropout1(output)
        src = self.norm1(src)

        output = self.pos_ffn(src)

        # add & norm
        output = src + self.dropout2(output)
        output = self.norm2(output)

        return output, self_attn


class TransformerEncoder(nn.Module):
    """This class implements the transformer encoder

    Arguements
    ----------
    num_layers : int
        Number of transformer layers to include
    nhead : int
        number of attention heads
    d_ffn : int
        Hidden size of self-attention Feed Forward layer
    input_shape : tuple
        Expected shape of an example input.
    embed_dim : int
        The dimension of the input embedding.
    kdim : int
        dimension for key (Optional)
    vdim : int
        dimension for value (Optional)
    dropout : float
        dropout for the encoder (Optional)
    input_module: torch class
        the module to process the source input feature to expected
        feature dimension (Optional)

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = TransformerEncoder(1, 8, 512, embed_dim=512)
    >>> output, _ = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        input_shape=None,
        embed_dim=None,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation=nn.ReLU,
    ):
        super().__init__()

        if input_shape is None and embed_dim is None:
            raise ValueError("Expected one of input_shape or embed_dim")

        reshape_first_layer = False
        if embed_dim is None:
            if len(input_shape) == 4:
                reshape_first_layer = True
                embed_dim = input_shape[-2] * input_shape[-1]
            else:
                embed_dim = input_shape[-1]

        self.layers = torch.nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    embed_dim=embed_dim,
                    reshape=reshape_first_layer if i == 0 else False,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = sb.nnet.LayerNorm([None, None, embed_dim], eps=1e-6)

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Arguements
        ----------
        src: tensor
            the sequence to the encoder layer (required).
        src_mask: tensor
            the mask for the src sequence (optional).
        src_key_padding_mask: tensor
            the mask for the src keys per batch (optional).
        """
        output = src
        attention_lst = []
        for enc_layer in self.layers:
            output, attention = enc_layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
            )
            attention_lst.append(attention)
        output = self.norm(output)

        return output, attention_lst


class TransformerDecoderLayer(nn.Module):
    """This class implements the self-attention decoder layer

    Arguements
    ----------
    d_ffn : int
        Hidden size of self-attention Feed Forward layer
    nhead : int
        number of attention heads
    embed_dim : int
        dimension of the model
    kdim : int
        dimension for key (optional)
    vdim : int
        dimension for value (optional)
    dropout : float
        dropout for the decoder (optional)

    Example
    -------
    >>> src = torch.rand((8, 60, 512))
    >>> tgt = torch.rand((8, 60, 512))
    >>> net = TransformerDecoderLayer(1024, 8, embed_dim=512)
    >>> output, self_attn, multihead_attn = net(src, tgt)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_ffn,
        nhead,
        embed_dim,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.self_attn = sb.nnet.MultiheadAttention(
            nhead=nhead,
            embed_dim=embed_dim,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
        )
        self.mutihead_attn = sb.nnet.MultiheadAttention(
            nhead=nhead,
            embed_dim=embed_dim,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
        )
        self.pos_ffn = sb.nnet.PositionalwiseFeedForward(
            d_ffn=d_ffn,
            input_size=embed_dim,
            dropout=dropout,
            activation=activation,
        )

        # normalization layers
        self.norm1 = torch.nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = torch.nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm3 = torch.nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        Arguements
        ----------
        tgt: tensor
            the sequence to the decoder layer (required).
        memory: tensor
            the sequence from the last layer of the encoder (required).
        tgt_mask: tensor
            the mask for the tgt sequence (optional).
        memory_mask: tensor
            the mask for the memory sequence (optional).
        tgt_key_padding_mask: tensor
            the mask for the tgt keys per batch (optional).
        memory_key_padding_mask: tensor
            the mask for the memory keys per batch (optional).
        """
        # self-attention over the target sequence
        tgt2, self_attn = self.self_attn(
            query=tgt,
            key=tgt,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )

        # add & norm
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # multi-head attention over the target sequence and encoder states
        tgt2, multihead_attention = self.mutihead_attn(
            query=tgt,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )

        # add & norm
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt = self.pos_ffn(tgt)

        # add & norm
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        return tgt, self_attn, multihead_attention


class TransformerDecoder(nn.Module):
    """This class implements the Transformer decoder

    Arguements
    ----------
    d_ffn : int
        Hidden size of self-attention Feed Forward layer
    nhead : int
        number of attention heads
    embed_dim : int
        dimension of the model
    kdim : int
        dimension for key (Optional)
    vdim : int
        dimension for value (Optional)
    dropout : float
        dropout for the decoder (Optional)

    Example
    -------
    >>> src = torch.rand((8, 60, 512))
    >>> tgt = torch.rand((8, 60, 512))
    >>> net = TransformerDecoder(1, 8, 1024, embed_dim=512)
    >>> output, _, _ = net(src, tgt)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        embed_dim,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    embed_dim=embed_dim,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = torch.nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        Arguements
        ----------
        tgt: tensor
            the sequence to the decoder layer (required).
        memory: tensor
            the sequence from the last layer of the encoder (required).
        tgt_mask: tensor
            the mask for the tgt sequence (optional).
        memory_mask: tensor
            the mask for the memory sequence (optional).
        tgt_key_padding_mask: tensor
            the mask for the tgt keys per batch (optional).
        memory_key_padding_mask: tensor
            the mask for the memory keys per batch (optional).
        """
        output = tgt
        self_attns, multihead_attns = [], []
        for dec_layer in self.layers:
            output, self_attn, multihead_attn = dec_layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            self_attns.append(self_attn)
            multihead_attns.append(multihead_attn)
        output = self.norm(output)

        return output, self_attns, multihead_attns


def get_key_padding_mask(padded_input, pad_idx):
    """Create a binary mask to prevent attention to padded locations

    Arguements
    ----------
    padded_input: int
        padded input
    pad_idx:
        idx for padding element

    Example
    -------
    >>> a = torch.LongTensor([[1,1,0], [2,3,0], [4,5,0]])
    >>> get_key_padding_mask(a, pad_idx=0)
    tensor([[False, False,  True],
            [False, False,  True],
            [False, False,  True]])
    """
    if len(padded_input.shape) == 4:
        bz, time, ch1, ch2 = padded_input.shape
        padded_input = padded_input.reshape(bz, time, ch1 * ch2)

    key_padded_mask = padded_input.eq(pad_idx)

    # if the input is more than 2d, mask the locations where they are silence
    # across all channels
    if len(padded_input.shape) > 2:
        key_padded_mask = key_padded_mask.float().prod(dim=-1).bool()
        return key_padded_mask.detach()

    return key_padded_mask.detach()


def get_lookahead_mask(padded_input):
    """Creates a binary mask for each sequence.

    Arguements
    ----------
    padded_input : tensor

    Example
    -------
    >>> a = torch.LongTensor([[1,1,0], [2,3,0], [4,5,0]])
    >>> get_lookahead_mask(a)
    tensor([[0., -inf, -inf],
            [0., 0., -inf],
            [0., 0., 0.]])
    """
    seq_len = padded_input.shape[1]
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask.detach().to(padded_input.device)
