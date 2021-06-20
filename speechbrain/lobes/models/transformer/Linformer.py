"""
SpeechBrain style implementation of: Linformer Encoder
"""

import torch
import torch.nn as nn
import speechbrain as sb
from typing import Optional
from speechbrain.nnet.attention import (
    LinearMultiheadAttention,
    get_EF,
)


class LinformerEncoderLayer(nn.Module):
    """This is an implementation of Linformer encoder layer.

    Arguements
    ----------
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    nhead : int
        Number of attention heads.
    d_model : int
        The expected size of the input embedding.
    dropout : float
        Dropout for the encoder (Optional).
    activation : PyTorch activation module
        Activation function to use
    normalize_before : bool
        If normalization is done before the current layer
    max_seq_len : int
        the maximum sequence lenght you want to use
    proj_k : int
        in which lower dimension you want to project your matrix
    param_sharing : str
        in you want to share parameters across
    method : str
        method -> convolution or learnable matrix
    layerwise_proj : str
        if you want to project layerwise

    Example
    -------
    # >>> import torch
    # >>> linf_enc = LinformerEncoderLayer(
    # >>> d_ffn=12,
    # >>> nhead=8,
    # >>> d_model=128,
    # >>> dropout=0.1,
    # >>> activation=nn.ReLU,
    # >>> normalize_before=False,
    # >>> max_seq_len=256,
    # >>> proj_k=256//2,
    # >>> param_sharing='none',
    # >>> method='learnable',
    # >>> layerwise_proj=None)
    # >>> x = torch.rand((2, 256, 128))
    # >>> linf_enc(x)[0].shape
    torch.Size([2, 256, 128])
    """

    def __init__(
        self,
        d_ffn,
        nhead,
        d_model=None,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation=nn.ReLU,
        normalize_before=False,
        max_seq_len=1000,
        proj_k=128,
        param_sharing="none",
        method="convolution",
        layerwise_proj=None,
    ):
        super().__init__()

        self.self_att = LinearMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=kdim,
            vdim=vdim,
            seq_len=max_seq_len,
            proj_k=proj_k,
            param_sharing=param_sharing,
            method=method,
            layerwise_proj=layerwise_proj,
        )

        self.pos_ffn = sb.nnet.attention.PositionalwiseFeedForward(
            d_ffn=d_ffn,
            input_size=d_model,
            dropout=dropout,
            activation=activation,
        )

        self.norm1 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.norm2 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Arguements
        ----------
        src : tensor
            The sequence to the encoder layer (required).
        src_mask : tensor
            The mask for the src sequence (optional).
        src_key_padding_mask : tensor
            The mask for the src keys per batch (optional).
        """
        if self.normalize_before:
            src1 = self.norm1(src)
        else:
            src1 = src

        output, self_attn = self.self_att(
            src1,
            src1,
            src1,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )

        # add & norm
        src = src + self.dropout1(output)
        if not self.normalize_before:
            src = self.norm1(src)

        if self.normalize_before:
            src1 = self.norm2(src)
        else:
            src1 = src
        output = self.pos_ffn(src1)

        # add & norm
        output = src + self.dropout2(output)
        if not self.normalize_before:
            output = self.norm2(output)

        return output, self_attn


class LinformerEncoder(nn.Module):
    """This class implements the Linformer encoder.

    Arguments
    ---------
    num_layers : int
        Number of linformer layers to have.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    nhead : int
        Number of attention heads.
    d_model : int
        The expected size of the input embedding.
    dropout : float
        Dropout for the encoder (Optional).
    activation : PyTorch activation module
        Activation function to use
    normalize_before : bool
        If normalization is done before the current layer
    max_seq_len : int
        the maximum sequence lenght you want to use
    proj_k : int
        in which lower dimension you want to project your matrix
    param_sharing : str
        in you want to share parameters across
    method : str
        method -> convolution or learnable matrix
    layerwise_proj : str
        if you want to project layerwise
    kdim : int
        Dimension of the key (Optional).
    vdim : int
        Dimension of the value (Optional).

    Example
    -------
    # >>> import torch
    # >>> linf_enc = LinformerEncoder(
    # >>> d_ffn=12,
    # >>> nhead=8,
    # >>> d_model=128,
    # >>> dropout=0.1,
    # >>> activation=nn.ReLU,
    # >>> normalize_before=False,
    # >>> max_seq_len=256,
    # >>> proj_k=256//2,
    # >>> param_sharing='none',
    # >>> method='learnable',
    # >>> num_layers=1)
    # >>> x = torch.rand((2, 256, 128))
    # >>> linf_enc(x)[0].shape
    # torch.Size([2, 256, 128])
    """

    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        input_shape=None,
        d_model=None,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation=nn.ReLU,
        normalize_before=False,
        max_seq_len=1000,
        proj_k=128,
        param_sharing="none",
        method="convolution",
    ):
        super().__init__()

        if input_shape is None and d_model is None:
            raise ValueError("Expected one of input_shape or d_model")

        if input_shape is not None and d_model is None:
            if len(input_shape) == 3:
                msg = (
                    "Input shape of the Transformer must be (batch, time, fea). Please revise the forward "
                    "function in TransformerInterface to handel arbitary shape of input."
                )
                raise ValueError(msg)
            d_model = input_shape[-1]

        self.layerwise_proj = None
        if param_sharing == "layerwise":
            self.layerwise_proj = get_EF(
                max_seq_len, proj_k, method=method, head_dim=d_model, bias=True
            )

        self.layers = torch.nn.ModuleList(
            [
                LinformerEncoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    max_seq_len=max_seq_len,
                    proj_k=proj_k,
                    param_sharing=param_sharing,
                    method=method,
                    layerwise_proj=self.layerwise_proj,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ----------
        src : tensor
            The sequence to the encoder layer (required).
        src_mask : tensor
            The mask for the src sequence (optional).
        src_key_padding_mask : tensor
            The mask for the src keys per batch (optional).
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
