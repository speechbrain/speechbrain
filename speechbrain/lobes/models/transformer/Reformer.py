import torch
import torch.nn as nn
from speechbrain.nnet.attention_utils.longformer_utilities import (
    longformer_src_padder,
)
import speechbrain as sb
from typing import Optional


class ReformerEncoder(nn.Module):
    """
    Reformer encoder interface implementation in the SpeechBrain style.
    Authors
    * Most of the code comes from: https://github.com/lucidrains/reformer-pytorch

    The architecture is based on the paper "Reformer: The Efficient Transformer":
    https://arxiv.org/abs/2001.04451

    * Modification to fit SpeechBrain's interface

    Arguments
    ---------
    num_layers : int
        Number of layers of attention to include.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    n_hashes : int
        Number of hashing rounds
    bucket_size : int
        bucket size to use during the hashing
    input_shape : tuple
        Expected shape of an example input.
    d_model : int
        The dimension of the input embedding.
    dropout : float
        Dropout for the encoder (Optional).
    activation : PyTorch activation module
        Activation function to use
    normalize_before : bool
        If normalization is done before the current layer

    # TODO: Fix the example for Unit Testing
    Example
    -------
    """

    def __init__(
        self,
        d_ffn,
        num_layers,
        nhead,
        n_hashes,
        bucket_size,
        attn_chunks,
        input_shape=None,
        d_model=None,
        dropout=0.1,
        activation=nn.ReLU,
        normalize_before=False,
    ):
        super().__init__()

        if input_shape is None and d_model is None:
            raise ValueError("Expected one of input_shape or d_model")

        if input_shape is not None and d_model is None:
            if len(input_shape) == 3:
                msg = (
                    "Input shape of the Transformer must be (batch, time, fea). Please revise the forward function "
                    "in TransformerInterface to handel arbitary shape of input."
                )
                raise ValueError(msg)
            d_model = input_shape[-1]

        self.layers = torch.nn.ModuleList(
            [
                ReformerEncoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    n_hashes=n_hashes,
                    attn_chunks=attn_chunks,
                    bucket_size=bucket_size,
                    d_model=d_model,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
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


class ReformerEncoderLayer(nn.Module):
    """
    Reformer encoder layer implementation in the SpeechBrain style.
    * Most of the code comes from: https://github.com/lucidrains/reformer-pytorch

    The architecture is based on the paper "Reformer: The Efficient Transformer":
    https://arxiv.org/abs/2001.04451

    * Modification to fit SpeechBrain's interface

    Arguments
    ---------
    num_layers : int
        Number of layers to include.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    layer_id : int
        which layer this corresponds to (linked to the number of encoder)
    attention_window : list
        List of the window size to use
    attention_mode : str
        Type of attention -> currently supports 'sliding_chunks' and 'sliding_chunks_no_overlap'
    d_model : int
        The dimension of the input embedding.
    dropout : float
        Dropout for the encoder (Optional).
    activation : PyTorch activation module
        Activation function to use
    normalize_before : bool
        If normalization is done before the current layer

    # TODO: Fix the example for Unit Testing
    Example
    -------
    """

    def __init__(
        self,
        d_ffn,
        nhead,
        n_hashes,
        bucket_size,
        attn_chunks,
        d_model=None,
        dropout=0.1,
        activation=nn.ReLU,
        normalize_before=False,
    ):
        super().__init__()
        self.attention_window = bucket_size
        self.self_att = sb.nnet.attention.LSHSelfAttention(
            heads=nhead,
            dim=d_model,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            attn_chunks=attn_chunks,
            causal=False,
            allow_duplicate_attention=True,
            attend_across_buckets=True,
            rehash_each_round=True,
            drop_for_hash_rate=0.0,
            random_rotations_per_head=False,
            return_attn=False,
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
        # TODO: Masks are not yet implemented within the Reformer but this doesn't cause any issue for ASR
        # TODO: introduce something similar that's being done within PyTorch library:
        #  https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py
        """
        Arguements
        ----------
        src : tensor
            The sequence to the encoder layer (required).
        src_mask : tensor
            not yet implemented
        src_key_padding_mask : tensor
            not yet implemented
        """
        if self.normalize_before:
            src1 = self.norm1(src)
        else:
            src1 = src

        src1 = longformer_src_padder(
            tens=src1,
            window_padding_size=self.attention_window,
            permutation=True,
        )

        output, self_attn = self.self_att(x=src1)
        src = longformer_src_padder(
            src, window_padding_size=self.attention_window, permutation=True
        )
        src = src + self.dropout1(output)

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
