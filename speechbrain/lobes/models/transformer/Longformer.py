import torch
import torch.nn as nn
from speechbrain.nnet.attention_utils.longformer_utilities import (
    longformer_src_padder,
)
import speechbrain as sb
from typing import Optional


class LongformerEncoder(nn.Module):
    """
    Longformer encoder interface implementation in the SpeechBrain style.
    Authors
    * Most of the code comes from: https://github.com/allenai/longformer
    Longformer is an open-source project developed by the Allen Institute for Artificial Intelligence (AI2).
    AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and
    engineering.

    The architecture is based on the paper "Longformer: The Long-Document Transformer":
    https://arxiv.org/pdf/2004.05150.pdf

    * Modification to fit SpeechBrain's interface by Jonathan Tremblay (2021)

    Arguments
    ---------
    num_layers : int
        Number of Longformer layers to include.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    attention_window : list
        List of the window size to use
    attention_mode : str
        Type of Longformer attention -> currently supports 'sliding_chunks' and 'sliding_chunks_no_overlap'
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
    # >>> import torch
    # >>> x = torch.rand((8, 60, 512))
    # >>> net = LongformerEncoder(
    # >>> d_ffn=128,
    # >>> nhead=4,
    # >>> num_layers=1,
    # >>> d_model=512,
    # >>> attention_window=[12],
    # >>> attention_mode='sliding_chunks',
    # >>> dropout=0.1,
    # >>> activation=nn.ReLU,
    # >>> normalize_before=False)
    # >>> output, _ = net(x)
    # >>> output.shape
    torch.Size([8, 72, 512])
    """

    def __init__(
        self,
        d_ffn,
        num_layers,
        nhead,
        attention_window: list,
        attention_mode: str = "sliding_chunks",
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
                LongformerEncoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    layer_id=i,
                    attention_window=attention_window,
                    attention_mode=attention_mode,
                    num_layers=num_layers,
                    d_model=d_model,
                    dropout=dropout,
                    normalize_before=normalize_before,
                    activation=activation,
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


class LongformerEncoderLayer(nn.Module):
    """
    This is an implementation of Longformer self-attention encoder layer in the SpeechBrain style.
    Authors
    * Most of the code comes from: https://github.com/allenai/longformer
    Longformer is an open-source project developed by the Allen Institute for Artificial Intelligence (AI2).
    AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and
    engineering.

    The architecture is based on the paper "Longformer: The Long-Document Transformer":
    https://arxiv.org/pdf/2004.05150.pdf

    * Modification to fit SpeechBrain's interface by Jonathan Tremblay (2021)

    Arguments
    ---------
    num_layers : int
        Number of Longformer layers to include.
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
    # >>> import torch
    # >>> x = torch.rand((8, 60, 512))
    # >>> net = LongformerEncoderLayer(
    # >>> d_ffn=128,
    # >>> nhead=4,
    # >>> num_layers=1,
    # >>> layer_id=0,
    # >>> d_model=512,
    # >>> attention_window=[12],
    # >>> attention_mode='sliding_chunks',
    # >>> dropout=0.1,
    # >>> activation=nn.ReLU,
    # >>> normalize_before=False)
    # >>> output, _ = net(x)
    # >>> output.shape
    torch.Size([8, 72, 512])
    """

    def __init__(
        self,
        d_ffn,
        nhead,
        layer_id,
        attention_window,
        attention_mode,
        num_layers,
        d_model=None,
        dropout=0.1,
        activation=nn.ReLU,
        normalize_before=False,
    ):
        super().__init__()
        self.attention_window = attention_window
        self.self_att = sb.nnet.attention.LongformerSelfAttention(
            layer_id=layer_id,
            num_attention_heads=nhead,
            hidden_size=d_model,
            attention_probs_dropout_prob=dropout,
            attention_window=attention_window,
            attention_mode=attention_mode,
            attention_dilation=[1] * num_layers,  # Not implemented yet
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
        # TODO: Masks are not yet implemented within the Longformer but this doesn't cause any issue for ASR
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
            window_padding_size=self.attention_window[0],
            permutation=True,
        )

        output, self_attn = self.self_att(
            hidden_states=src1, output_attentions=True
        )
        src = longformer_src_padder(
            src, window_padding_size=self.attention_window[0], permutation=True
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
