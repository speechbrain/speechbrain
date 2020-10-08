import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.nnet.attention import (
    RelativePosMultiHeadAttention,
    PositionalwiseFeedForward,
)

from speechbrain.nnet.group_layer_norm import GroupLayerNorm
from speechbrain.nnet.group_linear import GroupLinear
from speechbrain.lobes.models.transformer.group_communication import (
    GroupCommunication
)


class TransformerEncoderLayerRP(nn.Module):
    """ This is an implementation of self-attention encoder layer

    Arguements
    ----------
    d_ffn : int
        Hidden size of self-attention Feed Forward layer
    nhead : int
        number of attention heads
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
    >>> pos = torch.rand((60, 512))
    >>> net = TransformerEncoderLayerRP(512, 8)
    >>> output = net(x, pos, init_params=True)
    >>> print(output[0].shape)
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_ffn,
        nhead,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation=nn.ReLU,
        num_modules=1,
        use_group_comm=False,
    ):
        super().__init__()
        self.self_att = RelativePosMultiHeadAttention(
            nhead=nhead, dropout=dropout, kdim=kdim, vdim=vdim, nb=num_modules,
        )
        self.pos_ffn = PositionalwiseFeedForward(
            d_ffn=d_ffn, dropout=dropout, activation=activation, nb=num_modules,
        )

        self.num_modules = num_modules
        self.d_ffn = d_ffn

        self.norm1 = GroupLayerNorm(d_ffn, num_modules, eps=1e-6)
        self.norm2 = GroupLayerNorm(d_ffn, num_modules, eps=1e-6)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.use_group_comm = use_group_comm
        if use_group_comm:
            self.group_comm = GroupCommunication(d_ffn, num_modules)
            self.norm_comm = GroupLayerNorm(d_ffn, num_modules, eps=1e-6)
            self.dropout_comm = torch.nn.Dropout(dropout)

    def init_params(self, first_input):
        self.din = first_input.shape[-1]

        if self.num_modules > 1:
            self.competition = GroupLinear(
                self.din, self.num_modules, self.num_modules, a=0.05
            ).to(first_input.device)
        else:
            self.competition = None

    def forward(
        self, src, pos_embs, src_mask=None, src_key_padding_mask=None, init_params=False
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
        if init_params:
            self.init_params(src)

        if self.competition is not None:
            comp = self.competition(src)
            comp = F.softmax(comp, dim=2)
            self.comp_score = comp
            comp = comp.unsqueeze(-1).repeat(
                1, 1, 1, self.din // self.num_modules
            )
            comp = comp.view((src.shape[0], src.shape[1], self.din))
        else:
            comp = None
            self.comp_score = None

        output, self_attn = self.self_att(
            src,
            src,
            src,
            pos_embs,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            init_params=init_params,
        )

        # add & norm
        if comp is None:
            src = src + self.dropout1(output)
        else:
            src = src + self.dropout1(output) * comp

        src = self.norm1(src, init_params=init_params)

        output = self.pos_ffn(src, init_params)

        # add & norm
        output = src + self.dropout2(output)
        output = self.norm2(output, init_params=init_params)

        if self.use_group_comm:
            residual = output * 1.0
            output = self.group_comm(output, init_params=init_params)
            output = self.dropout_comm(output)
            output = self.norm_comm(output + residual, init_params=init_params)

        return output, self.comp_score



class TransformerEncoderRP(nn.Module):
    """This class implements the transformer encoder

    Arguements
    ----------
    d_ffn : int
        Hidden size of self-attention Feed Forward layer
    nhead : int
        number of attention heads
    kdim : int
        dimension for key (Optional)
    vdim : int
        dimension for value (Optional)
    dropout : float
        dropout for the encoder (Optional)
    input_module: torch class
        the module to process the source input feature to expected feature dimension (Optional)

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> pos = torch.rand((60, 512))
    >>> net = TransformerEncoderRP(1, 8, 512, 512)
    >>> output = net(x, pos, init_params=True)
    >>> print(output.shape)
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation=nn.ReLU,
        return_attention=False,
        num_modules=1,
        use_group_comm=False,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                TransformerEncoderLayerRP(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    num_modules=num_modules
                    if (j > 1 and j < num_layers - 1)
                    else 1,
                    use_group_comm=use_group_comm,
                )
                for j in range(num_layers)
            ]
        )
        self.norm = GroupLayerNorm(d_ffn, 1, eps=1e-6)
        self.return_attention = return_attention

    def forward(
        self, src, pos_embs, src_mask=None, src_key_padding_mask=None, init_params=False
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
                pos_embs,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                init_params=init_params,
            )
            attention_lst.append(attention)
        output = self.norm(output, init_params=init_params)

        if self.return_attention:
            return output, attention_lst
        return output

