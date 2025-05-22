"""This module mixes information from different tokens via HyperMixing.
It can be viewed as a linear-time drop-in replacement for (self-)attention.

source: https://arxiv.org/abs/2203.03691

Authors
 * Florian Mai 2023
 * Juan Pablo Zuluaga 2023
"""

import math
from typing import Optional

import torch
from torch import nn


class HyperMixing(nn.Module):
    """This class implements multi-head HyperMixing.
    It is an implementation of the token-mixing component in HyperMixer, a linear
    time drop-in replacement for self-attention. In contrast to the original HyperMixer,
    this module supports multiple heads, which improves the expressiveness of the model
    while decreasing the number of parameters.

    Reference: https://arxiv.org/abs/2203.03691

    Arguments
    ---------
    input_output_dim : int
        number of features in keys, queries, and values
    hypernet_size : int
        determines the size of the hidden layer of the token-mixing MLP.
    tied : bool
        If True, then the generated weight matrices of the token-mixing MLP are tied.
    num_heads : int
        parallel token-mixing MLPs.
    fix_tm_hidden_size : bool
        If True, the hidden-layer size is equal to hypernet_size rather than hypernet_size / num_heads.
    max_length : int
        Maximum number of input tokens. Needed for generating sufficiently large position embeddings.

    Example
    -------
    >>> import torch
    >>> inputs = torch.rand([8, 60, 512])
    >>> net = HyperMixing(512, 2048, num_heads=8)
    >>> outputs, attn = net(inputs, inputs, inputs)
    >>> outputs.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        input_output_dim: int,
        hypernet_size: int,
        tied: bool = False,
        num_heads: int = 1,
        fix_tm_hidden_size: bool = False,
        max_length: int = 3000,
    ) -> None:
        super().__init__()
        self.input_output_dim = input_output_dim
        self.hyper = HyperNetwork(
            input_output_dim,
            hypernet_size,
            tied=tied,
            num_heads=num_heads,
            keep_output_size=fix_tm_hidden_size,
        )
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(input_output_dim)
        self.num_heads = num_heads

        from speechbrain.lobes.models.transformer.Transformer import (
            PositionalEncoding,
        )

        # add pos encoding
        self.positional_encoding = PositionalEncoding(
            input_output_dim, max_length
        )

    def _mlp_pass_from_components(self, out, W1, W2, activation):
        """function to stick MLP1 together manually"""
        out = torch.bmm(out, W1)
        out = activation(out)
        out = torch.bmm(out, W2.transpose(1, 2))
        return out

    def forward(
        self,
        query,
        key,
        value,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn_weights: Optional[bool] = True,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """
        The signature of this method is deliberately chosen to be the same as for
        sb.nnet.attention.MultiHeadAttention for compatibility within SpeechBrain.

        NOTE: key, value, attn_mask and pos_embs have no effect. Query is used for
        all three. Thus, the module should only be used to replace self-attention at the moment.

        Arguments
        ----------
        query : torch.Tensor
            (B, L, E) where L is the target sequence length,
            B is the batch size, E is the embedding dimension.
        key : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
            Currently unused. All
        value : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
            Currently unused.
        attn_mask : torch.Tensor, optional
            NOTE: Currently has NO effect.
        key_padding_mask : torch.Tensor, optional
            (B, S) where B is the batch size, S is the source sequence
            length. If a ByteTensor is provided, the non-zero positions will
            be ignored while the position with the zero positions will be
            unchanged. If a BoolTensor is provided, the positions with the
            value of True will be ignored while the position with the value
            of False will be unchanged.
        return_attn_weights: torch.Tensor, optional
            NOTE: Currently has NO effect.
        pos_embs: torch.Tensor, optional
            NOTE: Currently has NO effect.

        Outputs
        -------
        attn_output : torch.Tensor
            (B, L, E) where L is the target sequence length, B is the
            batch size, E is the embedding dimension.
        attn_output_weights : torch.Tensor
            (B, L, S) where B is the batch size, L is the target
            sequence length, S is the source sequence length.
            NOTE: always returns all zeros.
        """

        # NOTE: We are ignoring keys and values, because HyperMixing can only be used in the encoder atm (where it's all the same)
        out = query

        bsize = out.size(0)
        seq_len = out.size(1)

        if key_padding_mask is not None:
            float_mask = (
                torch.logical_not(key_padding_mask).unsqueeze(-1).float()
            )
            out = out * float_mask

        # add position embedding before passing to hypernetwork
        hyp_input = out + self.positional_encoding(out)
        W1, W2 = self.hyper(
            hyp_input
        )  # [bsize, num_heads, seq_len, hypernet_size // num_heads]

        if key_padding_mask is not None:
            # mask the weights
            W1 = W1 * float_mask.unsqueeze(1)
            W2 = W2 * float_mask.unsqueeze(1)

        # reshape the num_heads into the batch dimension for parallelizing
        out = out.transpose(1, 2)  # [bsize, input_output_dim, seq_len]
        out = out.reshape(
            (
                bsize * self.num_heads,
                self.input_output_dim // self.num_heads,
                seq_len,
            )
        )  # [bsize * num_heads, input_output_dim // num_heads, seq_len]
        W1 = W1.reshape((bsize * self.num_heads, seq_len, -1))
        W2 = W2.reshape((bsize * self.num_heads, seq_len, -1))

        # we stick the token-mixing MLP together manually
        out = self._mlp_pass_from_components(out, W1, W2, self.activation)

        # concatenate heads
        out = out.reshape((bsize, self.input_output_dim, seq_len))

        # transpose back
        out = out.transpose(1, 2)

        # apply layer norm on outputs of the TM-MLP
        out = self.layer_norm(out)

        dummy_att_weights = torch.zeros(
            (bsize, seq_len, seq_len), device=out.device
        )
        return out, dummy_att_weights


class HyperNetwork(nn.Module):
    """This class implements The HyperNetwork. It is an approach of using a one network,
    also known as a hypernetwork, to generate the weights for another network.
    Here, it is used to generate the labels of linear layers.

    Reference: https://arxiv.org/abs/1609.09106

    Arguments
    ----------
    input_output_dim : int
        Dimension of the linear layers
    hypernet_size:
        Dimension of the HyperNetwork
    tied : bool, optional
        Define whether weights of layer 1 and layer 2 are shared
    num_heads: int, optional
        Number of heads, akin to heads in MultiHeadAttention
    keep_output_size: bool, optional
        Set whether to keep the same output size independent of number of heads
    """

    def __init__(
        self,
        input_output_dim: int,
        hypernet_size: int,
        tied=False,
        num_heads=1,
        keep_output_size=True,
    ) -> None:
        super(HyperNetwork, self).__init__()

        # Define whether the two linear layers have tied weights
        self.tied = tied
        self.w1_gen = ParallelMLPs(
            input_output_dim,
            input_output_dim,
            output_size=hypernet_size,
            num_mlps=num_heads,
            keep_output_size=keep_output_size,
        )
        if self.tied:
            self.w2_gen = self.w1_gen
        else:
            self.w2_gen = ParallelMLPs(
                input_output_dim,
                input_output_dim,
                output_size=hypernet_size,
                num_mlps=num_heads,
                keep_output_size=keep_output_size,
            )

    def forward(self, input_tensor: torch.Tensor):
        """Forward computation for a HyperNetwork.

        Arguments
        ----------
        input_tensor : [batchsize, max_positions, d]
            The HyperNetwork is supposed to generate an MLP of the form W_2(GELU(W1 x)), where
            W1 : N -> k and W2 : k -> N, so it has to return tensors W1 and W2

        Outputs
        -------
        W1 : torch.Tensor
            Generated weights of Layer 1
        W2 : torch.Tensor
            Generated weights of Layer 2
        """
        W1 = self.w1_gen(input_tensor)
        if self.tied:
            W2 = W1
        else:
            W2 = self.w2_gen(input_tensor)

        return W1, W2


class ParallelMLPs(nn.Module):
    """Class that implements the MultiHead HyperMixer or HyperConformer.

    Arguments
    ----------
    input_size : int
        Dimension of the linear layers
    hidden_size: int
        Dimension of the hidden layer
    output_size : int
        Dimension of the HyperNetwork
    num_mlps : int
        Number of heads, akin to heads in MultiHeadAttention
    keep_output_size : bool, optional
        Set whether to keep the same output size independent of number of heads
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size=None,
        num_mlps=1,
        keep_output_size=True,
    ) -> None:
        super(ParallelMLPs, self).__init__()

        if output_size is None:
            output_size = input_size

        self.original_in_size = input_size
        self.original_out_size = output_size

        assert input_size % num_mlps == 0
        assert output_size % num_mlps == 0
        assert hidden_size % num_mlps == 0
        input_size = input_size // num_mlps

        if not keep_output_size:
            output_size = output_size // num_mlps
        hidden_size = hidden_size // num_mlps

        self.input_size = input_size
        self.output_size = output_size

        self.num_mlps = num_mlps

        # set the weights and biases parameters
        self.fc1_weights = nn.Parameter(
            torch.empty(num_mlps, hidden_size, input_size)
        )
        self.fc1_biases = nn.Parameter(torch.empty(num_mlps, hidden_size))
        self.fc2_weights = nn.Parameter(
            torch.empty(num_mlps, output_size, hidden_size)
        )
        self.fc2_biases = nn.Parameter(torch.empty(num_mlps, output_size))

        # initialize the weights and biases
        nn.init.xavier_uniform_(self.fc1_weights, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.fc1_biases, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.fc2_weights, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.fc2_biases, gain=math.sqrt(2.0))

        self.activation = nn.GELU()

    def forward(self, x):
        """Performs the forward computation of multi parallel MLPs.

        Arguments
        ----------
        x : tensor
            Input tensor

        Outputs
        -------
        x : torch.Tensor
            return output tensor
        """

        # x [bsize, seq_len, num_features]
        bsize = x.size(0)
        seq_len = x.size(1)

        # Reshape the input tensor to match the number of parallel MLPs and their input size
        x = x.reshape((bsize, seq_len, self.num_mlps, self.input_size))

        # Perform the first linear transformation and add bias
        # Using einsum so we can do it for multiple MLPs in parallel
        x = torch.einsum(
            "blmf,mhf->bmlh", x, self.fc1_weights
        ) + self.fc1_biases.unsqueeze(0).unsqueeze(2)

        # Apply activation function and perform the second linear transformation and add bias
        x = self.activation(x)
        x = torch.einsum(
            "bmlh,mfh->bmlf", x, self.fc2_weights
        ) + self.fc2_biases.unsqueeze(0).unsqueeze(2)

        return x
