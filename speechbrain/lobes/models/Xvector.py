"""A popular speaker recognition and diarization model.

Authors
 * Nauman Dawalatabad 2020
 * Mirco Ravanelli 2020
"""

# import os
import torch  # noqa: F401
import torch.nn as nn
from speechbrain.nnet.pooling import StatisticsPooling
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.normalization import BatchNorm1d
from speechbrain.nnet.activations import Softmax


class Xvector(torch.nn.Module):
    """This model extracts XVectors for speaker recognition and diarization.

    Arguments
    ---------
    device : str
        Device used e.g. "cpu" or "cuda"
    activation : torch class
        A class for constructing the activation layers.
    tdnn_blocks : int
        Number of time delay neural (TDNN) layers.
    tdnn_channels : list of ints
        Output channels for TDNN layer.
    tdnn_kernel_sizes : list of ints
        List of kernel sizes for each TDNN layer.
    tdnn_dilations : list of ints
        List of dialations for kernels in each TDNN layer.
    lin_neurons : int
        Number of neurons in linear layers.

    Example
    -------
    >>> compute_xvect = Xvector('cpu')
    >>> input_feats = torch.rand([5, 10, 24])
    >>> outputs = compute_xvect(input_feats, init_params=True)
    >>> outputs.shape
    torch.Size([5, 1, 512])
    """

    def __init__(
        self,
        device="cpu",
        activation=torch.nn.LeakyReLU,
        tdnn_blocks=5,
        tdnn_channels=[512, 512, 512, 512, 1500],
        tdnn_kernel_sizes=[5, 3, 3, 1, 1],
        tdnn_dilations=[1, 2, 3, 1, 1],
        lin_neurons=512,
    ):

        super().__init__()
        self.blocks = nn.ModuleList()

        # TDNN layers
        for block_index in range(tdnn_blocks):
            self.blocks.extend(
                [
                    Conv1d(
                        out_channels=tdnn_channels[block_index],
                        kernel_size=tdnn_kernel_sizes[block_index],
                        dilation=tdnn_dilations[block_index],
                    ),
                    activation(),
                    BatchNorm1d(),
                ]
            )

        # Statistical pooling
        self.blocks.append(StatisticsPooling(device))

        # Final linear transformation
        self.blocks.append(
            Linear(n_neurons=lin_neurons, bias=True, combine_dims=False)
        )

    def init_params(self, first_input):
        """
        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        x = first_input

        for layer in self.blocks:
            try:
                x = layer(x, init_params=True)
            except TypeError:
                x = layer(x)

    def forward(self, x, lens=None, init_params=False):
        """Returns the x vectors.

        Arguments
        ---------
        x : torch.Tensor
        """
        if init_params:
            self.init_params(x)

        for layer in self.blocks:
            try:
                x = layer(x, lengths=lens)
            except TypeError:
                x = layer(x)

        return x


class Classifier(torch.nn.Module):
    """This class implements the last MLP on the top of xvector features.

    Arguments
    ---------
    device : str
        Device used e.g. "cpu" or "cuda"
    activation : torch class
        A class for constructing the activation layers.
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.
    out_neurons : int
        Number of ouput neurons.

    Example
    -------
    >>> compute_xvect = Xvector('cpu')
    >>> classify = Classifier('cpu')
    >>> input_feats = torch.rand([5, 10, 24])
    >>> xvects = compute_xvect(input_feats, init_params=True)
    >>> output = classify(xvects, init_params=True)
    >>> output.shape
    torch.Size([5, 1, 1211])
    """

    def __init__(
        self,
        device="cpu",
        activation=torch.nn.LeakyReLU,
        lin_blocks=1,
        lin_neurons=512,
        out_neurons=1211,
    ):

        super().__init__()
        self.blocks = nn.ModuleList()

        self.blocks.extend([activation(), BatchNorm1d()])

        for block_index in range(lin_blocks):
            self.blocks.extend(
                [
                    Linear(
                        n_neurons=lin_neurons, bias=True, combine_dims=False
                    ),
                    activation(),
                    BatchNorm1d(),
                ]
            )

        # Final Softmax classifier
        self.blocks.extend(
            [Linear(n_neurons=out_neurons, bias=True), Softmax(apply_log=True)]
        )

    def init_params(self, first_input):
        """
        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        x = first_input

        for layer in self.blocks:
            try:
                x = layer(x, init_params=True)
            except TypeError:
                x = layer(x)

    def forward(self, x, init_params=False):
        """Returns the output probabilities over speakers.

        Arguments
        ---------
        x : torch.Tensor
        """
        if init_params:
            self.init_params(x)

        for layer in self.blocks:
            try:
                x = layer(x, init_params=init_params)
            except TypeError:
                x = layer(x)
        return x


class Discriminator(torch.nn.Module):
    """This class implements a discriminator on the top of xvector features.

    Arguments
    ---------
    device : str
        Device used e.g. "cpu" or "cuda"
    activation : torch class
        A class for constructing the activation layers.
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.

    Example
    -------
    >>> compute_xvect = Xvector('cpu')
    >>> classify = Classifier('cpu')
    >>> input_feats = torch.rand([5, 10, 24])
    >>> xvects = compute_xvect(input_feats, init_params=True)
    >>> output = classify(xvects, init_params=True)
    >>> output.shape
    torch.Size([5, 1, 1211])
    """

    def __init__(
        self,
        device="cpu",
        activation=torch.nn.LeakyReLU,
        lin_blocks=1,
        lin_neurons=512,
        out_neurons=1,
    ):

        super().__init__()
        self.blocks = nn.ModuleList()

        for block_index in range(lin_blocks):
            self.blocks.extend(
                [
                    Linear(
                        n_neurons=lin_neurons, bias=True, combine_dims=False
                    ),
                    BatchNorm1d(),
                    activation(),
                ]
            )

        # Final Layer (sigmoid not included)
        self.blocks.extend([Linear(n_neurons=out_neurons)])

    def init_params(self, first_input):
        """
        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        x = first_input

        for layer in self.blocks:
            try:
                x = layer(x, init_params=True)
            except TypeError:
                x = layer(x)

    def forward(self, x, init_params=False):
        """Returns the output probabilities over speakers.

        Arguments
        ---------
        x : torch.Tensor
        """
        if init_params:
            self.init_params(x)

        for layer in self.blocks:
            try:
                x = layer(x, init_params=init_params)
            except TypeError:
                x = layer(x)

        return x
