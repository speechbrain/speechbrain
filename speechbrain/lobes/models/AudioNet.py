"""
A simple audio classification architecture for AudioMNIST https://arxiv.org/abs/1807.03418.
Used with a learnable gammatone frontend in "Learnable filter-banks
        for CNN-based audio applications", in Proc of NLDL 2022 (https://septentrio.uit.no/index.php/nldl/article/view/6279)
Authors
 * Nicolas Aspert 2024
"""

# import os
import torch  # noqa: F401
import torch.nn as nn
from torch.nn import Dropout, Flatten

import speechbrain as sb
from speechbrain.nnet.activations import Softmax
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.pooling import Pooling1d


class AudioNet(nn.Module):
    """This model extracts features for spoken digits recognition.

    Arguments
    ---------
    activation : torch class
        A class for constructing the activation layers.
    conv_blocks : int
        Number of convolutional layers.
    conv_channels : list of ints
        Output channels for conv layer.
    conv_kernel_sizes : list of ints
        List of kernel sizes for each conv layer.
    conv_dilations : list of ints
        List of dilations for kernels in each conv layer.
    max_pooling_kernel : list of ints
        Kernel size for max pooling in each conv layer. If zero, no pooling is applied.
    max_pooling_stride : list of ints
        Stride for max pooling in each conv layer. Only applied if corresponding value in max_pooling_kernel is not zero.
    in_channels : int
        Expected size of input features.

    Example
    -------
    >>> compute_audionet = AudioNet()
    >>> input_feats = torch.rand([5, 10, 40])
    >>> outputs = compute_audionet(input_feats)
    >>> outputs.shape
    torch.Size([5, 128])
    """

    def __init__(
        self,
        activation=torch.nn.ReLU,
        conv_blocks=3,
        conv_channels=[64, 128, 128],
        conv_kernel_sizes=[3, 3, 3],
        conv_dilations=[
            1,
            1,
            1,
        ],
        max_pooling_kernel=[2, 2, 2],
        max_pooling_stride=[2, 2, 2],
        in_channels=40,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.flatten = Flatten()
        # Conv layers
        for block_index in range(conv_blocks):
            out_channels = conv_channels[block_index]
            self.blocks.extend(
                self._get_conv_block(
                    in_channels,
                    out_channels,
                    conv_kernel_sizes[block_index],
                    conv_dilations[block_index],
                    activation,
                    max_pooling_kernel[block_index],
                    max_pooling_stride[block_index],
                )
            )
            in_channels = conv_channels[block_index]

    def _get_conv_block(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        activation,
        pool_size,
        pool_stride,
    ):
        """
        Builds the layers for a single convolutional block.

        Arguments
        ---------
        in_channels : int
            Expected size of input features.
        out_channels : int
            Number of output channels.
        kernel_size : int
            Kernel size for the convolutional layer.
        dilation : int
            Dilation for the convolutional layer.
        activation : torch class
            A class for constructing the activation layers. If None, no activation is applied.
        pool_size : int
            Kernel size for max pooling. If zero, no pooling is applied.
        pool_stride : int
            Stride for max pooling. Only applied if pool_size is not zero.

        Returns
        -------
        res : list
            List of layers in the block.
        """
        res = [
            Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            )
        ]
        if activation is not None:
            res.append(activation())
        if pool_size > 0:
            res.append(
                Pooling1d(
                    kernel_size=pool_size, pool_type="max", stride=pool_stride
                )
            )

        return res

    def forward(self, x, lens=None):
        """Computes the AudioNet features.

        Arguments
        ---------
        x : torch.Tensor
            Inputs features for extracting x-vectors.
        lens : torch.Tensor
            The corresponding relative lengths of the inputs.

        Returns
        -------
        x : torch.Tensor
            Features
        """

        for layer in self.blocks:
            try:
                x = layer(x, lengths=lens)
            except TypeError:
                x = layer(x)
        return self.flatten(x)


class Classifier(sb.nnet.containers.Sequential):
    """This class implements the classifier MLP on the top of AudioNet features.
    A softmax layer is used to predict the class probabilities after the MLP.
    Dropout of 50% is applied after each linear layer.

    Arguments
    ---------
    input_shape : tuple
        Expected shape of an example input.
    activation : torch class
        A class for constructing the activation layers.
    lin_blocks : int
        Number of linear layers.
    lin_neurons : list of ints
        Number of neurons in linear layers.
    out_neurons : int
        Number of output neurons in the final layer.

    Example
    -------
    >>> input_feats = torch.rand([5, 10, 40])
    >>> compute_audionet = AudioNet()
    >>> audionet_feats = compute_audionet(input_feats)
    >>> classify = Classifier(input_shape=audionet_feats.shape)
    >>> output = classify(audionet_feats)
    >>> output.shape
    torch.Size([5, 10])
    """

    def __init__(
        self,
        input_shape,
        activation=torch.nn.ReLU,
        lin_blocks=2,
        lin_neurons=[512, 256],
        out_neurons=10,
    ):
        super().__init__(input_shape=input_shape)

        if lin_blocks > 0:
            self.append(sb.nnet.containers.Sequential, layer_name="DNN")

        for block_index in range(lin_blocks):
            block_name = f"block_{block_index}"
            self.DNN.append(
                sb.nnet.containers.Sequential, layer_name=block_name
            )
            self.DNN[block_name].append(
                Linear,
                n_neurons=lin_neurons[block_index],
                bias=True,
                layer_name="linear",
            )
            self.DNN[block_name].append(Dropout(p=0.5), layer_name="dropout")
            if activation is not None:
                self.DNN[block_name].append(activation(), layer_name="act")

        # Final Softmax classifier
        self.append(Linear, n_neurons=out_neurons, layer_name="out")
        self.append(Softmax(apply_log=True), layer_name="softmax")


class AudioNetFrontend(nn.Module):
    """
    A simple frontend for AudioNet, using a single convolutional layer with max pooling.

    Arguments
    ----------

    in_channels : int
        Expected size of input features.
    conv_kernel_size : int
        Kernel size for the convolutional layer.
    conv_dilation : int
        Dilation for the convolutional layer.
    out_channels : int
        Number of output channels.
    activation : torch class
        A class for constructing the activation layers.
    """

    def __init__(
        self,
        in_channels,
        conv_kernel_size,
        conv_dilation,
        out_channels,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv_kernel_size,
            dilation=conv_dilation,
        )
        self.activ = activation()
        self.pool = Pooling1d(
            kernel_size=3, pool_type="max", stride=2, padding=1
        )

    def forward(self, x):
        """
        Computes the frontend features from raw audio data.

        Arguments
        ---------
        x : torch.Tensor
            Inputs audio data.

        Returns
        -------
        x : torch.Tensor
            Features
        """
        x = x.unsqueeze(1).transpose(1, -1)
        x = self.conv(x)
        x = self.activ(x)
        x = self.pool(x)
        return x
