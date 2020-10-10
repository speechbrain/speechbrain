"""This is a module to ensemble a convolution (depthwise) encoder with or without residule connection

Authors
 * Jianyuan Zhong 2020
"""
import torch
from speechbrain.nnet.CNN import Conv2d
from speechbrain.nnet.containers import Sequential
from speechbrain.nnet.normalization import BatchNorm2d


class ConvolutionFrontEnd(Sequential):
    """This is a module to ensemble a convolution (depthwise) encoder with or without residule connection

     Arguements
    ----------
    out_channels: int
        number of output channels of this model (default 640)
    out_channels: Optional(list[int])
        number of output channels for each of block.
    kernel_size: int
        kernel size of convolution layers (default 3)
    strides: Optional(list[int])
        striding factor for each block, this stride is applied at the last convolution layer at each block.
    num_blocks: int
        number of block (default 21)
    num_per_layers: int
        number of convolution layers for each block (default 5)
    dropout: float
        dropout (default 0.15)
    activation: torch class
        activation function for each block (default Swish)
    norm: torch class
        normalization to regularize the model (default BatchNorm1d)
    residuals: Optional(list[bool])
        whether apply residual connection at each block (default None)

    Example
    -------
    >>> x = torch.rand((8, 120, 40))
    >>> conv = ConvolutionFrontEnd(input_shape=x.shape)
    >>> out = conv(x)
    >>> out.shape
    torch.Size([8, 120, 40])
    """

    def __init__(
        self,
        input_shape,
        num_blocks=3,
        num_layers_per_block=5,
        out_channels=[128, 256, 512],
        kernel_sizes=[3, 3, 3],
        strides=[1, 2, 2],
        dilations=[1, 1, 1],
        residuals=[True, True, True],
        conv_module=Conv2d,
        activation=torch.nn.LeakyReLU,
        norm=BatchNorm2d,
        dropout=0.1,
    ):
        super().__init__(input_shape)
        for i in range(num_blocks):
            self.append(
                ConvBlock,
                num_layers=num_layers_per_block,
                out_channels=out_channels[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                dilation=dilations[i],
                residual=residuals[i],
                conv_module=conv_module,
                activation=activation,
                norm=norm,
                dropout=dropout,
            )


class ConvBlock(torch.nn.Module):
    """An implementation of convolution block with 1d or 2d convolutions (depthwise)

    Arguements
    ----------
    out_channels: int
        number of output channels of this model (default 640)
    kernel_size: int
        kernel size of convolution layers (default 3)
    strides: int
        striding factor for this block (default 1)
    num_layers: int
        number of depthwise convolution layers for this block
    activation: torch class
        activation function for this block
    norm: torch class
        normalization to regularize the model (default BatchNorm1d)
    residuals: bool
        whether apply residual connection at this block (default None)

    Example
    -------
    >>> x = torch.rand((8, 120, 40))
    >>> conv = ConvBlock(2, 128, input_shape=x.shape)
    >>> out = conv(x)
    >>> x.shape
    torch.Size([8, 120, 40])
    """

    def __init__(
        self,
        num_layers,
        out_channels,
        input_shape,
        kernel_size=3,
        stride=1,
        dilation=1,
        residual=False,
        conv_module=Conv2d,
        activation=torch.nn.LeakyReLU,
        norm=None,
        dropout=0.1,
    ):
        super().__init__()

        self.convs = Sequential(input_shape)

        for i in range(num_layers):
            self.convs.append(
                conv_module,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride if i == num_layers - 1 else 1,
                dilation=dilation,
            )
            if norm is not None:
                self.convs.append(norm)
            self.convs.append(activation())
            self.convs.append(torch.nn.Dropout(dropout))

        self.reduce_conv = None
        self.drop = None
        if residual:
            self.reduce_conv = Sequential(input_shape)
            self.reduce_conv.append(
                conv_module,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
            )
            self.reduce_conv.append(norm)
            self.drop = torch.nn.Dropout(dropout)

    def forward(self, x):
        out = self.convs(x)
        if self.reduce_conv:
            out = out + self.reduce_conv(x)
            out = self.drop(x)

        return out
