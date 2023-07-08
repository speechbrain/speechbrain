"""This is a module to ensemble a convolution (depthwise) encoder with or without residule connection.

Authors
 * Jianyuan Zhong 2020
 * Titouan Parcollet 2023
"""
import torch
from speechbrain.nnet.CNN import Conv2d, Conv1d
from speechbrain.nnet.containers import Sequential
from speechbrain.nnet.normalization import LayerNorm


class ConvolutionalSpatialGatingUnit(torch.nn.Module):
    """This module implementing CSGU as defined in:
    Branchformer: Parallel MLP-Attention Architectures
    to Capture Local and Global Context for Speech Recognition
    and Understanding"

    The code is heavily inspired from the original ESPNet
    implementation.

    Arguments
    ----------
    input_size: int
        Size of the feature (channel) dimension.
    kernel_size: int, optional
        Size of the kernel
    dropout: float, optional
        Dropout rate to be applied at the output
    use_linear_after_conv: bool, optional
        If True, will apply a linear transformation of size input_size//2
    activation: torch.class, optional
        Activation function to use on the gate, default is Identity.

    Example
    -------
    >>> x = torch.rand((8, 30, 10))
    >>> conv = ConvolutionalSpatialGatingUnit(input_size=x.shape[-1])
    >>> out = conv(x)
    >>> out.shape
    torch.Size([8, 30, 5])
    """

    def __init__(
        self,
        input_size,
        kernel_size=31,
        dropout=0.0,
        use_linear_after_conv=False,
        activation=torch.nn.Identity,
    ):
        super().__init__()

        self.input_size = input_size
        self.use_linear_after_conv = use_linear_after_conv
        self.activation = activation()

        if self.input_size % 2 != 0:
            raise ValueError("Input size must be divisible by 2!")

        n_channels = input_size // 2  # split input channels
        self.norm = LayerNorm(n_channels)
        self.conv = Conv1d(
            input_shape=(None, None, n_channels),
            out_channels=n_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
            groups=n_channels,
            conv_init="normal",
            skip_transpose=False,
        )

        if self.use_linear_after_conv:
            self.linear = torch.nn.Linear(n_channels, n_channels)
            torch.nn.init.normal_(self.linear.weight, std=1e-6)
            torch.nn.init.ones_(self.linear.bias)

        torch.nn.init.ones_(self.conv.conv.bias)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        """
        Arguments
        ----------
        x: torch.Tensor -> (B, T, D)

        """

        # We create two sequences where feat dim is halved
        x1, x2 = x.chunk(2, dim=-1)

        x2 = self.norm(x2)
        x2 = self.conv(x2)
        if self.use_linear_after_conv:
            x2 = self.linear(x2)
        x2 = self.activation(x2)

        return self.dropout(x2 * x1)


class ConvolutionFrontEnd(Sequential):
    """This is a module to ensemble a convolution (depthwise) encoder with or
    without residual connection.

    Arguments
    ----------
    out_channels: int
        Number of output channels of this model (default 640).
    out_channels: Optional(list[int])
        Number of output channels for each of block.
    kernel_size: int
        Kernel size of convolution layers (default 3).
    strides: Optional(list[int])
        Striding factor for each block, this stride is applied at the last convolution layer at each block.
    num_blocks: int
        Number of block (default 21).
    num_per_layers: int
        Number of convolution layers for each block (default 5).
    dropout: float
        Dropout (default 0.15).
    activation: torch class
        Activation function for each block (default Swish).
    norm: torch class
        Normalization to regularize the model (default BatchNorm1d).
    residuals: Optional(list[bool])
        Whether apply residual connection at each block (default None).

    Example
    -------
    >>> x = torch.rand((8, 30, 10))
    >>> conv = ConvolutionFrontEnd(input_shape=x.shape)
    >>> out = conv(x)
    >>> out.shape
    torch.Size([8, 8, 3, 512])
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
        norm=LayerNorm,
        dropout=0.1,
        conv_bias=True,
        padding="same",
        conv_init=None,
    ):
        super().__init__(input_shape=input_shape)
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
                layer_name=f"convblock_{i}",
                conv_bias=conv_bias,
                padding=padding,
                conv_init=conv_init,
            )


class ConvBlock(torch.nn.Module):
    """An implementation of convolution block with 1d or 2d convolutions (depthwise).

    Arguments
    ----------
    out_channels : int
        Number of output channels of this model (default 640).
    kernel_size : int
        Kernel size of convolution layers (default 3).
    strides : int
        Striding factor for this block (default 1).
    num_layers : int
        Number of depthwise convolution layers for this block.
    activation : torch class
        Activation function for this block.
    norm : torch class
        Normalization to regularize the model (default BatchNorm1d).
    residuals: bool
        Whether apply residual connection at this block (default None).

    Example
    -------
    >>> x = torch.rand((8, 30, 10))
    >>> conv = ConvBlock(2, 16, input_shape=x.shape)
    >>> out = conv(x)
    >>> x.shape
    torch.Size([8, 30, 10])
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
        conv_bias=True,
        padding="same",
        conv_init=None,
    ):
        super().__init__()
        self.convs = Sequential(input_shape=input_shape)

        for i in range(num_layers):
            self.convs.append(
                conv_module,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride if i == num_layers - 1 else 1,
                dilation=dilation,
                layer_name=f"conv_{i}",
                bias=conv_bias,
                padding=padding,
                conv_init=conv_init,
            )
            if norm is not None:
                self.convs.append(norm, layer_name=f"norm_{i}")
            self.convs.append(activation(), layer_name=f"act_{i}")
            self.convs.append(
                torch.nn.Dropout(dropout), layer_name=f"dropout_{i}"
            )

        self.reduce_conv = None
        self.drop = None
        if residual:
            self.reduce_conv = Sequential(input_shape=input_shape)
            self.reduce_conv.append(
                conv_module,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                layer_name="conv",
            )
            self.reduce_conv.append(norm, layer_name="norm")
            self.drop = torch.nn.Dropout(dropout)

    def forward(self, x):
        """ Processes the input tensor x and returns an output tensor."""
        out = self.convs(x)
        if self.reduce_conv:
            out = out + self.reduce_conv(x)
            out = self.drop(out)
        return out
