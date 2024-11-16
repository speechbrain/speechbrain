import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch.nn.utils import weight_norm, remove_weight_norm
from torch.autograd.function import InplaceFunction
import torchaudio
from omegaconf import OmegaConf
import logging
from collections import OrderedDict

import os
import zipfile

import requests
from tqdm import tqdm


def get_padding(kernel_size, dilation=1):
    """
    Computes the padding size for a given kernel size and dilation.

    Arguments
    ---------
    kernel_size : int
        Size of the convolutional kernel.
    dilation : int, optional
        Dilation factor for convolution (default is 1).

    Returns
    -------
    int
        Calculated padding size.
    """
    return int((kernel_size * dilation - dilation) / 2)

class round_func5(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(5*input)/5
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class round_func9(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(9*input)/9
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class round_func_binary(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class Heaviside(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        values = torch.tensor([0.0]).type_as(input)
        return torch.heaviside(input, values)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class PreProcessor(nn.Module):
    """
    A module for preprocessing input data through convolution and pooling operations.

    Arguments
    ---------
    n_in : int
        Number of input channels.
    n_out : int
        Number of output channels.
    num_samples : int
        Number of samples for pooling.
    kernel_size : int, optional
        Size of the convolutional kernel (default is 7).
    causal : bool, optional
        If True, applies causal convolution (default is False).
    """

    def __init__(self, n_in, n_out, num_samples, kernel_size=7, causal=False):
        super(PreProcessor, self).__init__()
        self.pooling = torch.nn.AvgPool1d(kernel_size=num_samples)
        self.conv = Conv1d(n_in, n_out, kernel_size=kernel_size, causal=causal)
        self.activation = nn.PReLU()

    def forward(self, x):
        """
        Applies convolution, activation, and pooling to the input data.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Processed output tensor.
        """
        output = self.activation(self.conv(x))
        output = self.pooling(output)
        return output

class PostProcessor(nn.Module):
    """
    A module for postprocessing data through convolution and reshaping.

    Arguments
    ---------
    n_in : int
        Number of input channels.
    n_out : int
        Number of output channels.
    num_samples : int
        Number of samples for repetition.
    kernel_size : int, optional
        Size of the convolutional kernel (default is 7).
    causal : bool, optional
        If True, applies causal convolution (default is False).
    """

    def __init__(self, n_in, n_out, num_samples, kernel_size=7, causal=False):
        super(PostProcessor, self).__init__()
        self.num_samples = num_samples
        self.conv = Conv1d(n_in, n_out, kernel_size=kernel_size, causal=causal)
        self.activation = nn.PReLU()

    def forward(self, x):
        """
        Applies reshaping, repetition, and convolution to the input data.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Processed output tensor.
        """
        x = torch.transpose(x, 1, 2)
        B, T, C = x.size()
        x = x.repeat(1, 1, self.num_samples).view(B, -1, C)
        x = torch.transpose(x, 1, 2)
        output = self.activation(self.conv(x))
        return output

class DownsampleLayer(nn.Module):
    """
    A downsampling layer that applies convolution, optional pooling, and activation.

    Arguments
    ---------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the convolutional kernel.
    stride : int, optional
        Stride of the convolution (default is 1).
    causal : bool, optional
        If True, applies causal convolution (default is False).
    activation : nn.Module, optional
        Activation function (default is PReLU).
    use_weight_norm : bool, optional
        If True, applies weight normalization to the convolution (default is True).
    pooling : bool, optional
        If True, applies an average pooling operation (default is False).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
        activation=nn.PReLU(),
        use_weight_norm: bool = True,
        pooling: bool = False,
    ):
        super(DownsampleLayer, self).__init__()
        self.pooling = pooling
        self.stride = stride
        self.activation = activation
        self.use_weight_norm = use_weight_norm
        if pooling:
            self.layer = Conv1d(
                in_channels, out_channels, kernel_size, causal=causal
            )
            self.pooling = nn.AvgPool1d(kernel_size=stride)
        else:
            self.layer = Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                causal=causal,
            )
        if use_weight_norm:
            self.layer = weight_norm(self.layer)

    def forward(self, x):
        """
        Applies convolution, optional pooling, and activation to the input data.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Processed output tensor.
        """
        x = self.layer(x)
        x = self.activation(x) if self.activation is not None else x
        if self.pooling:
            x = self.pooling(x)
        return x

    def remove_weight_norm(self):
        """
        Removes weight normalization from the convolutional layer.
        """
        if self.use_weight_norm:
            remove_weight_norm(self.layer)

class UpsampleLayer(nn.Module):
    """
    An upsampling layer that applies transposed convolution or repetition, with activation.

    Arguments
    ---------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the convolutional kernel.
    stride : int, optional
        Stride of the transposed convolution (default is 1).
    causal : bool, optional
        If True, applies causal convolution (default is False).
    activation : nn.Module, optional
        Activation function (default is PReLU).
    use_weight_norm : bool, optional
        If True, applies weight normalization to the convolution (default is True).
    repeat : bool, optional
        If True, applies repetition instead of transposed convolution (default is False).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
        activation=nn.PReLU(),
        use_weight_norm: bool = True,
        repeat: bool = False,
    ):
        super(UpsampleLayer, self).__init__()
        self.repeat = repeat
        self.stride = stride
        self.activation = activation
        self.use_weight_norm = use_weight_norm
        if repeat:
            self.layer = Conv1d(
                in_channels, out_channels, kernel_size, causal=causal
            )
        else:
            self.layer = ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                causal=causal,
            )
        if use_weight_norm:
            self.layer = weight_norm(self.layer)

    def forward(self, x):
        """
        Applies upsampling through transposed convolution or repetition, followed by activation.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Processed output tensor.
        """
        x = self.layer(x)
        x = self.activation(x) if self.activation is not None else x
        if self.repeat:
            x = torch.transpose(x, 1, 2)
            B, T, C = x.size()
            x = x.repeat(1, 1, self.stride).view(B, -1, C)
            x = torch.transpose(x, 1, 2)
        return x

    def remove_weight_norm(self):
        """
        Removes weight normalization from the convolutional layer.
        """
        if self.use_weight_norm:
            remove_weight_norm(self.layer)

class ResEncoderBlock(nn.Module):
    """
    A residual encoder block with multiple residual units and a downsampling layer.

    Arguments
    ---------
    n_in : int
        Number of input channels.
    n_out : int
        Number of output channels.
    stride : int
        Stride for the downsampling layer.
    down_kernel_size : int
        Kernel size for the downsampling layer.
    res_kernel_size : int, optional
        Size of the convolutional kernel for residual connections (default is 7).
    causal : bool, optional
        If True, applies causal convolution (default is False).
    """

    def __init__(
        self,
        n_in,
        n_out,
        stride,
        down_kernel_size,
        res_kernel_size=7,
        causal=False,
    ):
        super(ResEncoderBlock, self).__init__()
        self.convs = nn.ModuleList(
            [
                ResidualUnit(
                    n_in,
                    n_out // 2,
                    dilation=1,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out // 2,
                    n_out // 2,
                    dilation=3,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out // 2,
                    n_out // 2,
                    dilation=5,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out // 2,
                    n_out // 2,
                    dilation=7,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out // 2,
                    n_out // 2,
                    dilation=9,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
            ]
        )
        self.down_conv = DownsampleLayer(
            n_in, n_out, down_kernel_size, stride=stride, causal=causal
        )

    def forward(self, x):
        """
        Applies a series of residual units and a downsampling layer.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Processed output tensor.
        """
        for conv in self.convs:
            x = conv(x)
        x = self.down_conv(x)
        return x

class ResDecoderBlock(nn.Module):
    """
    A residual decoder block with upsampling and multiple residual units.

    Arguments
    ---------
    n_in : int
        Number of input channels.
    n_out : int
        Number of output channels.
    stride : int
        Stride for the upsampling layer.
    up_kernel_size : int
        Kernel size for the upsampling layer.
    res_kernel_size : int, optional
        Size of the convolutional kernel for residual connections (default is 7).
    causal : bool, optional
        If True, applies causal convolution (default is False).
    """

    def __init__(
        self,
        n_in,
        n_out,
        stride,
        up_kernel_size,
        res_kernel_size=7,
        causal=False,
    ):
        super(ResDecoderBlock, self).__init__()
        self.up_conv = UpsampleLayer(
            n_in,
            n_out,
            kernel_size=up_kernel_size,
            stride=stride,
            causal=causal,
            activation=None,
        )
        self.convs = nn.ModuleList(
            [
                ResidualUnit(
                    n_out,
                    n_out,
                    dilation=1,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out,
                    n_out,
                    dilation=3,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out,
                    n_out,
                    dilation=5,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out,
                    n_out,
                    dilation=7,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out,
                    n_out,
                    dilation=9,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
            ]
        )

    def forward(self, x):
        """
        Applies upsampling followed by a series of residual units.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Processed output tensor.
        """
        x = self.up_conv(x)
        for conv in self.convs:
            x = conv(x)
        return x

class ResidualUnit(nn.Module):
    """
    A residual unit with two convolutional layers and activation functions.

    Arguments
    ---------
    n_in : int
        Number of input channels.
    n_out : int
        Number of output channels.
    dilation : int
        Dilation factor for the first convolutional layer.
    res_kernel_size : int, optional
        Size of the convolutional kernel for residual connections (default is 7).
    causal : bool, optional
        If True, applies causal convolution (default is False).
    """

    def __init__(self, n_in, n_out, dilation, res_kernel_size=7, causal=False):
        super(ResidualUnit, self).__init__()
        self.conv1 = weight_norm(
            Conv1d(
                n_in,
                n_out,
                kernel_size=res_kernel_size,
                dilation=dilation,
                causal=causal,
            )
        )
        self.conv2 = weight_norm(
            Conv1d(n_in, n_out, kernel_size=1, causal=causal)
        )
        self.activation1 = nn.PReLU()
        self.activation2 = nn.PReLU()

    def forward(self, x):
        """
        Applies two convolutional layers with activations and adds the input for a residual connection.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor with residual connection applied.
        """
        output = self.activation1(self.conv1(x))
        output = self.activation2(self.conv2(output))
        return output + x

class ResEncoderBlock(nn.Module):
    """
    A residual encoder block with multiple residual units and a downsampling layer.

    Arguments
    ---------
    n_in : int
        Number of input channels.
    n_out : int
        Number of output channels.
    stride : int
        Stride for the downsampling layer.
    down_kernel_size : int
        Kernel size for the downsampling layer.
    res_kernel_size : int, optional
        Size of the convolutional kernel for residual connections (default is 7).
    causal : bool, optional
        If True, applies causal convolution (default is False).
    """

    def __init__(
        self,
        n_in,
        n_out,
        stride,
        down_kernel_size,
        res_kernel_size=7,
        causal=False,
    ):
        super(ResEncoderBlock, self).__init__()
        self.convs = nn.ModuleList(
            [
                ResidualUnit(
                    n_in,
                    n_out // 2,
                    dilation=1,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out // 2,
                    n_out // 2,
                    dilation=3,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out // 2,
                    n_out // 2,
                    dilation=5,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out // 2,
                    n_out // 2,
                    dilation=7,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out // 2,
                    n_out // 2,
                    dilation=9,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
            ]
        )
        self.down_conv = DownsampleLayer(
            n_in, n_out, down_kernel_size, stride=stride, causal=causal
        )

    def forward(self, x):
        """
        Applies a series of residual units and a downsampling layer.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Processed output tensor.
        """
        for conv in self.convs:
            x = conv(x)
        x = self.down_conv(x)
        return x

class ResDecoderBlock(nn.Module):
    """
    A residual decoder block with upsampling and multiple residual units.

    Arguments
    ---------
    n_in : int
        Number of input channels.
    n_out : int
        Number of output channels.
    stride : int
        Stride for the upsampling layer.
    up_kernel_size : int
        Kernel size for the upsampling layer.
    res_kernel_size : int, optional
        Size of the convolutional kernel for residual connections (default is 7).
    causal : bool, optional
        If True, applies causal convolution (default is False).
    """

    def __init__(
        self,
        n_in,
        n_out,
        stride,
        up_kernel_size,
        res_kernel_size=7,
        causal=False,
    ):
        super(ResDecoderBlock, self).__init__()
        self.up_conv = UpsampleLayer(
            n_in,
            n_out,
            kernel_size=up_kernel_size,
            stride=stride,
            causal=causal,
            activation=None,
        )
        self.convs = nn.ModuleList(
            [
                ResidualUnit(
                    n_out,
                    n_out,
                    dilation=1,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out,
                    n_out,
                    dilation=3,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out,
                    n_out,
                    dilation=5,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out,
                    n_out,
                    dilation=7,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out,
                    n_out,
                    dilation=9,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
            ]
        )

    def forward(self, x):
        """
        Applies upsampling followed by a series of residual units.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Processed output tensor.
        """
        x = self.up_conv(x)
        for conv in self.convs:
            x = conv(x)
        return x

class Conv1d(nn.Conv1d):
    """
    Custom 1D convolution layer with causal option.

    Arguments
    ---------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the convolutional kernel.
    stride : int, optional
        Stride of the convolution (default is 1).
    dilation : int, optional
        Dilation factor (default is 1).
    groups : int, optional
        Number of blocked connections (default is 1).
    padding_mode : str, optional
        Padding mode (default is 'zeros').
    bias : bool, optional
        If True, adds a learnable bias (default is True).
    padding : int, optional
        Explicit padding value.
    causal : bool, optional
        If True, applies causal convolution.
    w_init_gain : str, optional
        Gain value for Xavier initialization.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 padding_mode: str = 'zeros',
                 bias: bool = True,
                 padding = None,
                 causal: bool = False,
                 w_init_gain = None):
        self.causal = causal
        if padding is None:
            if causal:
                padding = 0
                self.left_padding = dilation * (kernel_size - 1)
            else:
                padding = get_padding(kernel_size, dilation)
        super(Conv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias=bias)
        if w_init_gain is not None:
            torch.nn.init.xavier_uniform_(
                self.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
    
    def forward(self, x):
        if self.causal:
            x = F.pad(x.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)

        return super(Conv1d, self).forward(x)
    
class ConvTranspose1d(nn.ConvTranspose1d):
    """
    Custom transposed 1D convolution layer with causal option.

    Arguments
    ---------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the convolutional kernel.
    stride : int, optional
        Stride of the convolution (default is 1).
    output_padding : int, optional
        Additional size added to one side of the output (default is 0).
    groups : int, optional
        Number of blocked connections (default is 1).
    bias : bool, optional
        If True, adds a learnable bias (default is True).
    dilation : int, optional
        Dilation factor (default is 1).
    padding : int, optional
        Explicit padding value (default is None).
    padding_mode : str, optional
        Padding mode (default is 'zeros').
    causal : bool, optional
        If True, applies causal convolution.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 output_padding: int = 0,
                 groups: int = 1,
                 bias: bool = True,
                 dilation: int = 1,
                 padding=None,
                 padding_mode: str = 'zeros',
                 causal: bool = False):
        if padding is None:
            padding = 0 if causal else (kernel_size - stride) // 2
        if causal:
            assert padding == 0, "padding is not allowed in causal ConvTranspose1d."
            assert kernel_size == 2 * stride, "kernel_size must be equal to 2*stride is not allowed in causal ConvTranspose1d."
        super(ConvTranspose1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode)
        self.causal = causal
        self.stride = stride

    def forward(self, x):
        """
        Applies the transposed convolution operation.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Transposed convolved output tensor.
        """
        x = super(ConvTranspose1d, self).forward(x)
        if self.causal:
            x = x[:, :, : -self.stride]
        return x

class ScalarModel(nn.Module):
    def __init__(self, num_bands, sample_rate, causal, num_samples, downsample_factors, downsample_kernel_sizes,
                       upsample_factors, upsample_kernel_sizes, latent_hidden_dim, default_kernel_size,
                       delay_kernel_size, init_channel, res_kernel_size):
        super(ScalarModel, self).__init__()
        # self.args = args
        self.encoder = []
        self.decoder = []
        self.vq = round_func_binary() # using 2
        # Encoder parts
        self.encoder.append(
            weight_norm(
                Conv1d(
                    num_bands,
                    init_channel,
                    kernel_size=default_kernel_size,
                    causal=causal
                )
            )
        )
        if num_samples > 1:
            # Downsampling
            self.encoder.append(
                PreProcessor(init_channel,
                             init_channel,
                             num_samples,
                             kernel_size=default_kernel_size,
                             causal=causal))
        for i, down_factor in enumerate(downsample_factors):
            self.encoder.append(
                ResEncoderBlock(init_channel * np.power(2, i),
                                init_channel * np.power(2, i+1),
                                down_factor,
                                downsample_kernel_sizes[i],
                                res_kernel_size,
                                causal=causal))
        self.encoder.append(
            weight_norm(Conv1d(
                init_channel * np.power(2, len(downsample_factors)),
                latent_hidden_dim,
                kernel_size=default_kernel_size,
                causal=causal)))
        # Decoder
        # look ahead 
        self.decoder.append(
            weight_norm(Conv1d(
                latent_hidden_dim,
                init_channel * np.power(2, len(upsample_factors)),
                kernel_size=delay_kernel_size)))
        for i, upsample_factor in enumerate(upsample_factors):
            self.decoder.append(
                ResDecoderBlock(init_channel * np.power(2, len(upsample_factors) - i),
                             init_channel * np.power(2, len(upsample_factors) - i - 1),
                             upsample_factor,
                             upsample_kernel_sizes[i],
                             res_kernel_size,
                             causal=causal))
        if num_samples > 1:
            self.decoder.append(
                PostProcessor(
                    init_channel,
                    init_channel,
                    num_samples,
                    kernel_size=default_kernel_size,
                    causal=causal))
        self.decoder.append(
            weight_norm(Conv1d(
                init_channel,
                num_bands,
                kernel_size=default_kernel_size,
                causal=causal)))
        self.encoder = nn.ModuleList(self.encoder)
        self.decoder = nn.ModuleList(self.decoder)

    def forward(self, x):
        for i, layer in enumerate(self.encoder):
            if i != len(self.encoder) - 1:
                x = layer(x)
            else:
                x = F.tanh(layer(x))
        # import pdb; pdb.set_trace()
        x = self.vq.apply(x) # vq
        for i, layer in enumerate(self.decoder):
            x = layer(x)
        return x

    def inference(self, x):
        for i, layer in enumerate(self.encoder):
            if i != len(self.encoder) - 1:
                x = layer(x)
            else:
                x = F.tanh(layer(x)) # reverse to tanh
        emb = x 
        # import pdb; pdb.set_trace()
        emb_quant = self.vq.apply(emb) # vq
        x = emb_quant
        for i, layer in enumerate(self.decoder):
            x = layer(x)
        return emb, emb_quant, x

    def encode(self, x):
        for i, layer in enumerate(self.encoder):
            if i != len(self.encoder) - 1:
                x = layer(x)
            else:
                x = F.tanh(layer(x)) # reverse to tanh
        emb = x 
        # import pdb; pdb.set_trace()
        emb_quant = self.vq.apply(emb) # vq
        #print('emb_quant org ', emb_quant.shape)
        # assert 1==2
        return emb_quant

    def decode(self, emb_quant):
        x = emb_quant
        for i, layer in enumerate(self.decoder):
            x = layer(x)
        return x

# def decimal_to_ternary_matrix(decimals, D):
#     """
#     Convert a tensor of decimal numbers to a D*N ternary matrix.
    
#     Parameters:
#     - decimals: a 1D tensor of decimal numbers, length N
#     - D: number of ternary digits to represent each number (depth)
    
#     Returns:
#     - A 2D tensor of shape (D, N) where each column is a ternary number.
#     """
#     #print('decimals ', decimals)
#     N = decimals.shape[0]
#     ternary_matrix = torch.zeros((D, N), dtype=torch.long)
#     for index, num in enumerate(decimals):
#         for pos in range(D):
#             ternary_matrix[pos, index] = num % 3  # Fill column from bottom to top
#             num //= 3
#     # print(ternary_matrix)
#     # assert 1==2
#     return ternary_matrix

# def decimal_to_ternary_matrix(decimals, D):
#     """
#     Converts a batch of decimal numbers into a ternary matrix representation using vectorized operations.

#     Parameters:
#     ----------
#     input_decimals : torch.Tensor
#         A 2D tensor of shape (B, N), where B is the batch size and N is the number of decimal inputs per batch.
#     depth : int
#         The number of ternary digits to represent each number.

#     Returns:
#     -------
#     torch.Tensor
#         A 3D tensor of shape (B, D, N), where each element represents a ternary-encoded version of the input decimals,
#         with D indicating the depth of the ternary representation.
#     """
#     B, T = decimals.shape
#     expanded_decimals = decimals.unsqueeze(1).expand(-1, D, -1)  # Shape: (B, D, N)
#     powers_of_three = (3 ** torch.arange(D).flip(0)).view(1, -1, 1).to(decimals.device)  # Shape: (1, D, 1)

#     ternary_matrix = (expanded_decimals // powers_of_three) % 3  # Vectorized ternary conversion

#     return ternary_matrix

def decimal_to_ternary_matrix(decimals, D):
    """
    Convert a tensor of decimal numbers to a D*T ternary matrix for each batch.

    Parameters:
    - decimals: a 2D tensor of decimal numbers with shape (B, T)
    - D: number of ternary digits to represent each number (depth)
    
    Returns:
    - A 3D tensor of shape (B, D, T) where each slice along the first dimension 
      corresponds to a batch and each column is a ternary number.
    """
    B, T = decimals.shape
    ternary_matrix = torch.zeros((B, D, T), dtype=torch.long)
    for pos in range(D):
        ternary_matrix[:, pos, :] = decimals % 3  # Modulo operation
        decimals //= 3  # Floor division for next ternary digit
    
    return ternary_matrix
# def ternary_matrix_to_decimal(matrix):
#     """
#     Convert a D*N ternary matrix to a list of decimal numbers.
    
#     Parameters:
#     - matrix: a 2D numpy array of shape (D, N) where each column is a ternary number.
    
#     Returns:
#     - A list of integers, each representing the decimal equivalent of a ternary number.
#     """
#     D, N = matrix.shape  # D is the number of digits, N is the number of ternary numbers
#     powers_of_three = 3 ** np.arange(D)  # [3^0, 3^1, ..., 3^(D-1)]
#     decimals = np.dot(powers_of_three, matrix)  # Matrix multiplication to convert all columns
#     return decimals.tolist()

def ternary_matrix_to_decimal(matrix):
    """
    Efficiently convert a B*D*N ternary matrix to a list of decimal numbers for each batch without using loops.

    Parameters:
    - matrix: a 3D numpy array of shape (B, D, N) where each column in the D*N slice is a ternary number.

    Returns:
    - A 2D numpy array of shape (B, N) where each value represents the decimal equivalent of a ternary number.
    """
    B, D, N = matrix.shape  # B is the batch size, D is the number of digits, N is the number of ternary numbers
    powers_of_three = 3 ** np.arange(D)  # [3^0, 3^1, ..., 3^(D-1)]

    # Reshape powers_of_three for broadcasting: [D] -> [1, D, 1]
    powers_of_three = powers_of_three[:, np.newaxis]  # Shape [D, 1]

    # Compute dot product using broadcasting: matrix * powers_of_three along D axis
    decimals = np.sum(matrix * powers_of_three, axis=1)  # Sum along the D axis

    return decimals

class SQCodec(nn.Module):
    def __init__(self, 
                source,
                filename,
                save_path=None,
                config="config.yaml",
                checkpoint="ckpt_00190000.pth",
                sample_rate=16000,
                dim_codebook = 19683,
                n_codebook = 4,
                bw =2,
                device=torch.device('cpu'), 
                clip_length=450,
                ):
        """ soundstream with fixed bandwidth of 4kbps 
            It encodes audio with 50 fps and 8-dim vector for each frame
            The value of each entry is in [0, 1023]
        """

        super(SQCodec, self).__init__()
        self.config_path = os.path.join(save_path,filename.split(".")[0], config)
        self.ckpt_path = os.path.join(save_path,filename.split(".")[0], checkpoint)
        if not os.path.exists(self.config_path) and not os.path.exists(self.ckpt_path):
            download_and_extract(source, filename, save_path)
        # GPU is only for offline tokenization
        # So, when distributed training is launched, this should still be on CPU
        self.device = device
        self.clip_length = clip_length

        logging.info(f"using config {self.config_path} and model {self.ckpt_path}")
        self.scalar_codec = self.build_codec_model(self.config_path).to(device)
        # properties
        self.sr = sample_rate
        self.dim_codebook = dim_codebook
        self.n_codebook = n_codebook
        self.bw = bw # bw=1.5 ---> 3 codebooks
        self.freq = self.n_codebook * 50
        self.mask_id = self.dim_codebook * self.n_codebook

    def build_codec_model(self, config):
        exp_model_config = OmegaConf.load(config)
        scalar_codec = ScalarModel(**exp_model_config.generator.config)  
        parameter_dict = torch.load(self.ckpt_path)
        new_state_dict = OrderedDict()
        scalar_codec.load_state_dict(parameter_dict['codec_model']) # load model
        return scalar_codec

    # def _flatten_codebooks(self, arr, offset_size=None):
    #     assert len(arr.shape) == 2
    #     arr = arr.copy()
    #     if offset_size is not None:
    #         for n in range(arr.shape[0]):
    #             arr[n, :] += (offset_size * n)
    #     flat_arr = arr.ravel("F")
        return flat_arr

    def _flatten_codebooks(self, arr, offset_size=None):
        """
        Flattens a 3D array (B, N, D) to a 1D array while applying an offset to each codebook if specified.

        Parameters:
        - arr: 3D numpy array of shape (B, N, D).
        - offset_size: int or None, optional
            The offset size to be applied to each codebook slice (default is None).

        Returns:
        - 1D numpy array representing the flattened codebooks.
        """
        assert len(arr.shape) == 3, "Input array must have 3 dimensions [B, N, D]"
        N, B, D = arr.shape
        arr = arr.copy()

        # Apply offset if specified
        if offset_size is not None:
            for n in range(N):
                arr[n, :, :] += (offset_size * n)

        # Flatten along the last dimension and sum to produce a (B, N) result
        flattened_arr = np.concatenate(arr, axis=1)

        return flattened_arr

    def encode(self, inputs, sr=16000):
        # wav, sr = torchaudio.load(wav_root)
        # if wav.numel() == 0:
        #     return None
        # wav = wav[0:1] # use the first channel
        # if sr != self.sr:
        #     wav = torchaudio.transforms.Resample(sr, self.sr)(wav)
        # wav = wav.unsqueeze(1).to(self.device) # (1,1,len)
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
        compressed = self.scalar_codec.encode(inputs) # B, dim, len
        # compressed = compressed.squeeze(0) # dim, len
        chunks = compressed.chunk(self.n_codebook, dim=1) # 
        codec_ls = []
        for i, chunk in enumerate(chunks):
            chunk = chunk.detach().cpu().numpy() #.int() + 1
            chunk= chunk.astype(np.int32) + 1 # .astype(np.int32)
            tmp_codec = ternary_matrix_to_decimal(chunk)
            codec_ls.append(tmp_codec)
        codec_ls = np.array(codec_ls) # 4*N
        flat_codec = self._flatten_codebooks(codec_ls, self.dim_codebook)
        flat_codec = torch.from_numpy(flat_codec)
        flat_codec = flat_codec.to(torch.int32)
        return flat_codec
    

    def decode(self, codes):
        assert codes.dim() == 2  # Ensure input shape is (B, T)
        B, T = codes.shape
        assert T % self.n_codebook == 0, "Length T must be divisible by n_codebook"
        
        # Reshape to (B, T // n_codebook, n_codebook) for processing
        codes = codes.view(B, self.n_codebook, -1).permute(1,0,2)  # shape (n_codebook, B, T // n_codebook)
        
        # Adjust codes by subtracting the offset
        for i in range(self.n_codebook):
            codes[i, :, :] -= i * self.dim_codebook
        
        emb_quant = []
        for i in range(self.n_codebook):  # Iterate over n_codebook
            tmp_list = decimal_to_ternary_matrix(codes[i,:,:], D=9) - 1
            emb_quant.append(tmp_list)  # Shape (B, T // n_codebook, D)
        
        emb_quant = torch.cat(emb_quant, dim=1)  # Concatenate along the second dimension: (B, n * T // n_codebook, D)
        # emb_quant = emb_quant.unsqueeze(0)  # Add a singleton dimension if needed: (1, B, n * T // n_codebook, D)
        
        # Pass through the scalar codec
        out = self.scalar_codec.decode(emb_quant.float().to(self.device))
        out = out.detach().cpu().squeeze(0)  # Remove the singleton dimension if needed
        return out

    def all(self, wav_root):
        wav, sr = torchaudio.load(wav_root)
        if wav.numel() == 0:
            return None
        if sr != self.sr:
            wav = torchaudio.transforms.Resample(sr, self.sr)(wav)
        wav = wav.unsqueeze(1).to(self.device) # (1,1,len)
        emb, emb_quant, x = self.scalar_codec.inference(wav)
        return x.detach().cpu().squeeze(0)

    @property
    def is_discrete(self):
        return True


    @property
    def codebook_length(self):
        return self.dim_codebook * self.n_codebook + 1

    # def find_length(self, x):
    #     return self.tokenize(x).shape[0] // self.n_codebook

def download_and_extract(repo_id, filename, save_path):
    """
    Downloads a ZIP file from a URL, extracts its contents, and removes the ZIP file.

    Arguments
    ---------
    url : str
        The URL of the ZIP file to download.
    save_path : str
        The directory where the contents will be saved.

    Returns
    -------
    None
    """
    # Ensure save_path directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    

    # Download the file with progress bar
    zip_filename = hf_hub_download(repo_id=repo_id,filename=filename ,cache_dir=save_path)

    # Extract the file
    with zipfile.ZipFile(zip_filename, "r") as zip_ref:
        zip_ref.extractall(save_path)

    # Remove the downloaded ZIP file
    os.remove(zip_filename)
    print(f"File downloaded, extracted to '{save_path}', and ZIP file removed.")


from huggingface_hub import hf_hub_download
if __name__ == '__main__':

    model_hub = "Dongchao/UniAudio"
    save_path = "savedir"
    filename="SQ-Codec.zip"
    config="config.yaml"
    checkpoint="ckpt_00190000.pth"
    model =SQCodec(model_hub, filename, save_path, config, checkpoint)
    # audio = torch.randn(3, 16000)
    wav, sr = torchaudio.load("target.wav")
    audio = torchaudio.transforms.Resample(sr, 16000)(wav)
    print(audio)
    length = torch.tensor([1.0, .5, .75, 1.0])

    tokens= model.encode(audio)
    print(tokens.shape)
    rec = model.decode(tokens)
    torchaudio.save('tmp2.wav', rec, 16000,bits_per_sample=16,  encoding='PCM_S')
    print(rec)

    print(rec.shape)


