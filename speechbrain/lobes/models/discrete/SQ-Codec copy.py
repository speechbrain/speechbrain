"""This lobe enables the integration of  pretrained SQ-Codec.

SQ-Codec is a scalar quantizatiod instrad of the Residual Vector Quantization (RVQ) in previous audio codec works.
audio codec works, named SQ-Codec. 
' 
The code is adopted from https://github.com/yangdongchao/SimpleSpeech/blob/main/ldm/models/scalar16k.py

Repository: https://github.com/yangdongchao/SimpleSpeech
Paper: https://arxiv.org/pdf/2406.02328

Authors
 * Pooneh Mousavi 2024
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm, remove_weight_norm
from torch.autograd.function import InplaceFunction
from omegaconf import OmegaConf
from huggingface_hub import snapshot_download


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

@torch.jit.script
def snake(x, alpha):
    """
    Computes the Snake activation function.

    Arguments
    ---------
    x : torch.Tensor
        Input tensor.
    alpha : torch.Tensor
        Scaling parameter.

    Returns
    -------
    torch.Tensor
        Output tensor after applying the Snake function.
    """
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x

class Snake1d(nn.Module):
    """
    Snake activation module for 1D data.

    Arguments
    ---------
    channels : int
        Number of input channels.
    """
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        """
        Applies the Snake activation function.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Transformed tensor.
        """
        return snake(x, self.alpha)

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
                 padding=None,
                 causal: bool = False,
                 w_init_gain=None):
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
        """
        Applies the convolution operation.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Convolved output tensor.
        """
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
            x = x[:, :, :-self.stride]
        return x

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
        self.conv1 = weight_norm(Conv1d(n_in, n_out, kernel_size=res_kernel_size, dilation=dilation, causal=causal))
        self.conv2 = weight_norm(Conv1d(n_in, n_out, kernel_size=1, causal=causal))
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
    def __init__(self, n_in, n_out, stride, down_kernel_size, res_kernel_size=7, causal=False):
        super(ResEncoderBlock, self).__init__()
        self.convs = nn.ModuleList([
            ResidualUnit(n_in, n_out // 2, dilation=1, res_kernel_size=res_kernel_size, causal=causal),
            ResidualUnit(n_out // 2, n_out // 2, dilation=3, res_kernel_size=res_kernel_size, causal=causal),
            ResidualUnit(n_out // 2, n_out // 2, dilation=5, res_kernel_size=res_kernel_size, causal=causal),
            ResidualUnit(n_out // 2, n_out // 2, dilation=7, res_kernel_size=res_kernel_size, causal=causal),
            ResidualUnit(n_out // 2, n_out // 2, dilation=9, res_kernel_size=res_kernel_size, causal=causal),
        ])
        self.down_conv = DownsampleLayer(
            n_in, n_out, down_kernel_size, stride=stride, causal=causal)

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
    def __init__(self, n_in, n_out, stride, up_kernel_size, res_kernel_size=7, causal=False):
        super(ResDecoderBlock, self).__init__()
        self.up_conv = UpsampleLayer(
            n_in, n_out, kernel_size=up_kernel_size, stride=stride, causal=causal, activation=None)
        self.convs = nn.ModuleList([
            ResidualUnit(n_out, n_out, dilation=1, res_kernel_size=res_kernel_size, causal=causal),
            ResidualUnit(n_out, n_out, dilation=3, res_kernel_size=res_kernel_size, causal=causal),
            ResidualUnit(n_out, n_out, dilation=5, res_kernel_size=res_kernel_size, causal=causal),
            ResidualUnit(n_out, n_out, dilation=7, res_kernel_size=res_kernel_size, causal=causal),
            ResidualUnit(n_out, n_out, dilation=9, res_kernel_size=res_kernel_size, causal=causal),
        ])

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
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 causal: bool = False,
                 activation=nn.PReLU(),
                 use_weight_norm: bool = True,
                 pooling: bool = False):
        super(DownsampleLayer, self).__init__()
        self.pooling = pooling
        self.stride = stride
        self.activation = activation
        self.use_weight_norm = use_weight_norm
        if pooling:
            self.layer = Conv1d(
                in_channels, out_channels, kernel_size, causal=causal)
            self.pooling = nn.AvgPool1d(kernel_size=stride)
        else:
            self.layer = Conv1d(
                in_channels, out_channels, kernel_size, stride=stride, causal=causal)
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
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 causal: bool = False,
                 activation=nn.PReLU(),
                 use_weight_norm: bool = True,
                 repeat: bool = False):
        super(UpsampleLayer, self).__init__()
        self.repeat = repeat
        self.stride = stride
        self.activation = activation
        self.use_weight_norm = use_weight_norm
        if repeat:
            self.layer = Conv1d(
                in_channels, out_channels, kernel_size, causal=causal)
        else:
            self.layer = ConvTranspose1d(
                in_channels, out_channels, kernel_size, stride=stride, causal=causal)
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

class round_func5(InplaceFunction):
    """
    Custom rounding function that rounds input values to the nearest multiple of 1/5.

    Methods
    -------
    forward(ctx, input)
        Applies forward pass for rounding.
    backward(ctx, grad_output)
        Computes gradients for backward pass.
    """
    @staticmethod
    def forward(ctx, input):
        """
        Forward pass that rounds input tensor values to the nearest multiple of 1/5.

        Arguments
        ---------
        ctx : context
            Context for storing information for backward computation.
        input : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Rounded tensor.
        """
        ctx.input = input
        return torch.round(5 * input) / 5

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass that propagates gradients through the rounding operation.

        Arguments
        ---------
        ctx : context
            Context from the forward pass.
        grad_output : torch.Tensor
            Gradient of the output.

        Returns
        -------
        torch.Tensor
            Gradient of the input.
        """
        grad_input = grad_output.clone()
        return grad_input

class round_func9(InplaceFunction):
    """
    Custom rounding function that rounds input values to the nearest multiple of 1/9.

    Methods
    -------
    forward(ctx, input)
        Applies forward pass for rounding.
    backward(ctx, grad_output)
        Computes gradients for backward pass.
    """
    @staticmethod
    def forward(ctx, input):
        """
        Forward pass that rounds input tensor values to the nearest multiple of 1/9.

        Arguments
        ---------
        ctx : context
            Context for storing information for backward computation.
        input : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Rounded tensor.
        """
        ctx.input = input
        return torch.round(9 * input) / 9

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass that propagates gradients through the rounding operation.

        Arguments
        ---------
        ctx : context
            Context from the forward pass.
        grad_output : torch.Tensor
            Gradient of the output.

        Returns
        -------
        torch.Tensor
            Gradient of the input.
        """
        grad_input = grad_output.clone()
        return grad_input

class round_func_binary(InplaceFunction):
    """
    Custom rounding function that rounds input values to the nearest integer (binary rounding).

    Methods
    -------
    forward(ctx, input)
        Applies forward pass for rounding.
    backward(ctx, grad_output)
        Computes gradients for backward pass.
    """
    @staticmethod
    def forward(ctx, input):
        """
        Forward pass that rounds input tensor values to the nearest integer.

        Arguments
        ---------
        ctx : context
            Context for storing information for backward computation.
        input : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Rounded tensor.
        """
        ctx.input = input
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass that propagates gradients through the rounding operation.

        Arguments
        ---------
        ctx : context
            Context from the forward pass.
        grad_output : torch.Tensor
            Gradient of the output.

        Returns
        -------
        torch.Tensor
            Gradient of the input.
        """
        grad_input = grad_output.clone()
        return grad_input

class ScalarModel(nn.Module):
    """
    A model for scalar quantization with encoder-decoder structure using convolutional layers.

    Arguments
    ---------
    num_bands : int
        Number of input bands.
    sample_rate : int
        Sample rate of the input data.
    causal : bool
        If True, applies causal convolution.
    num_samples : int
        Number of samples to process.
    downsample_factors : list
        List of downsample factors for each encoder layer.
    downsample_kernel_sizes : list
        List of kernel sizes for downsampling layers.
    upsample_factors : list
        List of upsample factors for each decoder layer.
    upsample_kernel_sizes : list
        List of kernel sizes for upsampling layers.
    latent_hidden_dim : int
        Dimension of the latent space.
    default_kernel_size : int
        Default kernel size for convolution layers.
    delay_kernel_size : int
        Kernel size for the delay layer.
    init_channel : int
        Initial number of channels for convolution layers.
    res_kernel_size : int
        Kernel size for residual layers.
    """
    def __init__(self, num_bands, sample_rate, causal, num_samples, downsample_factors, downsample_kernel_sizes,
                 upsample_factors, upsample_kernel_sizes, latent_hidden_dim, default_kernel_size,
                 delay_kernel_size, init_channel, res_kernel_size):
        super(ScalarModel, self).__init__()
        self.encoder = []
        self.decoder = []
        self.vq = round_func9()  # Using custom rounding function

        # Encoder components
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
            # Downsampling layer
            self.encoder.append(
                PreProcessor(init_channel, init_channel, num_samples, kernel_size=default_kernel_size, causal=causal))
        for i, down_factor in enumerate(downsample_factors):
            self.encoder.append(
                ResEncoderBlock(init_channel * np.power(2, i), init_channel * np.power(2, i + 1),
                                down_factor, downsample_kernel_sizes[i], res_kernel_size, causal=causal))
        self.encoder.append(
            weight_norm(Conv1d(
                init_channel * np.power(2, len(downsample_factors)),
                latent_hidden_dim,
                kernel_size=default_kernel_size,
                causal=causal)))

        # Decoder components
        self.decoder.append(
            weight_norm(Conv1d(
                latent_hidden_dim,
                init_channel * np.power(2, len(upsample_factors)),
                kernel_size=delay_kernel_size)))
        for i, upsample_factor in enumerate(upsample_factors):
            self.decoder.append(
                ResDecoderBlock(init_channel * np.power(2, len(upsample_factors) - i),
                                init_channel * np.power(2, len(upsample_factors) - i - 1),
                                upsample_factor, upsample_kernel_sizes[i], res_kernel_size, causal=causal))
        if num_samples > 1:
            self.decoder.append(
                PostProcessor(init_channel, init_channel, num_samples, kernel_size=default_kernel_size, causal=causal))
        self.decoder.append(
            weight_norm(Conv1d(
                init_channel,
                num_bands,
                kernel_size=default_kernel_size,
                causal=causal)))

        self.encoder = nn.ModuleList(self.encoder)
        self.decoder = nn.ModuleList(self.decoder)

    def forward(self, x):
        """
        Forward pass through the encoder-decoder structure.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after processing through the model.
        """
        for i, layer in enumerate(self.encoder):
            if i != len(self.encoder) - 1:
                x = layer(x)
            else:
                x = F.tanh(layer(x))
        x = self.vq.apply(x)  # Quantization
        for i, layer in enumerate(self.decoder):
            x = layer(x)
        return x

    def encode(self, x):
        """
        Encodes input data using the encoder layers.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Encoded tensor (latent representation).
        """
        for i, layer in enumerate(self.encoder):
            if i != len(self.encoder) - 1:
                x = layer(x)
            else:
                x = F.tanh(layer(x))
        emb = x
        emb_quant = self.vq.apply(emb)  # Quantization
        return emb , emb_quant

    def decode(self, x):
        """
        Decodes input data using the decoder layers.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor (latent representation).

        Returns
        -------
        torch.Tensor
            Decoded output tensor.
        """
        # x = self.vq.apply(x)  # Ensure similar distribution
        for i, layer in enumerate(self.decoder):
            x = layer(x)
        return x

    def inference(self, x):
        """
        Full inference pipeline (encoding and decoding).

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        tuple
            Tuple of (encoded tensor, quantized encoded tensor, final output).
        """
        for i, layer in enumerate(self.encoder):
            if i != len(self.encoder) - 1:
                x = layer(x)
            else:
                x = F.tanh(layer(x))
        emb = x
        emb_quant = self.vq.apply(emb)  # Quantization
        x = emb_quant
        for i, layer in enumerate(self.decoder):
            x = layer(x)
        return emb, emb_quant, x

class ScalarAE(nn.Module):
    """
    Autoencoder model that utilizes a ScalarModel for encoding and decoding.

    Arguments
    ---------
    scalar_config : str
        Path to the configuration file for the model.
    resume_path : str, optional
        Path to a checkpoint file for resuming training (default is None).
    num_codebooks : int, optional
        Number of codebooks for encoding (default is 4).
    Example
    -------
    >>> model_path = "16k_32dim_5/ckpt.pth"
    >>> config_file = "16k_32dim_5/config.yaml"
    >>> model = ScalarAE( config_file, model_path)
    >>> audio = torch.randn(4, 16000 )
    >>> length = torch.tensor([1.0, .5, .75, 1.0])
    >>> _, tokens = model.encode(audio)
    >>> print(tokens.shape)
    torch.Size([4, 32, 50])
    >>> rec = model.decode(tokens)
    >>> print(rec.shape)
    torch.Size([4, 1, 16000])
    """
    def __init__(self, scalar_config, resume_path=None, num_codebooks=4):
        super(ScalarAE, self).__init__()
        self.num_codebooks = num_codebooks
        self.scalar_config = scalar_config
        self.resume_path = resume_path
        exp_model_config = OmegaConf.load(self.scalar_config)
        self.model = ScalarModel(**exp_model_config.generator.config)
        if resume_path is not None:
            self.resume_model()

    def resume_model(self):
        """
        Loads model parameters from a checkpoint file.
        """
        parameter_dict = torch.load(self.resume_path)
        self.model.load_state_dict(parameter_dict['codec_model'])

    def encode(self, x):
        """
        Encodes input data using the ScalarModel encoder.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Encoded tensor with limited number of codebooks.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.model.encode(x)

    def decode(self, x):
        """
        Decodes input data using the ScalarModel decoder.

        Arguments
        ---------
        x : torch.Tensor
            Encoded tensor.

        Returns
        -------
        torch.Tensor
            Decoded output tensor.
        """
        return self.model.decode(x)

import os
import requests
import zipfile
from tqdm import tqdm

def download_and_extract(url, save_path):
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
    zip_filename = os.path.join(save_path, 'downloaded_file.zip')
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kilobyte

    with open(zip_filename, 'wb') as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=block_size):
            f.write(chunk)
            bar.update(len(chunk))

    # Extract the file
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(save_path)

    # Remove the downloaded ZIP file
    os.remove(zip_filename)
    print(f"File downloaded, extracted to '{save_path}', and ZIP file removed.")

# Example usage
# download_and_extract('https://huggingface.co/Dongchao/UniAudio/resolve/main/16k_50dim_9.zip', 'path/to/save')


model_path = "16k_32dim_5/ckpt.pth"
config_file = "16k_32dim_5/config.yaml"
num_codebooks = 7
model = ScalarAE( config_file, model_path, num_codebooks)
audio = torch.randn(4,  16000)
length = torch.tensor([1.0, .5, .75, 1.0])
_, tokens = model.encode(audio)
# _,tokens,rec = model.model.inference(audio)
print(tokens.shape)
rec = model.decode(tokens)
print(rec.shape)