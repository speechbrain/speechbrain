"""Library implementing convolutional neural networks.

Authors
 * Mirco Ravanelli 2020
 * Jianyuan Zhong 2020
 * Cem Subakan 2021
 * Davide Borra 2021
 * Andreas Nautsch 2022
 * Sarthak Yadav 2022
"""

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from speechbrain.processing.signal_processing import (
    gabor_impulse_response,
    gabor_impulse_response_legacy_complex,
)
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class SincConv(nn.Module):
    """This function implements SincConv (SincNet).

    M. Ravanelli, Y. Bengio, "Speaker Recognition from raw waveform with
    SincNet", in Proc. of  SLT 2018 (https://arxiv.org/abs/1808.00158)

    Arguments
    ---------
    out_channels : int
        It is the number of output channels.
    kernel_size: int
        Kernel size of the convolutional filters.
    input_shape : tuple
        The shape of the input. Alternatively use ``in_channels``.
    in_channels : int
        The number of input channels. Alternatively use ``input_shape``.
    stride : int
        Stride factor of the convolutional filters. When the stride factor > 1,
        a decimation in time is performed.
    dilation : int
        Dilation factor of the convolutional filters.
    padding : str
        (same, valid, causal). If "valid", no padding is performed.
        If "same" and stride is 1, output shape is the same as the input shape.
        "causal" results in causal (dilated) convolutions.
    padding_mode : str
        This flag specifies the type of padding. See torch.nn documentation
        for more information.
    sample_rate : int
        Sampling rate of the input signals. It is only used for sinc_conv.
    min_low_hz : float
        Lowest possible frequency (in Hz) for a filter. It is only used for
        sinc_conv.
    min_band_hz : float
        Lowest possible value (in Hz) for a filter bandwidth.

    Example
    -------
    >>> inp_tensor = torch.rand([10, 16000])
    >>> conv = SincConv(
    ...     input_shape=inp_tensor.shape, out_channels=25, kernel_size=11
    ... )
    >>> out_tensor = conv(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 16000, 25])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        input_shape=None,
        in_channels=None,
        stride=1,
        dilation=1,
        padding="same",
        padding_mode="reflect",
        sample_rate=16000,
        min_low_hz=50,
        min_band_hz=50,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # input shape inference
        if input_shape is None and self.in_channels is None:
            raise ValueError("Must provide one of input_shape or in_channels")

        if self.in_channels is None:
            self.in_channels = self._check_input_shape(input_shape)

        if self.out_channels % self.in_channels != 0:
            raise ValueError(
                "Number of output channels must be divisible by in_channels"
            )

        # Initialize Sinc filters
        self._init_sinc_conv()

    def forward(self, x):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve. 2d or 4d tensors are expected.

        Returns
        -------
        wx : torch.Tensor
            The convolved outputs.
        """
        x = x.transpose(1, -1)
        self.device = x.device

        unsqueeze = x.ndim == 2
        if unsqueeze:
            x = x.unsqueeze(1)

        if self.padding == "same":
            x = self._manage_padding(
                x, self.kernel_size, self.dilation, self.stride
            )

        elif self.padding == "causal":
            num_pad = (self.kernel_size - 1) * self.dilation
            x = F.pad(x, (num_pad, 0))

        elif self.padding == "valid":
            pass

        else:
            raise ValueError(
                "Padding must be 'same', 'valid' or 'causal'. Got %s."
                % (self.padding)
            )

        sinc_filters = self._get_sinc_filters()

        wx = F.conv1d(
            x,
            sinc_filters,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.in_channels,
        )

        if unsqueeze:
            wx = wx.squeeze(1)

        wx = wx.transpose(1, -1)

        return wx

    def _check_input_shape(self, shape):
        """Checks the input shape and returns the number of input channels."""

        if len(shape) == 2:
            in_channels = 1
        elif len(shape) == 3:
            in_channels = shape[-1]
        else:
            raise ValueError(
                "sincconv expects 2d or 3d inputs. Got " + str(len(shape))
            )

        # Kernel size must be odd
        if self.kernel_size % 2 == 0:
            raise ValueError(
                "The field kernel size must be an odd number. Got %s."
                % (self.kernel_size)
            )
        return in_channels

    def _get_sinc_filters(self):
        """This functions creates the sinc-filters to used for sinc-conv."""
        # Computing the low frequencies of the filters
        low = self.min_low_hz + torch.abs(self.low_hz_)

        # Setting minimum band and minimum freq
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2,
        )
        band = (high - low)[:, 0]

        # Passing from n_ to the corresponding f_times_t domain
        self.n_ = self.n_.to(self.device)
        self.window_ = self.window_.to(self.device)
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        # Left part of the filters.
        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low))
            / (self.n_ / 2)
        ) * self.window_

        # Central element of the filter
        band_pass_center = 2 * band.view(-1, 1)

        # Right part of the filter (sinc filters are symmetric)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        # Combining left, central, and right part of the filter
        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right], dim=1
        )

        # Amplitude normalization
        band_pass = band_pass / (2 * band[:, None])

        # Setting up the filter coefficients
        filters = band_pass.view(self.out_channels, 1, self.kernel_size)

        return filters

    def _init_sinc_conv(self):
        """Initializes the parameters of the sinc_conv layer."""

        # Initialize filterbanks such that they are equally spaced in Mel scale
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = torch.linspace(
            self._to_mel(self.min_low_hz),
            self._to_mel(high_hz),
            self.out_channels + 1,
        )

        hz = self._to_hz(mel)

        # Filter lower frequency and bands
        self.low_hz_ = hz[:-1].unsqueeze(1)
        self.band_hz_ = (hz[1:] - hz[:-1]).unsqueeze(1)

        # Maiking freq and bands learnable
        self.low_hz_ = nn.Parameter(self.low_hz_)
        self.band_hz_ = nn.Parameter(self.band_hz_)

        # Hamming window
        n_lin = torch.linspace(
            0, (self.kernel_size / 2) - 1, steps=int(self.kernel_size / 2)
        )
        self.window_ = 0.54 - 0.46 * torch.cos(
            2 * math.pi * n_lin / self.kernel_size
        )

        # Time axis  (only half is needed due to symmetry)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = (
            2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate
        )

    def _to_mel(self, hz):
        """Converts frequency in Hz to the mel scale."""
        return 2595 * np.log10(1 + hz / 700)

    def _to_hz(self, mel):
        """Converts frequency in the mel scale to Hz."""
        return 700 * (10 ** (mel / 2595) - 1)

    def _manage_padding(self, x, kernel_size: int, dilation: int, stride: int):
        """This function performs zero-padding on the time axis
        such that their lengths is unchanged after the convolution.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        kernel_size : int
            Size of kernel.
        dilation : int
            Dilation used.
        stride : int
            Stride.

        Returns
        -------
        x : torch.Tensor
        """

        # Detecting input shape
        L_in = self.in_channels

        # Time padding
        padding = get_padding_elem(L_in, stride, kernel_size, dilation)

        # Applying padding
        x = F.pad(x, padding, mode=self.padding_mode)

        return x


class Conv1d(nn.Module):
    """This function implements 1d convolution.

    Arguments
    ---------
    out_channels : int
        It is the number of output channels.
    kernel_size : int
        Kernel size of the convolutional filters.
    input_shape : tuple
        The shape of the input. Alternatively use ``in_channels``.
    in_channels : int
        The number of input channels. Alternatively use ``input_shape``.
    stride : int
        Stride factor of the convolutional filters. When the stride factor > 1,
        a decimation in time is performed.
    dilation : int
        Dilation factor of the convolutional filters.
    padding : str
        (same, valid, causal). If "valid", no padding is performed.
        If "same" and stride is 1, output shape is the same as the input shape.
        "causal" results in causal (dilated) convolutions.
    groups : int
        Number of blocked connections from input channels to output channels.
    bias : bool
        Whether to add a bias term to convolution operation.
    padding_mode : str
        This flag specifies the type of padding. See torch.nn documentation
        for more information.
    skip_transpose : bool
        If False, uses batch x time x channel convention of speechbrain.
        If True, uses batch x channel x time convention.
    weight_norm : bool
        If True, use weight normalization,
        to be removed with self.remove_weight_norm() at inference
    conv_init : str
        Weight initialization for the convolution network
    default_padding: str or int
        This sets the default padding mode that will be used by the pytorch Conv1d backend.

    Example
    -------
    >>> inp_tensor = torch.rand([10, 40, 16])
    >>> cnn_1d = Conv1d(
    ...     input_shape=inp_tensor.shape, out_channels=8, kernel_size=5
    ... )
    >>> out_tensor = cnn_1d(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 40, 8])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        input_shape=None,
        in_channels=None,
        stride=1,
        dilation=1,
        padding="same",
        groups=1,
        bias=True,
        padding_mode="reflect",
        skip_transpose=False,
        weight_norm=False,
        conv_init=None,
        default_padding=0,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode
        self.unsqueeze = False
        self.skip_transpose = skip_transpose

        if input_shape is None and in_channels is None:
            raise ValueError("Must provide one of input_shape or in_channels")

        if in_channels is None:
            in_channels = self._check_input_shape(input_shape)

        self.in_channels = in_channels

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=default_padding,
            groups=groups,
            bias=bias,
        )

        if conv_init == "kaiming":
            nn.init.kaiming_normal_(self.conv.weight)
        elif conv_init == "zero":
            nn.init.zeros_(self.conv.weight)
        elif conv_init == "normal":
            nn.init.normal_(self.conv.weight, std=1e-6)

        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve. 2d or 4d tensors are expected.

        Returns
        -------
        wx : torch.Tensor
            The convolved outputs.
        """
        if not self.skip_transpose:
            x = x.transpose(1, -1)

        if self.unsqueeze:
            x = x.unsqueeze(1)

        if self.padding == "same":
            x = self._manage_padding(
                x, self.kernel_size, self.dilation, self.stride
            )

        elif self.padding == "causal":
            num_pad = (self.kernel_size - 1) * self.dilation
            x = F.pad(x, (num_pad, 0))

        elif self.padding == "valid":
            pass

        else:
            raise ValueError(
                "Padding must be 'same', 'valid' or 'causal'. Got "
                + self.padding
            )

        wx = self.conv(x)

        if self.unsqueeze:
            wx = wx.squeeze(1)

        if not self.skip_transpose:
            wx = wx.transpose(1, -1)

        return wx

    def _manage_padding(self, x, kernel_size: int, dilation: int, stride: int):
        """This function performs zero-padding on the time axis
        such that their lengths is unchanged after the convolution.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        kernel_size : int
            Size of kernel.
        dilation : int
            Dilation used.
        stride : int
            Stride.

        Returns
        -------
        x : torch.Tensor
            The padded outputs.
        """

        # Detecting input shape
        L_in = self.in_channels

        # Time padding
        padding = get_padding_elem(L_in, stride, kernel_size, dilation)

        # Applying padding
        x = F.pad(x, padding, mode=self.padding_mode)

        return x

    def _check_input_shape(self, shape):
        """Checks the input shape and returns the number of input channels."""

        if len(shape) == 2:
            self.unsqueeze = True
            in_channels = 1
        elif self.skip_transpose:
            in_channels = shape[1]
        elif len(shape) == 3:
            in_channels = shape[2]
        else:
            raise ValueError(
                "conv1d expects 2d, 3d inputs. Got " + str(len(shape))
            )

        # Kernel size must be odd
        if not self.padding == "valid" and self.kernel_size % 2 == 0:
            raise ValueError(
                "The field kernel size must be an odd number. Got %s."
                % (self.kernel_size)
            )

        return in_channels

    def remove_weight_norm(self):
        """Removes weight normalization at inference if used during training."""
        self.conv = nn.utils.remove_weight_norm(self.conv)


class Conv2d(nn.Module):
    """This function implements 2d convolution.

    Arguments
    ---------
    out_channels : int
        It is the number of output channels.
    kernel_size : tuple
        Kernel size of the 2d convolutional filters over time and frequency
        axis.
    input_shape : tuple
        The shape of the input. Alternatively use ``in_channels``.
    in_channels : int
        The number of input channels. Alternatively use ``input_shape``.
    stride: int
        Stride factor of the 2d convolutional filters over time and frequency
        axis.
    dilation : int
        Dilation factor of the 2d convolutional filters over time and
        frequency axis.
    padding : str
        (same, valid, causal).
        If "valid", no padding is performed.
        If "same" and stride is 1, output shape is same as input shape.
        If "causal" then proper padding is inserted to simulate causal convolution on the first spatial dimension.
        (spatial dim 1 is dim 3 for both skip_transpose=False and skip_transpose=True)
    groups : int
        This option specifies the convolutional groups. See torch.nn
        documentation for more information.
    bias : bool
        If True, the additive bias b is adopted.
    padding_mode : str
        This flag specifies the type of padding. See torch.nn documentation
        for more information.
    max_norm : float
        kernel max-norm.
    swap : bool
        If True, the convolution is done with the format (B, C, W, H).
        If False, the convolution is dine with (B, H, W, C).
        Active only if skip_transpose is False.
    skip_transpose : bool
        If False, uses batch x spatial.dim2 x spatial.dim1 x channel convention of speechbrain.
        If True, uses batch x channel x spatial.dim1 x spatial.dim2 convention.
    weight_norm : bool
        If True, use weight normalization,
        to be removed with self.remove_weight_norm() at inference
    conv_init : str
        Weight initialization for the convolution network

    Example
    -------
    >>> inp_tensor = torch.rand([10, 40, 16, 8])
    >>> cnn_2d = Conv2d(
    ...     input_shape=inp_tensor.shape, out_channels=5, kernel_size=(7, 3)
    ... )
    >>> out_tensor = cnn_2d(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 40, 16, 5])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        input_shape=None,
        in_channels=None,
        stride=(1, 1),
        dilation=(1, 1),
        padding="same",
        groups=1,
        bias=True,
        padding_mode="reflect",
        max_norm=None,
        swap=False,
        skip_transpose=False,
        weight_norm=False,
        conv_init=None,
    ):
        super().__init__()

        # handle the case if some parameter is int
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode
        self.unsqueeze = False
        self.max_norm = max_norm
        self.swap = swap
        self.skip_transpose = skip_transpose

        if input_shape is None and in_channels is None:
            raise ValueError("Must provide one of input_shape or in_channels")

        if in_channels is None:
            in_channels = self._check_input(input_shape)

        self.in_channels = in_channels

        # Weights are initialized following pytorch approach
        self.conv = nn.Conv2d(
            self.in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=groups,
            bias=bias,
        )

        if conv_init == "kaiming":
            nn.init.kaiming_normal_(self.conv.weight)
        elif conv_init == "zero":
            nn.init.zeros_(self.conv.weight)

        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve. 2d or 4d tensors are expected.

        Returns
        -------
        x : torch.Tensor
            The output of the convolution.
        """
        if not self.skip_transpose:
            x = x.transpose(1, -1)
            if self.swap:
                x = x.transpose(-1, -2)

        if self.unsqueeze:
            x = x.unsqueeze(1)

        if self.padding == "same":
            x = self._manage_padding(
                x, self.kernel_size, self.dilation, self.stride
            )

        elif self.padding == "causal":
            num_pad = (self.kernel_size[0] - 1) * self.dilation[1]
            x = F.pad(x, (0, 0, num_pad, 0))

        elif self.padding == "valid":
            pass

        else:
            raise ValueError(
                "Padding must be 'same','valid' or 'causal'. Got "
                + self.padding
            )

        if self.max_norm is not None:
            self.conv.weight.data = torch.renorm(
                self.conv.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )

        wx = self.conv(x)

        if self.unsqueeze:
            wx = wx.squeeze(1)

        if not self.skip_transpose:
            wx = wx.transpose(1, -1)
            if self.swap:
                wx = wx.transpose(1, 2)
        return wx

    def _manage_padding(
        self,
        x,
        kernel_size: Tuple[int, int],
        dilation: Tuple[int, int],
        stride: Tuple[int, int],
    ):
        """This function performs zero-padding on the time and frequency axes
        such that their lengths is unchanged after the convolution.

        Arguments
        ---------
        x : torch.Tensor
            Input to be padded
        kernel_size : int
            Size of the kernel for computing padding
        dilation : int
            Dilation rate for computing padding
        stride: int
            Stride for computing padding

        Returns
        -------
        x : torch.Tensor
            The padded outputs.
        """
        # Detecting input shape
        L_in = self.in_channels

        # Time padding
        padding_time = get_padding_elem(
            L_in, stride[-1], kernel_size[-1], dilation[-1]
        )

        padding_freq = get_padding_elem(
            L_in, stride[-2], kernel_size[-2], dilation[-2]
        )
        padding = padding_time + padding_freq

        # Applying padding
        x = nn.functional.pad(x, padding, mode=self.padding_mode)

        return x

    def _check_input(self, shape):
        """Checks the input shape and returns the number of input channels."""

        if len(shape) == 3:
            self.unsqueeze = True
            in_channels = 1

        elif len(shape) == 4:
            in_channels = shape[3]

        else:
            raise ValueError(f"Expected 3d or 4d inputs. Got {len(shape)}")

        # Kernel size must be odd
        if not self.padding == "valid" and (
            self.kernel_size[0] % 2 == 0 or self.kernel_size[1] % 2 == 0
        ):
            raise ValueError(
                "The field kernel size must be an odd number. Got %s."
                % (self.kernel_size)
            )

        return in_channels

    def remove_weight_norm(self):
        """Removes weight normalization at inference if used during training."""
        self.conv = nn.utils.remove_weight_norm(self.conv)


class ConvTranspose1d(nn.Module):
    """This class implements 1d transposed convolution with speechbrain.
    Transpose convolution is normally used to perform upsampling.

    Arguments
    ---------
    out_channels : int
        It is the number of output channels.
    kernel_size : int
        Kernel size of the convolutional filters.
    input_shape : tuple
        The shape of the input. Alternatively use ``in_channels``.
    in_channels : int
        The number of input channels. Alternatively use ``input_shape``.
    stride : int
        Stride factor of the convolutional filters. When the stride factor > 1,
        upsampling in time is performed.
    dilation : int
        Dilation factor of the convolutional filters.
    padding : str or int
        To have in output the target dimension, we suggest tuning the kernel
        size and the padding properly. We also support the following function
        to have some control over the padding and the corresponding output
        dimensionality.
        if "valid", no padding is applied
        if "same", padding amount is inferred so that the output size is closest
        to possible to input size. Note that for some kernel_size / stride combinations
        it is not possible to obtain the exact same size, but we return the closest
        possible size.
        if "factor", padding amount is inferred so that the output size is closest
        to inputsize*stride. Note that for some kernel_size / stride combinations
        it is not possible to obtain the exact size, but we return the closest
        possible size.
        if an integer value is entered, a custom padding is used.
    output_padding : int,
        Additional size added to one side of the output shape
    groups: int
        Number of blocked connections from input channels to output channels.
        Default: 1
    bias: bool
        If True, adds a learnable bias to the output
    skip_transpose : bool
        If False, uses batch x time x channel convention of speechbrain.
        If True, uses batch x channel x time convention.
    weight_norm : bool
        If True, use weight normalization,
        to be removed with self.remove_weight_norm() at inference

    Example
    -------
    >>> from speechbrain.nnet.CNN import Conv1d, ConvTranspose1d
    >>> inp_tensor = torch.rand([10, 12, 40])  # [batch, time, fea]
    >>> convtranspose_1d = ConvTranspose1d(
    ...     input_shape=inp_tensor.shape,
    ...     out_channels=8,
    ...     kernel_size=3,
    ...     stride=2,
    ... )
    >>> out_tensor = convtranspose_1d(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 25, 8])

    >>> # Combination of Conv1d and ConvTranspose1d
    >>> from speechbrain.nnet.CNN import Conv1d, ConvTranspose1d
    >>> signal = torch.tensor([1, 100])
    >>> signal = torch.rand([1, 100])  # [batch, time]
    >>> conv1d = Conv1d(
    ...     input_shape=signal.shape, out_channels=1, kernel_size=3, stride=2
    ... )
    >>> conv_out = conv1d(signal)
    >>> conv_t = ConvTranspose1d(
    ...     input_shape=conv_out.shape,
    ...     out_channels=1,
    ...     kernel_size=3,
    ...     stride=2,
    ...     padding=1,
    ... )
    >>> signal_rec = conv_t(conv_out, output_size=[100])
    >>> signal_rec.shape
    torch.Size([1, 100])

    >>> signal = torch.rand([1, 115])  # [batch, time]
    >>> conv_t = ConvTranspose1d(
    ...     input_shape=signal.shape,
    ...     out_channels=1,
    ...     kernel_size=3,
    ...     stride=2,
    ...     padding="same",
    ... )
    >>> signal_rec = conv_t(signal)
    >>> signal_rec.shape
    torch.Size([1, 115])

    >>> signal = torch.rand([1, 115])  # [batch, time]
    >>> conv_t = ConvTranspose1d(
    ...     input_shape=signal.shape,
    ...     out_channels=1,
    ...     kernel_size=7,
    ...     stride=2,
    ...     padding="valid",
    ... )
    >>> signal_rec = conv_t(signal)
    >>> signal_rec.shape
    torch.Size([1, 235])

    >>> signal = torch.rand([1, 115])  # [batch, time]
    >>> conv_t = ConvTranspose1d(
    ...     input_shape=signal.shape,
    ...     out_channels=1,
    ...     kernel_size=7,
    ...     stride=2,
    ...     padding="factor",
    ... )
    >>> signal_rec = conv_t(signal)
    >>> signal_rec.shape
    torch.Size([1, 231])

    >>> signal = torch.rand([1, 115])  # [batch, time]
    >>> conv_t = ConvTranspose1d(
    ...     input_shape=signal.shape,
    ...     out_channels=1,
    ...     kernel_size=3,
    ...     stride=2,
    ...     padding=10,
    ... )
    >>> signal_rec = conv_t(signal)
    >>> signal_rec.shape
    torch.Size([1, 211])

    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        input_shape=None,
        in_channels=None,
        stride=1,
        dilation=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        skip_transpose=False,
        weight_norm=False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.unsqueeze = False
        self.skip_transpose = skip_transpose

        if input_shape is None and in_channels is None:
            raise ValueError("Must provide one of input_shape or in_channels")

        if in_channels is None:
            in_channels = self._check_input_shape(input_shape)

        if self.padding == "same":
            L_in = input_shape[-1] if skip_transpose else input_shape[1]
            padding_value = get_padding_elem_transposed(
                L_in,
                L_in,
                stride=stride,
                kernel_size=kernel_size,
                dilation=dilation,
                output_padding=output_padding,
            )
        elif self.padding == "factor":
            L_in = input_shape[-1] if skip_transpose else input_shape[1]
            padding_value = get_padding_elem_transposed(
                L_in * stride,
                L_in,
                stride=stride,
                kernel_size=kernel_size,
                dilation=dilation,
                output_padding=output_padding,
            )
        elif self.padding == "valid":
            padding_value = 0
        elif type(self.padding) is int:
            padding_value = padding
        else:
            raise ValueError("Not supported padding type")

        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=padding_value,
            groups=groups,
            bias=bias,
        )

        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x, output_size=None):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve. 2d or 4d tensors are expected.
        output_size : int
            The size of the output

        Returns
        -------
        x : torch.Tensor
            The convolved output
        """

        if not self.skip_transpose:
            x = x.transpose(1, -1)

        if self.unsqueeze:
            x = x.unsqueeze(1)

        wx = self.conv(x, output_size=output_size)

        if self.unsqueeze:
            wx = wx.squeeze(1)

        if not self.skip_transpose:
            wx = wx.transpose(1, -1)

        return wx

    def _check_input_shape(self, shape):
        """Checks the input shape and returns the number of input channels."""

        if len(shape) == 2:
            self.unsqueeze = True
            in_channels = 1
        elif self.skip_transpose:
            in_channels = shape[1]
        elif len(shape) == 3:
            in_channels = shape[2]
        else:
            raise ValueError(
                "conv1d expects 2d, 3d inputs. Got " + str(len(shape))
            )

        return in_channels

    def remove_weight_norm(self):
        """Removes weight normalization at inference if used during training."""
        self.conv = nn.utils.remove_weight_norm(self.conv)


class DepthwiseSeparableConv1d(nn.Module):
    """This class implements the depthwise separable 1d convolution.

    First, a channel-wise convolution is applied to the input
    Then, a point-wise convolution to project the input to output

    Arguments
    ---------
    out_channels : int
        It is the number of output channels.
    kernel_size : int
        Kernel size of the convolutional filters.
    input_shape : tuple
        Expected shape of the input.
    stride : int
        Stride factor of the convolutional filters. When the stride factor > 1,
        a decimation in time is performed.
    dilation : int
        Dilation factor of the convolutional filters.
    padding : str
        (same, valid, causal). If "valid", no padding is performed.
        If "same" and stride is 1, output shape is the same as the input shape.
        "causal" results in causal (dilated) convolutions.
    bias : bool
        If True, the additive bias b is adopted.

    Example
    -------
    >>> inp = torch.randn([8, 120, 40])
    >>> conv = DepthwiseSeparableConv1d(256, 3, input_shape=inp.shape)
    >>> out = conv(inp)
    >>> out.shape
    torch.Size([8, 120, 256])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        input_shape,
        stride=1,
        dilation=1,
        padding="same",
        bias=True,
    ):
        super().__init__()

        assert len(input_shape) == 3, "input must be a 3d tensor"

        bz, time, chn = input_shape

        self.depthwise = Conv1d(
            chn,
            kernel_size,
            input_shape=input_shape,
            stride=stride,
            dilation=dilation,
            padding=padding,
            groups=chn,
            bias=bias,
        )

        self.pointwise = Conv1d(
            out_channels,
            kernel_size=1,
            input_shape=input_shape,
        )

    def forward(self, x):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve. 3d tensors are expected.

        Returns
        -------
        The convolved outputs.
        """
        return self.pointwise(self.depthwise(x))


class DepthwiseSeparableConv2d(nn.Module):
    """This class implements the depthwise separable 2d convolution.

    First, a channel-wise convolution is applied to the input
    Then, a point-wise convolution to project the input to output

    Arguments
    ---------
    out_channels : int
        It is the number of output channels.
    kernel_size : int
        Kernel size of the convolutional filters.
    input_shape : tuple
        Expected shape of the input tensors.
    stride : int
        Stride factor of the convolutional filters. When the stride factor > 1,
        a decimation in time is performed.
    dilation : int
        Dilation factor of the convolutional filters.
    padding : str
        (same, valid, causal). If "valid", no padding is performed.
        If "same" and stride is 1, output shape is the same as the input shape.
        "causal" results in causal (dilated) convolutions.
    bias : bool
        If True, the additive bias b is adopted.

    Example
    -------
    >>> inp = torch.randn([8, 120, 40, 1])
    >>> conv = DepthwiseSeparableConv2d(256, (3, 3), input_shape=inp.shape)
    >>> out = conv(inp)
    >>> out.shape
    torch.Size([8, 120, 40, 256])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        input_shape,
        stride=(1, 1),
        dilation=(1, 1),
        padding="same",
        bias=True,
    ):
        super().__init__()

        # handle the case if some parameter is int
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        assert len(input_shape) in {3, 4}, "input must be a 3d or 4d tensor"
        self.unsqueeze = len(input_shape) == 3

        bz, time, chn1, chn2 = input_shape

        self.depthwise = Conv2d(
            chn2,
            kernel_size,
            input_shape=input_shape,
            stride=stride,
            dilation=dilation,
            padding=padding,
            groups=chn2,
            bias=bias,
        )

        self.pointwise = Conv2d(
            out_channels,
            kernel_size=(1, 1),
            input_shape=input_shape,
        )

    def forward(self, x):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve. 3d tensors are expected.

        Returns
        -------
        out : torch.Tensor
            The convolved output.
        """
        if self.unsqueeze:
            x = x.unsqueeze(1)

        out = self.pointwise(self.depthwise(x))

        if self.unsqueeze:
            out = out.squeeze(1)

        return out


class GaborConv1d(nn.Module):
    """
    This class implements 1D Gabor Convolutions from

    Neil Zeghidour, Olivier Teboul, F{\'e}lix de Chaumont Quitry & Marco Tagliasacchi, "LEAF: A LEARNABLE FRONTEND
    FOR AUDIO CLASSIFICATION", in Proc. of ICLR 2021 (https://arxiv.org/abs/2101.08596)

    Arguments
    ---------
    out_channels : int
        It is the number of output channels.
    kernel_size: int
        Kernel size of the convolutional filters.
    stride : int
        Stride factor of the convolutional filters. When the stride factor > 1,
        a decimation in time is performed.
    input_shape : tuple
        Expected shape of the input.
    in_channels : int
        Number of channels expected in the input.
    padding : str
        (same, valid). If "valid", no padding is performed.
        If "same" and stride is 1, output shape is the same as the input shape.
    padding_mode : str
        This flag specifies the type of padding. See torch.nn documentation
        for more information.
    sample_rate : int,
        Sampling rate of the input signals. It is only used for sinc_conv.
    min_freq : float
        Lowest possible frequency (in Hz) for a filter
    max_freq : float
        Highest possible frequency (in Hz) for a filter
    n_fft: int
        number of FFT bins for initialization
    normalize_energy: bool
        whether to normalize energy at initialization. Default is False
    bias : bool
        If True, the additive bias b is adopted.
    sort_filters: bool
        whether to sort filters by center frequencies. Default is False
    use_legacy_complex: bool
        If False, torch.complex64 data type is used for gabor impulse responses
        If True, computation is performed on two real-valued tensors
    skip_transpose: bool
        If False, uses batch x time x channel convention of speechbrain.
        If True, uses batch x channel x time convention.

    Example
    -------
    >>> inp_tensor = torch.rand([10, 8000])
    >>> # 401 corresponds to a window of 25 ms at 16000 kHz
    >>> gabor_conv = GaborConv1d(40, kernel_size=401, stride=1, in_channels=1)
    >>> #
    >>> out_tensor = gabor_conv(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 8000, 40])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        stride,
        input_shape=None,
        in_channels=None,
        padding="same",
        padding_mode="constant",
        sample_rate=16000,
        min_freq=60.0,
        max_freq=None,
        n_fft=512,
        normalize_energy=False,
        bias=False,
        sort_filters=False,
        use_legacy_complex=False,
        skip_transpose=False,
    ):
        super().__init__()
        self.filters = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.sort_filters = sort_filters
        self.sample_rate = sample_rate
        self.min_freq = min_freq
        if max_freq is None:
            max_freq = sample_rate / 2
        self.max_freq = max_freq
        self.n_fft = n_fft
        self.normalize_energy = normalize_energy
        self.use_legacy_complex = use_legacy_complex
        self.skip_transpose = skip_transpose

        if input_shape is None and in_channels is None:
            raise ValueError("Must provide one of input_shape or in_channels")

        if in_channels is None:
            in_channels = self._check_input_shape(input_shape)

        self.kernel = nn.Parameter(self._initialize_kernel())
        if bias:
            self.bias = torch.nn.Parameter(torch.ones(self.filters * 2))
        else:
            self.bias = None

    def forward(self, x):
        """Returns the output of the Gabor convolution.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve.

        Returns
        -------
        x : torch.Tensor
            The output of the Gabor convolution
        """
        if not self.skip_transpose:
            x = x.transpose(1, -1)

        unsqueeze = x.ndim == 2
        if unsqueeze:
            x = x.unsqueeze(1)

        kernel = self._gabor_constraint(self.kernel)
        if self.sort_filters:
            idxs = torch.argsort(kernel[:, 0])
            kernel = kernel[idxs, :]

        filters = self._gabor_filters(kernel)
        if not self.use_legacy_complex:
            temp = torch.view_as_real(filters)
            real_filters = temp[:, :, 0]
            img_filters = temp[:, :, 1]
        else:
            real_filters = filters[:, :, 0]
            img_filters = filters[:, :, 1]
        stacked_filters = torch.cat(
            [real_filters.unsqueeze(1), img_filters.unsqueeze(1)], dim=1
        )
        stacked_filters = torch.reshape(
            stacked_filters, (2 * self.filters, self.kernel_size)
        )
        stacked_filters = stacked_filters.unsqueeze(1)

        if self.padding == "same":
            x = self._manage_padding(x, self.kernel_size)
        elif self.padding == "valid":
            pass
        else:
            raise ValueError(
                "Padding must be 'same' or 'valid'. Got " + self.padding
            )

        output = F.conv1d(
            x, stacked_filters, bias=self.bias, stride=self.stride, padding=0
        )
        if not self.skip_transpose:
            output = output.transpose(1, -1)
        return output

    def _gabor_constraint(self, kernel_data):
        mu_lower = 0.0
        mu_upper = math.pi
        sigma_lower = (
            4
            * torch.sqrt(
                2.0 * torch.log(torch.tensor(2.0, device=kernel_data.device))
            )
            / math.pi
        )
        sigma_upper = (
            self.kernel_size
            * torch.sqrt(
                2.0 * torch.log(torch.tensor(2.0, device=kernel_data.device))
            )
            / math.pi
        )
        clipped_mu = torch.clamp(
            kernel_data[:, 0], mu_lower, mu_upper
        ).unsqueeze(1)
        clipped_sigma = torch.clamp(
            kernel_data[:, 1], sigma_lower, sigma_upper
        ).unsqueeze(1)
        return torch.cat([clipped_mu, clipped_sigma], dim=-1)

    def _gabor_filters(self, kernel):
        t = torch.arange(
            -(self.kernel_size // 2),
            (self.kernel_size + 1) // 2,
            dtype=kernel.dtype,
            device=kernel.device,
        )
        if not self.use_legacy_complex:
            return gabor_impulse_response(
                t, center=kernel[:, 0], fwhm=kernel[:, 1]
            )
        else:
            return gabor_impulse_response_legacy_complex(
                t, center=kernel[:, 0], fwhm=kernel[:, 1]
            )

    def _manage_padding(self, x, kernel_size):
        # this is the logic that gives correct shape that complies
        # with the original implementation at https://github.com/google-research/leaf-audio

        def get_padding_value(kernel_size):
            """Gets the number of elements to pad."""
            kernel_sizes = (kernel_size,)
            from functools import reduce
            from operator import __add__

            conv_padding = reduce(
                __add__,
                [
                    (k // 2 + (k - 2 * (k // 2)) - 1, k // 2)
                    for k in kernel_sizes[::-1]
                ],
            )
            return conv_padding

        pad_value = get_padding_value(kernel_size)
        x = F.pad(x, pad_value, mode=self.padding_mode, value=0)
        return x

    def _mel_filters(self):
        def _mel_filters_areas(filters):
            peaks, _ = torch.max(filters, dim=1, keepdim=True)
            return (
                peaks
                * (torch.sum((filters > 0).float(), dim=1, keepdim=True) + 2)
                * np.pi
                / self.n_fft
            )

        mel_filters = torchaudio.functional.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=self.min_freq,
            f_max=self.max_freq,
            n_mels=self.filters,
            sample_rate=self.sample_rate,
        )
        mel_filters = mel_filters.transpose(1, 0)
        if self.normalize_energy:
            mel_filters = mel_filters / _mel_filters_areas(mel_filters)
        return mel_filters

    def _gabor_params_from_mels(self):
        coeff = torch.sqrt(2.0 * torch.log(torch.tensor(2.0))) * self.n_fft
        sqrt_filters = torch.sqrt(self._mel_filters())
        center_frequencies = torch.argmax(sqrt_filters, dim=1)
        peaks, _ = torch.max(sqrt_filters, dim=1, keepdim=True)
        half_magnitudes = peaks / 2.0
        fwhms = torch.sum((sqrt_filters >= half_magnitudes).float(), dim=1)
        output = torch.cat(
            [
                (center_frequencies * 2 * np.pi / self.n_fft).unsqueeze(1),
                (coeff / (np.pi * fwhms)).unsqueeze(1),
            ],
            dim=-1,
        )
        return output

    def _initialize_kernel(self):
        return self._gabor_params_from_mels()

    def _check_input_shape(self, shape):
        """Checks the input shape and returns the number of input channels."""

        if len(shape) == 2:
            in_channels = 1
        elif len(shape) == 3:
            in_channels = 1
        else:
            raise ValueError(
                "GaborConv1d expects 2d or 3d inputs. Got " + str(len(shape))
            )

        # Kernel size must be odd
        if self.kernel_size % 2 == 0:
            raise ValueError(
                "The field kernel size must be an odd number. Got %s."
                % (self.kernel_size)
            )
        return in_channels


def get_padding_elem(L_in: int, stride: int, kernel_size: int, dilation: int):
    """This function computes the number of elements to add for zero-padding.

    Arguments
    ---------
    L_in : int
    stride: int
    kernel_size : int
    dilation : int

    Returns
    -------
    padding : int
        The size of the padding to be added
    """
    if stride > 1:
        padding = [math.floor(kernel_size / 2), math.floor(kernel_size / 2)]

    else:
        L_out = (
            math.floor((L_in - dilation * (kernel_size - 1) - 1) / stride) + 1
        )
        padding = [
            math.floor((L_in - L_out) / 2),
            math.floor((L_in - L_out) / 2),
        ]
    return padding


def get_padding_elem_transposed(
    L_out: int,
    L_in: int,
    stride: int,
    kernel_size: int,
    dilation: int,
    output_padding: int,
):
    """This function computes the required padding size for transposed convolution

    Arguments
    ---------
    L_out : int
    L_in : int
    stride: int
    kernel_size : int
    dilation : int
    output_padding : int

    Returns
    -------
    padding : int
        The size of the padding to be applied
    """

    padding = -0.5 * (
        L_out
        - (L_in - 1) * stride
        - dilation * (kernel_size - 1)
        - output_padding
        - 1
    )
    return int(padding)
