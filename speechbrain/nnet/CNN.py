"""Library implementing convolutional neural networks.

Author
    Mirco Ravanelli 2020
"""

import math
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SincConv(nn.Module):
    """This function implements SincConv (SincNet).

    M. Ravanelli, Y. Bengio, "Speaker Recognition from raw waveform with
    SincNet", in Proc. of  SLT 2018 (https://arxiv.org/abs/1808.00158)

    Arguments
    ---------
    out_channels: int
        It is the number of output channels.
    kernel_size: int
        Kernel size of the convolutional filters.
    stride: int
        Stride factor of the convolutional filters. When the stride factor > 1,
        a decimation in time is performed.
    dilation: int
        Dilation factor of the convolutional filters.
    padding: bool
        if True, zero-padding is performed.
    padding_mode: str
        This flag specifies the type of padding. See torch.nn documentation
        for more information.
    groups: int
        This option specifies the convolutional groups. See torch.nn
        documentation for more information.
    bias: bool
        If True, the additive bias b is adopted.
    sample_rate: int,
        Sampling rate of the input signals. It is only used for sinc_conv.
    min_low_hz: float
        Lowest possible frequency (in Hz) for a filter. It is only used for
        sinc_conv.
    min_low_hz: float
        Lowest possible value (in Hz) for a filter bandwidth.

    Example
    -------
    >>> inp_tensor = torch.rand([10, 16000])
    >>> conv = SincConv(out_channels=25, kernel_size=11)
    >>> out_tensor = conv(inp_tensor, init_params=True)
    >>> out_tensor.shape
    torch.Size([10, 16000, 25])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        padding=True,
        groups=1,
        bias=True,
        padding_mode="reflect",
        sample_rate=16000,
        min_low_hz=50,
        min_band_hz=50,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.unsqueeze = False
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

    def init_params(self, first_input):
        """
        Initializes the parameters of the conv1d layer.

        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """

        self.in_channels = self._check_input(first_input)
        self.device = first_input.device

        # Initialize Sinc filters
        self._init_sinc_conv(first_input)

    def forward(self, x, init_params=False):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve. 2d or 4d tensors are expected.

        """
        if init_params:
            self.init_params(x)

        x = x.transpose(1, -1)

        if self.unsqueeze:
            x = x.unsqueeze(1)

        if self.padding:
            x = self._manage_padding(
                x, self.kernel_size, self.dilation, self.stride
            )

        sinc_filters = self._get_sinc_filters()

        wx = F.conv1d(
            x,
            sinc_filters,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )

        if self.unsqueeze:
            wx = wx.squeeze(1)

        wx = wx.transpose(1, -1)

        return wx

    def _get_sinc_filters(self,):
        """This functions creates the sinc-filters to used for sinc-conv.
        """
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
        filters = (
            (band_pass)
            .view(self.out_channels, 1, self.kernel_size)
            .to(self.device)
        )

        return filters

    def _init_sinc_conv(self, first_input):
        """
        Initializes the parameters of the sinc_conv layer.

        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """

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
        self.low_hz_ = nn.Parameter(self.low_hz_).to(self.device)
        self.band_hz_ = nn.Parameter(self.band_hz_).to(self.device)

        # Hamming window
        n_lin = torch.linspace(
            0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)),
        )
        self.window_ = 0.54 - 0.46 * torch.cos(
            2 * math.pi * n_lin / self.kernel_size
        ).to(self.device)

        # Time axis  (only half is needed due to symmetry)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = (
            2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate
        ).to(self.device)

    def _to_mel(self, hz):
        """Converts frequency in Hz to the mel scale.
        """
        return 2595 * np.log10(1 + hz / 700)

    def _to_hz(self, mel):
        """Converts frequency in the mel scale to Hz.
        """
        return 700 * (10 ** (mel / 2595) - 1)

    def _manage_padding(self, x, kernel_size, dilation, stride):
        """This function performs zero-padding on the time axis
        such that their lengths is unchanged after the convolution.

        Arguments
        ---------
        x : torch.Tensor
        kernel_size : int
        dilation : int
        stride: int
        """

        # Detecting input shape
        L_in = x.shape[-1]

        # Time padding
        padding = get_padding_elem(L_in, stride, kernel_size, dilation)

        # Applying padding
        x = nn.functional.pad(x, tuple(padding), mode=self.padding_mode)

        return x

    def _check_input(self, x):
        """
        Checks the input and returns the number of input channels.
        """
        if len(x.shape) == 2:
            self.unsqueeze = True
            in_channels = 1
        elif len(x.shape) == 3:
            in_channels = x.shape[2]
        else:
            raise ValueError("conv1d expects 2d, 3d inputs. Got " + len(x))
        # Kernel size must be odd
        if self.kernel_size % 2 == 0:
            raise ValueError(
                "The field kernel size must be an odd number. Got %s."
                % (self.kernel_size)
            )
        return in_channels


class Conv1d(nn.Module):
    """This function implements 1d convolution.

    Arguments
    ---------
    out_channels: int
        It is the number of output channels.
    kernel_size: int
        Kernel size of the convolutional filters.
    stride: int
        Stride factor of the convolutional filters. When the stride factor > 1,
        a decimation in time is performed.
    dilation: int
        Dilation factor of the convolutional filters.
    padding: bool
        if True, zero-padding is performed.
    padding_mode: str
        This flag specifies the type of padding. See torch.nn documentation
        for more information.
    groups: int
        This option specifies the convolutional groups. See torch.nn
        documentation for more information.
    bias: bool
        If True, the additive bias b is adopted.

    Example
    -------
    >>> inp_tensor = torch.rand([10, 40, 16])
    >>> cnn_1d = Conv1d(out_channels=8, kernel_size=5)
    >>> out_tensor = cnn_1d(inp_tensor, init_params=True)
    >>> out_tensor.shape
    torch.Size([10, 40, 8])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        padding=True,
        groups=1,
        bias=True,
        padding_mode="reflect",
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.unsqueeze = False

    def init_params(self, first_input):
        """
        Initializes the parameters of the conv1d layer.

        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """

        self.in_channels = self._check_input(first_input)

        self.conv = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=0,
            groups=self.groups,
            bias=self.bias,
        ).to(first_input.device)

    def forward(self, x, init_params=False):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve. 2d or 4d tensors are expected.

        """
        if init_params:
            self.init_params(x)

        x = x.transpose(1, -1)

        if self.unsqueeze:
            x = x.unsqueeze(1)

        if self.padding:
            x = self._manage_padding(
                x, self.kernel_size, self.dilation, self.stride
            )

        wx = self.conv(x)

        if self.unsqueeze:
            wx = wx.squeeze(1)

        wx = wx.transpose(1, -1)

        return wx

    def _manage_padding(self, x, kernel_size, dilation, stride):
        """This function performs zero-padding on the time axis
        such that their lengths is unchanged after the convolution.

        Arguments
        ---------
        x : torch.Tensor
        kernel_size : int
        dilation : int
        stride: int
        """

        # Detecting input shape
        L_in = x.shape[-1]

        # Time padding
        padding = get_padding_elem(L_in, stride, kernel_size, dilation)

        # Applying padding
        x = nn.functional.pad(x, tuple(padding), mode=self.padding_mode)

        return x

    def _check_input(self, x):
        """
        Checks the input and returns the number of input channels.
        """

        if len(x.shape) == 2:
            self.unsqueeze = True
            in_channels = 1
        elif len(x.shape) == 3:
            in_channels = x.shape[2]
        else:
            raise ValueError("conv1d expects 2d, 3d inputs. Got " + len(x))
        # Kernel size must be odd
        if self.kernel_size % 2 == 0:
            raise ValueError(
                "The field kernel size must be an odd number. Got %s."
                % (self.kernel_size)
            )
        return in_channels


class Conv2d(nn.Module):
    """This function implements 1d convolution.

    Arguments
    ---------
    out_channels: int
        It is the number of output channels.
    kernel_size: tuple
        Kernel size of the 2d convolutional filters over time and frequency
        axis.
    stride: int
        Stride factor of the 2d convolutional filters over time and frequency
        axis.
    dilation: int
        Dilation factor of the 2d convolutional filters over time and
        frequency axis.
    padding: bool
        if True, zero-padding is performed.
    padding_mode: str
        This flag specifies the type of padding. See torch.nn documentation
        for more information.
    groups: int
        This option specifies the convolutional groups. See torch.nn
        documentation for more information.
    bias: bool
        If True, the additive bias b is adopted.

    Example
    -------
    >>> inp_tensor = torch.rand([10, 40, 16, 8])
    >>> cnn_2d = Conv2d(out_channels=5, kernel_size=(7,3))
    >>> out_tensor = cnn_2d(inp_tensor, init_params=True)
    >>> out_tensor.shape
    torch.Size([10, 40, 16, 5])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        stride=(1, 1),
        dilation=(1, 1),
        padding=True,
        groups=1,
        bias=True,
        padding_mode="reflect",
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.unsqueeze = False

    def init_params(self, first_input):
        """
        Initializes the parameters of the conv1d layer.

        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """

        self.in_channels = self._check_input(first_input)

        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias,
        ).to(first_input.device)

    def forward(self, x, init_params=False):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve. 2d or 4d tensors are expected.

        """
        if init_params:
            self.init_params(x)

        x = x.transpose(1, -1)

        if self.unsqueeze:
            x = x.unsqueeze(1)

        if self.padding:
            x = self._manage_padding(
                x, self.kernel_size, self.dilation, self.stride
            )

        wx = self.conv(x)

        if self.unsqueeze:
            wx = wx.squeeze(1)

        wx = wx.transpose(1, -1)

        return wx

    def _manage_padding(self, x, kernel_size, dilation, stride):
        """This function performs zero-padding on the time and frequency axises
        such that their lengths is unchanged after the convolution.

        Arguments
        ---------
        x : torch.Tensor
        kernel_size : int
        dilation : int
        stride: int
        """
        # Detecting input shape
        L_in = x.shape[-1]

        # Time padding
        padding_time = get_padding_elem(
            L_in, stride[-1], kernel_size[-1], dilation[-1]
        )

        padding_freq = get_padding_elem(
            L_in, stride[-2], kernel_size[-2], dilation[-2]
        )
        padding = padding_time + padding_freq

        # Applying padding
        x = nn.functional.pad(x, tuple(padding), mode=self.padding_mode)

        return x

    def _check_input(self, x):
        """
        Checks the input and returns the number of input channels.
        """
        if len(x.shape) == 3:
            self.unsqueeze = True
            in_channels = 1

        elif len(x.shape) == 4:
            in_channels = x.shape[3]

        else:
            raise ValueError("conv1d expects 3d or 4d inputs. Got " + len(x))

        # Kernel size must be odd
        if self.kernel_size[0] % 2 == 0 or self.kernel_size[1] % 2 == 0:
            raise ValueError(
                "The field kernel size must be an odd number. Got %s."
                % (self.kernel_size)
            )

        return in_channels


def get_padding_elem(L_in, stride, kernel_size, dilation):
    """This function computes the number of elements to add for zero-padding.

    Arguments
    ---------
    L_in : int
    stride: int
    kernel_size : int
    dilation : int
    """
    if stride > 1:
        n_steps = math.ceil(((L_in - kernel_size * dilation) / stride) + 1)
        L_out = stride * (n_steps - 1) + kernel_size * dilation
        padding = [kernel_size // 2, kernel_size // 2]

    else:
        L_out = (L_in - dilation * (kernel_size - 1) - 1) / stride + 1
        L_out = int(L_out)

        padding = [(L_in - L_out) // 2, (L_in - L_out) // 2]
    return padding
