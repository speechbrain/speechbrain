"""Library implementing complex-valued convolutional neural networks.

Authors
 * Titouan Parcollet 2020
"""
import torch
import torch.nn as nn
import logging
import torch.nn.functional as F
from speechbrain.nnet.CNN import get_padding_elem
from speechbrain.nnet.complex_networks.complex_ops import (
    complex_convolution,
    check_conv_input,
)

logger = logging.getLogger(__name__)


class ComplexConv1d(torch.nn.Module):
    """This function implements complex-valued 1d convolution.

    Arguments
    ---------
    out_channels: int
        Number of output channels. Please note
        that these are complex-valued neurons. If 256
        channels are specified, the output dimension
        will be 512.
    kernel_size: int
        Kernel size of the convolutional filters.
    stride: int, optional
        Default: 1.
        Stride factor of the convolutional filters.
    dilation: int, optional
        Default: 1.
        Dilation factor of the convolutional filters.
    padding: str, optional
        Default: same.
        (same, valid, causal). If "valid", no padding is performed.
        If "same" and stride is 1, output shape is same as input shape.
        "causal" results in causal (dilated) convolutions.
    padding_mode: str, optional
        Default: reflect.
        This flag specifies the type of padding. See torch.nn documentation
        for more information.
    groups: int, optional
        Default: 1
        This option specifies the convolutional groups. See torch.nn
        documentation for more information.
    bias: bool, optional
        Default: True.
        If True, the additive bias b is adopted.
    init_criterion: str , optional
        Default: he.
        (glorot, he).
        This parameter controls the initialization criterion of the weights.
        It is combined with weights_init to build the initialization method of
        the complex-valued weights.
    weight_init: str, optional
        Default: complex.
        (complex, unitary).
        This parameter defines the initialization procedure of the
        complex-valued weights. "complex" will generate random complex-valued
        weights following the init_criterion and the complex polar form.
        "unitary" will normalize the weights to lie on the unit circle.
        More details in: "Deep Complex Networks", Trabelsi C. et al.

    Example
    -------
    >>> inp_tensor = torch.rand([10, 16, 30])
    >>> cnn_1d = ComplexConv1d(
    ...     input_shape=inp_tensor.shape, out_channels=12, kernel_size=5
    ... )
    >>> out_tensor = cnn_1d(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 16, 24])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        input_shape,
        stride=1,
        dilation=1,
        padding="same",
        groups=1,
        bias=True,
        padding_mode="reflect",
        init_criterion="glorot",
        weight_init="complex",
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
        self.init_criterion = init_criterion
        self.weight_init = weight_init

        check_conv_input(input_shape, channels_axis=-1)
        self.in_channels = self._check_input(input_shape) // 2

        self.conv = complex_convolution(
            self.in_channels,
            self.out_channels,
            conv1d=True,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=0,
            groups=self.groups,
            bias=self.bias,
            init_criterion=self.init_criterion,
            weight_init=self.weight_init,
        )

    def forward(self, x):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve. 3d or 4d tensors are expected.

        """
        # (batch, channel, time)
        x = x.transpose(1, -1)
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

        wx = self.conv(x)

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
        x = F.pad(x, tuple(padding), mode=self.padding_mode)

        return x

    def _check_input(self, input_shape):
        """
        Checks the input and returns the number of input channels.
        """

        if len(input_shape) == 3:
            in_channels = input_shape[2]
        else:
            raise ValueError("conv1d expects 3d inputs. Got " + input_shape)

        # Kernel size must be odd
        if self.kernel_size % 2 == 0:
            raise ValueError(
                "The field kernel size must be an odd number. Got %s."
                % (self.kernel_size)
            )

        return in_channels


class ComplexConv2d(nn.Module):
    """This function implements complex-valued 1d convolution.

    Arguments
    ---------
    out_channels: int
        Number of output channels. Please note
        that these are complex-valued neurons. If 256
        channels are specified, the output dimension
        will be 512.
    kernel_size: int
        Kernel size of the convolutional filters.
    stride: int, optional
        Default: 1.
        Stride factor of the convolutional filters.
    dilation: int, optional
        Default: 1.
        Dilation factor of the convolutional filters.
    padding: str, optional
        Default: same.
        (same, valid, causal). If "valid", no padding is performed.
        If "same" and stride is 1, output shape is same as input shape.
        "causal" results in convolutions which are causal on time axis,
        and "same" on feature axis.
    padding_mode: str, optional
        Default: reflect.
        This flag specifies the type of padding. See torch.nn documentation
        for more information.
    groups: int, optional
        Default: 1
        This option specifies the convolutional groups. See torch.nn
        documentation for more information.
    bias: bool, optional
        Default: True.
        If True, the additive bias b is adopted.
    init_criterion: str , optional
        Default: he.
        (glorot, he).
        This parameter controls the initialization criterion of the weights.
        It is combined with weights_init to build the initialization method of
        the complex-valued weights.
    weight_init: str, optional
        Default: complex.
        (complex, unitary).
        This parameter defines the initialization procedure of the
        complex-valued weights. "complex" will generate random complex-valued
        weights following the init_criterion and the complex polar form.
        "unitary" will normalize the weights to lie on the unit circle.
        More details in: "Deep Complex Networks", Trabelsi C. et al.

    Example
    -------
    >>> inp_tensor = torch.rand([10, 16, 30, 30])
    >>> cnn_2d = ComplexConv2d(
    ...     input_shape=inp_tensor.shape, out_channels=12, kernel_size=5
    ... )
    >>> out_tensor = cnn_2d(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 16, 30, 24])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        input_shape,
        stride=1,
        dilation=1,
        padding="same",
        groups=1,
        bias=True,
        padding_mode="reflect",
        init_criterion="glorot",
        weight_init="complex",
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
        self.init_criterion = init_criterion
        self.weight_init = weight_init

        # k -> [k,k]
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size, self.kernel_size]

        if isinstance(self.dilation, int):
            self.dilation = [self.dilation, self.dilation]

        if isinstance(self.stride, int):
            self.stride = [self.stride, self.stride]

        self.in_channels = self._check_input(input_shape) // 2

        self.conv = complex_convolution(
            self.in_channels,
            self.out_channels,
            conv1d=False,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=0,
            groups=self.groups,
            bias=self.bias,
            init_criterion=self.init_criterion,
            weight_init=self.weight_init,
        )

    def forward(self, x, init_params=False):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : torch.Tensor (batch, time, feature, channels)
            input to convolve. 3d or 4d tensors are expected.

        """
        if init_params:
            self.init_params(x)

        # If (batch, time, feature) -> (batch, time, feature, 1)
        if self.unsqueeze:
            x = x.unsqueeze(1)

        # (batch, channel, time, feature)
        x = x.permute(0, 3, 1, 2)

        if self.padding == "same":
            x = self._manage_padding(
                x, self.kernel_size, self.dilation, self.stride
            )

        elif self.padding == "causal":
            # Padding on time axis for causal
            num_pad = (self.kernel_size[0] - 1) * self.dilation[0]
            x = F.pad(x, (0, 0, num_pad, 0))

            # Padding on feature axis
            x = self._manage_padding(
                x,
                [1, self.kernel_size[1]],
                [1, self.dilation[1]],
                [1, self.stride[1]],
            )

        elif self.padding == "valid":
            pass

        else:
            raise ValueError(
                "Padding must be 'same', 'valid' or 'causal'. Got %s."
                % (self.padding)
            )

        wx = self.conv(x)

        wx = wx.permute(0, 2, 3, 1)

        if self.unsqueeze:
            wx = wx.squeeze(-1)

        return wx

    def _manage_padding(self, x, kernel_size, dilation, stride):
        """This function performs zero-padding on the time and frequency axes
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
            L_in, stride[-2], kernel_size[-2], dilation[-2]
        )

        padding_freq = get_padding_elem(
            L_in, stride[-1], kernel_size[-1], dilation[-1]
        )
        padding = padding_freq + padding_time

        # Applying padding
        x = nn.functional.pad(x, tuple(padding), mode=self.padding_mode)

        return x

    def _check_input(self, input_shape):
        """
        Checks the input and returns the number of input channels.
        """
        if len(input_shape) == 3:
            self.unsqueeze = True
            in_channels = 1

        elif len(input_shape) == 4:
            in_channels = input_shape[3]

        else:
            raise ValueError("Expected 3d or 4d inputs. Got " + input_shape)

        # Kernel size must be odd
        if self.kernel_size[0] % 2 == 0 or self.kernel_size[1] % 2 == 0:
            raise ValueError(
                "The field kernel size must be an odd number. Got %s."
                % (self.kernel_size)
            )

        return in_channels
