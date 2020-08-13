"""This library implements different operations needed by complex-
 valued architectures.
 This work is inspired by: "Deep Complex Networks" from Trabelsi C.
 et al.

Authors
 * Titouan Parcollet 2020
"""

import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np


class complex_linear(nn.Module):
    """ This class implements a fully connected complex-valued
        linear layer: y = Wx + b. y, W, x and b are thus complex
        numbers. A complex number is written as: r + xi. A tensor of
        complex numbers x = [batch, 32] can be understood as
        [batch, 0:15] = R and [batch, 16:31] = Xi. Thus the features
        dimension is cut in half (must be dividible by 2).

    Arguments
    ---------
    inp_size: int
        input size in terms of complex numbers. If input_size
        is equal to 512 real numbers, then 256 must be given.
    n_neurons: int
        number of output neurons (i.e, the dimensionality of the output).
        Please note that these are complex-valued neurons. If 256 neurons
        are specified, the output dimension will be 512.
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
    device: str, optional
        Defines in the pytorch style the device that should be used for
        computations.

    Example
    -------
    >>> model = complex_linear(70, 200, bias=True)
    >>> inp_tensor = torch.rand([100,4,140])
    >>> out_tensor = model(inp_tensor)
    >>> out_tensor.shape
    torch.Size([100, 4, 400])
    """

    def __init__(
        self,
        inp_size,
        n_neurons,
        bias,
        init_criterion="glorot",
        weight_init="complex",
        device="cpu",
    ):
        super(complex_linear, self).__init__()

        # Setting parameters
        self.inp_size = inp_size
        self.n_neurons = n_neurons
        self.bias = bias
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.device = device

        # Two weight matrices are created for the real and imaginary parts of
        # the weights. This will also allow an easier complex product.
        self.real_weight = Parameter(torch.Tensor(inp_size, n_neurons))
        self.imag_weight = Parameter(torch.Tensor(inp_size, n_neurons))

        if self.bias:
            self.b = Parameter(torch.Tensor(2 * n_neurons))
        else:
            self.b = torch.tensor(2 * n_neurons, requires_grad=False)

        # Managing the weight initialization and bias
        self.winit = {"complex": complex_init, "unitary": unitary_init}[
            self.weight_init
        ]

        affect_init(
            self.real_weight, self.imag_weight, self.winit, init_criterion
        )

        if self.b.requires_grad:
            self.b.data.zero_()

    def forward(self, x):
        """Returns the output of the linear operation.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)

        """

        wx = complex_linear_op(x, self.real_weight, self.imag_weight, self.b)

        return wx


class complex_convolution(Module):
    """ This class implements a complex-valued convolutional layer:
        y = w*x + b. y, W, x and b are thus complex numbers.
        A complex number is written as: r + xi. A tensor of
        complex numbers x = [batch, 32] can be understood as
        [batch, 0:15] = R and [batch, 16:31] = Xi.

        Arguments
        ---------

        in_channels: int
            Number of input channels. Please note
            that these are complex-valued neurons. If 256
            channels are specified, the real input dimension
            is 512.
        out_channels: int
            Number of output channels. Please note
            that these are complex-valued neurons. If 256
            channels are specified, the output dimension
            will be 512.
        conv1d: bool
            If True a 1D convolution will be applied. If False,
            a 2D convolution will be used.
        kernel_size: int
            Kernel size of the convolutional filters.
        stride: int, optional
            Default: 1.
            Stride factor of the convolutional filters.
        dilation: int, optional
            Default: 1.
            Dilation factor of the convolutional filters.
        padding: int, optional
            Default: 0.
            Amount of padding to add.
        groups: int, optional
            Default: 1.
            This option specifies the convolutional groups.
            See torch.nn documentation for more information.
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
     >>> conv2 = complex_convolution(8, 64, conv1d=False, kernel_size=[3,3])
     >>> inp_tensor = torch.rand([10, 16, 64, 64])
     >>> out_tensor = conv2(inp_tensor)
     >>> out_tensor.shape
     torch.Size([10, 128, 62, 62])
     """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv1d,
        kernel_size,
        stride=1,
        dilation=1,
        padding=0,
        groups=1,
        bias=True,
        init_criterion="glorot",
        weight_init="complex",
    ):
        super(complex_convolution, self).__init__()

        # Setting parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1d = conv1d
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.init_criterion = init_criterion
        self.weight_init = weight_init

        # Managing the weight initialization and bias by directly setting the
        # correct function
        self.winit = {"complex": complex_init, "unitary": unitary_init}[
            self.weight_init
        ]

        (self.k_shape, self.w_shape) = get_kernel_and_weight_shape(
            self.conv1d, self.in_channels, self.out_channels, self.kernel_size
        )

        self.real_weight = Parameter(torch.Tensor(*self.w_shape))
        self.imag_weight = Parameter(torch.Tensor(*self.w_shape))

        if self.bias:
            self.b = Parameter(torch.Tensor(2 * self.out_channels))
        else:
            self.b = torch.Tensor(2 * self.out_channels, requires_grad=False)

        affect_conv_init(
            self.real_weight,
            self.imag_weight,
            self.kernel_size,
            self.winit,
            self.init_criterion,
        )

        if self.b.requires_grad:
            self.b.data.zero_()

    def forward(self, x):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve.
        """

        wx = complex_conv_op(
            x,
            self.conv1d,
            self.real_weight,
            self.imag_weight,
            self.b,
            self.stride,
            0,
            self.dilation,
        )
        return wx


def check_complex_input(input):
    """Check the complex-valued shape for a linear layer.

    Arguments
    ---------
    input : torch.Tensor (batch, time, channel)
    """
    if input.dim() not in {2, 3}:
        raise Exception(
            "Complex linear accepts only input of dimension 2 or 3."
            " input.dim = " + str(input.dim())
        )

    nb_hidden = input.size()[-1]

    if nb_hidden % 1 != 0:
        raise Exception(
            "Complex Tensors must have an even number of hidden dimensions."
            " input.size()[1] = " + str(nb_hidden)
        )


def check_conv_input(input, channels_axis=1):
    """Check the complex-valued shape for a convolutional layer.

    Arguments
    ---------
    input : torch.Tensor (batch, time, channel)
    channels_axis : int, index of the channel axis.
    """
    if input.dim() not in {3, 4, 5}:
        raise Exception(
            "Complex convolution accepts only input of dimension 3, 4 or 5."
            " input.dim = " + str(input.dim())
        )

    nb_channels = input.size(channels_axis)
    if nb_channels % 2 != 0:
        print("input.size()" + str(input.size()))
        raise Exception(
            "Complex Tensors must have an even number of feature maps."
            " input.size()[1] = " + str(nb_channels)
        )


def get_real(input, input_type="linear", channels_axis=1):
    """Returns the real components of the complex-valued input.

    Arguments
    ---------
    input : torch.Tensor (batch, time, channel)
    input_type: str, (linear, convolution)
    channels_axis : int, index of the channel axis.
    """
    if input_type == "linear":
        check_complex_input(input)
    elif input_type == "convolution":
        check_conv_input(input, channels_axis=channels_axis)
    else:
        raise Exception(
            "Input_type must be either 'convolution' or 'linear'."
            " Found input_type = " + str(input_type)
        )

    if input_type == "linear":
        nb_hidden = input.size()[-1]
        if input.dim() == 2:
            return input.narrow(
                1, 0, nb_hidden // 2
            )  # input[:, :nb_hidden / 2]
        elif input.dim() == 3:
            return input.narrow(
                2, 0, nb_hidden // 2
            )  # input[:, :, :nb_hidden / 2]
    else:
        nb_featmaps = input.size(channels_axis)
        return input.narrow(channels_axis, 0, nb_featmaps // 2)


def get_imag(input, input_type="linear", channels_axis=1):
    """Returns the imaginary components of the complex-valued input.

    Arguments
    ---------
    input : torch.Tensor (batch, time, channel)
    input_type: str, (linear, convolution)
    channels_axis : int, index of the channel axis.
    """
    if input_type == "linear":
        check_complex_input(input)
    elif input_type == "convolution":
        check_conv_input(input, channels_axis=channels_axis)
    else:
        raise Exception(
            "Input_type must be either 'convolution' or 'linear'."
            " Found input_type = " + str(input_type)
        )

    if input_type == "linear":
        nb_hidden = input.size()[-1]
        if input.dim() == 2:
            return input.narrow(
                1, nb_hidden // 2, nb_hidden // 2
            )  # input[:, :nb_hidden / 2]
        elif input.dim() == 3:
            return input.narrow(
                2, nb_hidden // 2, nb_hidden // 2
            )  # input[:, :, :nb_hidden / 2]
    else:
        nb_featmaps = input.size(channels_axis)
        return input.narrow(channels_axis, nb_featmaps // 2, nb_featmaps // 2)


def get_conjugate(input, input_type="linear", channels_axis=1):
    """Returns the conjugate (z = r - xi) of the input complex numbers.

    Arguments
    ---------
    input : torch.Tensor (batch, time, channel)
    input_type: str, (linear, convolution)
    channels_axis : int, index of the channel axis.
    """
    input_imag = get_imag(input, input_type, channels_axis)
    input_real = get_real(input, input_type, channels_axis)
    if input_type == "linear":
        return torch.cat([input_real, -input_imag], dim=-1)
    elif input_type == "convolution":
        return torch.cat([input_real, -input_imag], dim=channels_axis)


def complex_linear_op(input, real_weight, imag_weight, bias):
    """
    Applies a complex linear transformation to the incoming data.

    Arguments
    ---------
    input: torch.Tensor, (batch_size, nb_complex_in * 2)
    real_weight: torch.Parameters, (nb_complex_in, nb_complex_out)
    imag_weight: torch.Parameters, (nb_complex_in, nb_complex_out)
    bias: torch.Parameters, (nb_complex_out * 2)
    """

    cat_real = torch.cat([real_weight, -imag_weight], dim=0)
    cat_imag = torch.cat([imag_weight, real_weight], dim=0)
    cat_complex = torch.cat([cat_real, cat_imag], dim=1)
    if bias.requires_grad:
        if input.dim() == 3:
            if input.size(0) == 1:
                input = input.squeeze(0)
                return torch.addmm(bias, input, cat_complex)
            else:
                return input.matmul(cat_complex) + bias
        else:
            return torch.addmm(bias, input, cat_complex)
    else:
        if input.dim() == 3:
            return input.matmul(cat_complex)
        else:
            return input.mm(cat_complex)


def complex_conv_op(
    input, conv1d, real_weight, imag_weight, bias, stride, padding, dilation
):
    """Applies a complex convolution to the incoming data.

    Arguments
    ---------
    input: torch.Tensor, (batch_size, nb_complex_in * 2, *signal_length)
    real_weight: torch.Parameters, (nb_complex_out, nb_complex_in, *kernel)
    imag_weight: torch.Parameters, (nb_complex_out, nb_complex_in, *kernel)
    bias: torch.Parameters, (nb_complex_out * 2)
    stride: int
    padding: int
    dilation: int
    """
    cat_real = torch.cat([real_weight, -imag_weight], dim=1)
    cat_imag = torch.cat([imag_weight, real_weight], dim=1)
    cat_complex = torch.cat([cat_real, cat_imag], dim=0)

    if conv1d:
        convfunc = F.conv1d
    else:
        convfunc = F.conv2d

    return convfunc(input, cat_complex, bias, stride, padding, dilation)


def unitary_init(
    in_features, out_features, kernel_size=None, criterion="glorot"
):
    """ Returns a matrice of unitary complex numbers.

    Arguments
    ---------
    in_features: int
    out_features: int
    kernel_size: int
    criterion: str, (glorot, he)
    """

    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    number_of_weights = np.prod(kernel_shape)
    v_r = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_i = np.random.uniform(-1.0, 1.0, number_of_weights)

    # Unitary quaternion
    for i in range(0, number_of_weights):
        norm = np.sqrt(v_r[i] ** 2 + v_i[i] ** 2) + 0.0001
        v_r[i] /= norm
        v_i[i] /= norm

    v_r = v_r.reshape(kernel_shape)
    v_i = v_i.reshape(kernel_shape)

    return (v_r, v_i)


def complex_init(
    in_features, out_features, kernel_size=None, criterion="glorot"
):
    """ Returns a matrice of complex numbers initialized as described in:
    "Deep Complex Networks", Trabelsi C. et al.

    Arguments
    ---------
    in_features: int
    out_features: int
    kernel_size: int
    criterion: str, (glorot, he)
    """

    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_out = out_features * receptive_field
        fan_in = in_features * receptive_field
    else:
        fan_out = out_features
        fan_in = in_features
    if criterion == "glorot":
        s = 1.0 / (fan_in + fan_out)
    else:
        s = 1.0 / fan_in

    if kernel_size is None:
        size = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            size = (out_features, in_features) + tuple((kernel_size,))
        else:
            size = (out_features, in_features) + (*kernel_size,)

    modulus = np.random.rayleigh(scale=s, size=size)
    phase = np.random.uniform(-np.pi, np.pi, size)
    weight_real = modulus * np.cos(phase)
    weight_imag = modulus * np.sin(phase)

    return (weight_real, weight_imag)


def affect_init(real_weight, imag_weight, init_func, criterion):
    """ Applies the weight initialization function given to the parameters.

    Arguments
    ---------
    real_weight: torch.Parameters, (nb_complex_in, nb_complex_out)
    imag_weight: torch.Parameters, (nb_complex_in, nb_complex_out)
    init_func: function, (unitary_init, complex_init)
    criterion: str, (glorot, he)
    """
    a, b = init_func(real_weight.size(0), real_weight.size(1), None, criterion)
    a, b = torch.from_numpy(a), torch.from_numpy(b)
    real_weight.data = a.type_as(real_weight.data)
    imag_weight.data = b.type_as(imag_weight.data)


def affect_conv_init(
    real_weight, imag_weight, kernel_size, init_func, criterion
):
    """ Applies the weight initialization function given to the parameters.
    This is specificaly written for convolutional layers.

    Arguments
    ---------
    real_weight: torch.Parameters, (nb_complex_out, nb_complex_in, *kernel)
    imag_weight: torch.Parameters, (nb_complex_out, nb_complex_in, *kernel)
    kernel_size: int
    init_func: function, (unitary_init, complex_init)
    criterion: str, (glorot, he)
    """
    in_channels = real_weight.size(1)
    out_channels = real_weight.size(0)
    a, b = init_func(
        in_channels, out_channels, kernel_size=kernel_size, criterion=criterion,
    )
    a, b = torch.from_numpy(a), torch.from_numpy(b)
    real_weight.data = a.type_as(real_weight.data)
    imag_weight.data = b.type_as(imag_weight.data)


def get_kernel_and_weight_shape(conv1d, in_channels, out_channels, kernel_size):
    """ Returns the kernel size and weight shape for convolutional layers.

    Arguments
    ---------
    conv1d: bool
    in_channels: int
    out_channels: int
    kernel_size: int
    """
    if conv1d:
        ks = kernel_size
        w_shape = (out_channels, in_channels) + tuple((ks,))
    else:  # in case it is 2d
        ks = (kernel_size[0], kernel_size[1])
        w_shape = (out_channels, in_channels) + (*ks,)
    return ks, w_shape


# The following mean function using a list of reduced axes is taken from:
# https://discuss.pytorch.org/t/sum-mul-over-multiple-axes/1882/8
def multi_mean(input, axes, keepdim=False):
    """
    Performs `torch.mean` over multiple dimensions of `input`.
    """
    axes = sorted(axes)
    m = input
    for axis in reversed(axes):
        m = m.mean(axis, keepdim)
    return m


def complex_concat(input, input_type="linear", channels_axis=1):
    """
    Applies complex concatenation on the channel axis.

    Arguments
    ---------
    input : torch.Tensor (batch, time, channel)
    input_type: str, (linear, convolution)
    channels_axis : int, index of the channel axis.
    """
    real_part = [get_real(x, input_type, channels_axis) for x in input]
    imag_part = [get_imag(x, input_type, channels_axis) for x in input]

    cat_real = torch.cat(real_part, dim=channels_axis)
    cat_imag = torch.cat(imag_part, dim=channels_axis)

    return torch.cat([cat_real, cat_imag], dim=channels_axis)
