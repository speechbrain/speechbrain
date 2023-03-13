"""Library implementing pooling.

Authors
 * Titouan Parcollet 2020
 * Mirco Ravanelli 2020
 * Nauman Dawalatabad 2020
 * Jianyuan Zhong 2020
 * Sarthak Yadav 2022
"""

import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Pooling1d(nn.Module):
    """This function implements 1d pooling of the input tensor.

    Arguments
    ---------
    pool_type : str
        It is the type of pooling function to use ('avg','max').
    kernel_size : int
        It is the kernel size that defines the pooling dimension.
        For instance, kernel size=3 applies a 1D Pooling with a size=3.
    input_dims : int
        The count of dimensions expected in the input.
    pool_axis : int
        The axis where the pooling is applied.
    stride : int
        It is the stride size.
    padding : int
        It is the number of padding elements to apply.
    dilation : int
        Controls the dilation factor of pooling.
    ceil_mode : bool
        When True, will use ceil instead of floor to compute the output shape.

    Example
    -------
    >>> pool = Pooling1d('max',3)
    >>> inputs = torch.rand(10, 12, 40)
    >>> output=pool(inputs)
    >>> output.shape
    torch.Size([10, 4, 40])
    """

    def __init__(
        self,
        pool_type,
        kernel_size,
        input_dims=3,
        pool_axis=1,
        ceil_mode=False,
        padding=0,
        dilation=1,
        stride=None,
    ):
        super().__init__()
        self.pool_axis = pool_axis

        if stride is None:
            stride = kernel_size

        if pool_type == "avg":
            if input_dims == 3:
                self.pool_layer = torch.nn.AvgPool1d(
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    ceil_mode=ceil_mode,
                )
            elif input_dims == 4:
                self.pool_layer = torch.nn.AvgPool2d(
                    (1, kernel_size),
                    stride=(1, stride),
                    padding=(0, padding),
                    ceil_mode=ceil_mode,
                )
            else:
                raise ValueError("input_dims must be 3 or 4")

        elif pool_type == "max":
            if input_dims == 3:
                self.pool_layer = torch.nn.MaxPool1d(
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    ceil_mode=ceil_mode,
                )
            elif input_dims == 4:
                self.pool_layer = torch.nn.MaxPool2d(
                    (1, kernel_size),
                    stride=(1, stride),
                    padding=(0, padding),
                    dilation=(1, dilation),
                    ceil_mode=ceil_mode,
                )
            else:
                raise ValueError("input_dims must be 3 or 4")

        else:
            raise ValueError("pool_type must be 'avg' or 'max'")

    def forward(self, x):
        """Performs 1d pooling to the input tensor.

        Arguments
        ---------
        x : torch.Tensor
            It represents a tensor for a mini-batch.
        """
        # Put the pooling axes as the last dimension for torch.nn.pool
        x = x.transpose(-1, self.pool_axis)

        # Apply pooling
        x = self.pool_layer(x)

        # Recover input shape
        x = x.transpose(-1, self.pool_axis)

        return x


class Pooling2d(nn.Module):
    """This function implements 2d pooling of the input tensor.

    Arguments
    ---------
    pool_type : str
        It is the type of pooling function to use ('avg','max').
    pool_axis : tuple
        It is a list containing the axis that will be considered
        during pooling.
    kernel_size : int
        It is the kernel size that defines the pooling dimension.
        For instance, kernel size=3,3 performs a 2D Pooling with a 3x3 kernel.
    stride : int
        It is the stride size.
    padding : int
        It is the number of padding elements to apply.
    dilation : int
        Controls the dilation factor of pooling.
    ceil_mode : bool
        When True, will use ceil instead of floor to compute the output shape.

    Example
    -------
    >>> pool = Pooling2d('max',(5,3))
    >>> inputs = torch.rand(10, 15, 12)
    >>> output=pool(inputs)
    >>> output.shape
    torch.Size([10, 3, 4])
    """

    def __init__(
        self,
        pool_type,
        kernel_size,
        pool_axis=(1, 2),
        ceil_mode=False,
        padding=0,
        dilation=1,
        stride=None,
    ):
        super().__init__()
        self.pool_type = pool_type
        self.kernel_size = kernel_size
        self.pool_axis = pool_axis
        self.ceil_mode = ceil_mode
        self.padding = padding
        self.dilation = dilation

        if stride is None:
            self.stride = kernel_size
        else:
            self.stride = stride

        if self.pool_type == "avg":
            self.pool_layer = torch.nn.AvgPool2d(
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                ceil_mode=self.ceil_mode,
            )
        else:
            self.pool_layer = torch.nn.MaxPool2d(
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                ceil_mode=self.ceil_mode,
            )

    def forward(self, x):
        """Performs 2d pooling to the input tensor.

        Arguments
        ---------
        x : torch.Tensor
            It represents a tensor for a mini-batch.
        """
        # Add extra two dimension at the last two, and then swap the pool_axis to them
        # Example: pool_axis=[1,2]
        # [a,b,c,d] => [a,b,c,d,1,1]
        # [a,b,c,d,1,1] => [a,1,c,d,b,1]
        # [a,1,c,d,b,1] => [a,1,1,d,b,c]
        # [a,1,1,d,b,c] => [a,d,b,c]
        x = (
            x.unsqueeze(-1)
            .unsqueeze(-1)
            .transpose(-2, self.pool_axis[0])
            .transpose(-1, self.pool_axis[1])
            .squeeze(self.pool_axis[1])
            .squeeze(self.pool_axis[0])
        )

        # Apply pooling
        x = self.pool_layer(x)

        # Swap back the pool_axis from the last two dimension
        # Example: pool_axis=[1,2]
        # [a,d,b,c] => [a,1,d,b,c]
        # [a,1,d,b,c] => [a,1,1,d,b,c]
        # [a,1,1,d,b,c] => [a,b,1,d,1,c]
        # [a,b,1,d,1,c] => [a,b,c,d,1,1]
        # [a,b,c,d,1,1] => [a,b,c,d]
        x = (
            x.unsqueeze(self.pool_axis[0])
            .unsqueeze(self.pool_axis[1])
            .transpose(-2, self.pool_axis[0])
            .transpose(-1, self.pool_axis[1])
            .squeeze(-1)
            .squeeze(-1)
        )

        return x


class StatisticsPooling(nn.Module):
    """This class implements a statistic pooling layer.

    It returns the mean and/or std of input tensor.

    Arguments
    ---------
    return_mean : True
         If True, the average pooling will be returned.
    return_std : True
         If True, the standard deviation will be returned.

    Example
    -------
    >>> inp_tensor = torch.rand([5, 100, 50])
    >>> sp_layer = StatisticsPooling()
    >>> out_tensor = sp_layer(inp_tensor)
    >>> out_tensor.shape
    torch.Size([5, 1, 100])
    """

    def __init__(self, return_mean=True, return_std=True):
        super().__init__()

        # Small value for GaussNoise
        self.eps = 1e-5
        self.return_mean = return_mean
        self.return_std = return_std
        if not (self.return_mean or self.return_std):
            raise ValueError(
                "both of statistics are equal to False \n"
                "consider enabling mean and/or std statistic pooling"
            )

    def forward(self, x, lengths=None):
        """Calculates mean and std for a batch (input tensor).

        Arguments
        ---------
        x : torch.Tensor
            It represents a tensor for a mini-batch.
        """
        if lengths is None:
            if self.return_mean:
                mean = x.mean(dim=1)
            if self.return_std:
                std = x.std(dim=1)
        else:
            mean = []
            std = []
            for snt_id in range(x.shape[0]):
                # Avoiding padded time steps
                actual_size = int(torch.round(lengths[snt_id] * x.shape[1]))

                # computing statistics
                if self.return_mean:
                    mean.append(
                        torch.mean(x[snt_id, 0:actual_size, ...], dim=0)
                    )
                if self.return_std:
                    std.append(torch.std(x[snt_id, 0:actual_size, ...], dim=0))
            if self.return_mean:
                mean = torch.stack(mean)
            if self.return_std:
                std = torch.stack(std)

        if self.return_mean:
            gnoise = self._get_gauss_noise(mean.size(), device=mean.device)
            gnoise = gnoise
            mean += gnoise
        if self.return_std:
            std = std + self.eps

        # Append mean and std of the batch
        if self.return_mean and self.return_std:
            pooled_stats = torch.cat((mean, std), dim=1)
            pooled_stats = pooled_stats.unsqueeze(1)
        elif self.return_mean:
            pooled_stats = mean.unsqueeze(1)
        elif self.return_std:
            pooled_stats = std.unsqueeze(1)

        return pooled_stats

    def _get_gauss_noise(self, shape_of_tensor, device="cpu"):
        """Returns a tensor of epsilon Gaussian noise.

        Arguments
        ---------
        shape_of_tensor : tensor
            It represents the size of tensor for generating Gaussian noise.
        """
        gnoise = torch.randn(shape_of_tensor, device=device)
        gnoise -= torch.min(gnoise)
        gnoise /= torch.max(gnoise)
        gnoise = self.eps * ((1 - 9) * gnoise + 9)

        return gnoise


class AdaptivePool(nn.Module):
    """This class implements the adaptive average pooling.

    Arguments
    ---------
    delations : output_size
        The size of the output.

    Example
    -------
    >>> pool = AdaptivePool(1)
    >>> inp = torch.randn([8, 120, 40])
    >>> output = pool(inp)
    >>> output.shape
    torch.Size([8, 1, 40])
    """

    def __init__(self, output_size):
        super().__init__()

        condition = (
            isinstance(output_size, int)
            or isinstance(output_size, tuple)
            or isinstance(output_size, list)
        )
        assert condition, "output size must be int, list or tuple"

        if isinstance(output_size, tuple) or isinstance(output_size, list):
            assert (
                len(output_size) == 2
            ), "len of output size must not be greater than 2"

        if isinstance(output_size, int):
            self.pool = nn.AdaptiveAvgPool1d(output_size)
        else:
            self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        """Performs adpative pooling to the input tensor.

        Arguments
        ---------
        x : torch.Tensor
            It represents a tensor for a mini-batch.
        """
        if x.ndim == 3:
            return self.pool(x.permute(0, 2, 1)).permute(0, 2, 1)

        if x.ndim == 4:
            return self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class GaussianLowpassPooling(nn.Module):
    """
    This class implements a learnable Gaussian lowpass pooling from

    Neil Zeghidour, Olivier Teboul, F{\'e}lix de Chaumont Quitry & Marco Tagliasacchi, "LEAF: A LEARNABLE FRONTEND
    FOR AUDIO CLASSIFICATION", in Proc. of ICLR 2021 (https://arxiv.org/abs/2101.08596)

    Arguments
    ---------
    in_channels : int
        The number of input channels.
    kernel_size: int
        Kernel size of the gaussian lowpass filters.
    stride : int
        Stride factor of the convolutional filters. When the stride factor > 1,
        a decimation in time is performed.
    padding : str
        (same, valid). If "valid", no padding is performed.
        If "same" and stride is 1, output shape is the same as the input shape.
    padding_mode : str
        This flag specifies the type of padding. See torch.nn documentation
        for more information.
    bias : bool
        If True, the additive bias b is adopted.
    skip_transpose : bool
        If False, uses batch x time x channel convention of speechbrain.
        If True, uses batch x channel x time convention.

    Example
    -------
    >>> inp_tensor = torch.rand([10, 8000, 40])
    >>> low_pass_pooling = GaussianLowpassPooling(
    ...     40, kernel_size=401, stride=160,
    ... )
    >>> # parameters corresponding to a window of 25 ms and stride 10 ms at 16000 kHz
    >>> out_tensor = low_pass_pooling(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 50, 40])
    """

    def __init__(
        self,
        in_channels,
        kernel_size,
        stride=1,
        initialization_constant=0.4,
        padding="same",
        padding_mode="constant",
        bias=True,
        skip_transpose=False,
    ):
        super(GaussianLowpassPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.in_channels = in_channels
        self.skip_transpose = skip_transpose
        self.weights = nn.Parameter(
            torch.ones((1, 1, in_channels, 1)) * initialization_constant
        )

        if bias:
            self._bias = torch.nn.Parameter(torch.ones(in_channels,))
        else:
            self._bias = None

    def _get_impulse_responses(self, sigma):
        filter_size = self.kernel_size
        sigma = torch.clamp(sigma, min=(2.0 / filter_size), max=0.5)
        t = torch.arange(0, filter_size, dtype=sigma.dtype, device=sigma.device)
        t = torch.reshape(t, (1, filter_size, 1, 1))
        numerator = t - 0.5 * (filter_size - 1)
        denominator = sigma * 0.5 * (filter_size - 1)
        return torch.exp(-0.5 * (numerator / denominator) ** 2)

    def forward(self, x):
        """Performs GaussianLowpass Pooling.

        Arguments
        ---------
        x : torch.Tensor
            3D tensor in input [batch,time,channels].
        """
        if not self.skip_transpose:
            x = x.transpose(1, -1)

        kernel = self._get_impulse_responses(self.weights)
        kernel = kernel.reshape(-1, self.kernel_size, self.in_channels)
        kernel = kernel.permute(2, 0, 1)

        if self.padding == "same":
            x = self._manage_padding(x, self.kernel_size)
        elif self.padding == "valid":
            pass
        else:
            raise ValueError(
                "Padding must be 'same' or 'valid'. Got " + self.padding
            )
        outputs = F.conv1d(
            x,
            kernel,
            bias=self._bias,
            stride=self.stride,
            padding=0,
            groups=self.in_channels,
        )
        if not self.skip_transpose:
            outputs = outputs.transpose(1, -1)
        return outputs

    def _manage_padding(self, x, kernel_size):
        # this is the logic that gives correct shape that complies
        # with the original implementation at https://github.com/google-research/leaf-audio

        def get_padding_value(kernel_size):
            """Get number of elements to pad."""
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
