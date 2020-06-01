"""A multi-time scale atrous pooling (MATConv) layers to perform pooling on the input feature.

   Authors: Jianyuan Zhong 2020
"""

import torch  # noqa: F401
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.nnet.containers import Sequential
from speechbrain.nnet.CNN import Conv
from speechbrain.nnet.normalization import Normalize
from speechbrain.nnet.dropout import Dropout


class MATConvModule(Sequential):
    """This is the module for astrous (dilated convolution) on time or time-frequency domain

    Arguments
    ---------
    overrides : mapping
        Additional parameters overriding the MATConv block parameters.

    MATConv Block Parameters
    ------------------------
        ..include:: MATConv_block1.yaml

    Example
    -------
    >>> import torch
    >>> model = MATConvModule(128, (1,), (1,))
    >>> input = torch.rand([10, 120, 60])
    >>> output = model(input, init_params=True)
    >>> len(output.shape)
    3
    """

    def __init__(
        self, out_channels, kernel_size, dilation=(1, 1),
    ):
        self.dilation = dilation
        self.out_channels = out_channels
        layers = (
            Conv(
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=dilation,
                bias=False,
            ),
            Normalize("batchnorm"),
            nn.LeakyReLU(),
        )

        super().__init__(*layers)

    def forward(self, x, init_params=False):
        """
        Arguments
        ---------
        x : tensor
            the input tensor to run through the network.
        """
        if not self._check_input_size(x):
            return x
        if init_params:
            #  init weights with other params
            output = super(MATConvModule, self).forward(x, init_params)
            self._init_weight()
            return output
        else:
            if not hasattr(self.layers[0], "conv"):
                output = super(MATConvModule, self).forward(x, True)
                self._init_weight()
                return output

            return super(MATConvModule, self).forward(x, init_params)

    def _check_input_size(self, x):
        if isinstance(self.dilation, int):
            return x.shape[1] >= self.dilation
        else:
            for i, dil in enumerate(self.dilation):
                if dil >= x.shape[1 + i]:
                    return False
            return True

    def _init_weight(self):
        for m in self.layers:
            if isinstance(m, Conv):
                torch.nn.init.kaiming_normal_(m.conv.weight)


class MATConvPool(nn.Module):
    """This model is the MATConv Pooling.
    It performs multi-resolutions Pooling on the input vectors
    via first convolution layers with differient dilation rates and a
    global average pooling, Then project the outputs of above
    layers to a single vector.

    Arguments
    ---------
    delations : list
        Delation delation rates to be used in MATConv modules

    Examples
    --------
    >>> import torch
    >>> model = MATConvPool(128, (1,))
    >>> input = torch.rand([10, 120, 60])
    >>> output = model(input, init_params=True)
    >>> output.shape
    torch.Size([10, 120, 128])
    """

    def __init__(
        self,
        out_channels,
        stride,
        matconv_channels=256,
        pool_axis=(1, 2),
        dilations=[1, 6, 12, 18],
        droupout=0.15,
    ):
        super().__init__()

        if isinstance(stride, int):
            stride = (stride,)
        if isinstance(pool_axis, int):
            pool_axis = (pool_axis,)

        self.pool_axis = pool_axis
        self.matConv1d = self._check_dimentinality(stride)
        self.matConvs = nn.ModuleList()

        for (i, dilation) in enumerate(dilations):
            if i == 0:
                self.matConvs.append(
                    MATConvModule(
                        out_channels=matconv_channels,
                        kernel_size=self._make_params(1),
                        dilation=self._make_params(dilation),
                    )
                )
            else:
                self.matConvs.append(
                    MATConvModule(
                        out_channels=matconv_channels,
                        kernel_size=self._make_params(3),
                        dilation=self._make_params(dilation),
                    )
                )

        self.global_avg_pool = Sequential(
            AdaptivePool(self._make_params(1)),
            Conv(
                out_channels=out_channels,
                kernel_size=self._make_params(1),
                bias=False,
            ),
            Normalize("batchnorm"),
            nn.LeakyReLU(),
        )

        self.conv = Sequential(
            Conv(
                out_channels=out_channels,
                kernel_size=self._make_params(1),
                stride=stride,
                bias=False,
            ),
            Normalize("batchnorm"),
            nn.LeakyReLU(),
            Dropout(droupout),
        )

    def forward(self, x, init_params=True):
        """
        Arguments
        ---------
        x : tensor
            the input tensor to run through the network.
        """

        if init_params:
            self._check_params(x)

        if self.combine_batch_time:
            x = x.transpose(-1, self.pool_axis[0])
            new_shape = x.shape
            x = x.reshape(
                new_shape[0] * new_shape[1], new_shape[2], -1
            ).transpose(1, -1)

        xs = [conv(x, init_params) for conv in self.matConvs]
        x_avg = self.global_avg_pool(x, init_params)

        if self.matConv1d:
            x_avg = F.interpolate(
                x_avg.permute(0, 2, 1),
                xs[-1].permute(0, 2, 1).size()[2:],
                mode="linear",
                align_corners=True,
            ).permute(0, 2, 1)
        else:
            x_avg = F.interpolate(
                x_avg.permute(0, 3, 1, 2),
                xs[-1].permute(0, 3, 1, 2).size()[2:],
                mode="bilinear",
                align_corners=True,
            ).permute(0, 2, 3, 1)

        x = torch.cat((*xs, x_avg), dim=-1)
        x = self.conv(x, init_params)

        if self.combine_batch_time:
            x = x.permute(0, 2, 1)
            x = x.reshape(new_shape[0], new_shape[1], x.shape[1], -1)
            x = x.transpose(-1, self.pool_axis[0])

        if init_params:
            self._init_weight()

        # print(x.shape, 'MATConv output size')

        return x

    def _check_params(self, x):
        self.combine_batch_time = False
        if self.matConv1d:
            if len(x.shape) == 4:
                self.combine_batch_time = True

            if len(self.pool_axis) != 1:
                err_msg = (
                    "pool_axes must corresponds to the pooling dimension. "
                    "The pooling dimension is 1 and %s axes are specified."
                    % (str(len(self.pool_axis)))
                )
                raise ValueError(err_msg)

            if self.pool_axis[0] >= len(x.shape):
                err_msg = (
                    "pool_axes is greater than the number of dimensions. "
                    "The tensor dimension is %s and the specified pooling "
                    "axis is %s." % (str(len(x.shape)), str(self.pool_axis))
                )
                raise ValueError(err_msg)

        else:
            if len(self.pool_axis) != 2:
                err_msg = (
                    "pool_axes must corresponds to the pooling dimension. "
                    "The pooling dimension is 2 and %s axes are specified."
                    % (str(len(self.pool_axis)))
                )
                raise ValueError(err_msg)

            dims = len(x.shape)
            if self.pool_axis[0] >= dims or self.pool_axis[1] >= dims:
                err_msg = (
                    "pool_axes is greater than the number of dimensions. "
                    "The tensor dimension is %s and the specified pooling "
                    "axis are %s." % (str(len(x.shape)), str(self.pool_axis))
                )
                raise ValueError(err_msg)

    def _check_dimentinality(self, stride):
        return len(stride) == 1

    def _make_params(self, params, make_norm_layer=False):
        if self.matConv1d:
            return (params,)
        else:
            return (params, params)

    def _init_weight(self):
        for m in self.global_avg_pool.layers:
            if isinstance(m, Conv):
                torch.nn.init.kaiming_normal_(m.conv.weight)

        for m in self.conv.layers:
            if isinstance(m, Conv):
                torch.nn.init.kaiming_normal_(m.conv.weight)


class AdaptivePool(nn.Module):
    """This model is the adaptive average pooling

    Arguments
    ---------
    delations : output_size
        the size of the output

    example
    -------
    """

    def __init__(self, output_size):
        super().__init__()

        if len(output_size) == 1:
            output_size = output_size[0]

        if isinstance(output_size, int):
            self.pool = nn.AdaptiveAvgPool1d(output_size)
        else:
            self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        if len(x.shape) == 3:
            return self.pool(x.permute(0, 2, 1)).permute(0, 2, 1)

        if len(x.shape) == 4:
            return self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
