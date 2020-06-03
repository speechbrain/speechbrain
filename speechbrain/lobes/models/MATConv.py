"""A multi-time scale atrous pooling (MATConv) layers to perform pooling on the input feature.

   Authors: Jianyuan Zhong 2020
"""

import torch  # noqa: F401
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.nnet.containers import Sequential
from speechbrain.nnet.CNN import Conv1d, Conv2d
from speechbrain.nnet.normalization import BatchNorm1d, BatchNorm2d
from speechbrain.nnet.dropout import Dropout2d


class MATConvModule1d(Sequential):
    """This is the module for astrous (dilated convolution) on time or time-frequency domain

    Arguments
    ---------
    out_channels : int
        number of channels for the output feature
    kernel_size : int
        the kernel size of the dilated convolution layer
    activation : torch class
        activation function is going to be used in this block
    norm : torch class
        regulariztion method for this nueral block
    dilation : int
        size of dilation

    Example
    -------
    >>> import torch
    >>> model = MATConvModule1d(128, (1,), (1,))
    >>> input = torch.rand([10, 120, 60])
    >>> output = model(input, init_params=True)
    >>> len(output.shape)
    3
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        activation=nn.LeakyReLU,
        norm=BatchNorm1d,
        dilation=1,
    ):
        self.dilation = dilation
        self.out_channels = out_channels
        layers = (
            Conv1d(
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=dilation,
                bias=False,
            ),
            norm(),
            activation(),
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
            output = super(MATConvModule1d, self).forward(x, init_params)
            self._init_weight()
            return output
        else:
            if not hasattr(self.layers[0], "conv"):
                output = super(MATConvModule1d, self).forward(x, True)
                self._init_weight()
                return output

            return super(MATConvModule1d, self).forward(x, init_params)

    def _check_input_size(self, x):
        if isinstance(self.dilation, int):
            return x.shape[1] >= self.dilation
        else:
            return x.shape[1] >= self.dilation[0]

    def _init_weight(self):
        for m in self.layers:
            if isinstance(m, Conv1d):
                torch.nn.init.kaiming_normal_(m.conv.weight)
            if isinstance(m, BatchNorm1d):
                m.norm.weight.data.fill_(1)
                m.norm.bias.data.zero_()


class MATConvModule2d(Sequential):
    """This is the module for astrous (dilated convolution) on time or time-frequency domain

    Arguments
    ---------
    out_channels : int
        number of channels for the output feature
    kernel_size : int or tuple
        the kernel size of the dilated convolution layer
    activation : torch class
        activation function is going to be used in this block
    norm : torch class
        regulariztion method for this nueral block
    dilation : int
        size of dilation

    MATConv Block Parameters
    ------------------------
        ..include:: MATConv_block1.yaml

    Example
    -------
    >>> import torch
    >>> model = MATConvModule2d(128, (1,1), (1,))
    >>> input = torch.rand([10, 120, 60])
    >>> output = model(input, init_params=True)
    >>> len(output.shape)
    3
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        activation=nn.LeakyReLU,
        norm=BatchNorm2d,
        dilation=(1, 1),
    ):

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.dilation = dilation
        self.out_channels = out_channels
        layers = (
            Conv2d(
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=dilation,
                bias=False,
            ),
            norm(),
            activation(),
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
            output = super(MATConvModule2d, self).forward(x, init_params)
            self._init_weight()
            return output
        else:
            if not hasattr(self.layers[0], "conv"):
                output = super(MATConvModule2d, self).forward(x, True)
                self._init_weight()
                return output

            return super(MATConvModule2d, self).forward(x, init_params)

    def _check_input_size(self, x):
        for i, dil in enumerate(self.dilation):
            if dil >= x.shape[1 + i]:
                return False
        return True

    def _init_weight(self):
        for m in self.layers:
            if isinstance(m, Conv2d):
                torch.nn.init.kaiming_normal_(m.conv.weight)
            if isinstance(m, BatchNorm2d):
                m.norm.weight.data.fill_(1)
                m.norm.bias.data.zero_()


class MATConvPool1d(nn.Module):
    """This model is the MATConv Pooling.
    It performs multi-resolutions Pooling on the input vectors
    via first convolution layers with differient dilation rates and a
    global average pooling, Then project the outputs of above
    layers to a single vector.

    Arguments
    ---------
    out_channels: int
        number of channels for the output feature
    matpool_channels : int
        number of channels for each dilated convolution layers
    stride : int
        convolution stride (recommoned value is one)
    activation : torch class
        activation function is going to be used in this block
    norm : torch class
        regulariztion method for this nueral block
    pool_axis: int or tuple
        the axis of the input tensor that the pooling will be performed on
    delations : list
        Delation delation rates to be used in MATConv modules
    dropuout : int

    Examples
    --------
    >>> import torch
    >>> model = MATConvPool(128, (1,), pool_axis=(1,))
    >>> input = torch.rand([10, 120, 60])
    >>> output = model(input, init_params=True)
    >>> output.shape
    torch.Size([10, 120, 128])
    """

    def __init__(
        self,
        out_channels,
        matpool_channels=None,
        stride=1,
        activation=nn.LeakyReLU,
        norm=BatchNorm1d,
        pool_axis=1,
        dilations=[1, 6, 12, 18],
        droupout=0.15,
    ):
        super().__init__()

        if isinstance(stride, int):
            stride = (stride,)
        if matpool_channels is None:
            matpool_channels = out_channels

        self.pool_axis = pool_axis
        self.matConvs = nn.ModuleList()

        for (i, dilation) in enumerate(dilations):
            if i == 0:
                self.matConvs.append(
                    MATConvModule1d(
                        out_channels=matpool_channels,
                        kernel_size=1,
                        activation=activation,
                        dilation=dilation,
                    )
                )
            else:
                self.matConvs.append(
                    MATConvModule1d(
                        out_channels=matpool_channels,
                        kernel_size=3,
                        activation=activation,
                        dilation=dilation,
                    )
                )

        self.global_avg_pool = Sequential(
            AdaptivePool(1),
            Conv1d(out_channels=out_channels, kernel_size=1, bias=False,),
            norm(),
            activation(),
        )

        self.conv = Sequential(
            Conv1d(
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            norm(),
            activation(),
            Dropout2d(droupout),
        )

    def forward(self, x, init_params=False):
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

        x_avg = F.interpolate(
            x_avg.permute(0, 2, 1),
            xs[-1].permute(0, 2, 1).size()[2:],
            mode="linear",
            align_corners=True,
        ).permute(0, 2, 1)

        x = torch.cat((*xs, x_avg), dim=-1)
        x = self.conv(x, init_params)

        if self.combine_batch_time:
            x = x.permute(0, 2, 1)
            x = x.reshape(new_shape[0], new_shape[1], x.shape[1], -1)
            x = x.transpose(-1, self.pool_axis[0])

        if init_params:
            self._init_weight()

        return x

    def _check_params(self, x):
        self.combine_batch_time = False
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

    def _init_weight(self):
        for m in self.global_avg_pool.layers:
            if isinstance(m, Conv1d):
                torch.nn.init.kaiming_normal_(m.conv.weight)
            if isinstance(m, BatchNorm1d):
                m.norm.weight.data.fill_(1)
                m.norm.bias.data.zero_()

        for m in self.conv.layers:
            if isinstance(m, Conv1d):
                torch.nn.init.kaiming_normal_(m.conv.weight)
            if isinstance(m, BatchNorm1d):
                m.norm.weight.data.fill_(1)
                m.norm.bias.data.zero_()


class MATConvPool2d(nn.Module):
    """This model is the MATConv Pooling.
    It performs multi-resolutions Pooling on the input vectors
    via first convolution layers with differient dilation rates and a
    global average pooling, Then project the outputs of above
    layers to a single vector.

    Arguments
    ---------
    out_channels: int
        number of channels for the output feature
    matpool_channels : int
        number of channels for each dilated convolution layers
    stride : int
        convolution stride (recommoned value is one)
    activation : torch class
        activation function is going to be used in this block
    norm : torch class
        regulariztion method for this nueral block
    pool_axis: int or tuple
        the axis of the input tensor that the pooling will be performed on
    delations : list
        Delation delation rates to be used in MATConv modules
    dropuout : int

    Examples
    --------
    >>> import torch
    >>> model = MATConvPool(128, (1,), pool_axis=(1,))
    >>> input = torch.rand([10, 120, 60])
    >>> output = model(input, init_params=True)
    >>> output.shape
    torch.Size([10, 120, 128])
    """

    def __init__(
        self,
        out_channels,
        activation=nn.LeakyReLU,
        norm=BatchNorm2d,
        stride=(1, 1),
        matpool_channels=256,
        pool_axis=(1, 2),
        dilations=[1, 6, 12, 18],
        droupout=0.15,
    ):
        super().__init__()

        if isinstance(stride, int):
            stride = (stride, stride)

        self.pool_axis = pool_axis
        self.matConvs = nn.ModuleList()

        for (i, dilation) in enumerate(dilations):
            if i == 0:
                self.matConvs.append(
                    MATConvModule2d(
                        out_channels=matpool_channels,
                        kernel_size=(1, 1),
                        activation=activation,
                        dilation=dilation,
                    )
                )
            else:
                self.matConvs.append(
                    MATConvModule2d(
                        out_channels=matpool_channels,
                        kernel_size=(3, 3),
                        activation=activation,
                        dilation=dilation,
                    )
                )

        self.global_avg_pool = Sequential(
            AdaptivePool((1, 1)),
            Conv2d(out_channels=out_channels, kernel_size=(1, 1), bias=False,),
            norm(),
            activation(),
        )

        self.conv = Sequential(
            Conv2d(
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=stride,
                bias=False,
            ),
            norm(),
            activation(),
            torch.nn.Dropout(droupout),
        )

    def forward(self, x, init_params=False):
        """
        Arguments
        ---------
        x : tensor
            the input tensor to run through the network.
        """

        if init_params:
            self._check_params(x)

        xs = [conv(x, init_params) for conv in self.matConvs]
        x_avg = self.global_avg_pool(x, init_params)

        x_avg = F.interpolate(
            x_avg.permute(0, 3, 1, 2),
            xs[-1].permute(0, 3, 1, 2).size()[2:],
            mode="bilinear",
            align_corners=True,
        ).permute(0, 2, 3, 1)

        x = torch.cat((*xs, x_avg), dim=-1)
        x = self.conv(x, init_params)

        if init_params:
            self._init_weight()

        return x

    def _check_params(self, x):
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

    def _init_weight(self):
        for m in self.global_avg_pool.layers:
            if isinstance(m, Conv2d):
                torch.nn.init.kaiming_normal_(m.conv.weight)
            if isinstance(m, BatchNorm2d):
                m.norm.weight.data.fill_(1)
                m.norm.bias.data.zero_()

        for m in self.conv.layers:
            if isinstance(m, Conv2d):
                torch.nn.init.kaiming_normal_(m.conv.weight)
            if isinstance(m, BatchNorm2d):
                m.norm.weight.data.fill_(1)
                m.norm.bias.data.zero_()


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
