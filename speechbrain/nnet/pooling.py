"""Library implementing pooling.

Author
    Titouan Parcollet 2020  Mirco Ravanelli 2020
"""

import torch
import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


class Pooling1d(nn.Module):
    """This function implements 1d pooling of the input tensor

    Arguments
    ---------
    pool_type : str
        It is the type of pooling function to use ('avg','max')
    pool_axis : int
        Axis where pooling is applied
    kernel_size : int
        It is the kernel size that defines the pooling dimension.
        For instance, kernel size=3 applies a 1D Pooling with a size=3.
    stride : int
        It is the stride size.
    padding : int
        It is the number of padding elements to apply.
    dilation : int
        Controls the dilation factor of pooling.
    ceil_mode : int
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
        pool_axis=1,
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
        self.combine_batch_time = False

        if stride is None:
            self.stride = kernel_size
        else:
            self.stride = stride

        if self.pool_type == "avg":

            self.pool_layer = torch.nn.AvgPool1d(
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                ceil_mode=self.ceil_mode,
            )

        else:
            self.pool_layer = torch.nn.MaxPool1d(
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                ceil_mode=self.ceil_mode,
            )

    def forward(self, x, init_params=False):

        # Put the pooling axes as the last dimension for torch.nn.pool
        # If the input tensor is 4 dimensional, combine the first and second
        # axes together to respect the input shape of torch.nn.pool1d
        x = x.transpose(-1, self.pool_axis)
        new_shape = x.shape
        if len(x.shape) == 4:
            x = x.reshape(new_shape[0] * new_shape[1], new_shape[2], -1)
            self.combine_batch_time = True

        # Apply pooling
        x = self.pool_layer(x)

        # Recover input shape
        if self.combine_batch_time:
            x = x.reshape(new_shape[0], new_shape[1], new_shape[2], -1)
        x = x.transpose(-1, self.pool_axis)

        return x


class Pooling2d(nn.Module):
    """This function implements 2d pooling of the input tensor

    Arguments
    ---------
    pool_type : str
        It is the type of pooling function to use ('avg','max')
    pool_axis : tuple
        It is a list containing the axis that will be considered
        during pooling.
    kernel_size : int
        It is the kernel size that defines the pooling dimension.
        For instance, kernel size=3,3 performs a 2D Pooling with a 3x3 kernel.
    stride : int
        It is the stride size.
    padding : int
        It is the numbe of padding elements to apply.
    dilation : int
        Controls the dilation factor of pooling.
    ceil_mode : int
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

    def forward(self, x, init_params=False):

        # Add extra two dimension at the last two, and then swap the pool_axis to them
        # Example: pool_axis=[1,2]
        # [a,b,c,d] => [a,b,c,d,1,1]
        # [a,b,c,d,1,1] => [a,1,c,d,b,1]
        # [a,1,c,d,b,1] => [a,1,1,d,b,c]
        # [a,1,1,d,b,c] => [a,d,b,c]
        x = (
            x.reshape(*x.shape, 1, 1)
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
