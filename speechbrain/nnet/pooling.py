"""Library implementing pooling.

Author
    Titouan Parcollet 2020
"""

import torch
import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


class Pooling(nn.Module):
    """This function implements pooling of the input tensor

    Arguments
    ---------
    pool_type : str
        It is the type of pooling function to use ('avg','max')
    pool_axis : list
        It is a list containing the axis that will be considered
        during pooling. It must match the dimensionality of the pooling.
        If the pooling is 2D, then a list of 2 indices is expected.
    kernel_size : list
        It is the kernel size. Note that it also defines the pooling dimension.
        For instance kernel size=3 applies a 1D Pooling with a size=3,
        while kernel size=3,3 performs a 2D Pooling with a 3x3 kernel.
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
    >>> pool = Pooling('max',3)
    >>> inputs = torch.rand(10, 50, 40)
    >>> pool.init_params(inputs)
    >>> output=pool(inputs)
    >>> output.shape
    torch.Size([10, 50, 13])
    """

    def __init__(
        self,
        pool_type,
        kernel_size,
        pool_axis=2,
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
        self.pool1d = False
        self.pool2d = False
        self.combine_batch_time = False

        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size,)
        if isinstance(self.pool_axis, int):
            self.pool_axis = (self.pool_axis,)

        if stride is None:
            self.stride = kernel_size
        else:
            self.stride = stride

    def init_params(self, first_input):
        """
        Arguments
        ---------
        first_input : tensor
            A dummy input of the right shape for initializing parameters.
        """

        # Check parameters
        self.check_params(first_input)

        # Pooling initialization
        if self.pool_type == "avg":

            # Check Pooling dimension
            if self.pool1d:
                self.pool_layer = torch.nn.AvgPool1d(
                    self.kernel_size[0],
                    stride=self.stride,
                    padding=self.padding,
                    ceil_mode=self.ceil_mode,
                )

            else:
                self.pool_layer = torch.nn.AvgPool2d(
                    self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    ceil_mode=self.ceil_mode,
                )
        else:
            if self.pool1d:
                self.pool_layer = torch.nn.MaxPool1d(
                    self.kernel_size[0],
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

    def check_params(self, first_input):

        # Check that enough pooling axes are specified
        if len(self.kernel_size) == 1:
            self.pool1d = True

            # In the case of a 4 dimensional input vector, we need to
            # combine the batch and time dimension together because
            # torch.nn.pool1d only accepts 3D vectors as inputs.
            if len(first_input.shape) > 3:
                self.combine_batch_time = True

            if len(self.pool_axis) != 1:
                err_msg = (
                    "pool_axes must corresponds to the pooling dimension. "
                    "The pooling dimension is 1 and %s axes are specified."
                    % (str(len(self.pool_axis)))
                )
                raise ValueError(err_msg)

            if self.pool_axis[0] >= len(first_input.shape):
                err_msg = (
                    "pool_axes is greater than the number of dimensions. "
                    "The tensor dimension is %s and the specified pooling "
                    "axis is %s."
                    % (str(len(first_input.shape)), str(self.pool_axis))
                )
                raise ValueError(err_msg)

        if len(self.kernel_size) == 2:
            self.pool2d = True

            if len(self.pool_axis) != 2:
                err_msg = (
                    "pool_axes must corresponds to the pooling dimension. "
                    "The pooling dimension is 2 and %s axes are specified."
                    % (str(len(self.pool_axis)))
                )
                raise ValueError(err_msg)

            dims = len(first_input.shape)
            if self.pool_axis[0] >= dims or self.pool_axis[1] >= dims:
                err_msg = (
                    "pool_axes is greater than the number of dimensions. "
                    "The tensor dimension is %s and the specified pooling "
                    "axis are %s."
                    % (str(len(first_input.shape)), str(self.pool_axis))
                )
                raise ValueError(err_msg)

    def forward(self, x, init_params=False):
        if init_params:
            self.init_params(x)

        # Put the pooling axes as the last dimension for torch.nn.pool
        # If the input tensor is 4 dimensional, combine the first and second
        # axes together to respect the input shape of torch.nn.pool1d
        if self.pool1d:
            x = x.transpose(-1, self.pool_axis[0])
            new_shape = x.shape
            if self.combine_batch_time:
                x = x.reshape(new_shape[0] * new_shape[1], new_shape[2], -1)

        if self.pool2d:
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

        # Recover input shape
        if self.pool1d:
            if self.combine_batch_time:
                x = x.reshape(new_shape[0], new_shape[1], new_shape[2], -1)
            x = x.transpose(-1, self.pool_axis[0])

        if self.pool2d:
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
