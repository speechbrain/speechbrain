import torch


class Pooling(torch.nn.Module):
    """This function implements pooling of the input tensor.

    Args:
        pool_type: Function to use for pooling, one of [max, avg].
        kernel_size: A list of sizes used for the convolutional kernel,
            one integer for 1D pooling, two integers for 2D pooling.
        pool_axis: A list of the axes that are considered during pooling,
            one axis for 1D pooling, two axes for 2D pooling.
        stride: The size of the stride.
        padding: The size of padding to apply during pooling.
        dilation: The size of dilation to use during pooling.
        ceil_mode: When True, use ceil instead of floor to compute the shape
            of the output.

     Example:
        >>> import torch
        >>> inp_tensor = torch.rand([1,4,1,4])
        >>> pool = Pooling(pool_type='max', kernel_size=[2,2])
        >>> out_tensor = pool(inp_tensor)
        >>> out_tensor.shape

    Author:
        Titouan Percollet 2020
    """

    def __init__(
        self,
        pool_type,
        kernel_size,
        pool_axis=1,
        ceil_mode=False,
        padding=0,
        stride=1,
    ):
        super().__init__()
        self.pool_type = pool_type
        self.kernel_size = kernel_size
        self.pool_axis = pool_axis
        self.ceil_mode = ceil_mode
        self.padding = padding
        self.stride = stride

        if not isinstance(kernel_size, list):
            self.kernel_size = [self.kernel_size]
        if not isinstance(pool_axis, list):
            self.pool_axis = [self.pool_axis]

        # Option for pooling
        self.pool1d = False
        self.pool2d = False
        self.combine_batch_time = False

        # Check that enough pooling axes are specified
        def hook(self, first_input):
            if len(self.kernel_size) == 1:
                self.pool1d = True

                # In the case of a 4 dimensional input vector, we need to
                # combine the batch and time dimension together because
                # torch.nn.pool1d only accepts 3D vectors as inputs.
                if len(first_input[0].shape) > 3:
                    self.combine_batch_time = True

                if len(self.pool_axis) != 1:
                    err_msg = (
                        'pool_axes must corresponds to the pooling dimension. '
                        "The pooling dimension is 1 and %s axes are specified."
                        % (str(len(self.pool_axis)))
                    )
                    raise ValueError(err_msg)

                if self.pool_axis[0] >= len(first_input[0].shape):
                    err_msg = (
                        'pool_axes is greater than the number of dimensions. '
                        "The tensor dimension is %s and the specified pooling "
                        "axis is %s."
                        % (str(len(first_input[0].shape)), str(self.pool_axis))
                    )
                    raise ValueError(err_msg)

            if len(self.kernel_size) == 2:
                self.pool2d = True

                if len(self.pool_axis) != 2:
                    err_msg = (
                        'pool_axes must corresponds to the pooling dimension. '
                        "The pooling dimension is 2 and %s axes are specified."
                        % (str(len(self.pool_axis)))
                    )
                    raise ValueError(err_msg)

                if self.pool_axis[0] >= len(first_input[0].shape) or \
                   self.pool_axis[1] >= len(first_input[0].shape):
                    err_msg = (
                        'pool_axes is greater than the number of dimensions. '
                        "The tensor dimension is %s and the specified pooling "
                        "axis are %s."
                        % (str(len(first_input[0].shape)), str(self.pool_axis))
                    )
                    raise ValueError(err_msg)

            # Pooling initialization
            if self.pool_type == "avg":

                # Check Pooling dimension
                if self.pool1d:
                    self.pool_layer = torch.nn.AvgPool1d(
                        self.kernel_size[0],
                        stride=self.stride,
                        padding=self.padding,
                        ceil_mode=self.ceil_mode)

                else:
                    self.pool_layer = torch.nn.AvgPool2d(
                        self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                        ceil_mode=self.ceil_mode)
            else:
                if self.pool1d:
                    self.pool_layer = torch.nn.MaxPool1d(
                        self.kernel_size[0],
                        stride=self.stride,
                        padding=self.padding,
                        ceil_mode=self.ceil_mode)

                else:
                    self.pool_layer = torch.nn.MaxPool2d(
                        self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                        ceil_mode=self.ceil_mode)
            self.hook.remove()

        self.hook = self.register_forward_pre_hook(hook)

    def forward(self, x):

        # Reading input list
        or_shape = x.shape

        # Put the pooling axes as the last dimension for torch.nn.pool
        # If the input tensor is 4 dimensional, combine the first and second
        # axes together to respect the input shape of torch.nn.pool1d
        if self.pool1d:
            x = x.transpose(len(or_shape)-1, self.pool_axis[0])
            new_shape = x.shape
            if self.combine_batch_time:
                x = x.reshape(new_shape[0]*new_shape[1], new_shape[2], -1)

        if self.pool2d:
            x = x.transpose(len(or_shape)-2, self.pool_axis[0]).transpose(
                            len(or_shape)-1, self.pool_axis[1])

        # Apply pooling
        x = self.pool_layer(x)

        # Recover input shape
        if self.pool1d:
            if self.combine_batch_time:
                x = x.reshape(new_shape[0], new_shape[1], new_shape[2], -1)
            x = x.transpose(len(or_shape)-1, self.pool_axis[0])

        if self.pool2d:
            x = x.transpose(len(or_shape)-2, self.pool_axis[0]).transpose(
                            len(or_shape)-1, self.pool_axis[1])

        return x
