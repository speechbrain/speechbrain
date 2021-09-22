"""Library implementing pooling.

Authors
 * Titouan Parcollet 2020
 * Mirco Ravanelli 2020
 * Nauman Dawalatabad 2020
 * Jianyuan Zhong 2020
"""

import torch
import logging
import torch.nn as nn

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
    ceil_mode : int
        When True, will use ceil instead of floor to compute the output shape.

    Example
    -------
    >>> pool = Pooling1d('max',3)
    >>> inputs = torch.rand(10, 12, 40)
    >>> mask = torch.zeros(10, 12, 40).bool()
    >>> output, mask = pool(inputs, mask)
    >>> output.shape
    torch.Size([10, 4, 40])
    >>> mask.shape
    torch.Size([10, 4, 40])
    """

    def __init__(
        self,
        pool_type,
        kernel_size,
        input_dims=3,
        pool_axis=1,
        ceil_mode=True,
        padding=0,
        dilation=1,
        stride=None,
    ):
        super().__init__()
        self.pool_axis = pool_axis

        if stride is None:
            stride = kernel_size

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.pool_type = pool_type
        self.input_dims = input_dims
        self.inf = 1e23

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

    def forward(self, x, mask=None):

        if mask is not None:
            if self.pool_type == "max":
                # Avoid edge effect from paddings
                x = x.masked_fill(mask, -self.inf)
            elif self.pool_type == "avg":
                x = torch.cat([x, ~mask.expand(x.size())], dim=0)

        # Put the pooling axes as the last dimension for torch.nn.pool
        x = x.transpose(-1, self.pool_axis)

        # Apply pooling
        x = self.pool_layer(x)

        # Recover input shape
        x = x.transpose(-1, self.pool_axis)

        if mask is not None:
            # Avoid edge effect from paddings
            if self.pool_type == "avg":
                x_sum, mask_sum = x.split(x.size(0) // 2, dim=0)
                x = x_sum / (mask_sum + 1 / self.inf)

            mask = self.compute_mask(mask)
            x.masked_fill_(mask, 0.0)

        return x, mask

    def compute_mask(self, mask):

        if not self.ceil_mode:
            raise ValueError("Masking when ceil_mode=False is not supported.")

        mask = mask.transpose(-1, self.pool_axis)
        mask = mask[..., :: self.stride]
        mask = mask.transpose(-1, self.pool_axis)

        return mask


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
    ceil_mode : int
        When True, will use ceil instead of floor to compute the output shape.

    Example
    -------
    >>> pool = Pooling2d('max',(5,3))
    >>> inputs = torch.rand(10, 15, 12)
    >>> mask = torch.zeros(10, 15, 12).bool()
    >>> output, mask =pool(inputs, mask)
    >>> output.shape
    torch.Size([10, 3, 4])
    >>> mask.shape
    torch.Size([10, 3, 4])
    """

    def __init__(
        self,
        pool_type,
        kernel_size,
        pool_axis=(1, 2),
        ceil_mode=True,  # False,
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
        self.inf = 1e23

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

    def forward(self, x, mask=None):

        if mask is not None:
            if self.pool_type == "max":
                # Avoid edge effect from paddings
                x = x.masked_fill(mask, -self.inf)
            elif self.pool_type == "avg":
                x = torch.cat([x, ~mask.expand(x.size())], dim=0)

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

        if mask is not None:
            # Avoid edge effect from paddings
            if self.pool_type == "avg":
                x_sum, mask_sum = x.split(x.size(0) // 2, dim=0)
                x = x_sum / (mask_sum + 1 / self.inf)

            mask = self.compute_mask(mask)
            x.masked_fill_(mask, 0.0)

        return x, mask

    def compute_mask(self, mask):

        if not self.ceil_mode:
            raise ValueError("Masking when ceil_mode=False is not supported.")

        mask = (
            mask.unsqueeze(-1)
            .unsqueeze(-1)
            .transpose(-2, self.pool_axis[0])
            .transpose(-1, self.pool_axis[1])
            .squeeze(self.pool_axis[1])
            .squeeze(self.pool_axis[0])
        )

        # Subsample mask
        mask = mask[..., :: self.stride[-2], :: self.stride[-1]]

        mask = (
            mask.unsqueeze(self.pool_axis[0])
            .unsqueeze(self.pool_axis[1])
            .transpose(-2, self.pool_axis[0])
            .transpose(-1, self.pool_axis[1])
            .squeeze(-1)
            .squeeze(-1)
        )

        return mask


class StatisticsPooling(nn.Module):
    """This class implements a statistic pooling layer.

    It returns the concatenated mean and std of input tensor.

    Example
    -------
    >>> inp_tensor = torch.rand([5, 100, 50])
    >>> inp_mask = torch.zeros([5, 100, 1]).bool()
    >>> sp_layer = StatisticsPooling()
    >>> out_tensor, out_mask = sp_layer(inp_tensor, inp_mask)
    >>> out_tensor.shape
    torch.Size([5, 1, 100])
    >>> out_mask.shape
    torch.Size([5, 100, 1])
    """

    def __init__(self):
        super().__init__()

        # Small value for GaussNoise
        self.eps = 1e-5

    def forward(self, x, mask=None):
        """Calculates mean and std for a batch (input tensor).

        Arguments
        ---------
        x : torch.Tensor
            It represents a tensor for a mini-batch.
        """
        if mask is None:
            mean = x.mean(dim=1)
            std = x.std(dim=1)
        else:
            mean = []
            std = []
            # Avoiding padded time steps
            lengths = torch.sum(~mask, dim=1).squeeze(-1)
            for snt_id in range(x.shape[0]):
                actual_size = lengths[snt_id]

                # computing statistics
                mean.append(
                    torch.mean(x[snt_id, 1 : actual_size - 1, ...], dim=0)
                )
                std.append(
                    torch.std(x[snt_id, 1 : actual_size - 1, ...], dim=0)
                )

            mean = torch.stack(mean)
            std = torch.stack(std)

        gnoise = self._get_gauss_noise(mean.size(), device=mean.device)
        gnoise = gnoise
        mean += gnoise
        std = std + self.eps

        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(1)

        # Create summy mask for output
        if mask is not None:
            mask = torch.zeros([pooled_stats.size(0), 1, 1], dtype=torch.bool)

        return pooled_stats, mask

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

        if x.ndim == 3:
            return self.pool(x.permute(0, 2, 1)).permute(0, 2, 1)

        if x.ndim == 4:
            return self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
