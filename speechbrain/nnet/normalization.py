"""Library implementing normalization.

Authors
 * Mirco Ravanelli 2020
"""
import torch
import torch.nn as nn


class BatchNorm1d(nn.Module):
    """Applies 1d batch normalization to the input tensor.

    Arguments
    ---------
    eps : float
        This value is added to std deviation estimationto improve the numerical
        stability.
    momentum : float
        It is a value used for the running_mean and running_var computation.
    affine : bool
        When set to True, the affine parameters are learned.
    track_running_stats : bool
        When set to True, this module tracks the running mean and variance,
        and when set to False, this module does not track such statistics.
    combine_batch_time : bool
        When true, it combines batch an time axis.


    Example
    -------
    >>> input = torch.randn(100, 10)
    >>> norm = BatchNorm1d()
    >>> output = norm(input, init_params=True)
    >>> output.shape
    torch.Size([100, 10])
    """

    def __init__(
        self,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        combine_batch_time=False,
    ):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.combine_batch_time = combine_batch_time

    def init_params(self, first_input):
        """
        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        fea_dim = first_input.shape[-1]

        self.norm = nn.BatchNorm1d(
            fea_dim,
            eps=self.eps,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
        ).to(first_input.device)

    def forward(self, x, init_params=False):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, [channels])
            input to normalize. 2d or 3d tensors are expected in input
            4d tensors can be used when combine_dims=True.
        """
        if init_params:
            self.init_params(x)

        if self.combine_batch_time:
            shape_or = x.shape
            if len(x.shape) == 3:
                x = x.reshape(shape_or[0] * shape_or[1], shape_or[2])
            else:
                x = x.reshape(
                    shape_or[0] * shape_or[1], shape_or[3], shape_or[2]
                )

        else:
            x = x.transpose(-1, 1)

        x_n = self.norm(x)

        if self.combine_batch_time:
            x_n = x_n.reshape(shape_or)
        else:
            x_n = x_n.transpose(1, -1)

        return x_n


class BatchNorm2d(nn.Module):
    """Applies 2d batch normalization to the input tensor.

    Arguments
    ---------
    eps : float
        This value is added to std deviation estimationto improve the numerical
        stability.
    momentum : float
        It is a value used for the running_mean and running_var computation.
    affine : bool
        When set to True, the affine parameters are learned.
    track_running_stats : bool
        When set to True, this module tracks the running mean and variance,
        and when set to False, this module does not track such statistics.

    Example
    -------
    >>> input = torch.randn(100, 10, 5, 20)
    >>> norm = BatchNorm2d()
    >>> output = norm(input, init_params=True)
    >>> output.shape
    torch.Size([100, 10, 5, 20])
    """

    def __init__(
        self, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,
    ):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

    def init_params(self, first_input):
        """
        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        fea_dim = first_input.shape[-1]

        self.norm = nn.BatchNorm2d(
            fea_dim,
            eps=self.eps,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
        ).to(first_input.device)

    def forward(self, x, init_params=False):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel1, channel2)
            input to normalize. 4d tensors are expected.
        """
        if init_params:
            self.init_params(x)

        x = x.transpose(-1, 1)

        x_n = self.norm(x)

        x_n = x_n.transpose(1, -1)

        return x_n


class LayerNorm(nn.Module):
    """Applies layer normalization to the input tensor.

    Arguments
    ---------
    eps : float
        This value is added to std deviation estimationto improve the numerical
        stability.
    elementwise_affine : bool
        If True, this module has learnable per-element affine parameters
        initialized to ones (for weights) and zeros (for biases).

    Example
    -------
    >>> input = torch.randn(100, 101, 128)
    >>> norm = LayerNorm()
    >>> output = norm(input, init_params=True)
    >>> output.shape
    torch.Size([100, 101, 128])
    """

    def __init__(self, eps=1e-05, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine

    def init_params(self, first_input):
        """
        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        self.norm = torch.nn.LayerNorm(
            first_input.size()[2:],
            eps=self.eps,
            elementwise_affine=self.elementwise_affine,
        ).to(first_input.device)

    def forward(self, x, init_params=False):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channels)
            input to normalize. 3d or 4d tensors are expected.
        """
        if init_params:
            self.init_params(x)
        x_n = self.norm(x)
        return x_n


class InstanceNorm1d(nn.Module):
    """Applies 1d instance normalization to the input tensor.

    Arguments
    ---------
    eps : float
        This value is added to std deviation estimationto improve the numerical
        stability.
    momentum : float
        It is a value used for the running_mean and running_var computation.
    track_running_stats : bool
        When set to True, this module tracks the running mean and variance,
        and when set to False, this module does not track such statistics.

    Example
    -------
    >>> input = torch.randn(100, 10, 20)
    >>> norm = InstanceNorm1d()
    >>> output = norm(input, init_params=True)
    >>> output.shape
    torch.Size([100, 10, 20])
    """

    def __init__(
        self, eps=1e-05, momentum=0.1, track_running_stats=True,
    ):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats

    def init_params(self, first_input):
        """
        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        fea_dim = first_input.shape[-1]

        self.norm = nn.InstanceNorm1d(
            fea_dim,
            eps=self.eps,
            momentum=self.momentum,
            track_running_stats=self.track_running_stats,
        ).to(first_input.device)

    def forward(self, x, init_params=False):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channels)
            input to normalize. 3d tensors are expected.
        """
        if init_params:
            self.init_params(x)

        x = x.transpose(-1, 1)

        x_n = self.norm(x)

        x_n = x_n.transpose(1, -1)

        return x_n


class InstanceNorm2d(nn.Module):
    """Applies 2d instance normalization to the input tensor.

    Arguments
    ---------
    eps : float
        This value is added to std deviation estimationto improve the numerical
        stability.
    momentum : float
        It is a value used for the running_mean and running_var computation.
    track_running_stats : bool
        When set to True, this module tracks the running mean and variance,
        and when set to False, this module does not track such statistics.

    Example
    -------
    >>> input = torch.randn(100, 10, 20, 2)
    >>> norm = InstanceNorm2d()
    >>> output = norm(input, init_params=True)
    >>> output.shape
    torch.Size([100, 10, 20, 2])
    """

    def __init__(
        self, eps=1e-05, momentum=0.1, track_running_stats=True,
    ):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats

    def init_params(self, first_input):
        """
        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        fea_dim = first_input.shape[-1]

        self.norm = nn.InstanceNorm2d(
            fea_dim,
            eps=self.eps,
            momentum=self.momentum,
            track_running_stats=self.track_running_stats,
        ).to(first_input.device)

    def forward(self, x, init_params=False):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel1, channel2)
            input to normalize. 4d tensors are expected.
        """
        if init_params:
            self.init_params(x)

        x = x.transpose(-1, 1)

        x_n = self.norm(x)

        x_n = x_n.transpose(1, -1)

        return x_n
