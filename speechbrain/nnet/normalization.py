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
    input_shape : tuple
        The expected shape of the input. Alternatively, use ``input_size``.
    input_size : int
        The expected size of the input. Alternatively, use ``input_shape``.
    eps : float
        This value is added to std deviation estimation to improve the numerical
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
    >>> norm = BatchNorm1d(input_shape=input.shape)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 10])
    """

    def __init__(
        self,
        input_shape=None,
        input_size=None,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        combine_batch_time=False,
        skip_transpose=False,
    ):
        super().__init__()
        self.combine_batch_time = combine_batch_time
        self.skip_transpose = skip_transpose

        if input_size is None and skip_transpose:
            input_size = input_shape[1]
        elif input_size is None:
            input_size = input_shape[-1]

        self.norm = nn.BatchNorm1d(
            input_size,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, [channels])
            input to normalize. 2d or 3d tensors are expected in input
            4d tensors can be used when combine_dims=True.
        """
        shape_or = x.shape
        if self.combine_batch_time:
            if x.ndim == 3:
                x = x.reshape(shape_or[0] * shape_or[1], shape_or[2])
            else:
                x = x.reshape(
                    shape_or[0] * shape_or[1], shape_or[3], shape_or[2]
                )

        elif not self.skip_transpose:
            x = x.transpose(-1, 1)

        x_n = self.norm(x)

        if self.combine_batch_time:
            x_n = x_n.reshape(shape_or)
        elif not self.skip_transpose:
            x_n = x_n.transpose(1, -1)

        return x_n


class BatchNorm2d(nn.Module):
    """Applies 2d batch normalization to the input tensor.

    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input. Alternatively, use ``input_size``.
    input_size : int
        The expected size of the input. Alternatively, use ``input_shape``.
    eps : float
        This value is added to std deviation estimation to improve the numerical
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
    >>> norm = BatchNorm2d(input_shape=input.shape)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 10, 5, 20])
    """

    def __init__(
        self,
        input_shape=None,
        input_size=None,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__()

        if input_shape is None and input_size is None:
            raise ValueError("Expected input_shape or input_size as input")

        if input_size is None:
            input_size = input_shape[-1]

        self.norm = nn.BatchNorm2d(
            input_size,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel1, channel2)
            input to normalize. 4d tensors are expected.
        """
        x = x.transpose(-1, 1)
        x_n = self.norm(x)
        x_n = x_n.transpose(1, -1)

        return x_n


class LayerNorm(nn.Module):
    """Applies layer normalization to the input tensor.

    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input.
    eps : float
        This value is added to std deviation estimation to improve the numerical
        stability.
    elementwise_affine : bool
        If True, this module has learnable per-element affine parameters
        initialized to ones (for weights) and zeros (for biases).

    Example
    -------
    >>> input = torch.randn(100, 101, 128)
    >>> norm = LayerNorm(input_shape=input.shape)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 101, 128])
    """

    def __init__(
        self,
        input_size=None,
        input_shape=None,
        eps=1e-05,
        elementwise_affine=True,
    ):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if input_shape is not None:
            input_size = input_shape[2:]

        self.norm = torch.nn.LayerNorm(
            input_size,
            eps=self.eps,
            elementwise_affine=self.elementwise_affine,
        )

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channels)
            input to normalize. 3d or 4d tensors are expected.
        """
        return self.norm(x)


class InstanceNorm1d(nn.Module):
    """Applies 1d instance normalization to the input tensor.

    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input. Alternatively, use ``input_size``.
    input_size : int
        The expected size of the input. Alternatively, use ``input_shape``.
    eps : float
        This value is added to std deviation estimation to improve the numerical
        stability.
    momentum : float
        It is a value used for the running_mean and running_var computation.
    track_running_stats : bool
        When set to True, this module tracks the running mean and variance,
        and when set to False, this module does not track such statistics.
    affine : bool
        A boolean value that when set to True, this module has learnable
        affine parameters, initialized the same way as done for
        batch normalization. Default: False.

    Example
    -------
    >>> input = torch.randn(100, 10, 20)
    >>> norm = InstanceNorm1d(input_shape=input.shape)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 10, 20])
    """

    def __init__(
        self,
        input_shape=None,
        input_size=None,
        eps=1e-05,
        momentum=0.1,
        track_running_stats=True,
        affine=False,
    ):
        super().__init__()

        if input_shape is None and input_size is None:
            raise ValueError("Expected input_shape or input_size as input")

        if input_size is None:
            input_size = input_shape[-1]

        self.norm = nn.InstanceNorm1d(
            input_size,
            eps=eps,
            momentum=momentum,
            track_running_stats=track_running_stats,
            affine=affine,
        )

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channels)
            input to normalize. 3d tensors are expected.
        """
        x = x.transpose(-1, 1)
        x_n = self.norm(x)
        x_n = x_n.transpose(1, -1)

        return x_n


class InstanceNorm2d(nn.Module):
    """Applies 2d instance normalization to the input tensor.

    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input. Alternatively, use ``input_size``.
    input_size : int
        The expected size of the input. Alternatively, use ``input_shape``.
    eps : float
        This value is added to std deviation estimation to improve the numerical
        stability.
    momentum : float
        It is a value used for the running_mean and running_var computation.
    track_running_stats : bool
        When set to True, this module tracks the running mean and variance,
        and when set to False, this module does not track such statistics.
    affine : bool
        A boolean value that when set to True, this module has learnable
        affine parameters, initialized the same way as done for
        batch normalization. Default: False.

    Example
    -------
    >>> input = torch.randn(100, 10, 20, 2)
    >>> norm = InstanceNorm2d(input_shape=input.shape)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 10, 20, 2])
    """

    def __init__(
        self,
        input_shape=None,
        input_size=None,
        eps=1e-05,
        momentum=0.1,
        track_running_stats=True,
        affine=False,
    ):
        super().__init__()

        if input_shape is None and input_size is None:
            raise ValueError("Expected input_shape or input_size as input")

        if input_size is None:
            input_size = input_shape[-1]

        self.norm = nn.InstanceNorm2d(
            input_size,
            eps=eps,
            momentum=momentum,
            track_running_stats=track_running_stats,
            affine=affine,
        )

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel1, channel2)
            input to normalize. 4d tensors are expected.
        """
        x = x.transpose(-1, 1)
        x_n = self.norm(x)
        x_n = x_n.transpose(1, -1)

        return x_n

class GroupScaling1D(nn.Module):
    """Scales inputs by the second moment for the entire layer
    Arguments
    ---------
    eps : float
        This value is added for numerical stability.
    group_num : int
        Number of groups to separate the channels into. In the case of
        transformers, it is number of heads in multihead attention
    """

    def __init__(self, eps=1e-5, group_num=4):
        super(GroupScaling1D, self).__init__()
        self.eps = eps
        self.group_num = group_num

    def forward(self, input):
        """Returns the scaled input by the second moment.
        Arguments
        ---------
        input : torch.Tensor (batch, time, features)
            input to scale. 3d tensors are expected.
        """
        # calculate second moment
        # different group use different mean
        T, B, C = input.shape
        Cg = C // self.group_num
        gn_input = input.contiguous().reshape(T, B, self.group_num, Cg)
        moment2 = (
            torch.repeat_interleave(
                torch.mean(gn_input * gn_input, dim=3, keepdim=True),
                repeats=Cg,
                dim=-1,
            )
            .contiguous()
            .reshape(T, B, C)
        )
        # divide out second moment
        return input / torch.sqrt(moment2 + self.eps)


class PowerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        running_phi,
        eps,
        afwd,
        abkw,
        ema_gz,
        warmup_iters,
        current_iter,
        input_denom,
    ):
        """Computes normalized inputs with the running statistics for power norm
        in forward propagation
        Arguments
        ---------
        ctx : object of torch.autograd.function.PowerFunctionBackward
            Context object to stash information for backward computation
        x : torch.Tensor (batch, time, features)
            input to normalize. 3d tensors are expected.
        eps : float
            This value is added for numerical stability.
        weight : float
            Weights parameter for the affine computation
        bias : float
            Bias parameter for the affine computation
        running_phi : torch.Tensor
            Running quadratic mean of the previous iteration
        afwd : float
            Moving average coefficient in the forward propagation
        abkw : float
            Moving average coefficient in the backward propagation
        ema_gz : torch.Tensor
            Exponentially decaying average.
        warmup_iters : int
            Warmup steps for the learning rate schedule in the optimizer.
        current_iter : int
            Current iteration step
        input_denom : torch.Tensor (batch, time, features)
            input to compute the squares for the quadratic mean
        """
        ctx.eps = eps
        current_iter = current_iter.item()
        ctx.current_iter = current_iter
        ctx.warmup_iters = warmup_iters
        ctx.abkw = abkw
        N, C, H, W = x.shape
        x2 = (input_denom * input_denom).mean(dim=0)

        var = x2.reshape(1, C, 1, 1)
        if current_iter <= warmup_iters:
            z = x / (var + eps).sqrt()
        else:
            z = x / (running_phi + eps).sqrt()

        y = z
        ctx.save_for_backward(z, var, weight, ema_gz)

        if current_iter < warmup_iters:
            running_phi.copy_(
                running_phi * (current_iter - 1) / current_iter
                + var.mean(dim=0, keepdim=True) / current_iter
            )
        running_phi.copy_(
            afwd * running_phi + (1 - afwd) * var.mean(dim=0, keepdim=True)
        )
        y = weight.reshape(1, C, 1, 1) * y + bias.reshape(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """Computes the gradient of the loss w.r.t quadratic mean with
        running statistics for power norm in backward propagation
        Arguments
        ---------
        grad_output: torch.Tensor
        Tensor containing the gradient of the loss
        with respect to the output
        """
        eps = ctx.eps
        abkw = ctx.abkw

        N, C, H, W = grad_output.size()
        z, var, weight, ema_gz = ctx.saved_variables

        y = z
        g = grad_output * weight.reshape(1, C, 1, 1)
        g = g * 1

        approx_grad_g = g - (1 - abkw) * ema_gz * z
        ema_gz.add_(
            (approx_grad_g * z)
            .mean(dim=3, keepdim=True)
            .mean(dim=2, keepdim=True)
            .mean(dim=0, keepdim=True)
        )

        gx = 1.0 / torch.sqrt(var + eps) * approx_grad_g
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class PowerNorm(nn.Module):
    """Applies power normalization to the input tensor,
    a modification of batch normalization, used for testing the numerical
    stability.
    Most parts of the code are from the repo https://github.com/sIncerass/powernorm
    adapted to SpeechBrain style of implementation.
    The idea comes from the paper PowerNorm: Rethinking Batch Normalization in Transformers
    https://arxiv.org/pdf/2003.07845.pdf
    Arguments
    ---------
    num_features : int
        The number of features.
    eps : float
        This value is added to running statistics for numerical stability.
    alpha_fwd : float
        Moving average coefficient in the forward propagation
    alpha_bkw : float
        Moving average coefficient in the backward propagation
    warmup_iters : int
        Warmup steps for the learning rate schedule in the optimizer.
    affine : bool
        When set to True, the affine parameters are learned.
    group_num : int
        Number of groups to separate the channels into.
        In the case of transformers, it is number of heads in multihead attention
    Example
    -------
    >>> input = torch.randn(16, 63, 512)
    >>> norm = PowerNorm(512)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([16, 63, 512])
    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        alpha_fwd=0.9,
        alpha_bkw=0.9,
        affine=True,
        warmup_iters=10000,
        group_num=1,
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        self.register_parameter(
            "weight", nn.Parameter(torch.ones(num_features))
        )
        self.register_parameter("bias", nn.Parameter(torch.zeros(num_features)))
        self.register_buffer("running_phi", torch.ones(1, num_features, 1, 1))
        self.register_buffer("ema_gz", torch.zeros(1, num_features, 1, 1))
        self.register_buffer("iters", torch.zeros(1).type(torch.LongTensor))

        self.afwd = alpha_fwd
        self.abkw = alpha_bkw
        self.warmup_iters = warmup_iters
        self.gp = GroupScaling1D(group_num=group_num)

    def forward(self, input):
        """Returns the normalized input tensor.
        Arguments
        ---------
        input : torch.Tensor (batch, time, features)
            input to normalize. 3d tensors are expected.
        """
        assert len(input.shape) == 3
        input = self.gp(input)

        input_denom = input.clone()
        input_denom = input_denom.reshape(-1, self.num_features)

        input = input.permute(1, 2, 0).contiguous()
        input_shape = input.shape
        input = input.unsqueeze(-1)

        if self.training:
            self.iters.copy_(self.iters + 1)
            output = PowerFunction.apply(
                input,
                self.weight,
                self.bias,
                self.running_phi,
                self.eps,
                self.afwd,
                self.abkw,
                self.ema_gz,
                self.warmup_iters,
                self.iters,
                input_denom,
            )

        else:
            N, C, H, W = input.shape
            var = self.running_phi
            output = input / (var + self.eps).sqrt()
            output = self.weight.reshape(
                1, C, 1, 1
            ) * output + self.bias.reshape(1, C, 1, 1)

        output = output.reshape(input_shape)
        output = output.permute(2, 0, 1).contiguous()
        return output
