"""Library implementing activation functions.

Authors
 * Mirco Ravanelli 2020
 * Jianyuan Zhong 2020
"""

import torch
import torch.nn.functional as F

from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class Softmax(torch.nn.Module):
    """Computes the softmax of a 2d, 3d, or 4d input tensor.

    Arguments
    ---------
    apply_log : bool
        Whether to apply the log function before softmax.
    dim : int
        If the dimension where softmax is applied.
    reshape: bool
        whether to apply reshaping (true by default)
    dtype: torch.dtype
        dtype of the output tensor

    Example
    -------
    >>> classifier = Softmax()
    >>> inputs = torch.rand(10, 50, 40)
    >>> output = classifier(inputs)
    >>> output.shape
    torch.Size([10, 50, 40])
    """

    def __init__(self, apply_log=False, dim=-1, reshape=True, dtype=torch.float32):
        super().__init__()

        if apply_log:
            self.act = F.log_softmax
        else:
            self.act = F.softmax

        self.dim = dim
        self.reshape = reshape
        self.dtype = dtype

    def forward(self, x):
        """Returns the softmax of the input tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        x_act : torch.Tensor
            The softmax outputs.
        """
        # Reshaping the tensors
        dims = x.shape

        if self.reshape:
            if len(dims) == 3:
                x = x.reshape(dims[0] * dims[1], dims[2])

            if len(dims) == 4:
                x = x.reshape(dims[0] * dims[1], dims[2], dims[3])

        x_act = self.act(x, dim=self.dim, dtype=self.dtype)

        # Retrieving the original shape format
        if self.reshape:
            if len(dims) == 3:
                x_act = x_act.reshape(dims[0], dims[1], dims[2])

            if len(dims) == 4:
                x_act = x_act.reshape(dims[0], dims[1], dims[2], dims[3])

        return x_act


class GumbelSoftmax(torch.nn.Module):
    """Samples from the Gumbel-Softmax distribution and optionally discretizes.

    Reference: https://arxiv.org/abs/1611.00712, https://arxiv.org/abs/1611.01144

    Arguments
    ---------
    tau: float
        non-negative scalar temperature
    hard: bool
        if True, the returned samples will be discretized as one-hot vectors, but will be differentiated as if it is the soft sample in autograd
    apply_log: bool
        if True, returns the log of the softmax outputs.

    Example
    -------
    >>> x = torch.randn((8, 40, 120))
    >>> act = GumbelSoftmax(0.8, True)
    >>> x = act(x)
    """

    def __init__(self, tau, hard=False, apply_log=False):
        super().__init__()
        self.tau = tau
        self.hard = hard
        self.apply_log = apply_log

    def forward(self, x):
        """Returns the Gumbel softmax of the input tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        The Gumbel softmax output.
        """
        if self.apply_log:
            return torch.log(F.gumbel_softmax(x, tau=self.tau, hard=self.hard))
        return F.gumbel_softmax(x, tau=self.tau, hard=self.hard)


class Swish(torch.nn.Module):
    """The class implements the Swish activation function from
    https://arxiv.org/pdf/2005.03191.pdf

    given input x. Swish(x) = x / (1 + exp(beta * x))

    Arguments
    ---------
    beta: float
        Beta value.

    Example
    -------
    >>> x = torch.randn((8, 40, 120))
    >>> act = Swish()
    >>> x = act(x)
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        self.silu = torch.nn.SiLU()

    def forward(self, x):
        """Returns the Swished input tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        The swished output.
        """
        if self.beta != 1:  # slow path
            x = x * self.beta

        return self.silu(x)
