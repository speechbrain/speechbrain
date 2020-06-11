"""Library implementing activation functions.

Authors
 * Mirco Ravanelli 2020
"""

import torch
import logging

logger = logging.getLogger(__name__)


class Softmax(torch.nn.Module):
    """Computes the softmax of a 2d, 3d, or 4d input tensor.

    Arguments
    ---------
    apply_log : bool
        Whether to apply the log function before softmax.
    dim : int
        if the dimension where softmax is applied.

    Example
    -------
    >>> classifier = Softmax()
    >>> inputs = torch.rand(10, 50, 40)
    >>> output = classifier(inputs)
    >>> output.shape
    torch.Size([10, 50, 40])
    """

    def __init__(self, apply_log=False, dim=-1):
        super().__init__()

        if apply_log:
            self.act = torch.nn.LogSoftmax(dim=dim)
        else:
            self.act = torch.nn.Softmax(dim=dim)

    def forward(self, x):
        """Returns the softmax of the input tensor.

        Arguments
        ---------
        x : torch.Tensor
        """
        # Reshaping the tensors
        dims = x.shape

        if len(dims) == 3:
            x = x.reshape(dims[0] * dims[1], dims[2])

        if len(dims) == 4:
            x = x.reshape(dims[0] * dims[1], dims[2], dims[3])

        x_act = self.act(x)

        # Retrieving the original shape format
        if len(dims) == 3:
            x_act = x_act.reshape(dims[0], dims[1], dims[2])

        if len(dims) == 4:
            x_act = x_act.reshape(dims[0], dims[1], dims[2], dims[3])

        return x_act
