"""Library implementing activation functions.

Author
    Mirco Ravanelli 2020
"""

import torch
import logging

logger = logging.getLogger(__name__)


class softmax(torch.nn.Module):
    """Computes the softmax of a 2d, 3d, or 4d input tensor.

    Arguments
    ---------
    softmax_type : str
        it is the type of softmax to use ('softmax', 'log_softmax')
    dim : int
        if the dimension where softmax is applied.

    Example
    -------
    >>> classifier = softmax()
    >>> inputs = torch.rand(10, 50, 40)
    >>> output = classifier(inputs)
    >>> output.shape
    torch.Size([10, 50, 40])
    """

    def __init__(self, softmax_type="log_softmax", dim=-1):
        super().__init__()

        if softmax_type == "softmax":
            self.act = torch.nn.Softmax(dim=dim)

        if softmax_type == "log_softmax":
            self.act = torch.nn.LogSoftmax(dim=dim)

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
