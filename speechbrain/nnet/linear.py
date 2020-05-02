"""Library implementing linear transformation.

Author
    Mirco Ravanelli 2020
"""

import torch
import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


class linear(torch.nn.Module):
    """Computes a linear transformation y = wx + b.

    Arguments
    ---------
    n_neurons : int
        it is the number of output neurons (i.e, the dimensionality of the
        output)
    bias : bool
        if True, the additive bias b is adopted.

    Example
    -------
    >>> lin_t = linear(n_neurons=100)
    >>> inputs = torch.rand(10, 50, 40)
    >>> output = lin_t(inputs,init_params=True)
    >>> output.shape
    torch.Size([10, 50, 100])
    """

    def __init__(self, n_neurons, bias=True):
        super().__init__()
        self.n_neurons = n_neurons
        self.bias = bias

    def init_params(self, first_input):
        """
        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        fea_dim = first_input.shape[2]
        self.w = nn.Linear(fea_dim, self.n_neurons, bias=self.bias)
        self.w.to(first_input.device)

    def forward(self, x, init_params=False):
        """Returns the linear transformation of input tensor.

        Arguments
        ---------
        x : torch.Tensor
            input to transform linearly.
        """
        if init_params:
            self.init_params(x)

        # Transposing tensor (features always at the end)
        x = x.transpose(2, -1)

        wx = self.w(x)

        # Going back to the original shape format
        wx = wx.transpose(2, -1)
        return wx
