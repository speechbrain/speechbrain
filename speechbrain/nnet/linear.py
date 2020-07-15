"""Library implementing linear transformation.

Authors
 * Mirco Ravanelli 2020
"""

import torch
import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


class Linear(torch.nn.Module):
    """Computes a linear transformation y = wx + b.

    Arguments
    ---------
    n_neurons : int
        it is the number of output neurons (i.e, the dimensionality of the
        output)
    bias : bool
        if True, the additive bias b is adopted.
    combine_dims : bool
        if True and the input is 4D, combine 3rd and 4th dimensions of input.
    act_fct_for_init : str, optional
        Default: leaky_relu
        This parameter is used to compute gain for the weight Initialization.
        See the official pytorch documentation on initialization.

    Example
    -------
    >>> lin_t = Linear(n_neurons=100)
    >>> inputs = torch.rand(10, 50, 40)
    >>> output = lin_t(inputs,init_params=True)
    >>> output.shape
    torch.Size([10, 50, 100])
    """

    def __init__(
        self,
        n_neurons,
        bias=True,
        combine_dims=False,
        act_fct_for_init="leaky_relu",
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.bias = bias
        self.combine_dims = combine_dims
        self.act_fct_for_init = act_fct_for_init

    def init_params(self, first_input):
        """
        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        fea_dim = first_input.shape[-1]
        if len(first_input.shape) == 4 and self.combine_dims:
            fea_dim = first_input.shape[2] * first_input.shape[3]

        self.w = nn.Linear(fea_dim, self.n_neurons, bias=self.bias)
        # Initialize weights and biases
        nn.init.xavier_uniform_(
            self.w.weight.data,
            gain=nn.init.calculate_gain(self.act_fct_for_init),
        )
        self.w.bias.data.fill_(0)
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

        if len(x.shape) == 4 and self.combine_dims:
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        wx = self.w(x)

        return wx
