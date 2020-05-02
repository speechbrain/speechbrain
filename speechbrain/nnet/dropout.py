"""Library implementing dropout.

Author
    Mirco Ravanelli 2020
"""
import torch  # noqa: F401
import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


class dropout(nn.Module):
    """This function implements droput.

    This function implements droput of the input tensor. In particular,
    1d dropout (nn.Dropout) is activated with 2d or 3d input tensors, while
    nn.Dropout2d is activated with 4d input tensors.


    Arguments
    ---------
    dropout_rate : float
        It is the dropout factor (between 0 and 1).
    inplace : bool
        If True, it uses inplace operations.

    Example
    -------
    >>> drop = dropout(drop_rate=0.5)
    >>> inputs = torch.rand(10, 50, 40)
    >>> drop.init_params(inputs)
    >>> output=drop(inputs)
    >>> output.shape
    torch.Size([10, 50, 40])
    """

    def __init__(
        self, drop_rate, inplace=False,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.inplace = inplace

    def init_params(self, first_input):
        """
        Arguments
        ---------
        first_input : tensor
            A dummy input of the right shape for initializing parameters.
        """

        if len(first_input.shape) <= 3:
            self.drop = nn.Dropout(p=self.drop_rate, inplace=self.inplace)

        if len(first_input.shape) == 4:
            self.drop = nn.Dropout2d(p=self.drop_rate, inplace=self.inplace)

        if len(first_input.shape) == 5:
            self.drop = nn.Dropout3d(p=self.drop_rate, inplace=self.inplace)

    def forward(self, x, init_params=False):
        """Applies dropout to the input tensor.

        Arguments
        ---------
        x : torch.Tensor
        """
        if init_params:
            self.init_params(x)

        if self.drop_rate == 0.0:
            return x

        # time must be the last
        x = x.transpose(1, 2).transpose(2, -1)

        x_drop = self.drop(x)

        # Getting original dimensionality
        x_drop = x_drop.transpose(-1, 1).transpose(2, -1)

        return x_drop
