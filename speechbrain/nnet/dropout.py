"""Library implementing dropout.

Authors
 * Mirco Ravanelli 2020
"""

import torch  # noqa: F401
import torch.nn as nn

from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class Dropout2d(nn.Module):
    """This function implements dropout 2d. It randomly put zeros on
    entire channels.

    Arguments
    ---------
    drop_rate : float
        It is the dropout factor (between 0 and 1).
    inplace : bool
        If True, it uses inplace operations.

    Example
    -------
    >>> drop = Dropout2d(drop_rate=0.5)
    >>> inputs = torch.rand(10, 50, 40)
    >>> output = drop(inputs)
    >>> output.shape
    torch.Size([10, 50, 40])
    """

    def __init__(self, drop_rate, inplace=False):
        super().__init__()
        self.drop_rate = drop_rate
        self.inplace = inplace
        self.drop = nn.Dropout2d(p=self.drop_rate, inplace=self.inplace)

    def forward(self, x):
        """Applies dropout 2d to the input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel1, channel2)
            input to normalize. 4d tensors are expected.

        Returns
        -------
        x_drop : torch.Tensor
            The tensor with channels zeroed out.
        """

        # time must be the last
        x = x.transpose(1, 2).transpose(2, -1)
        x_drop = self.drop(x)
        x_drop = x_drop.transpose(-1, 1).transpose(2, -1)

        return x_drop
