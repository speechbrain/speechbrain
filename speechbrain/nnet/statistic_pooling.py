"""Library implementing statistics (mean and std) pooling.

Author
    Nauman Dawlatabad 2020
"""

import torch
import torch.nn as nn


class StatisticsPooling(nn.Module):
    """This function implements a statistic pooling layer.

    This class implements Statistics Pooling layer:
    It returns the concatenated mean and std of input tensor

    Arguments
    ---------
    tensor: torch.Tensor
        It is usually a set of features or the output of neural network layer

    Example
    -------
    >>> inp_tensor = torch.rand([5, 100, 20])
    >>> sp = StatisticsPooling()
    >>> out_tensor = sp(inp_tensor)
    >>> out_tensor.shape
    torch.Size([5, 1, 40])
    """

    def __init__(self):
        self.eps = 1e-5
        super(StatisticsPooling, self).__init__()

    def _get_gauss_noise(self, shape_of_tensor):
        """Returns a tensor of epsilon Gaussian noise

        Arguments
        ---------
        shap_of_tensor : tensor
            It represents the size of tensor for making Gaussian noise.
        """
        gnoise = torch.randn(shape_of_tensor)
        gnoise -= torch.min(gnoise)
        gnoise /= torch.max(gnoise)
        gnoise = self.eps * ((1 - 9) * gnoise + 9)

        return gnoise

    def forward(self, x):
        """Calculates mean and std for a batch (input tensor).

        Arguments
        ---------
        x : torch.Tensor
            It represents a tensor for a mini-batch
        """
        mean = x.mean(dim=1)
        gnoise = self._get_gauss_noise(mean.size())
        if torch.cuda.is_available():
            gnoise = gnoise.to(device="cuda:0")
        mean += gnoise
        std = x.std(dim=1) + self.eps
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(1)

        return pooled_stats
