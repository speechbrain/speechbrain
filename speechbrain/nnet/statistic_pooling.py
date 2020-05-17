"""Library implementing convolutional neural networks.

Author
    Nauman Dawlatabad 2020, Dannynis 2020
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
        super(StatisticsPooling, self).__init__()

    def forward(self, varying_length_tensor_for_a_batch):
        mean = varying_length_tensor_for_a_batch.mean(dim=1)
        std = varying_length_tensor_for_a_batch.std(dim=1)
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(0)
        pooled_stats = pooled_stats.transpose(0, 1).contiguous()

        return pooled_stats
