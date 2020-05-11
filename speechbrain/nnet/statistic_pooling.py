import torch
import torch.nn as nn


class StatisticsPooling(nn.Module):
    """
    -------------------------------------------------------------------------

    This function implements a statistic pooling layer (example as used in TDNN).

    Arguments
    -------
    Input (init):
                   - tensor (type: torch.Tensor, mandatory):
                       it is usually a set of faetures or the output of neural network layer


    Output (call): - concatenated mean and variance (type, torch.Tensor, mandatory):
                      it is the TDNN output. Time-delayed convolved input and kernel.

    Authors:
    -------
    Nauman Dawlatabad 2020, Dannynis 2020
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
