"""Calculate accuracy.

Authors
* Jianyuan Zhong 2020
"""
import torch
from speechbrain.data_io.data_io import length_to_mask


def Accuracy(log_probablities, targets, length=None):
    """Calculate accuarcy for predicted log probabilities and targets in a batch

    Arguements
    ----------
    log_probablities: tensor
        predicted log probabilities (batch_size, time, feature)
    targets: tensor
        target (batch_size, time)
    length: tensor
        length of target (batch_size,)

    Example
    -------
    >>> probs = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]]).unsqueeze(0)
    >>> acc = Accuracy(torch.log(probs), torch.tensor([1, 1, 0]).unsqueeze(0), torch.tensor([2/3]))
    >>> print(acc)
    0.5
    """
    if length is not None:
        mask = length_to_mask(
            length * targets.shape[1], max_len=targets.shape[1],
        ).bool()
        if len(targets.shape) == 3:
            mask = mask.unsqueeze(2).repeat(1, 1, targets.shape[2])

    padded_pred = log_probablities.argmax(-1)

    if length is not None:
        numerator = torch.sum(
            padded_pred.masked_select(mask) == targets.masked_select(mask)
        )
        denominator = torch.sum(mask)
    else:
        numerator = torch.sum(padded_pred == targets)
        denominator = targets.shape[1]
    return float(numerator) / float(denominator)


class AccuracyStats:
    def __init__(self):
        self.acc_list = []

    def append(self, log_probablities, targets, length=None):
        self.acc_list.append(Accuracy(log_probablities, targets, length))

    def summarize(self):
        return sum(self.acc_list) / len(self.acc_list)
