"""
Library implementing learnable exponential moving average layer
in compliance with LEAF

Neil Zeghidour, Olivier Teboul, F{\'e}lix de Chaumont Quitry & Marco Tagliasacchi, "LEAF: A LEARNABLE FRONTEND
FOR AUDIO CLASSIFICATION", in Proc of ICLR 2021 (https://arxiv.org/abs/2101.08596)

Authors
 * Sarthak Yadav 2022
"""
import torch
from torch import nn


class ExponentialMovingAverage(nn.Module):
    """
    Applies learnable exponential moving average
    Arguments
    ---------
    input_size : int
        The expected size of the input.
    coeff_init: float
        Initial smoothing coefficient value
    per_channel: bool
        Controls whether every smoothing coefficients are learned
        independently for every input channel
    trainable: bool
        whether to learn the PCEN parameters or use fixed
    skip_transpose : bool
        If False, uses batch x time x channel convention of speechbrain.
        If True, uses batch x channel x time convention.
    """

    def __init__(
        self,
        input_size: int,
        coeff_init: float = 0.04,
        per_channel: bool = False,
        trainable: bool = True,
        skip_transpose: bool = False,
    ):
        super(ExponentialMovingAverage, self).__init__()
        self._coeff_init = coeff_init
        self._per_channel = per_channel
        self.skip_transpose = skip_transpose
        self.trainable = trainable
        weights = (
            torch.ones(input_size,) if self._per_channel else torch.ones(1,)
        )
        self._weights = nn.Parameter(
            weights * self._coeff_init, requires_grad=trainable
        )

    def forward(self, x):
        if not self.skip_transpose:
            x = x.transpose(1, -1)
        w = torch.clamp(self._weights, min=0.0, max=1.0)
        initial_state = x[:, :, 0]

        def scan(init_state, x, w):
            x = x.permute(2, 0, 1)
            acc = init_state
            results = []
            for ix in range(x.shape[0]):
                acc = (w * x[ix]) + ((1.0 - w) * acc)
                results.append(acc.unsqueeze(0))
            results = torch.cat(results, dim=0)
            results = results.permute(1, 2, 0)
            return results

        output = scan(initial_state, x, w)
        if not self.skip_transpose:
            output = output.transpose(1, -1)
        return output
