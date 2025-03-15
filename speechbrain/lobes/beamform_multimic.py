"""Beamformer for multi-mic processing.

Authors
 * Nauman Dawalatabad
"""

import torch

from speechbrain.processing.features import ISTFT, STFT
from speechbrain.processing.multi_mic import Covariance, DelaySum, GccPhat


class DelaySum_Beamformer(torch.nn.Module):
    """Generate beamformed signal from multi-mic data using DelaySum beamforming.

    Arguments
    ---------
    sampling_rate : int (default: 16000)
        Sampling rate of audio signals.
    """

    def __init__(self, sampling_rate=16000):
        super().__init__()
        self.fs = sampling_rate
        self.stft = STFT(sample_rate=self.fs)
        self.cov = Covariance()
        self.gccphat = GccPhat()
        self.delaysum = DelaySum()
        self.istft = ISTFT(sample_rate=self.fs)

    def forward(self, mics_signals):
        """Returns beamformed signal using multi-mic data.

        Arguments
        ---------
        mics_signals : torch.Tensor
            Set of audio signals to be transformed.

        Returns
        -------
        sig : torch.Tensor
        """
        with torch.no_grad():
            Xs = self.stft(mics_signals)
            XXs = self.cov(Xs)
            tdoas = self.gccphat(XXs)
            Ys_ds = self.delaysum(Xs, tdoas)
            sig = self.istft(Ys_ds)

        return sig
