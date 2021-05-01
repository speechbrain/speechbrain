import torch
from speechbrain.processing.features import (
    STFT,
    ISTFT,
)

from speechbrain.processing.multi_mic(
    Covariance,
    GccPhat,
    DelaySum,
)

class DelaySum_Beamformer(torch.nn.Module):

    def __init__(
        self,
        sampling_rate=16000,
    ):
        self.fs = sampling_rate
        self.stft = STFT(sample_rate=fs)
        self.cov = Covariance()
        self.gccphat = GccPhat()
        self.delaysum = DelaySum()
        self.istft = ISTFT(sample_rate=fs)

    def forward(self, mics_signals):
        with torch.no_grad():
            Xs = self.stft(mics_signals)
            XXs = self.cov(Xs)
            tdoas = self.gccphat(XXs)
            Ys_ds = self.delaysum(Xs, tdoas)
            sig = self.istft(Ys_ds)

        return sig

