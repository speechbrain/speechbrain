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

class Fbank(torch.nn.Module):

    def __init__(self,
    ):
        super().__init__()

    def forward(self, wav):

        pass
        return beamformed

