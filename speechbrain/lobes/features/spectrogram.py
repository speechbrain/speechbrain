"""
Spectrogram features

Authors
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
"""
import torch
from speechbrain.processing.features import (
    STFT,
    spectral_magnitude,
)


class Spectrogram(torch.nn.Module):
    """Generate features for input to the speech pipeline.

    Arguments
    ---------
    sample_rate : int
        Sampling rate for the input waveforms.
    n_fft : int
        Number of samples to use in each stft.

    Example
    -------
    >>> import torch
    >>> inputs = torch.randn([10, 16000])
    >>> feature_maker = Spectrogram()
    >>> feats = feature_maker(inputs, init_params=True)
    >>> feats.shape
    torch.Size([10, 101, 201])
    """

    def __init__(
        self, sample_rate=16000, n_fft=400,
    ):
        super().__init__()
        self.compute_STFT = STFT(sample_rate=sample_rate, n_fft=n_fft)

    def forward(self, wav, init_params=False):
        """Returns a set of features generated from the input waveforms.

        Arguments
        ---------
        wav : tensor
            A batch of audio signals to transform to features.
        """
        STFT = self.compute_STFT(wav)
        return spectral_magnitude(STFT)
