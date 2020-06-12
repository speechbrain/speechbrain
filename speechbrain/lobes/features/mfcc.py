"""
MFCC Features

Authors
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
"""
import torch
from speechbrain.processing.features import (
    STFT,
    spectral_magnitude,
    Filterbank,
    DCT,
    Deltas,
    ContextWindow,
)


class MFCC(torch.nn.Module):
    """Generate features for input to the speech pipeline.

    Arguments
    ---------
    deltas : bool
        Whether or not to append derivatives and second derivatives
        to the features.
    context : bool
        Whether or not to append forward and backward contexts to
        the features.
    requires_grad : bool
        Whether to allow parameters (i.e. fbank centers and
        spreads) to update during training.
    sample_rate : int
        Sampling rate for the input waveforms.
    n_fft : int
        Number of samples to use in each stft.
    n_mels : int
        Number of filters to use for creating filterbank.
    n_mfcc : int
        Number of output coefficients
    left_frames : int
        Number of frames of left context to add.
    right_frames : int
        Number of frames of right context to add.

    Example
    -------
    >>> import torch
    >>> inputs = torch.randn([10, 16000])
    >>> feature_maker = MFCC()
    >>> feats = feature_maker(inputs, init_params=True)
    >>> feats.shape
    torch.Size([10, 101, 660])
    """

    def __init__(
        self,
        deltas=True,
        context=True,
        requires_grad=False,
        sample_rate=16000,
        n_fft=400,
        n_mels=23,
        n_mfcc=20,
        left_frames=5,
        right_frames=5,
    ):
        super().__init__()
        self.deltas = deltas
        self.context = context
        self.requires_grad = requires_grad

        self.compute_STFT = STFT(sample_rate=sample_rate, n_fft=n_fft)
        self.compute_fbanks = Filterbank(
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=0,
            f_max=sample_rate / 2,
            freeze=not requires_grad,
        )
        self.compute_dct = DCT(n_out=n_mfcc)
        self.compute_deltas = Deltas()
        self.context_window = ContextWindow(
            left_frames=left_frames, right_frames=right_frames,
        )

    def forward(self, wav, init_params=False):
        """Returns a set of mfccs generated from the input waveforms.

        Arguments
        ---------
        wav : tensor
            A batch of audio signals to transform to features.
        """
        STFT = self.compute_STFT(wav)
        mag = spectral_magnitude(STFT)
        fbanks = self.compute_fbanks(mag, init_params)
        mfccs = self.compute_dct(fbanks, init_params)

        if self.deltas:
            delta1 = self.compute_deltas(mfccs, init_params)
            delta2 = self.compute_deltas(delta1)
            mfccs = torch.cat([mfccs, delta1, delta2], dim=2)

        if self.context:
            mfccs = self.context_window(mfccs)

        return mfccs
