"""
Basic feature pipelines

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


class Fbank(torch.nn.Module):
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
    filter_shape : str
        Shape of the filters ('triangular', 'rectangular', 'gaussian').
    param_change_factor: bool
        If freeze=False, this parameter affects the speed at which the filter
        parameters (i.e., central_freqs and bands) can be changed.  When high
        (e.g., param_change_factor=1) the filters change a lot during training.
        When low (e.g. param_change_factor=0.1) the filter parameters are more
        stable during training
    param_rand_factor: float
        This parameter can be used to randomly change the filter parameters
        (i.e, central frequencies and bands) during training.  It is thus a
        sort of regularization. param_rand_factor=0 does not affect, while
        param_rand_factor=0.15 allows random variations within +-15% of the
        standard values of the filter parameters (e.g., if the central freq
        is 100 Hz, we can randomly change it from 85 Hz to 115 Hz).
    left_frames : int
        Number of frames of left context to add.
    right_frames : int
        Number of frames of right context to add.

    Example
    -------
    >>> import torch
    >>> inputs = torch.randn([10, 16000])
    >>> feature_maker = Fbank()
    >>> feats = feature_maker(inputs)
    >>> feats.shape
    torch.Size([10, 101, 40])
    """

    def __init__(
        self,
        deltas=False,
        context=False,
        requires_grad=False,
        sample_rate=16000,
        n_fft=400,
        n_mels=40,
        filter_shape="triangular",
        param_change_factor=1.0,
        param_rand_factor=0.0,
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
            filter_shape=filter_shape,
            param_change_factor=param_change_factor,
            param_rand_factor=param_rand_factor,
        )
        self.compute_deltas = Deltas(input_size=n_mels)
        self.context_window = ContextWindow(
            left_frames=left_frames, right_frames=right_frames,
        )

    def forward(self, wav):
        """Returns a set of features generated from the input waveforms.

        Arguments
        ---------
        wav : tensor
            A batch of audio signals to transform to features.
        """
        with torch.no_grad():
            STFT = self.compute_STFT(wav)
            mag = spectral_magnitude(STFT)
            fbanks = self.compute_fbanks(mag)

            if self.deltas:
                delta1 = self.compute_deltas(fbanks)
                delta2 = self.compute_deltas(delta1)
                fbanks = torch.cat([fbanks, delta1, delta2], dim=2)

            if self.context:
                fbanks = self.context_window(fbanks)

        return fbanks


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
    filter_shape : str
        Shape of the filters ('triangular', 'rectangular', 'gaussian').
    param_change_factor: bool
        If freeze=False, this parameter affects the speed at which the filter
        parameters (i.e., central_freqs and bands) can be changed.  When high
        (e.g., param_change_factor=1) the filters change a lot during training.
        When low (e.g. param_change_factor=0.1) the filter parameters are more
        stable during training
    param_rand_factor: float
        This parameter can be used to randomly change the filter parameters
        (i.e, central frequencies and bands) during training.  It is thus a
        sort of regularization. param_rand_factor=0 does not affect, while
        param_rand_factor=0.15 allows random variations within +-15% of the
        standard values of the filter parameters (e.g., if the central freq
        is 100 Hz, we can randomly change it from 85 Hz to 115 Hz).
    left_frames : int
        Number of frames of left context to add.
    right_frames : int
        Number of frames of right context to add.

    Example
    -------
    >>> import torch
    >>> inputs = torch.randn([10, 16000])
    >>> feature_maker = MFCC()
    >>> feats = feature_maker(inputs)
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
        filter_shape="triangular",
        param_change_factor=1.0,
        param_rand_factor=0.0,
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
            filter_shape=filter_shape,
            param_change_factor=param_change_factor,
            param_rand_factor=param_rand_factor,
        )
        self.compute_dct = DCT(input_size=n_mels, n_out=n_mfcc)
        self.compute_deltas = Deltas(input_size=n_mfcc)
        self.context_window = ContextWindow(
            left_frames=left_frames, right_frames=right_frames,
        )

    def forward(self, wav):
        """Returns a set of mfccs generated from the input waveforms.

        Arguments
        ---------
        wav : tensor
            A batch of audio signals to transform to features.
        """
        with torch.no_grad():
            STFT = self.compute_STFT(wav)
            mag = spectral_magnitude(STFT)
            fbanks = self.compute_fbanks(mag)
            mfccs = self.compute_dct(fbanks)

            if self.deltas:
                delta1 = self.compute_deltas(mfccs)
                delta2 = self.compute_deltas(delta1)
                mfccs = torch.cat([mfccs, delta1, delta2], dim=2)

            if self.context:
                mfccs = self.context_window(mfccs)

        return mfccs
