"""Low-level feature pipeline components

This library gathers functions that compute popular speech  features over
batches of data. All the classes are of type nn.Module. This gives the
possibility to have end-to-end  differentiability and to backpropagate the
gradient through them. Our functions are a modified version the ones
in torch audio toolkit (https://github.com/pytorch/audio).

Example
-------
>>> import torch
>>> from speechbrain.dataio.dataio import read_audio
>>> signal = read_audio("tests/samples/single-mic/example1.wav")
>>> signal = signal.unsqueeze(0)
>>> compute_STFT = STFT(
...     sample_rate=16000, win_length=25, hop_length=10, n_fft=400
... )
>>> features = compute_STFT(signal)
>>> features = spectral_magnitude(features)
>>> compute_fbanks = Filterbank(n_mels=40)
>>> features = compute_fbanks(features)
>>> compute_mfccs = DCT(input_size=40, n_out=20)
>>> features = compute_mfccs(features)
>>> compute_deltas = Deltas(input_size=20)
>>> delta1 = compute_deltas(features)
>>> delta2 = compute_deltas(delta1)
>>> features = torch.cat([features, delta1, delta2], dim=2)
>>> compute_cw = ContextWindow(left_frames=5, right_frames=5)
>>> features = compute_cw(features)
>>> norm = InputNormalization()
>>> features = norm(features, torch.tensor([1]).float())

Authors
 * Mirco Ravanelli 2020
 * Peter Plantinga 2025
 * Rogier van Dalen 2025
"""

import math
from typing import Optional, Tuple, Union

import torch
from torch.distributed import ReduceOp

from speechbrain.utils.checkpoints import (
    mark_as_loader,
    mark_as_saver,
    mark_as_transfer,
    register_checkpoint_hooks,
)
from speechbrain.utils.distributed import ddp_all_reduce
from speechbrain.utils.filter_analysis import FilterProperties
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class STFT(torch.nn.Module):
    """computes the Short-Term Fourier Transform (STFT).

    This class computes the Short-Term Fourier Transform of an audio signal.
    It supports multi-channel audio inputs (batch, time, channels).

    Arguments
    ---------
    sample_rate : int
        Sample rate of the input audio signal (e.g 16000).
    win_length : float
        Length (in ms) of the sliding window used to compute the STFT.
    hop_length : float
        Length (in ms) of the hope of the sliding window used to compute
        the STFT.
    n_fft : int
        Number of fft point of the STFT. It defines the frequency resolution
        (n_fft should be <= than win_len).
    window_fn : function
        A function that takes an integer (number of samples) and outputs a
        tensor to be multiplied with each window before fft.
    normalized_stft : bool
        If True, the function returns the  normalized STFT results,
        i.e., multiplied by win_length^-0.5 (default is False).
    center : bool
        If True (default), the input will be padded on both sides so that the
        t-th frame is centered at time t×hop_length. Otherwise, the t-th frame
        begins at time t×hop_length.
    pad_mode : str
        It can be 'constant','reflect','replicate', 'circular', 'reflect'
        (default). 'constant' pads the input tensor boundaries with a
        constant value. 'reflect' pads the input tensor using the reflection
        of the input boundary. 'replicate' pads the input tensor using
        replication of the input boundary. 'circular' pads using  circular
        replication.
    onesided : True
        If True (default) only returns nfft/2 values. Note that the other
        samples are redundant due to the Fourier transform conjugate symmetry.

    Example
    -------
    >>> import torch
    >>> compute_STFT = STFT(
    ...     sample_rate=16000, win_length=25, hop_length=10, n_fft=400
    ... )
    >>> inputs = torch.randn([10, 16000])
    >>> features = compute_STFT(inputs)
    >>> features.shape
    torch.Size([10, 101, 201, 2])
    """

    def __init__(
        self,
        sample_rate,
        win_length=25,
        hop_length=10,
        n_fft=400,
        window_fn=torch.hamming_window,
        normalized_stft=False,
        center=True,
        pad_mode="constant",
        onesided=True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.normalized_stft = normalized_stft
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided

        # Convert win_length and hop_length from ms to samples
        self.win_length = int(
            round((self.sample_rate / 1000.0) * self.win_length)
        )
        self.hop_length = int(
            round((self.sample_rate / 1000.0) * self.hop_length)
        )

        self.window = window_fn(self.win_length)

    def forward(self, x):
        """Returns the STFT generated from the input waveforms.

        Arguments
        ---------
        x : torch.Tensor
            A batch of audio signals to transform.

        Returns
        -------
        stft : torch.Tensor
        """
        # Managing multi-channel stft
        or_shape = x.shape
        if len(or_shape) == 3:
            x = x.transpose(1, 2)
            x = x.reshape(or_shape[0] * or_shape[2], or_shape[1])

        stft = torch.stft(
            x,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window.to(x.device),
            self.center,
            self.pad_mode,
            self.normalized_stft,
            self.onesided,
            return_complex=True,
        )

        stft = torch.view_as_real(stft)

        # Retrieving the original dimensionality (batch,time, channels)
        if len(or_shape) == 3:
            stft = stft.reshape(
                or_shape[0],
                or_shape[2],
                stft.shape[1],
                stft.shape[2],
                stft.shape[3],
            )
            stft = stft.permute(0, 3, 2, 4, 1)
        else:
            # (batch, time, channels)
            stft = stft.transpose(2, 1)

        return stft

    def get_filter_properties(self) -> FilterProperties:
        if not self.center:
            raise ValueError(
                "ValueProperties cannot model a non-centered STFT, as it "
                "assumes either centering or causality"
            )

        return FilterProperties(
            window_size=self.win_length, stride=self.hop_length
        )


class ISTFT(torch.nn.Module):
    """Computes the Inverse Short-Term Fourier Transform (ISTFT)

    This class computes the Inverse Short-Term Fourier Transform of
    an audio signal. It supports multi-channel audio inputs
    (batch, time_step, n_fft, 2, n_channels [optional]).

    Arguments
    ---------
    sample_rate : int
        Sample rate of the input audio signal (e.g. 16000).
    n_fft : int
        Number of points in FFT.
    win_length : float
        Length (in ms) of the sliding window used when computing the STFT.
    hop_length : float
        Length (in ms) of the hope of the sliding window used when computing
        the STFT.
    window_fn : function
        A function that takes an integer (number of samples) and outputs a
        tensor to be used as a window for ifft.
    normalized_stft : bool
        If True, the function assumes that it's working with the normalized
        STFT results. (default is False)
    center : bool
        If True (default), the function assumes that the STFT result was padded
        on both sides.
    onesided : True
        If True (default), the function assumes that there are n_fft/2 values
        for each time frame of the STFT.
    epsilon : float
        A small value to avoid division by 0 when normalizing by the sum of the
        squared window. Playing with it can fix some abnormalities at the
        beginning and at the end of the reconstructed signal. The default value
        of epsilon is 1e-12.

    Example
    -------
    >>> import torch
    >>> compute_STFT = STFT(
    ...     sample_rate=16000, win_length=25, hop_length=10, n_fft=400
    ... )
    >>> compute_ISTFT = ISTFT(sample_rate=16000, win_length=25, hop_length=10)
    >>> inputs = torch.randn([10, 16000])
    >>> outputs = compute_ISTFT(compute_STFT(inputs))
    >>> outputs.shape
    torch.Size([10, 16000])
    """

    def __init__(
        self,
        sample_rate,
        n_fft=None,
        win_length=25,
        hop_length=10,
        window_fn=torch.hamming_window,
        normalized_stft=False,
        center=True,
        onesided=True,
        epsilon=1e-12,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.normalized_stft = normalized_stft
        self.center = center
        self.onesided = onesided
        self.epsilon = epsilon

        # Convert win_length and hop_length from ms to samples
        self.win_length = int(
            round((self.sample_rate / 1000.0) * self.win_length)
        )
        self.hop_length = int(
            round((self.sample_rate / 1000.0) * self.hop_length)
        )

        # Create window using provided function
        self.window = window_fn(self.win_length)

    def forward(self, x, sig_length=None):
        """Returns the ISTFT generated from the input signal.

        Arguments
        ---------
        x : torch.Tensor
            A batch of audio signals in the frequency domain to transform.
        sig_length : int
            The length of the output signal in number of samples. If not
            specified will be equal to: (time_step - 1) * hop_length + n_fft

        Returns
        -------
        istft : torch.Tensor
        """
        or_shape = x.shape

        # Infer n_fft if not provided
        if self.n_fft is None and self.onesided:
            n_fft = (x.shape[2] - 1) * 2
        elif self.n_fft is None and not self.onesided:
            n_fft = x.shape[2]
        else:
            n_fft = self.n_fft

        # Changing the format for (batch, time_step, n_fft, 2, n_channels)
        if len(or_shape) == 5:
            x = x.permute(0, 4, 2, 1, 3)

            # Lumping batch and channel dimension, because torch.istft
            # doesn't support batching.
            x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        elif len(or_shape) == 4:
            x = x.permute(0, 2, 1, 3)

        # isft ask complex input
        x = torch.complex(x[..., 0], x[..., 1])

        istft = torch.istft(
            input=x,
            n_fft=n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(x.device),
            center=self.center,
            onesided=self.onesided,
            length=sig_length,
        )

        # Convert back to (time, time_step, n_channels)
        if len(or_shape) == 5:
            istft = istft.reshape(or_shape[0], or_shape[4], -1)
            istft = istft.transpose(1, 2)

        return istft


def spectral_magnitude(
    stft, power: float = 1, log: bool = False, eps: float = 1e-14
):
    """Returns the magnitude of a complex spectrogram.

    Arguments
    ---------
    stft : torch.Tensor
        A tensor, output from the stft function.
    power : int
        What power to use in computing the magnitude.
        Use power=1 for the power spectrogram.
        Use power=0.5 for the magnitude spectrogram.
    log : bool
        Whether to apply log to the spectral features.
    eps : float
        A small value to prevent square root of zero.

    Returns
    -------
    spectr : torch.Tensor

    Example
    -------
    >>> a = torch.Tensor([[3, 4]])
    >>> spectral_magnitude(a, power=0.5)
    tensor([5.])
    """
    spectr = stft.pow(2).sum(-1)

    # Add eps avoids NaN when spectr is zero
    if power < 1:
        spectr = spectr + eps
    spectr = spectr.pow(power)

    if log:
        return torch.log(spectr + eps)
    return spectr


class Filterbank(torch.nn.Module):
    """computes filter bank (FBANK) features given spectral magnitudes.

    Arguments
    ---------
    n_mels : float
        Number of Mel filters used to average the spectrogram.
    log_mel : bool
        If True, it computes the log of the FBANKs.
    filter_shape : str
        Shape of the filters ('triangular', 'rectangular', 'gaussian').
    f_min : int
        Lowest frequency for the Mel filters.
    f_max : int
        Highest frequency for the Mel filters.
    n_fft : int
        Number of fft points of the STFT. It defines the frequency resolution
        (n_fft should be<= than win_len).
    sample_rate : int
        Sample rate of the input audio signal (e.g, 16000)
    power_spectrogram : float
        Exponent used for spectrogram computation.
    amin : float
        Minimum amplitude (used for numerical stability).
    ref_value : float
        Reference value used for the dB scale.
    top_db : float
        Minimum negative cut-off in decibels.
    param_change_factor : bool
        If freeze=False, this parameter affects the speed at which the filter
        parameters (i.e., central_freqs and bands) can be changed.  When high
        (e.g., param_change_factor=1) the filters change a lot during training.
        When low (e.g. param_change_factor=0.1) the filter parameters are more
        stable during training
    param_rand_factor : float
        This parameter can be used to randomly change the filter parameters
        (i.e, central frequencies and bands) during training.  It is thus a
        sort of regularization. param_rand_factor=0 does not affect, while
        param_rand_factor=0.15 allows random variations within +-15% of the
        standard values of the filter parameters (e.g., if the central freq
        is 100 Hz, we can randomly change it from 85 Hz to 115 Hz).
    freeze : bool
        If False, it the central frequency and the band of each filter are
        added into nn.parameters. If True, the standard frozen features
        are computed.

    Example
    -------
    >>> import torch
    >>> compute_fbanks = Filterbank()
    >>> inputs = torch.randn([10, 101, 201])
    >>> features = compute_fbanks(inputs)
    >>> features.shape
    torch.Size([10, 101, 40])
    """

    def __init__(
        self,
        n_mels=40,
        log_mel=True,
        filter_shape="triangular",
        f_min=0,
        f_max=8000,
        n_fft=400,
        sample_rate=16000,
        power_spectrogram=2,
        amin=1e-10,
        ref_value=1.0,
        top_db=80.0,
        param_change_factor=1.0,
        param_rand_factor=0.0,
        freeze=True,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.log_mel = log_mel
        self.filter_shape = filter_shape
        self.f_min = f_min
        self.f_max = f_max
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.power_spectrogram = power_spectrogram
        self.amin = amin
        self.ref_value = ref_value
        self.top_db = top_db
        self.freeze = freeze
        self.n_stft = self.n_fft // 2 + 1
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))
        self.device_inp = torch.device("cpu")
        self.param_change_factor = param_change_factor
        self.param_rand_factor = param_rand_factor

        if self.power_spectrogram == 2:
            self.multiplier = 10
        else:
            self.multiplier = 20

        # Make sure f_min < f_max
        if self.f_min >= self.f_max:
            err_msg = "Require f_min: %f < f_max: %f" % (
                self.f_min,
                self.f_max,
            )
            logger.error(err_msg, exc_info=True)

        # Filter definition
        mel = torch.linspace(
            self._to_mel(self.f_min), self._to_mel(self.f_max), self.n_mels + 2
        )
        hz = self._to_hz(mel)

        # Computation of the filter bands
        band = hz[1:] - hz[:-1]
        self.band = band[:-1]
        self.f_central = hz[1:-1]

        # Adding the central frequency and the band to the list of nn param
        if not self.freeze:
            self.f_central = torch.nn.Parameter(
                self.f_central / (self.sample_rate * self.param_change_factor)
            )
            self.band = torch.nn.Parameter(
                self.band / (self.sample_rate * self.param_change_factor)
            )

        # Frequency axis
        all_freqs = torch.linspace(0, self.sample_rate // 2, self.n_stft)

        # Replicating for all the filters
        self.all_freqs_mat = all_freqs.repeat(self.f_central.shape[0], 1)

    def forward(self, spectrogram):
        """Returns the FBANks.

        Arguments
        ---------
        spectrogram : torch.Tensor
            A batch of spectrogram tensors.

        Returns
        -------
        fbanks : torch.Tensor
        """
        # Computing central frequency and bandwidth of each filter
        f_central_mat = self.f_central.repeat(
            self.all_freqs_mat.shape[1], 1
        ).transpose(0, 1)
        band_mat = self.band.repeat(self.all_freqs_mat.shape[1], 1).transpose(
            0, 1
        )

        # Uncomment to print filter parameters
        # print(self.f_central*self.sample_rate * self.param_change_factor)
        # print(self.band*self.sample_rate* self.param_change_factor)

        # Creation of the multiplication matrix. It is used to create
        # the filters that average the computed spectrogram.
        if not self.freeze:
            f_central_mat = f_central_mat * (
                self.sample_rate
                * self.param_change_factor
                * self.param_change_factor
            )
            band_mat = band_mat * (
                self.sample_rate
                * self.param_change_factor
                * self.param_change_factor
            )

        # Regularization with random changes of filter central frequency and band
        elif self.param_rand_factor != 0 and self.training:
            rand_change = (
                1.0
                + torch.rand(2) * 2 * self.param_rand_factor
                - self.param_rand_factor
            )
            f_central_mat = f_central_mat * rand_change[0]
            band_mat = band_mat * rand_change[1]

        fbank_matrix = self._create_fbank_matrix(f_central_mat, band_mat).to(
            spectrogram.device
        )

        sp_shape = spectrogram.shape

        # Managing multi-channels case (batch, time, channels)
        if len(sp_shape) == 4:
            spectrogram = spectrogram.permute(0, 3, 1, 2)
            spectrogram = spectrogram.reshape(
                sp_shape[0] * sp_shape[3], sp_shape[1], sp_shape[2]
            )

        # FBANK computation
        fbanks = torch.matmul(spectrogram, fbank_matrix)
        if self.log_mel:
            fbanks = self._amplitude_to_DB(fbanks)

        # Reshaping in the case of multi-channel inputs
        if len(sp_shape) == 4:
            fb_shape = fbanks.shape
            fbanks = fbanks.reshape(
                sp_shape[0], sp_shape[3], fb_shape[1], fb_shape[2]
            )
            fbanks = fbanks.permute(0, 2, 3, 1)

        return fbanks

    @staticmethod
    def _to_mel(hz):
        """Returns mel-frequency value corresponding to the input
        frequency value in Hz.

        Arguments
        ---------
        hz : float
            The frequency point in Hz.

        Returns
        -------
        The mel-frequency value
        """
        return 2595 * math.log10(1 + hz / 700)

    @staticmethod
    def _to_hz(mel):
        """Returns hz-frequency value corresponding to the input
        mel-frequency value.

        Arguments
        ---------
        mel : float
            The frequency point in the mel-scale.

        Returns
        -------
        The hz-frequency value
        """
        return 700 * (10 ** (mel / 2595) - 1)

    def _triangular_filters(self, all_freqs, f_central, band):
        """Returns fbank matrix using triangular filters.

        Arguments
        ---------
        all_freqs : torch.Tensor
            torch.Tensor gathering all the frequency points.
        f_central : torch.Tensor
            torch.Tensor gathering central frequencies of each filter.
        band : torch.Tensor
            torch.Tensor gathering the bands of each filter.

        Returns
        -------
        fbank_matrix : torch.Tensor
        """
        # Computing the slops of the filters
        slope = (all_freqs - f_central) / band
        left_side = slope + 1.0
        right_side = -slope + 1.0

        # Adding zeros for negative values
        zero = torch.zeros(1, device=self.device_inp)
        fbank_matrix = torch.max(
            zero, torch.min(left_side, right_side)
        ).transpose(0, 1)

        return fbank_matrix

    def _rectangular_filters(self, all_freqs, f_central, band):
        """Returns fbank matrix using rectangular filters.

        Arguments
        ---------
        all_freqs : torch.Tensor
            torch.Tensor gathering all the frequency points.
        f_central : torch.Tensor
            torch.Tensor gathering central frequencies of each filter.
        band : torch.Tensor
            torch.Tensor gathering the bands of each filter.

        Returns
        -------
        fbank_matrix : torch.Tensor
        """
        # cut-off frequencies of the filters
        low_hz = f_central - band
        high_hz = f_central + band

        # Left/right parts of the filter
        left_side = right_size = all_freqs.ge(low_hz)
        right_size = all_freqs.le(high_hz)

        fbank_matrix = (left_side * right_size).float().transpose(0, 1)

        return fbank_matrix

    def _gaussian_filters(
        self, all_freqs, f_central, band, smooth_factor=torch.tensor(2)
    ):
        """Returns fbank matrix using gaussian filters.

        Arguments
        ---------
        all_freqs : torch.Tensor
            torch.Tensor gathering all the frequency points.
        f_central : torch.Tensor
            torch.Tensor gathering central frequencies of each filter.
        band : torch.Tensor
            torch.Tensor gathering the bands of each filter.
        smooth_factor: torch.Tensor
            Smoothing factor of the gaussian filter. It can be used to employ
            sharper or flatter filters.

        Returns
        -------
        fbank_matrix : torch.Tensor
        """
        fbank_matrix = torch.exp(
            -0.5 * ((all_freqs - f_central) / (band / smooth_factor)) ** 2
        ).transpose(0, 1)

        return fbank_matrix

    def _create_fbank_matrix(self, f_central_mat, band_mat):
        """Returns fbank matrix to use for averaging the spectrum with
           the set of filter-banks.

        Arguments
        ---------
        f_central_mat : torch.Tensor
            torch.Tensor gathering central frequencies of each filter.
        band_mat : torch.Tensor
            torch.Tensor gathering the bands of each filter.

        Returns
        -------
        fbank_matrix : torch.Tensor
        """
        if self.filter_shape == "triangular":
            fbank_matrix = self._triangular_filters(
                self.all_freqs_mat, f_central_mat, band_mat
            )

        elif self.filter_shape == "rectangular":
            fbank_matrix = self._rectangular_filters(
                self.all_freqs_mat, f_central_mat, band_mat
            )

        else:
            fbank_matrix = self._gaussian_filters(
                self.all_freqs_mat, f_central_mat, band_mat
            )

        return fbank_matrix

    def _amplitude_to_DB(self, x):
        """Converts  linear-FBANKs to log-FBANKs.

        Arguments
        ---------
        x : torch.Tensor
            A batch of linear FBANK tensors.

        Returns
        -------
        x_db : torch.Tensor
        """
        x_db = self.multiplier * torch.log10(torch.clamp(x, min=self.amin))
        x_db -= self.multiplier * self.db_multiplier

        # Setting up dB max. It is the max over time and frequency,
        # Hence, of a whole sequence (sequence-dependent)
        new_x_db_max = x_db.amax(dim=(-2, -1)) - self.top_db

        # Clipping to dB max. The view is necessary as only a scalar is obtained
        # per sequence.
        x_db = torch.max(x_db, new_x_db_max.view(x_db.shape[0], 1, 1))

        return x_db


class DCT(torch.nn.Module):
    """Computes the discrete cosine transform.

    This class is primarily used to compute MFCC features of an audio signal
    given a set of FBANK features as input.

    Arguments
    ---------
    input_size : int
        Expected size of the last dimension in the input.
    n_out : int
        Number of output coefficients.
    ortho_norm : bool
        Whether to use orthogonal norm.

    Example
    -------
    >>> import torch
    >>> inputs = torch.randn([10, 101, 40])
    >>> compute_mfccs = DCT(input_size=inputs.size(-1))
    >>> features = compute_mfccs(inputs)
    >>> features.shape
    torch.Size([10, 101, 20])
    """

    def __init__(self, input_size, n_out=20, ortho_norm=True):
        super().__init__()

        if n_out > input_size:
            raise ValueError(
                "Cannot select more DCT coefficients than inputs "
                "(n_out=%i, n_in=%i)" % (n_out, input_size)
            )

        # Generate matrix for DCT transformation
        n = torch.arange(float(input_size))
        k = torch.arange(float(n_out)).unsqueeze(1)
        dct = torch.cos(math.pi / float(input_size) * (n + 0.5) * k)

        if ortho_norm:
            dct[0] *= 1.0 / math.sqrt(2.0)
            dct *= math.sqrt(2.0 / float(input_size))
        else:
            dct *= 2.0

        self.dct_mat = dct.t()

    def forward(self, x):
        """Returns the DCT of the input tensor.

        Arguments
        ---------
        x : torch.Tensor
            A batch of tensors to transform, usually fbank features.

        Returns
        -------
        dct : torch.Tensor
        """
        # Managing multi-channels case
        input_shape = x.shape
        if len(input_shape) == 4:
            x = x.reshape(x.shape[0] * x.shape[3], x.shape[1], x.shape[2])

        # apply the DCT transform
        dct = torch.matmul(x, self.dct_mat.to(x.device))

        # Reshape in the case of multi-channels
        if len(input_shape) == 4:
            dct = dct.reshape(
                input_shape[0], dct.shape[1], dct.shape[2], input_shape[3]
            )

        return dct


class Deltas(torch.nn.Module):
    """Computes delta coefficients (time derivatives).

    Arguments
    ---------
    input_size : int
        The expected size of the inputs for parameter initialization.
    window_length : int
        Length of the window used to compute the time derivatives.

    Example
    -------
    >>> inputs = torch.randn([10, 101, 20])
    >>> compute_deltas = Deltas(input_size=inputs.size(-1))
    >>> features = compute_deltas(inputs)
    >>> features.shape
    torch.Size([10, 101, 20])
    """

    def __init__(self, input_size, window_length=5):
        super().__init__()
        self.n = (window_length - 1) // 2
        self.denom = self.n * (self.n + 1) * (2 * self.n + 1) / 3

        self.register_buffer(
            "kernel",
            torch.arange(
                -self.n,
                self.n + 1,
                dtype=torch.float32,
            ).repeat(input_size, 1, 1),
        )

    def forward(self, x):
        """Returns the delta coefficients.

        Arguments
        ---------
        x : torch.Tensor
            A batch of tensors.

        Returns
        -------
        delta_coeff : torch.Tensor
        """
        # Managing multi-channel deltas reshape tensor (batch*channel,time)
        x = x.transpose(1, 2).transpose(2, -1)
        or_shape = x.shape
        if len(or_shape) == 4:
            x = x.reshape(or_shape[0] * or_shape[2], or_shape[1], or_shape[3])

        # Padding for time borders
        x = torch.nn.functional.pad(x, (self.n, self.n), mode="replicate")

        # Derivative estimation (with a fixed convolutional kernel)
        delta_coeff = (
            torch.nn.functional.conv1d(
                x, self.kernel.to(x.device), groups=x.shape[1]
            )
            / self.denom
        )

        # Retrieving the original dimensionality (for multi-channel case)
        if len(or_shape) == 4:
            delta_coeff = delta_coeff.reshape(
                or_shape[0], or_shape[1], or_shape[2], or_shape[3]
            )
        delta_coeff = delta_coeff.transpose(1, -1).transpose(2, -1)

        return delta_coeff


class ContextWindow(torch.nn.Module):
    """Computes the context window.

    This class applies a context window by gathering multiple time steps
    in a single feature vector. The operation is performed with a
    convolutional layer based on a fixed kernel designed for that.

    Arguments
    ---------
    left_frames : int
         Number of left frames (i.e, past frames) to collect.
    right_frames : int
        Number of right frames (i.e, future frames) to collect.

    Example
    -------
    >>> import torch
    >>> compute_cw = ContextWindow(left_frames=5, right_frames=5)
    >>> inputs = torch.randn([10, 101, 20])
    >>> features = compute_cw(inputs)
    >>> features.shape
    torch.Size([10, 101, 220])
    """

    def __init__(self, left_frames=0, right_frames=0):
        super().__init__()
        self.left_frames = left_frames
        self.right_frames = right_frames
        self.context_len = self.left_frames + self.right_frames + 1
        self.kernel_len = 2 * max(self.left_frames, self.right_frames) + 1

        # Kernel definition
        self.kernel = torch.eye(self.context_len, self.kernel_len)

        if self.right_frames > self.left_frames:
            lag = self.right_frames - self.left_frames
            self.kernel = torch.roll(self.kernel, lag, 1)

        self.first_call = True

    def forward(self, x):
        """Returns the tensor with the surrounding context.

        Arguments
        ---------
        x : torch.Tensor
            A batch of tensors.

        Returns
        -------
        cw_x : torch.Tensor
            The context-enriched tensor
        """
        x = x.transpose(1, 2)

        if self.first_call is True:
            self.first_call = False
            self.kernel = (
                self.kernel.repeat(x.shape[1], 1, 1)
                .view(x.shape[1] * self.context_len, self.kernel_len)
                .unsqueeze(1)
            )

        # Managing multi-channel case
        or_shape = x.shape
        if len(or_shape) == 4:
            x = x.reshape(or_shape[0] * or_shape[2], or_shape[1], or_shape[3])

        # Compute context (using the estimated convolutional kernel)
        cw_x = torch.nn.functional.conv1d(
            x,
            self.kernel.to(x.device),
            groups=x.shape[1],
            padding=max(self.left_frames, self.right_frames),
        )

        # Retrieving the original dimensionality (for multi-channel case)
        if len(or_shape) == 4:
            cw_x = cw_x.reshape(
                or_shape[0], cw_x.shape[1], or_shape[2], cw_x.shape[-1]
            )

        cw_x = cw_x.transpose(1, 2)

        return cw_x


def gaussian_statistics(
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dim: Union[int, tuple, None] = None,
):
    """
    Compute first- and second-order moments of data, and return them as the
    count, mean, and variance of a vector over one or more dimensions.

    Arguments
    ---------
    x: torch.Tensor
        The tensor to compute the statistics over.
    mask: torch.Tensor
        Padding mask to exclude padding from the statistics computation.
        For dimensions in `dim`, the mask size should exactly match `x`.
        All dimensions other than `dim` should be ones (e.g. [B, T, 1, ...])
        Ones / trues are valid positions, and zeros / falses are padding positions.
    dim: int | tuple | None
        The dimension or dimensions that the statistics should be computed over.
        The other dimensions are retained in the output.
        If None, then scalar-valued statistics will be returned.

    Returns
    -------
    count: int
        The number of values in the statistics computation, without padding
        this is just the product of the lengths of the dimensions in `dim`.
    mean: torch.Tensor
        The mean of the non-padding values over the dimensions in `dim`.
    variance: torch.Tensor
        The (biased) variance of the non-padding values over `dim`.

    Example
    -------
    >>> x = torch.tensor([[1.0, 3.0, 0.0]])
    >>> mask = torch.tensor([[True, True, False]])
    >>> dim = (0, 1)
    >>> count, mean, variance = gaussian_statistics(x, mask, dim)
    >>> count
    2
    >>> mean
    tensor(2.)
    >>> variance
    tensor(1.)
    """

    def normalise_dimensions(
        x: torch.Tensor, dim: Union[int, tuple, None]
    ) -> Tuple[tuple, tuple]:
        """Normalise "dim" and return (reduce_dimensions, keep_dimensions)."""
        all_dimensions = range(len(x.shape))
        if dim is None or dim == ():
            # dim == () is an exceptional case and replicates the strangeness
            # of torch.sum(.., dim=()) and friends.
            return (tuple(d for d in all_dimensions), ())
        elif isinstance(dim, int):
            return ((dim,), tuple(d for d in all_dimensions if d != dim))
        else:
            assert isinstance(dim, tuple)
            return (dim, tuple(d for d in all_dimensions if d not in dim))

    (reduce_dimensions, keep_dimensions) = normalise_dimensions(x, dim)

    # Check that the mask is shaped correctly.
    if mask is not None:
        assert len(mask.shape) == len(x.shape)
        for d in reduce_dimensions:
            assert mask.size(d) == x.size(d)
        for d in keep_dimensions:
            assert mask.size(d) == 1

    if mask is None:
        number = math.prod(x.size(d) for d in reduce_dimensions)
    else:
        number = int(torch.sum(mask))

    masked_data = x if mask is None else mask * x

    # First keep the dimensions so that broadcasting works.
    # If number == 0, the following will generate a warning, as it should.
    mean_with_dims = (
        torch.sum(masked_data, dim=reduce_dimensions, keepdim=True) / number
    )
    mean = torch.squeeze(mean_with_dims, dim=reduce_dimensions)

    central_squared_data = torch.square(x - mean_with_dims)
    masked_squared_data = (
        central_squared_data if mask is None else mask * central_squared_data
    )
    variance = torch.sum(masked_squared_data, dim=reduce_dimensions) / number

    return (number, mean, variance)


def combine_gaussian_statistics(
    left_statistics: Tuple[int, torch.Tensor, Optional[torch.Tensor]],
    right_statistics: Tuple[int, torch.Tensor, Optional[torch.Tensor]],
):
    """
    Combine the first- and second-order moments from two pieces of data.
    The data and the result is in the form (count, mean, variance).
    The result is the mean and variance as if they have been computed on the
    concatenation of the data for left_statistics and the data for
    right_statistics.

    Arguments
    ---------
    left_statistics: Tuple[int, torch.Tensor, Optional[torch.Tensor]]
        One set of gaussian stats: count, mean, variance
    right_statistics: Tuple[int, torch.Tensor, Optional[torch.Tensor]]
        Another set of gaussian stats: count, mean, variance

    Returns
    -------
    count
        The total number of elements in the data.
    mean
        The combined mean.
    variance
        The combined variance, relative to the new mean.
        Returns None if either statistics set has variance of None
    """
    left_count, left_mean, left_variance = left_statistics
    right_count, right_mean, right_variance = right_statistics
    assert left_mean.shape == right_mean.shape
    assert left_mean.shape == left_variance.shape
    assert left_variance.shape == right_variance.shape

    count = left_count + right_count

    left_weight = left_count / count
    right_weight = right_count / count

    mean = left_weight * left_mean + right_weight * right_mean

    # Reconstruct the left and right variances relative to "mean".
    compensated_left_variance = left_variance + torch.square(mean - left_mean)
    compensated_right_variance = right_variance + torch.square(
        mean - right_mean
    )

    variance = (
        left_weight * compensated_left_variance
        + right_weight * compensated_right_variance
    )

    return count, mean, variance


def combine_gaussian_statistics_distributed(
    statistics: Tuple[int, torch.Tensor, torch.Tensor],
):
    """
    Combine the first- and second-order moments from multiple pieces of data
    using torch.distributed.
    The data and the result is in the form (count, mean, variance).
    The result is the mean and variance as if they have been computed on the
    concatenation of the data for statistics for all parallel processes.

    Arguments
    ---------
    statistics: Tuple[int, torch.Tensor, torch.Tensor]
        A set of gaussian statistics to reduce across all processes.
        The three elements of the tuple represent the count, mean, and variance.

    Returns
    -------
    count
        The total number of elements in the data across processes.
    mean
        The combined mean.
    variance
        The combined variance, relative to the new mean.
    """
    # This is the DDP version of combine_gaussian_statistics above.
    local_count, local_mean, local_variance = statistics
    global_count = ddp_all_reduce(
        torch.tensor(local_count, device=local_mean.device), ReduceOp.SUM
    )
    global_count = global_count.item()

    local_weight = local_count / global_count
    global_mean = ddp_all_reduce(local_weight * local_mean, ReduceOp.SUM)

    compensated_local_variance = local_variance + torch.square(
        local_mean - global_mean
    )
    global_variance = ddp_all_reduce(
        local_weight * compensated_local_variance, ReduceOp.SUM
    )

    return (global_count, global_mean, global_variance)


def mean_std_update(
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: Union[int, tuple, None],
    run_count: int,
    run_mean: torch.Tensor,
    run_std: torch.Tensor,
):
    """Update the mean and variance statistics run_mean and run_std that
    have been computed on run_count samples to integrate the new samples x.

    WARNING: Must be called in sync across processes.

    Arguments
    ---------
    x : torch.Tensor
        The new values to add to the running stats.
    mask : torch.Tensor
        Padding mask to exclude padding from the statistics computation.
        All dimensions other than batch and time should be ones (e.g. [B, T, 1, ...])
        Ones / trues are valid positions, and zeros / falses are padding positions.
    dim : tuple or int
        The dimension or dimensions to reduce (e.g. 1 for length).
    run_count : float or torch.Tensor
        The running number of samples seen so far.
    run_mean : float or torch.Tensor
        The running mean of samples seen so far.
    run_std : float or torch.Tensor
        The running standard deviations from the mean.

    Returns
    -------
    new_run_count : torch.Tensor
        Updated count all samples, now including x.
    new_run_mean : torch.Tensor
        Updated running mean of all samples, now including x.
    new_run_std : torch.Tensor
        Updated running standard deviations of all samples, now including x.

    Example
    -------
    >>> input_tensor = torch.tensor([[-1.0, 0.0, 1.0, 0.0]])
    >>> input_length = torch.tensor([0.75])
    >>> input_length_dim = 1
    >>> input_mask = make_padding_mask(
    ...     input_tensor, input_length, input_length_dim
    ... )
    >>> dim = (0, input_length_dim)
    >>> run_count, run_mean, run_std = 0, torch.tensor(0.0), torch.tensor(1.0)
    >>> run_count, run_mean, run_std = mean_std_update(
    ...     input_tensor, input_mask, dim, run_count, run_mean, run_std
    ... )
    >>> run_count
    3
    >>> run_mean
    tensor(0.)
    >>> run_std
    tensor(0.8165)
    """

    new_statistics = combine_gaussian_statistics_distributed(
        gaussian_statistics(x, mask=mask, dim=dim)
    )

    current_statistics = (run_count, run_mean, run_std.square())
    (count, mean, variance) = combine_gaussian_statistics(
        current_statistics, new_statistics
    )

    return count, mean, variance.sqrt()


@register_checkpoint_hooks
class InputNormalization(torch.nn.Module):
    """Performs mean and variance normalization over the time and possibly
    the (global) batch dimension of the input.

    When the default norm_type of "global" is used, running mean and variance
    statistics are computed and stored incorporating all the samples seen.

    WARNING: at first, the running statistics do not represent the "true" mean
    and variance, but are estimates based on the data seen so far. Once enough
    data has been seen, the stats should closely approximate the "true" values.

    WARNING: Using global normalization, the first call of `forward()` will
    throw an error if no updates have been performed (including the current batch),
    i.e. on first call the `epoch >= update_until_epoch` or the module
    is first called in `.eval()` mode.

    Arguments
    ---------
    mean_norm : bool, default True
        If True, the mean will be normalized. Passing `False` is deprecated.
    std_norm : bool, default True
        If True, the variance will be normalized.
    norm_type : str, default "global"
        String parameter whose value defines how the statistics are computed:
         * 'sentence' computes norms per utterance (no running stats)
         * 'batch' computes norms per input tensor (no running stats)
         * 'global' computes norms over all inputs (single mean, variance)
         * 'speaker' - DEPRECATED
    avg_factor : float, optional
        Passing avg_factor is DEPRECATED as this exactly matches the
        behavior of BatchNorm. To maintain this behavior, use
        `speechbrain.nnet.normalization.BatchNorm1d(momentum=avg_factor)`.
    length_dim : int, default 1
        The dimension for which to mask out the padding positions.
    update_until_epoch : int, default 2
        The epoch for which updates to the norm stats should stop.
        By default, stops after one epoch of updates, as when
        epoch == update_until_epoch then the updates stop immediately.
    avoid_padding_norm : bool, default False
        Regardless of the value passed here, padding is ignored for statistics
        computation. However, if False is passed for `avoid_padding_norm`, padding
        will get normalized along with the rest of the input tensor. If True,
        the padding will not be affected by this normalization operation.
    epsilon : float, default 1e-10
        A small value to improve the numerical stability of the variance.
    device : str or torch.device
        The device on which to create the global statistics. Can be changed
        later with `.to(device)`.

    Example
    -------
    >>> import torch
    >>> inputs = torch.arange(9).view(3, 3).float()
    >>> inputs
    tensor([[0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]])
    >>> input_lens = torch.ones(3)
    >>> norm = InputNormalization(norm_type="sentence")
    >>> features = norm(inputs, input_lens)
    >>> features
    tensor([[-1.2247,  0.0000,  1.2247],
            [-1.2247,  0.0000,  1.2247],
            [-1.2247,  0.0000,  1.2247]])
    >>> norm = InputNormalization(norm_type="batch")
    >>> features = norm(inputs, input_lens)
    >>> features
    tensor([[-1.5492, -1.1619, -0.7746],
            [-0.3873,  0.0000,  0.3873],
            [ 0.7746,  1.1619,  1.5492]])
    >>> norm = InputNormalization(norm_type="global")
    >>> features = norm(inputs, input_lens)
    >>> features.mean() < 1e-7
    tensor(True)
    >>> features = norm(inputs + 1, input_lens)
    >>> features.mean()
    tensor(0.1901)
    >>> features = norm(inputs, input_lens)
    >>> features.mean()
    tensor(-0.1270)
    >>> features = norm(inputs - 1, input_lens)
    >>> features.mean()
    tensor(-0.3735)
    >>> features = norm(inputs, input_lens)
    >>> features.mean() < 1e-7
    tensor(True)
    """

    from typing import Dict

    spk_dict_mean: Dict[int, torch.Tensor]
    spk_dict_std: Dict[int, torch.Tensor]
    spk_dict_count: Dict[int, int]
    NORM_TYPES = ("global", "batch", "sentence")

    def __init__(
        self,
        mean_norm=True,
        std_norm=True,
        norm_type="global",
        avg_factor=None,
        length_dim=1,
        update_until_epoch=2,
        avoid_padding_norm=False,
        epsilon=1e-10,
        device="cpu",
    ):
        super().__init__()

        # Validate and store input arguments
        if not mean_norm:
            raise ValueError("Passing `False` for `mean_norm` is deprecated.")
        if avg_factor is not None:
            raise ValueError(
                "Passing avg_factor is DEPRECATED as this exactly matches the "
                "behavior of BatchNorm. To maintain this behavior, use "
                "`speechbrain.nnet.normalization.BatchNorm1d(momentum=avg_factor)`."
            )
        if norm_type == "speaker":
            raise ValueError("per-speaker normalization is deprecated.")
        elif norm_type not in self.NORM_TYPES:
            raise ValueError(f"norm_type must be one of {self.NORM_TYPES}.")

        self.std_norm = std_norm
        self.norm_type = norm_type
        self.avoid_padding_norm = avoid_padding_norm
        self.epsilon = epsilon
        self.device = device
        self.length_dim = length_dim

        # Set a suitably huge epoch if None is passed
        self.update_until_epoch = update_until_epoch or torch.inf

        # Containers for running mean/variance calculation
        # These will be initialized based on the first input tensor
        self.glob_mean = torch.empty(0)
        self.glob_std = torch.empty(0)
        self.count = 0

    def forward(self, x, lengths=None, epoch=None):
        """Normalizes the input tensor, x, according to the `norm_type`.

        Excludes the padded portion of the tensor by using the passed relative lengths.
        Automatically updates running mean, variance if "global" or "speaker" norm is used.

        Arguments
        ---------
        x : torch.Tensor
            The input tensor to normalize.
        lengths : torch.Tensor, optional
            The relative length of each sentence (e.g, `[0.7, 0.9, 1.0]`), used
            to avoid computing stats on the padding part of the tensor.
        epoch : int, optional
            The current epoch count, used to stop updates to global stats after
            enough samples have been seen (e.g. one epoch).

        Returns
        -------
        x : torch.Tensor
            The normalized tensor.
        """
        # Padding mask is used to protect padding elements from updates
        mask = make_padding_mask(x, lengths, length_dim=1)

        # Global stats should be updated before performing normalization
        if self.norm_type == "global":
            if self._should_update(epoch):
                self._update_global_stats(x, mask)
            mean, std = self.glob_mean, self.glob_std

        # Local stats are computed over self.length_dim
        elif self.norm_type == "sentence":
            mean, std = self._compute_current_stats(x, mask, self.length_dim)
        elif self.norm_type == "batch":
            _, mean, var = gaussian_statistics(x, mask, (0, self.length_dim))
            std = var.clamp(min=self.epsilon).sqrt()

        if self.std_norm is False:
            std = torch.ones_like(mean)

        # Add back reduced dimensions (avoiding padding if needed)
        if self.norm_type in ["global", "batch"]:
            mean, std = mean.unsqueeze(0), std.unsqueeze(0)
        mean = mean.unsqueeze(self.length_dim)
        std = std.unsqueeze(self.length_dim)
        if self.avoid_padding_norm:
            mean = mean.masked_fill(~mask, 0.0)
            std = std.masked_fill(~mask, 1.0)

        # Normalize using collected stats and avoiding division by 0
        return (x - mean) / std.clamp(min=self.epsilon)

    def _should_update(self, epoch):
        """Whether to perform an update, based on epoch count."""
        still_training = epoch is None or epoch < self.update_until_epoch
        return still_training and self.training

    def _update_global_stats(self, x, mask):
        """Use input tensor to update global statistics."""
        dim = (0, self.length_dim)
        if self.count == 0:
            # Initialize with the mean, std of the first batch
            _, self.glob_mean, var = gaussian_statistics(x, mask, dim=dim)
            self.glob_std = var.clamp(min=self.epsilon).sqrt()

        self.count, self.glob_mean, self.glob_std = mean_std_update(
            x, mask, dim, self.count, self.glob_mean, self.glob_std
        )

    def _compute_current_stats(self, x, mask, dim):
        """Computes masked mean and std of an input tensor along the given dimension(s)."""
        n = mask.sum(dim, keepdim=True)
        mean = (x * mask).sum(dim, keepdim=True) / n
        if self.std_norm:
            var = ((x - mean) * mask).square().sum(dim, keepdim=True) / n
        else:
            var = torch.ones_like(mean)
        return mean.squeeze(dim), var.squeeze(dim).sqrt()

    def _statistics_dict(self):
        """Fills the dictionary containing the normalization statistics."""
        state = {}
        state["count"] = self.count
        state["glob_mean"] = self.glob_mean
        state["glob_std"] = self.glob_std

        return state

    def _load_statistics_dict(self, state):
        """Loads the dictionary containing the statistics.

        Arguments
        ---------
        state : dict
            A dictionary containing the normalization statistics.

        Returns
        -------
        state : dict
        """
        self.count = state["count"]
        self.glob_mean = state["glob_mean"]
        self.glob_std = state["glob_std"]

        return state

    def to(self, device):
        """Puts the needed tensors in the right device."""
        self.device = device
        self = super(InputNormalization, self).to(device)
        self.glob_mean = self.glob_mean.to(device)
        self.glob_std = self.glob_std.to(device)

        return self

    @mark_as_saver
    def _save(self, path):
        """Save statistic dictionary.

        Arguments
        ---------
        path : str
            A path where to save the dictionary.
        """
        stats = self._statistics_dict()
        torch.save(stats, path)

    @mark_as_transfer
    @mark_as_loader
    def _load(self, path, end_of_epoch=False):
        """Load statistic dictionary.

        Arguments
        ---------
        path : str
            The path of the statistic dictionary
        end_of_epoch : bool
            Whether this is the end of an epoch.
            Here for compatibility, but not used.
        """
        del end_of_epoch  # Unused here.
        stats = torch.load(path, map_location=self.device)
        self._load_statistics_dict(stats)


def make_padding_mask(x, lengths=None, length_dim=1, eps=1e-6):
    """Create a mask from relative lengths along a given dimension.

    Arguments
    ---------
    x : torch.Tensor
        The input tensor demonstrating the size of the target mask.
    lengths : torch.Tensor, optional
        The relative lengths of an input batch of utterances.
        If None, all positions are considered valid (i.e. mask is all `True`).
    length_dim : int, default 1
        The dimension for which the lengths indicate padded positions.
    eps : float, default 1e-8
        A small constant to avoid floating point errors in computation of
        the padding mask.

    Returns
    -------
    padding_mask : torch.Tensor
        A boolean tensor with `True` for valid positions and `False`
        for padding positions. The `padding_mask` can be multiplied with
        `x` via broadcasting, as all dimensions other than length and batch
        are singleton dimensions.

    Example
    -------
    >>> input_tensor = torch.arange(3 * 4 * 2).view(3, 4, 2)
    >>> lengths = torch.tensor([1.0, 0.75, 0.5])
    >>> mask = make_padding_mask(input_tensor, lengths)
    >>> mask.shape
    torch.Size([3, 4, 1])
    >>> input_tensor * mask
    tensor([[[ 0,  1],
             [ 2,  3],
             [ 4,  5],
             [ 6,  7]],
    <BLANKLINE>
            [[ 8,  9],
             [10, 11],
             [12, 13],
             [ 0,  0]],
    <BLANKLINE>
            [[16, 17],
             [18, 19],
             [ 0,  0],
             [ 0,  0]]])
    """
    if lengths is None:
        lengths = torch.ones(x.size(0), device=x.device)

    # Convert relative lengths to absolute lengths, then compute boolean mask
    max_len = x.size(length_dim)
    abs_lengths = (lengths * max_len - eps).unsqueeze(1)
    mask = torch.arange(max_len, device=x.device).unsqueeze(0) < abs_lengths

    # Add dimensions other than (batch, length) back into the mask
    for dim in range(1, x.ndim):
        if dim != length_dim:
            mask = mask.unsqueeze(dim)

    # Leave the non-masked dimensions as singletons, which can be broadcast
    return mask


class GlobalNorm(torch.nn.Module):
    """A global normalization module - computes a single mean and standard deviation
    for the entire batch across unmasked positions and uses it to normalize the
    inputs to the desired mean and standard deviation.

    This normalization is reversible - it is possible to use the .denormalize()
    method to recover the original values.

    Arguments
    ---------
    norm_mean: float, default 0.0
        the desired normalized mean
    norm_std: float, default 1.0
        the desired normalized standard deviation
    update_steps: float, optional
        the number of steps over which statistics will be collected
    length_dim: int, default 2
        the dimension used to represent the length
    mask_value: float, default 0.0
        the value with which to fill masked positions
        without a mask_value, the masked positions would be normalized,
        which might not be desired

    Example
    -------
    >>> import torch
    >>> from speechbrain.processing.features import GlobalNorm
    >>> global_norm = GlobalNorm(
    ...     norm_mean=0.5, norm_std=0.2, update_steps=3, length_dim=1
    ... )
    >>> x = torch.tensor([[1.0, 2.0, 3.0]])
    >>> x_norm = global_norm(x)
    >>> x_norm
    tensor([[0.2551, 0.5000, 0.7449]])
    >>> x = torch.tensor([[5.0, 10.0, -4.0]])
    >>> x_norm = global_norm(x)
    >>> x_norm
    tensor([[0.6027, 0.8397, 0.1761]])
    >>> x_denorm = global_norm.denormalize(x_norm)
    >>> x_denorm
    tensor([[ 5.0000, 10.0000, -4.0000]])
    >>> x = torch.tensor([[100.0, -100.0, -50.0]])
    >>> global_norm.freeze()
    >>> global_norm(x)
    tensor([[ 5.1054, -4.3740, -2.0041]])
    >>> global_norm.denormalize(x_norm)
    tensor([[ 5.0000, 10.0000, -4.0000]])
    >>> global_norm.unfreeze()
    >>> global_norm(x)
    tensor([[ 5.1054, -4.3740, -2.0041]])
    >>> global_norm.denormalize(x_norm)
    tensor([[ 5.0000, 10.0000, -4.0000]])
    """

    def __init__(
        self,
        norm_mean=0.0,
        norm_std=1.0,
        update_steps=None,
        length_dim=2,
        mask_value=0.0,
    ):
        super().__init__()

        running_mean = torch.tensor(0.0)
        running_std = torch.tensor(0.0)
        weight = torch.tensor(0.0)
        self.register_buffer("running_mean", running_mean)
        self.register_buffer("running_std", running_std)
        self.register_buffer("weight", weight)
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.mask_value = mask_value
        self.step_count = 0
        self.update_steps = update_steps
        self.length_dim = length_dim
        self.frozen = False

    def forward(self, x, lengths=None, mask_value=None, skip_update=False):
        """Normalizes the tensor provided

        Arguments
        ---------
        x: torch.Tensor
            the tensor to normalize
        lengths: torch.Tensor, optional
            a tensor of relative lengths (padding will not
            count towards normalization)
        mask_value: float, optional
            the value to use for masked positions
        skip_update: bool, default False
            whether to skip updates to the norm

        Returns
        -------
        result: torch.Tensor
            the normalized tensor
        """
        if lengths is None:
            lengths = torch.ones(len(x))
        if mask_value is None:
            mask_value = self.mask_value

        # Expand mask to all dims because GlobalNorm is over all
        mask = make_padding_mask(x, lengths, self.length_dim).expand_as(x)

        # Update statistics using this tensor if needed
        if not skip_update and self.should_update():
            self.weight, self.running_mean, self.running_std = mean_std_update(
                x=x,
                mask=mask,
                dim=None,
                run_count=self.weight,
                run_mean=self.running_mean,
                run_std=self.running_std,
            )

        # Perform normalization using running stats to desired mean and std
        x = self.normalize(x)

        # Fill the mask with the normalized mask value
        if not torch.is_tensor(mask_value):
            mask_value = torch.tensor(mask_value, device=x.device)
        mask_value_norm = self.normalize(mask_value)
        x = x.masked_fill(~mask, mask_value_norm)

        # Count steps so we know when to stop
        self.step_count += 1

        return x

    def should_update(self):
        """Whether to perform an update."""
        if self.frozen:
            return False
        if self.update_steps is None:
            return True
        return self.step_count < self.update_steps

    def normalize(self, x):
        """Performs the normalization operation against the running
        mean and standard deviation

        Arguments
        ---------
        x: torch.Tensor
            the tensor to normalize

        Returns
        -------
        result: torch.Tensor
            the normalized tensor
        """
        x = (x - self.running_mean) / self.running_std
        x = (x * self.norm_std) + self.norm_mean
        return x

    def denormalize(self, x):
        """Reverses the normalization process

        Arguments
        ---------
        x: torch.Tensor
            a normalized tensor

        Returns
        -------
        result: torch.Tensor
            a denormalized version of x
        """
        x = (x - self.norm_mean) / self.norm_std
        x = x * self.running_std + self.running_mean
        return x

    def freeze(self):
        """Stops updates to the running mean/std"""
        self.frozen = True

    def unfreeze(self):
        """Resumes updates to the running mean/std"""
        self.frozen = False


class MinLevelNorm(torch.nn.Module):
    """A commonly used normalization for the decibel scale

    The scheme is as follows

    x_norm = (x - min_level_db)/-min_level_db * 2 - 1

    The rationale behind the scheme is as follows:

    The top of the scale is assumed to be 0db.
    x_rel = (x - min) / (max - min) gives the relative position on the scale
    between the minimum and the maximum where the minimum is 0. and the
    maximum is 1.

    The subsequent rescaling (x_rel * 2 - 1) puts it on a scale from -1. to 1.
    with the middle of the range centered at zero.

    Arguments
    ---------
    min_level_db: float
        the minimum level

    Example
    -------
    >>> norm = MinLevelNorm(min_level_db=-100.0)
    >>> x = torch.tensor([-50.0, -20.0, -80.0])
    >>> x_norm = norm(x)
    >>> x_norm
    tensor([ 0.0000,  0.6000, -0.6000])
    """

    def __init__(self, min_level_db):
        super().__init__()
        self.min_level_db = min_level_db

    def forward(self, x):
        """Normalizes audio features in decibels (usually spectrograms)

        Arguments
        ---------
        x: torch.Tensor
            input features

        Returns
        -------
        normalized_features: torch.Tensor
            the normalized features
        """
        x = (x - self.min_level_db) / -self.min_level_db
        x *= 2.0
        x = x - 1.0
        x = torch.clip(x, -1, 1)
        return x

    def denormalize(self, x):
        """Reverses the min level normalization process

        Arguments
        ---------
        x: torch.Tensor
            the normalized tensor

        Returns
        -------
        result: torch.Tensor
            the denormalized tensor
        """
        x = torch.clip(x, -1, 1)
        x = (x + 1.0) / 2.0
        x *= -self.min_level_db
        x += self.min_level_db
        return x


class DynamicRangeCompression(torch.nn.Module):
    """Dynamic range compression for audio signals - clipped log scale
    with an optional multiplier

    Arguments
    ---------
    multiplier: float
        the multiplier constant
    clip_val: float
        the minimum accepted value (values below this
        minimum will be clipped)

    Example
    -------
    >>> drc = DynamicRangeCompression()
    >>> x = torch.tensor([10.0, 20.0, 0.0, 30.0])
    >>> drc(x)
    tensor([  2.3026,   2.9957, -11.5129,   3.4012])
    >>> drc = DynamicRangeCompression(2.0)
    >>> x = torch.tensor([10.0, 20.0, 0.0, 30.0])
    >>> drc(x)
    tensor([  2.9957,   3.6889, -10.8198,   4.0943])
    """

    def __init__(self, multiplier=1, clip_val=1e-5):
        super().__init__()
        self.multiplier = multiplier
        self.clip_val = clip_val

    def forward(self, x):
        """Performs the forward pass

        Arguments
        ---------
        x: torch.Tensor
            the source signal

        Returns
        -------
        result: torch.Tensor
            the result
        """
        return torch.log(torch.clamp(x, min=self.clip_val) * self.multiplier)
