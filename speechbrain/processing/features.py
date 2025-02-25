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
>>> signal =read_audio('tests/samples/single-mic/example1.wav')
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
>>> features  = compute_cw(features)
>>> norm = InputNormalization()
>>> features = norm(features, torch.tensor([1]).float())

Authors
 * Mirco Ravanelli 2020
 * Peter Plantinga 2025
"""

import math
import warnings

import torch

from speechbrain.utils.checkpoints import (
    mark_as_loader,
    mark_as_saver,
    mark_as_transfer,
    register_checkpoint_hooks,
)
from speechbrain.utils.distributed import (
    ddp_all_reduce,
    is_distributed_initialized,
)
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
    >>> compute_ISTFT = ISTFT(
    ...     sample_rate=16000, win_length=25, hop_length=10
    ... )
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
    stft, power: int = 1, log: bool = False, eps: float = 1e-14
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


@register_checkpoint_hooks
class InputNormalization(torch.nn.Module):
    """Performs mean and variance normalization of the input tensor.

    Running mean and running variance calculation is done using Welford's
    Algorithm. A slight adaptation is used where each sample is weighted
    equally, rather than each value in each tensor.

    WARNING: at first, the running statistics do not represent the "true" mean
    and variance, but are estimates based on the data seen so far. Once enough
    data has been seen, the stats should closely approximate the "true" values.

    WARNING: Using global normalization, `forward()` will throw a division
    by zero error if no updates have been performed (including the current
    batch), i.e. on first call the epoch >= update_until_epoch. This is
    because the variance is undefined on initialization.

    Arguments
    ---------
    mean_norm : True
        If True, the mean will be normalized.
    std_norm : True
        If True, the variance will be normalized.
    norm_type : str, default "global"
        String parameter whose value defines how the statistics are computed:
         * 'sentence' computes norms per utterance (no running stats)
         * 'batch' computes norms per input tensor (no running stats)
         * 'global' computes norms over all inputs (single mean, variance)
         * 'speaker' computes norms per input id (mean, variance per id)
    avg_factor : float, optional
        It can be used to manually set the weighting factor between
        current statistics and accumulated ones. Stats are affected as:
        `(1 - avg_factor) * old + avg_factor * new`, so use values close to 0.
        Compared to global stats, is biased towards recent values when
        used over long-enough horizons (i.e. 1 / avg_factor samples).
    update_until_epoch : int, default 2
        The epoch for which updates to the norm stats should stop.
        By default, stops after one epoch of updates.
    epsilon : float, default 1e-10
        A small value to improve the numerical stability of the variance.
    spk_process_split : bool
        During distributed processing, whether each speaker has been assigned
        to a specific process, used to disable the warning about speaker stats
        not communicated across processes.

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
    tensor([[-1.,  0.,  1.],
            [-1.,  0.,  1.],
            [-1.,  0.,  1.]])
    >>> norm = InputNormalization(norm_type="batch")
    >>> features = norm(inputs, input_lens)
    >>> features
    tensor([[-1.4606, -1.0954, -0.7303],
            [-0.3651,  0.0000,  0.3651],
            [ 0.7303,  1.0954,  1.4606]])
    >>> norm = InputNormalization(norm_type="global")
    >>> features = norm(inputs, input_lens)
    >>> features.mean()
    tensor(0.)
    >>> features = norm(inputs + 1, input_lens)
    >>> features.mean()
    tensor(0.1792)
    >>> features = norm(inputs, input_lens)
    >>> features.mean()
    tensor(-0.1197)
    >>> features = norm(inputs - 1, input_lens)
    >>> features.mean()
    tensor(-0.3522)
    >>> features = norm(inputs, input_lens)
    >>> features.mean() < 1e-7
    tensor(True)
    """

    from typing import Dict

    spk_dict_mean: Dict[int, torch.Tensor]
    spk_dict_std: Dict[int, torch.Tensor]
    spk_dict_count: Dict[int, int]
    NORM_TYPES = ("global", "batch", "sentence", "speaker")

    def __init__(
        self,
        mean_norm=True,
        std_norm=True,
        norm_type="global",
        avg_factor=None,
        update_until_epoch=2,
        epsilon=1e-10,
        spk_process_split=False,
    ):
        super().__init__()

        # Validate and store input arguments
        if not mean_norm:
            raise ValueError("Passing `False` for `mean_norm` is deprecated.")
        if norm_type not in self.NORM_TYPES:
            raise ValueError(f"norm_type must be one of {self.NORM_TYPES}.")

        self.mean_norm = mean_norm
        self.std_norm = std_norm
        self.norm_type = norm_type
        self.avg_factor = avg_factor
        self.epsilon = epsilon

        # Show warning if speaker & distributed & not turned off
        self.spk_warning = not spk_process_split and norm_type == "speaker"

        # Set a suitably huge epoch if None is passed
        self.update_until_epoch = update_until_epoch or torch.inf

        # Containers for running mean/variance calculation
        self.glob_mean = torch.tensor([0.0])
        self.glob_var = torch.tensor([0.0])
        self.spk_dict_mean = {}
        self.spk_dict_var = {}
        self.spk_dict_count = {}
        self.count = 0

    def forward(self, x, lengths=None, spk_ids=None, epoch=None):
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
        spk_ids : torch.Tensor, optional
            Contains the ids of each speaker (e.g, `[0 10 6]`), used to
            perform per-speaker normalization when `norm_type='speaker'`.
        epoch : int, optional
            The current epoch count, used to stop updates to global stats after
            enough samples have been seen (e.g. one epoch).

        Returns
        -------
        x : torch.Tensor
            The normalized tensor.
        """
        if self.spk_warning and is_distributed_initialized():
            warnings.warn(
                "InputNormalization(norm_type='speaker') does not currently "
                "support distributed statistics. Proceeding with statistics "
                "computed individually for each process (this is okay if the "
                "speakers are not shared across processes.)"
            )

        # Padding mask is used to protect padding elements from updates
        mask = get_mask(x, lengths, length_dim=1)

        # For full-batch norms, compute stats on all non-padding values
        if self.norm_type == "global" or self.norm_type == "batch":
            if self.norm_type == "batch":
                mean, var = self._compute_current_stats(x[mask])
            elif self.norm_type == "global":
                if self._should_update(epoch):
                    self._update_global_stats(x[mask], weight=x.size(0))
                mean, var = self.glob_mean, self.glob_var

            # Normalize mean and standard deviation ignoring padding
            mean = mean.masked_fill(~mask, 0.0)
            std = var.sqrt().masked_fill(~mask, 1.0)
            return (x - mean) / std

        # For utterance-based norm_types "sentence" and "speaker" we require stats
        # on variable length tensors, which are not yet supported by built-in
        # `mean()` and `var()` functions in torch, so compute them manually.
        elif self.norm_type == "sentence" or self.norm_type == "speaker":

            # Compute mean over non-first dimensions (per-utterance)
            dims = tuple(range(1, x.ndim))
            mean = (x * mask).sum(dim=dims) / mask.sum(dim=dims)
            if self.norm_type == "speaker":
                mean = self._update_collect_spk_mean(mean, spk_ids, epoch)

            # Broadcast mean to all dims, but exclude padding
            view = (-1,) + (1,) * (x.ndim - 1)
            mean = mean.view(view).masked_fill(~mask, 0.0)

            # Compute variance per-utterance if requested
            if self.std_norm:
                square_diff = ((x - mean) * mask).square()
                var = square_diff.sum(dim=dims) / (mask.sum(dim=dims) - 1)
                if self.norm_type == "speaker":
                    var = self._update_collect_spk_variance(var, spk_ids, epoch)
            else:
                var = torch.ones(x.size(0), device=x.device)

            # Also broadcast standard deviation excluding padding
            return (x - mean) / var.sqrt().view(view).masked_fill(~mask, 1.0)

        # Should never get here, this is simply for safety
        else:
            raise ValueError(
                "Only 'global', 'batch', 'sentence', and 'speaker' norm_type supported"
            )

    def _should_update(self, epoch):
        """Whether to perform an update, based on epoch count."""
        still_training = epoch is None or epoch < self.update_until_epoch
        return still_training and self.training

    def _compute_current_stats(self, x):
        """Computes mean and variance of an input tensor."""
        if self.std_norm:
            var, mean = torch.var_mean(x)

            # Clamp variance for numerical stability
            var = var.clamp(min=self.epsilon)
        else:
            mean = torch.mean(x)
            var = torch.tensor(1.0, device=x.device)

        return mean, var

    def _update_global_stats(self, x, weight):
        """Use input tensor to update global statistics."""
        if self.avg_factor is not None:
            weight, self.count = self.avg_factor, (1 - self.avg_factor)

        if self.std_norm:
            self.count, self.glob_mean, self.glob_var = global_norm_update(
                x, weight, self.count, self.glob_mean, self.glob_var
            )

        else:
            self.count, self.glob_mean = global_norm_update(
                x, weight, self.count, self.glob_mean
            )

    def _update_collect_spk_mean(self, utterance_means, spk_ids, epoch):
        """Perform a per-speaker mean update, returning per-utterance means."""
        if self._should_update(epoch):
            for mean, spk_id in zip(utterance_means, spk_ids):
                if spk_id not in self.spk_dict_count:
                    self.spk_dict_count[spk_id] = 1
                    self.spk_dict_mean[spk_id] = mean
                else:
                    self.spk_dict_count[spk_id] += 1
                    self.spk_dict_mean[spk_id] = self._compute_spk_stat_update(
                        self.spk_dict_mean[spk_id], mean, spk_id
                    )

        for i, spk_id in enumerate(spk_ids):
            utterance_means[i] = self.spk_dict_mean[spk_id]
        return utterance_means

    def _update_collect_spk_variance(self, utterance_vars, spk_ids, epoch):
        """Perform a per-speaker variance update, returning per-utterance variances."""
        if self._should_update(epoch):
            for var, spk_id in zip(utterance_vars, spk_ids):
                if spk_id not in self.spk_dict_var:
                    self.spk_dict_var[spk_id] = var
                else:
                    self.spk_dict_var[spk_id] = self._compute_spk_stat_update(
                        self.spk_dict_var[spk_id], var, spk_id
                    )

        for i, spk_id in enumerate(spk_ids):
            utterance_vars[i] = self.spk_dict_var[spk_id]
        return utterance_vars

    def _compute_spk_stat_update(self, old_value, new_value, spk_id):
        """Compute weighted update according to avg_factor or count."""
        count = self.spk_dict_count[spk_id]
        if self.avg_factor:
            new_w, old_w = self.avg_factor, (1 - self.avg_factor)
        else:
            new_w, old_w = 1 / count, (count - 1) / count

        return old_value * old_w + new_value * new_w

    def _statistics_dict(self):
        """Fills the dictionary containing the normalization statistics.

        Standard deviation is stored instead of variance for backward compatibility.
        """
        state = {}
        state["count"] = self.count
        state["glob_mean"] = self.glob_mean
        state["glob_std"] = self.glob_var.sqrt()
        state["spk_dict_count"] = self.spk_dict_count
        state["spk_dict_mean"] = self.spk_dict_mean
        state["spk_dict_std"] = {
            spk: var.sqrt() for spk, var in self.spk_dict_var.items()
        }

        return state

    def _load_statistics_dict(self, state):
        """Loads the dictionary containing the statistics.

        Standard deviation is loaded instead of variance for backward compatibility.

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
        self.glob_var = state["glob_std"] ** 2

        # Loading the speaker-specific stats
        self.spk_dict_count = state["spk_dict_count"]
        self.spk_dict_mean = state["spk_dict_mean"].copy()
        self.spk_dict_var = {
            spk: std**2 for spk, std in state["spk_dict_std"].items()
        }

        return state

    def to(self, device):
        """Puts the needed tensors in the right device."""
        self = super(InputNormalization, self).to(device)
        self.glob_mean = self.glob_mean.to(device)
        self.glob_var = self.glob_var.to(device)
        for spk in self.spk_dict_mean:
            self.spk_dict_mean[spk] = self.spk_dict_mean[spk].to(device)
            self.spk_dict_var[spk] = self.spk_dict_var[spk].to(device)
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
        device = "cpu"
        stats = torch.load(path, map_location=device)
        self._load_statistics_dict(stats)


def global_norm_update(new_tensor, new_weight, weight, mean, var=None):
    """Welford's algorithm for running mean, squared-diff (from Wikipedia)

    Incorporate a new tensor into running statistics. Allows for weighting
    each tensor differently, e.g. by length. Handles sync across processes.

    WARNING: Must be called in sync across processes. Does not handle normalizing
    separate statistics on separate processes (e.g. per-speaker statistics).

    Arguments
    ---------
    new_tensor : torch.Tensor
        The new values to add to the running stats
    new_weight : float or torch.Tensor
        The amount to weight this sample (e.g. 1.0 or relative length)
    weight : float or torch.Tensor
        The running weight of samples seen so far
    mean : float or torch.Tensor
        The running mean of samples seen so far
    var : float or torch.Tensor, optional
        The running variance (mean squared-diff) from the mean
        If None, the variance is not computed.

    Returns
    -------
    weight : torch.Tensor
        Updated weight sum for all samples so far.
    mean : torch.Tensor
        Updated running mean of all samples so far.
    var : torch.Tensor (if passed)
        Updated running variance of all samples so far.
    """
    # Update mean for use in variance
    delta = new_tensor - mean
    mean += delta.mean() * new_weight / (weight + new_weight)

    # Update variance, including sync
    if var is not None:
        delta2 = new_tensor - mean
        new_var = (delta * delta2).sum() / (delta.numel() - 1)
        var = (var * weight + new_var * new_weight) / (weight + new_weight)
        var = ddp_all_reduce(var, torch.distributed.ReduceOp.AVG)

    # Communicate mean, weight across processes
    # Wait til after variance update to avoid negative variance
    mean = ddp_all_reduce(mean, torch.distributed.ReduceOp.AVG)
    weight += ddp_all_reduce(new_weight, torch.distributed.ReduceOp.SUM)

    return (weight, mean, var) if var is not None else (weight, mean)


def get_mask(x, lengths=None, length_dim=1):
    """Create a mask from relative lengths along a given dimension.

    Arguments
    ---------
    x : torch.Tensor
        The input tensor demonstrating the size of the target mask.
    lengths : torch.Tensor, optional
        The relative lengths of an input batch of utterances.
    length_dim : int, default 1
        The dimension for which the lengths indicate padded positions.

    Returns
    -------
    mask : torch.Tensor
        A boolean tensor with `True` for valid positions and `False`
        for padding positions.
    """
    if lengths is None:
        lengths = torch.ones(x.size(0), device=x.device)

    max_len = x.size(length_dim)
    abs_lengths = lengths.unsqueeze(1) * max_len
    mask = torch.arange(max_len, device=x.device).unsqueeze(0) < abs_lengths

    # Add extra dimensions back in to the mask
    for dim in range(1, x.ndim):
        if dim != length_dim:
            mask = mask.unsqueeze(dim)

    # Repeat values (without copying) into singleton dimensions
    mask = mask.expand_as(x)

    return mask


class GlobalNorm(torch.nn.Module):
    """A global normalization module - computes a single mean and standard deviation
    for the entire batch across unmasked positions and uses it to normalize the
    inputs to the desired mean and standard deviation.

    This normalization is reversible - it is possible to use the .denormalize()
    method to recover the original values.

    Arguments
    ---------
    norm_mean: float
        the desired normalized mean
    norm_std: float
        the desired normalized standard deviation
    update_steps: float
        the number of steps over which statistics will be collected
    length_dim: int
        the dimension used to represent the length
    mask_value: float
        the value with which to fill masked positions
        without a mask_value, the masked positions would be normalized,
        which might not be desired

    Example
    -------
    >>> import torch
    >>> from speechbrain.processing.features import GlobalNorm
    >>> global_norm = GlobalNorm(
    ...     norm_mean=0.5,
    ...     norm_std=0.2,
    ...     update_steps=3,
    ...     length_dim=1
    ... )
    >>> x = torch.tensor([[1., 2., 3.]])
    >>> x_norm = global_norm(x)
    >>> x_norm
    tensor([[0.3000, 0.5000, 0.7000]])
    >>> x = torch.tensor([[5., 10., -4.]])
    >>> x_norm = global_norm(x)
    >>> x_norm
    tensor([[0.5838, 0.7773, 0.2356]])
    >>> x_denorm = global_norm.denormalize(x_norm)
    >>> x_denorm
    tensor([[ 5.0000, 10.0000, -4.0000]])
    >>> x = torch.tensor([[100., -100., -50.]])
    >>> global_norm.freeze()
    >>> global_norm(x)
    tensor([[ 4.2603, -3.4796, -1.5446]])
    >>> global_norm.denormalize(x_norm)
    tensor([[ 5.0000, 10.0000, -4.0000]])
    >>> global_norm.unfreeze()
    >>> global_norm(x)
    tensor([[ 4.2603, -3.4796, -1.5446]])
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
        running_var = torch.tensor(0.0)
        weight = torch.tensor(0.0)
        self.register_buffer("running_mean", running_mean)
        self.register_buffer("running_var", running_var)
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
        lengths: torch.Tensor
            a tensor of relative lengths (padding will not
            count towards normalization)
        mask_value: float
            the value to use for masked positions
        skip_update: false
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

        mask = get_mask(x, lengths, self.length_dim)

        # Update statistics using this tensor if needed
        if not skip_update and self.should_update():
            self.perform_update(x[mask], weight=lengths.sum())

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

    def perform_update(self, tensor, weight):
        """Update running statistics with new tensor data."""
        self.weight, self.running_mean, self.running_var = global_norm_update(
            tensor, weight, self.weight, self.running_mean, self.running_var
        )

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
        x = (x - self.running_mean) / self.running_var.sqrt()
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
        x = x * self.running_var.sqrt() + self.running_mean
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
    >>> norm = MinLevelNorm(min_level_db=-100.)
    >>> x = torch.tensor([-50., -20., -80.])
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
    >>> x = torch.tensor([10., 20., 0., 30.])
    >>> drc(x)
    tensor([  2.3026,   2.9957, -11.5129,   3.4012])
    >>> drc = DynamicRangeCompression(2.)
    >>> x = torch.tensor([10., 20., 0., 30.])
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
