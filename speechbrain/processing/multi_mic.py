""" Multi-microphone components

This library contains functions for multi-microphone signal processing.

Example
-------
>>> import soundfile as sf
>>> import torch
>>>
>>> from speechbrain.processing.features import STFT, ISTFT
>>> from speechbrain.processing.multi_mic import Covariance
>>> from speechbrain.processing.multi_mic import GccPhat
>>> from speechbrain.processing.multi_mic import DelaySum, Mvdr, Gev
>>>
>>> xs_speech, fs = sf.read(
...    'samples/audio_samples/multi_mic/speech_-0.82918_0.55279_-0.082918.flac'
... )
>>> xs_noise_diff, _ = sf.read('samples/audio_samples/multi_mic/noise_diffuse.flac')
>>> xs_noise_loc, _ = sf.read('samples/audio_samples/multi_mic/noise_0.70225_-0.70225_0.11704.flac')

>>> ss = torch.tensor(xs_speech).unsqueeze(0).float()
>>> nn_diff = torch.tensor(0.05 * xs_noise_diff).unsqueeze(0).float()
>>> nn_loc = torch.tensor(0.05 * xs_noise_loc).unsqueeze(0).float()
>>> xs_diffused_noise = ss + nn_diff
>>> xs_localized_noise = ss + nn_loc

>>> # Delay-and-Sum Beamforming
>>> stft = STFT(sample_rate=fs)
>>> cov = Covariance()
>>> gccphat = GccPhat()
>>> delaysum = DelaySum()
>>> istft = ISTFT(sample_rate=fs)
>>> Xs = stft(xs_diffused_noise)
>>> XXs = cov(Xs)
>>> tdoas = gccphat(XXs)
>>> Ys_ds = delaysum(Xs, tdoas)
>>> ys_ds = istft(Ys_ds)

>>> # Mvdr Beamforming
>>> mvdr = Mvdr()
>>> Ys_mvdr = mvdr(Xs, XXs, tdoas)
>>> ys_mvdr = istft(Ys_mvdr)

>>> # GeV Beamforming
>>> gev = Gev()
>>> Xs = stft(xs_localized_noise)
>>> Ss = stft(ss)
>>> Nn = stft(nn_loc)
>>> SSs = cov(Ss)
>>> NNs = cov(Nn)
>>> Ys_gev = gev(Xs, SSs, NNs)
>>> ys_gev = istft(Ys_gev)

Authors:
 * William Aris
 * Francois Grondin

"""

import torch
import speechbrain.processing.decomposition as eig


class Covariance(torch.nn.Module):
    """ Computes the covariance matrices of the signals.

    Arguments:
    ----------
    average : boolean
        Informs the module if it should return an average
        (computed on the time dimension) of the covariance
        matrices. Default value is True.

    Example
    -------
    >>> import soundfile as sf
    >>> import torch
    >>>
    >>> from speechbrain.processing.features import STFT
    >>> from speechbrain.processing.multi_mic import Covariance
    >>>
    >>> xs_speech, fs = sf.read(
    ...    'samples/audio_samples/multi_mic/speech_-0.82918_0.55279_-0.082918.flac'
    ... )
    >>> xs_noise, _ = sf.read('samples/audio_samples/multi_mic/noise_diffuse.flac')
    >>> xs = xs_speech + 0.05 * xs_noise
    >>> xs = torch.tensor(xs).unsqueeze(0).float()
    >>>
    >>> stft = STFT(sample_rate=fs)
    >>> cov = Covariance()
    >>>
    >>> Xs = stft(xs)
    >>> XXs = cov(Xs)
    >>> XXs.shape
    torch.Size([1, 1001, 201, 2, 10])
    """

    def __init__(self, average=True):

        super().__init__()
        self.average = average

    def forward(self, Xs):
        """ This method uses the utility function _cov to compute covariance
        matrices. Therefore, the result has the following format:
        (batch, time_step, n_fft/2 + 1, 2, n_mics + n_pairs).

        The order on the last dimension corresponds to the triu_indices for a
        square matrix. For instance, if we have 4 channels, we get the following
        order: (0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3)
        and (3, 3). Therefore, XXs[..., 0] corresponds to channels (0, 0) and XXs[..., 1]
        corresponds to channels (0, 1).

        Arguments:
        ----------
        Xs : tensor
            A batch of audio signals in the frequency domain.
            The tensor must have the following format:
            (batch, time_step, n_fft/2 + 1, 2, n_mics)
        """

        XXs = Covariance._cov(Xs=Xs, average=self.average)
        return XXs

    @staticmethod
    def _cov(Xs, average=True):
        """ Computes the covariance matrices (XXs) of the signals. The result will
        have the following format: (batch, time_step, n_fft/2 + 1, 2, n_mics + n_pairs).

        Arguments:
        ----------
        Xs : tensor
            A batch of audio signals in the frequency domain.
            The tensor must have the following format:
            (batch, time_step, n_fft/2 + 1, 2, n_mics)

        average : boolean
            Informs the function if it should return an average
            (computed on the time dimension) of the covariance
            matrices. Default value is True.
        """

        # Get useful dimensions
        n_mics = Xs.shape[4]

        # Formating the real and imaginary parts
        Xs_re = Xs[..., 0, :].unsqueeze(4)
        Xs_im = Xs[..., 1, :].unsqueeze(4)

        # Computing the covariance
        Rxx_re = torch.matmul(Xs_re, Xs_re.transpose(3, 4)) + torch.matmul(
            Xs_im, Xs_im.transpose(3, 4)
        )

        Rxx_im = torch.matmul(Xs_re, Xs_im.transpose(3, 4)) - torch.matmul(
            Xs_im, Xs_re.transpose(3, 4)
        )

        # Selecting the upper triangular part of the covariance matrices
        idx = torch.triu_indices(n_mics, n_mics)

        XXs_re = Rxx_re[..., idx[0], idx[1]]
        XXs_im = Rxx_im[..., idx[0], idx[1]]

        XXs = torch.stack((XXs_re, XXs_im), 3)

        # Computing the average if desired
        if average is True:
            n_time_frames = XXs.shape[1]
            XXs = torch.mean(XXs, 1, keepdim=True)
            XXs = XXs.repeat(1, n_time_frames, 1, 1, 1)

        return XXs


class DelaySum(torch.nn.Module):
    """ Performs delay and sum beamforming by using the TDOAs and
        the first channel as a reference.

        Example
        -------
        >>> import soundfile as sf
        >>> import torch
        >>>
        >>> from speechbrain.processing.features import STFT, ISTFT
        >>> from speechbrain.processing.multi_mic import Covariance
        >>> from speechbrain.processing.multi_mic import GccPhat, DelaySum
        >>>
        >>> xs_speech, fs = sf.read(
        ...    'samples/audio_samples/multi_mic/speech_-0.82918_0.55279_-0.082918.flac'
        ... )
        >>> xs_noise, _ = sf.read('samples/audio_samples/multi_mic/noise_diffuse.flac')
        >>> xs = xs_speech + 0.05 * xs_noise
        >>> xs = torch.tensor(xs).unsqueeze(0).float()
        >>>
        >>> stft = STFT(sample_rate=fs)
        >>> cov = Covariance()
        >>> gccphat = GccPhat()
        >>> delaysum = DelaySum()
        >>> istft = ISTFT(sample_rate=fs)
        >>>
        >>> Xs = stft(xs)
        >>> XXs = cov(Xs)
        >>> tdoas = gccphat(XXs)
        >>> Ys = delaysum(Xs, tdoas)
        >>> ys = istft(Ys)
    """

    def __init__(self):

        super().__init__()

    def forward(self, Xs, tdoas):
        """ This method computes a steering vector by using the TDOAs and
        then calls the utility function _delaysum to perform beamforming.
        The result has the following format: (batch, time_step, n_fft, 2, 1).

        Arguments
        ---------
        Xs : tensor
            A batch of audio signals in the frequency domain.
            The tensor must have the following format:
            (batch, time_step, n_fft/2 + 1, 2, n_mics)

        tdoas : tensor
            The time difference of arrival (TDOA) (in samples) for
            each timestamp. The tensor has the format
            (batch, time_steps, n_mics + n_pairs)
        """

        # Get useful dimensions
        n_fft = Xs.shape[2]

        # Convert the tdoas to taus
        taus = tdoas2taus(tdoas=tdoas)

        # Generate the steering vector
        As = steering(taus=taus, n_fft=n_fft)

        # Apply delay and sum
        Ys = DelaySum._delaysum(Xs=Xs, As=As)

        return Ys

    @staticmethod
    def _delaysum(Xs, As):
        """ Perform delay and sum beamforming. The result has
        the following format: (batch, time_step, n_fft, 2, 1).

        Arguments
        ---------
        Xs : tensor
            A batch of audio signals in the frequency domain.
            The tensor must have the following format:
            (batch, time_step, n_fft/2 + 1, 2, n_mics)

        As : tensor
            The steering vector to point in the direction of
            the target source. The tensor must have the format
            (batch, time_step, n_fft/2 + 1, 2, n_mics)
        """

        # Get useful dimensions
        n_mics = Xs.shape[4]

        # Generate unmixing coefficients
        Ws_re = As[..., 0, :] / n_mics
        Ws_im = -1 * As[..., 1, :] / n_mics

        # Get input signal
        Xs_re = Xs[..., 0, :]
        Xs_im = Xs[..., 1, :]

        # Applying delay and sum
        Ys_re = torch.sum((Ws_re * Xs_re - Ws_im * Xs_im), dim=3, keepdim=True)
        Ys_im = torch.sum((Ws_re * Xs_im + Ws_im * Xs_re), dim=3, keepdim=True)

        # Assembling the result
        Ys = torch.stack((Ys_re, Ys_im), 3)

        return Ys


class Mvdr(torch.nn.Module):
    """ Perform minimum variance distortionless response (MVDR) beamforming
    by using an input signal in the frequency domain, its covariance matrices
    and tdoas (to compute a steering vector).

        Example
        -------
        >>> import soundfile as sf
        >>> import torch
        >>>
        >>> from speechbrain.processing.features import STFT, ISTFT
        >>> from speechbrain.processing.multi_mic import Covariance
        >>> from speechbrain.processing.multi_mic import GccPhat, Mvdr
        >>>
        >>> xs_speech, fs = sf.read(
        ...    'samples/audio_samples/multi_mic/speech_-0.82918_0.55279_-0.082918.flac'
        ... )
        >>> xs_noise, _ = sf.read('samples/audio_samples/multi_mic/noise_diffuse.flac')
        >>> xs = xs_speech + 0.05 * xs_noise
        >>> xs = torch.tensor(xs).unsqueeze(0).float()
        >>>
        >>> stft = STFT(sample_rate=fs)
        >>> cov = Covariance()
        >>> gccphat = GccPhat()
        >>> mvdr = Mvdr()
        >>> istft = ISTFT(sample_rate=fs)
        >>>
        >>> Xs = stft(xs)
        >>> XXs = cov(Xs)
        >>> tdoas = gccphat(XXs)
        >>> Ys = mvdr(Xs, XXs, tdoas)
        >>> ys = istft(Ys)
    """

    def __init__(self, eps=1e-20):

        super().__init__()

        self.eps = eps

    def forward(self, Xs, XXs, tdoas):
        """ This method computes a steering vector before using the
        utility function _mvdr to perform beamforming. The result has
        the following format: (batch, time_step, n_fft, 2, 1).

        Arguments
        ---------
        Xs : tensor
            A batch of audio signals in the frequency domain.
            The tensor must have the following format:
            (batch, time_step, n_fft/2 + 1, 2, n_mics)

        XXs : tensor
            The covariance matrices of the input signal. The tensor must
            have the format (batch, time_steps, n_fft/2 + 1, 2, n_mics + n_pairs)

        tdoas : tensor
            The time difference of arrival (TDOA) (in samples) for
            each timestamp. The tensor has the format
            (batch, time_steps, n_mics + n_pairs)
        """
        # Get useful dimensions
        n_fft = Xs.shape[2]

        # Convert the tdoas to taus
        taus = tdoas2taus(tdoas=tdoas)

        # Generate the steering vector
        As = steering(taus=taus, n_fft=n_fft)

        # Perform mvdr
        Ys = Mvdr._mvdr(Xs=Xs, XXs=XXs, As=As)

        return Ys

    @staticmethod
    def _mvdr(Xs, XXs, As, eps=1e-20):
        """ Perform minimum variance distortionless response beamforming.

        Arguments
        ---------
        Xs : tensor
            A batch of audio signals in the frequency domain.
            The tensor must have the following format:
            (batch, time_step, n_fft/2 + 1, 2, n_mics)

        XXs : tensor
            The covariance matrices of the input signal. The tensor must
            have the format (batch, time_steps, n_fft/2 + 1, 2, n_mics + n_pairs)

        As : tensor
            The steering vector to point in the direction of
            the target source. The tensor must have the format
            (batch, time_step, n_fft/2 + 1, 2, n_mics)
        """

        # Get unique covariance values to reduce the number of computations
        XXs_val, XXs_idx = torch.unique(XXs, return_inverse=True, dim=1)

        # Inverse covariance matrices
        XXs_inv = eig.inv(XXs_val)

        # Capture real and imaginary parts, and restore time steps
        XXs_inv_re = XXs_inv[..., 0][:, XXs_idx]
        XXs_inv_im = XXs_inv[..., 1][:, XXs_idx]

        # Decompose steering vector
        AsC_re = As[..., 0, :].unsqueeze(4)
        AsC_im = 1.0 * As[..., 1, :].unsqueeze(4)
        AsT_re = AsC_re.transpose(3, 4)
        AsT_im = -1.0 * AsC_im.transpose(3, 4)

        # Project
        XXs_inv_AsC_re = torch.matmul(XXs_inv_re, AsC_re) - torch.matmul(
            XXs_inv_im, AsC_im
        )
        XXs_inv_AsC_im = torch.matmul(XXs_inv_re, AsC_im) + torch.matmul(
            XXs_inv_im, AsC_re
        )

        # Compute the gain
        alpha = 1.0 / (
            torch.matmul(AsT_re, XXs_inv_AsC_re)
            - torch.matmul(AsT_im, XXs_inv_AsC_im)
        )

        # Get the unmixing coefficients
        Ws_re = torch.matmul(XXs_inv_AsC_re, alpha).squeeze(4)
        Ws_im = -torch.matmul(XXs_inv_AsC_im, alpha).squeeze(4)

        # Applying MVDR
        Xs_re = Xs[..., 0, :]
        Xs_im = Xs[..., 1, :]

        Ys_re = torch.sum((Ws_re * Xs_re - Ws_im * Xs_im), dim=3, keepdim=True)
        Ys_im = torch.sum((Ws_re * Xs_im + Ws_im * Xs_re), dim=3, keepdim=True)

        Ys = torch.stack((Ys_re, Ys_im), -2)

        return Ys


class Gev(torch.nn.Module):
    """ Generalized EigenValue decomposition (GEV) Beamforming

    Example
    -------
    >>> import soundfile as sf
    >>> import torch
    >>>
    >>> from speechbrain.processing.features import STFT, ISTFT
    >>> from speechbrain.processing.multi_mic import Covariance
    >>> from speechbrain.processing.multi_mic import Gev
    >>>
    >>> xs_speech, fs = sf.read(
    ...    'samples/audio_samples/multi_mic/speech_-0.82918_0.55279_-0.082918.flac'
    ... )
    >>> xs_noise, _ = sf.read('samples/audio_samples/multi_mic/noise_0.70225_-0.70225_0.11704.flac')
    >>> ss = torch.tensor(xs_speech).unsqueeze(0).float()
    >>> nn = torch.tensor(0.05 * xs_noise).unsqueeze(0).float()
    >>> xs = ss + nn
    >>>
    >>> stft = STFT(sample_rate=fs)
    >>> cov = Covariance()
    >>> gev = Gev()
    >>> istft = ISTFT(sample_rate=fs)
    >>>
    >>> Ss = stft(ss)
    >>> Nn = stft(nn)
    >>> Xs = stft(xs)
    >>>
    >>> SSs = cov(Ss)
    >>> NNs = cov(Nn)
    >>>
    >>> Ys = gev(Xs, SSs, NNs)
    >>> ys = istft(Ys)
    """

    def __init__(self):

        super().__init__()

    def forward(self, Xs, SSs, NNs):
        """ This method uses the utility function _gev to perform generalized
        eigenvalue decomposition beamforming. Therefore, the result has
        the following format: (batch, time_step, n_fft, 2, 1).

        Arguments
        ---------
        Xs : tensor
            A batch of audio signals in the frequency domain.
            The tensor must have the following format:
            (batch, time_step, n_fft/2 + 1, 2, n_mics)

        SSs : tensor
            The covariance matrices of the target signal. The tensor must
            have the format (batch, time_steps, n_fft/2 + 1, 2, n_mics + n_pairs)

        NNs : tensor
            The covariance matrices of the noise signal. The tensor must
            have the format (batch, time_steps, n_fft/2 + 1, 2, n_mics + n_pairs)
        """

        Ys = Gev._gev(Xs=Xs, SSs=SSs, NNs=NNs)

        return Ys

    @staticmethod
    def _gev(Xs, SSs, NNs):
        """ Perform generalized eigenvalue decomposition beamforming. The result
        has the following format: (batch, time_step, n_fft, 2, 1).

        Arguments
        ---------
        Xs : tensor
            A batch of audio signals in the frequency domain.
            The tensor must have the following format:
            (batch, time_step, n_fft/2 + 1, 2, n_mics)

        SSs : tensor
            The covariance matrices of the target signal. The tensor must
            have the format (batch, time_steps, n_fft/2 + 1, 2, n_mics + n_pairs)

        NNs : tensor
            The covariance matrices of the noise signal. The tensor must
            have the format (batch, time_steps, n_fft/2 + 1, 2, n_mics + n_pairs)
        """

        # Get useful dimensions
        n_mics = Xs.shape[4]
        n_mics_pairs = SSs.shape[4]

        # Computing the eigenvectors
        SSs_NNs = torch.cat((SSs, NNs), dim=4)
        SSs_NNs_val, SSs_NNs_idx = torch.unique(
            SSs_NNs, return_inverse=True, dim=1
        )

        SSs = SSs_NNs_val[..., range(0, n_mics_pairs)]
        NNs = SSs_NNs_val[..., range(n_mics_pairs, 2 * n_mics_pairs)]
        NNs = eig.pos_def(NNs)
        Vs, Ds = eig.gevd(SSs, NNs)

        # Beamforming
        F_re = Vs[..., (n_mics - 1), 0]
        F_im = Vs[..., (n_mics - 1), 1]

        # Normalize
        F_norm = 1.0 / (
            torch.sum(F_re ** 2 + F_im ** 2, dim=3, keepdim=True) ** 0.5
        ).repeat(1, 1, 1, n_mics)
        F_re *= F_norm
        F_im *= F_norm

        Ws_re = F_re[:, SSs_NNs_idx]
        Ws_im = F_im[:, SSs_NNs_idx]

        Xs_re = Xs[..., 0, :]
        Xs_im = Xs[..., 1, :]

        Ys_re = torch.sum((Ws_re * Xs_re - Ws_im * Xs_im), dim=3, keepdim=True)
        Ys_im = torch.sum((Ws_re * Xs_im + Ws_im * Xs_re), dim=3, keepdim=True)

        # Assembling the output
        Ys = torch.stack((Ys_re, Ys_im), 3)

        return Ys


class GccPhat(torch.nn.Module):
    """ Generalized Cross-Correlation with Phase Transform localization

    Arguments
    ---------
    tdoa_max : int
        Specifies a range to search for delays. For example, if
        tdoa_max = 10, the method will restrict its search for delays
        between -10 and 10 samples. This parameter is optional and its
        default value is None. When tdoa_max is None, the method will
        search for delays between -n_fft/2 and n_fft/2 (full range).

    eps : float
        A small value to avoid divisions by 0 with the phase transform. The
        default value is 1e-20.

    Example
    -------
    >>> import soundfile as sf
    >>> import torch
    >>>
    >>> from speechbrain.processing.features import STFT
    >>> from speechbrain.processing.multi_mic import Covariance
    >>> from speechbrain.processing.multi_mic import GccPhat
    >>>
    >>> xs_speech, fs = sf.read(
    ...    'samples/audio_samples/multi_mic/speech_-0.82918_0.55279_-0.082918.flac'
    ... )
    >>> xs_noise, _ = sf.read('samples/audio_samples/multi_mic/noise_diffuse.flac')
    >>> xs = xs_speech + 0.05 * xs_noise
    >>> xs = torch.tensor(xs).unsqueeze(0).float()
    >>>
    >>> stft = STFT(sample_rate=fs)
    >>> cov = Covariance()
    >>> gccphat = GccPhat()
    >>> Xs = stft(xs)
    >>> XXs = cov(Xs)
    >>> tdoas = gccphat(XXs)
    """

    def __init__(self, tdoa_max=None, eps=1e-20):

        super().__init__()
        self.tdoa_max = tdoa_max
        self.eps = eps

    def forward(self, XXs):
        """ Perform generalized cross-correlation with phase transform localization
        by using the utility function _gcc_phat and by extracting the delays (in samples)
        before perfoming a quadratic interpolation to improve the accuracy.
        The result has the format: (batch, time_steps, n_mics + n_pairs).

        The order on the last dimension corresponds to the triu_indices for a
        square matrix. For instance, if we have 4 channels, we get the following
        order: (0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3)
        and (3, 3). Therefore, delays[..., 0] corresponds to channels (0, 0) and delays[..., 1]
        corresponds to channels (0, 1).

        Arguments:
        ----------
        XXs : tensor
            The covariance matrices of the input signal. The tensor must
            have the format (batch, time_steps, n_fft/2 + 1, 2, n_mics + n_pairs)
        """

        xxs = GccPhat._gcc_phat(XXs=XXs, eps=self.eps)
        delays = GccPhat._extract_delays(xxs=xxs, tdoa_max=self.tdoa_max)
        tdoas = GccPhat._interpolate(xxs=xxs, delays=delays)

        return tdoas

    @staticmethod
    def _gcc_phat(XXs, eps=1e-20):
        """ Evaluate GCC-PHAT for each timestamp. It returns the result in the time
        domain. The result has the format: (batch, time_steps, n_fft, n_mics + n_pairs)

        Arguments
        ---------
        XXs : tensor
            The covariance matrices of the input signal. The tensor must
            have the format (batch, time_steps, n_fft/2 + 1, 2, n_mics + n_pairs)

        eps : float
            A small value to avoid divisions by 0 with the phase transform. The
            default value is 1e-20.
        """

        # Get useful dimensions
        n_samples = int((XXs.shape[2] - 1) * 2)

        # Extracting the tensors needed
        XXs_val, XXs_idx = torch.unique(XXs, return_inverse=True, dim=4)

        XXs_re = XXs_val[..., 0, :]
        XXs_im = XXs_val[..., 1, :]

        # Applying the phase transform
        XXs_abs = torch.sqrt(XXs_re ** 2 + XXs_im ** 2) + eps
        XXs_re_phat = XXs_re / XXs_abs
        XXs_im_phat = XXs_im / XXs_abs
        XXs_phat = torch.stack((XXs_re_phat, XXs_im_phat), 4)

        # Returning in the temporal domain
        XXs_phat = XXs_phat.transpose(2, 3)
        xxs = torch.irfft(XXs_phat, signal_ndim=1, signal_sizes=[n_samples])
        xxs = xxs[..., XXs_idx, :]

        # Formatting the output
        xxs = xxs.transpose(2, 3)

        return xxs

    @staticmethod
    def _extract_delays(xxs, tdoa_max=None):
        """ Extract the rounded delays from the cross-correlation for each timestamp.
        The result has the format: (batch, time_steps, n_mics + n_pairs).

        Arguments
        ---------
        xxs : tensor
            The correlation signals obtained after a gcc-phat operation. The tensor
            must have the format (batch, time_steps, n_fft, n_mics + n_pairs)

        tdoa_max : int
            Specifies a range to search for delays. For example, if
            tdoa_max = 10, the method will restrict its search for delays
            between -10 and 10 samples. This parameter is optional and its
            default value is None. When tdoa_max is None, the method will
            search for delays between -n_fft/2 and +n_fft/2 (full range).
        """

        # Get useful dimensions
        n_fft = xxs.shape[2]

        # If no tdoa specified, cover the whole frame
        if tdoa_max is None:
            tdoa_max = n_fft // 2

        # Splitting the GCC-PHAT values to search in the range
        slice_1 = xxs[..., 0:tdoa_max, :]
        slice_2 = xxs[..., -tdoa_max:, :]

        xxs_sliced = torch.cat((slice_1, slice_2), 2)

        # Extracting the delays in the range
        _, delays = torch.max(xxs_sliced, 2)

        # Adjusting the delays that were affected by the slicing
        offset = n_fft - xxs_sliced.shape[2]
        idx = delays >= slice_1.shape[2]
        delays[idx] += offset

        # Centering the delays around 0
        delays[idx] -= n_fft

        return delays

    @staticmethod
    def _interpolate(xxs, delays):
        """ Perform quadratic interpolation on the cross-correlation to
        improve the tdoa accuracy. The result has the format:
        (batch, time_steps, n_mics + n_pairs)

        Arguments
        ---------
        xxs : tensor
            The correlation signals obtained after a gcc-phat operation. The tensor
            must have the format (batch, time_steps, n_fft, n_mics + n_pairs)

        delays : tensor
            The rounded tdoas obtained by selecting the sample with the highest
            amplitude. The tensor must have the format
            (batch, time_steps, n_mics + n_pairs)
        """

        # Get useful dimensions
        n_fft = xxs.shape[2]

        # Get the max amplitude and its neighbours
        tp = torch.fmod((delays - 1) + n_fft, n_fft).unsqueeze(2)
        y1 = torch.gather(xxs, 2, tp).squeeze(2)
        tp = torch.fmod(delays + n_fft, n_fft).unsqueeze(2)
        y2 = torch.gather(xxs, 2, tp).squeeze(2)
        tp = torch.fmod((delays + 1) + n_fft, n_fft).unsqueeze(2)
        y3 = torch.gather(xxs, 2, tp).squeeze(2)

        # Add a fractional part to the initially rounded delay
        delays_frac = delays + (y1 - y3) / (2 * y1 - 4 * y2 + 2 * y3)

        return delays_frac


class SrpPhat(torch.nn.Module):
    """ Steered-Response Power with Phase Transform (SRP-PHAT) localization
    """

    def __init__(self):

        super().__init__()

    def forward(self):

        pass


class Music(torch.nn.Module):
    """ Multpile SIgnal Classification (MUSIC) localization
    """

    def __init__(self):

        super().__init__()

    def forward(self):

        pass


def tdoas2taus(tdoas):
    """ This function selects the tdoas of each channel and put them
    in a tensor. The result has the following format:
    (batch, time_steps, n_mics).

    Arguments:
    ----------
    tdoas : tensor
       The time difference of arrival (TDOA) (in samples) for
       each timestamp. The tensor has the format
       (batch, time_steps, n_mics + n_pairs)

    Example
    -------
    >>> import soundfile as sf
    >>> import torch
    >>>
    >>> from speechbrain.processing.features import STFT
    >>> from speechbrain.processing.multi_mic import Covariance
    >>> from speechbrain.processing.multi_mic import GccPhat, tdoas2taus
    >>>
    >>> xs_speech, fs = sf.read(
    ...    'samples/audio_samples/multi_mic/speech_-0.82918_0.55279_-0.082918.flac'
    ... )
    >>> xs_noise, _ = sf.read('samples/audio_samples/multi_mic/noise_diffuse.flac')
    >>> xs = xs_speech + 0.05 * xs_noise
    >>> xs = torch.tensor(xs).unsqueeze(0).float()
    >>>
    >>> stft = STFT(sample_rate=fs)
    >>> cov = Covariance()
    >>> gccphat = GccPhat()
    >>>
    >>> Xs = stft(xs)
    >>> XXs = cov(Xs)
    >>> tdoas = gccphat(XXs)
    >>> taus = tdoas2taus(tdoas)
    """

    n_pairs = tdoas.shape[len(tdoas.shape) - 1]
    n_channels = int(((1 + 8 * n_pairs) ** 0.5 - 1) / 2)
    taus = tdoas[..., range(0, n_channels)]

    return taus


def steering(taus, n_fft):
    """ This function computes a steering vector by using the time differences
    of arrival for each channel (in samples) and the number of bins (n_fft).
    The result has the following format: (batch, time_step, n_fft/2 + 1, 2, n_mics).

    Arguments:
    ----------
    taus : tensor
        The time differences of arrival for each channel. The tensor must have
        the following format: (batch, time_steps, n_mics).

    n_fft : int
        The number of bins resulting of the STFT. It is assumed that the
        argument "onesided" was set to True for the STFT.

    Example:
    --------
    >>> import soundfile as sf
    >>> import torch
    >>>
    >>> from speechbrain.processing.features import STFT
    >>> from speechbrain.processing.multi_mic import Covariance
    >>> from speechbrain.processing.multi_mic import GccPhat, tdoas2taus, steering
    >>>
    >>> xs_speech, fs = sf.read(
    ...    'samples/audio_samples/multi_mic/speech_-0.82918_0.55279_-0.082918.flac'
    ... )
    >>> xs_noise, _ = sf.read('samples/audio_samples/multi_mic/noise_diffuse.flac')
    >>> xs = xs_speech + 0.05 * xs_noise
    >>> xs = torch.tensor(xs).unsqueeze(0).float()
    >>>
    >>> stft = STFT(sample_rate=fs)
    >>> cov = Covariance()
    >>> gccphat = GccPhat()
    >>>
    >>> Xs = stft(xs)
    >>> n_fft = Xs.shape[2]
    >>> XXs = cov(Xs)
    >>> tdoas = gccphat(XXs)
    >>> taus = tdoas2taus(tdoas)
    >>> As = steering(taus, n_fft)
    """

    # Collecting useful numbers
    pi = 3.141592653589793

    frame_size = int((n_fft - 1) * 2)

    # Computing the different parts of the steering vector
    omegas = 2 * pi * torch.arange(0, n_fft, device=taus.device) / frame_size
    omegas = omegas.repeat(taus.shape + (1,))
    taus = taus.unsqueeze(len(taus.shape)).repeat(
        (1,) * len(taus.shape) + (n_fft,)
    )

    # Assembling the steering vector
    a_re = torch.cos(-omegas * taus)
    a_im = torch.sin(-omegas * taus)
    a = torch.stack((a_re, a_im), len(a_re.shape))
    a = a.transpose(len(a.shape) - 3, len(a.shape) - 1).transpose(
        len(a.shape) - 3, len(a.shape) - 2
    )

    return a
