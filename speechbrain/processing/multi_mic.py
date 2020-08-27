""" Multi-microphone components

This library contains functions for multi-microphone signal processing.

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

Authors:
 * William Aris
 * Francois Grondin

"""

import torch


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
        """ Computes the covariance matrices (XXs) of the signals. The result will
        have the following format: (batch, time_step, n_fft, 2, n_mics + n_pairs).

        The order on the last dimension corresponds to the triu_indices for a
        square matrix. For instance, if we have 4 channels, we get the
        following order: (0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3),
        (2, 2), (2, 3) and (3, 3). Therefore, XXs[..., 0] corresponds
        to channels (0, 0) and XXs[..., 1] corresponds to channels (0, 1).

        Arguments:
        ----------
        Xs : tensor
            A batch of audio signals in the frequency domain.
            The tensor must have the following format:
            (batch, time_step, n_fft, 2, n_mics)
        """

        # Formating the real and imaginary parts
        Xs_re = Xs[..., 0, :].unsqueeze(4)
        Xs_im = Xs[..., 1, :].unsqueeze(4)

        # Computing the covariance
        XXs_re = torch.matmul(Xs_re, Xs_re.transpose(3, 4)) + torch.matmul(
            Xs_im, Xs_im.transpose(3, 4)
        )

        XXs_im = torch.matmul(Xs_im, Xs_re.transpose(3, 4)) - torch.matmul(
            Xs_re, Xs_im.transpose(3, 4)
        )

        # Selecting the upper triangular part of the covariance matrices
        n_channels = Xs.shape[4]
        indices = torch.triu_indices(n_channels, n_channels)

        XXs_re = XXs_re[..., indices[0], indices[1]]
        XXs_im = XXs_im[..., indices[0], indices[1]]

        XXs = torch.stack((XXs_re, XXs_im), 3)

        if self.average is True:
            n_time_frames = XXs.shape[1]
            XXs = torch.mean(XXs, 1, keepdim=True)
            XXs = XXs.repeat(1, n_time_frames, 1, 1, 1)

        return XXs


class DelaySum(torch.nn.Module):
    """ Performs delay and sum beamforming using the TDOAs and
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
        """ Performs delay and sum beamforming using the TDOAs and
        the first channel as a reference. It returns the result
        in the frequency domain in the format
        (batch, time_step, n_fft, 2, 1)

        Arguments
        ---------
        Xs : tensor
            A batch of audio signals in the frequency domain, in
            the format (batch, time_step, n_fft, 2, n_mics)

        tdoas : tensor
            The time difference of arrival (TDOA) (in samples) for
            each timestamp. The tensor has the format
            (batch, time_steps, n_mics + n_pairs)
        """

        pi = 3.141592653589793

        n_batches = Xs.shape[0]
        n_time_frames = Xs.shape[1]
        n_fft = Xs.shape[2]
        n_channels = Xs.shape[4]

        N = int((n_fft - 1) * 2)

        # Computing the different parts of the steering vector
        omegas = 2 * pi * torch.arange(0, n_fft, device=Xs.device) / N
        omegas = omegas.unsqueeze(0).unsqueeze(-1)
        omegas = omegas.repeat(n_batches, n_time_frames, 1, n_channels)
        tdoas = tdoas[:, :, range(0, n_channels)]
        tdoas = tdoas.unsqueeze(2)
        tdoas = tdoas.repeat(1, 1, n_fft, 1)

        # Assembling the steering vector
        As_re = torch.cos(-1.0 * omegas * tdoas)
        As_im = -1.0 * torch.sin(-1.0 * omegas * tdoas)
        As = torch.stack((As_re, As_im), 3)

        # Computing beamforming coefficients
        Ws_re = As[..., 0, :] / n_channels
        Ws_im = -1.0 * As[..., 1, :] / n_channels

        # Applying delay and sum
        Xs_re = Xs[..., 0, :]
        Xs_im = Xs[..., 1, :]
        Ys_re = torch.sum((Ws_re * Xs_re - Ws_im * Xs_im), dim=-1, keepdim=True)
        Ys_im = torch.sum((Ws_re * Xs_im + Ws_im * Xs_re), dim=-1, keepdim=True)

        # Assembling the result
        Ys = torch.stack((Ys_re, Ys_im), 3)

        return Ys


class Mvdr(torch.nn.Module):
    """ Minimum Variance Distortionless Response (MVDR) Beamforming
    """

    def __init__(self):

        super().__init__()

    def forward(self):

        pass


class Gev(torch.nn.Module):
    """ Generalized EigenValue decomposition (GEV) Beamforming
    """

    def __init__(self):

        super().__init__()

    def forward(self):

        pass


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
        """ Evaluate the time difference of arrival (TDOA) (in samples)
        for each timestamp. It returns delays for every possible pair of
        microphones (including each microphone compared to itself, which
        gives a TDOA of 0 in this case). The result has the format:
        (batch, time_steps, n_mics + n_pairs)

        The order on the last dimension corresponds to the triu_indices for a
        square matrix. For instance, if we have 4 channels, we get the following
        order: (0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3)
        and (3, 3). Therefore, tdoas[..., 0] corresponds to channels (0, 0) and tdoas[..., 1]
        corresponds to channels (0, 1).

        Arguments
        ---------
        XXs : tensor
            The covariance matrices of the input signal. The tensor must
            have the format (batch, time_steps, n_fft/2, 2, n_mics + n_pairs)
        """

        # Extracting the tensors for the operations
        XXs_values, XXs_indices = torch.unique(XXs, return_inverse=True, dim=1)

        XXs_re = XXs_values[..., 0, :]
        XXs_im = XXs_values[..., 1, :]

        # Phase transform
        XXs_abs = torch.sqrt(XXs_re ** 2 + XXs_im ** 2) + self.eps

        XXs_re_phat = XXs_re / XXs_abs
        XXs_im_phat = XXs_im / XXs_abs

        XXs_phat = torch.stack((XXs_re_phat, XXs_im_phat), 4)

        # Returning in the temporal domain
        XXs_phat = XXs_phat.transpose(2, 3)
        n_samples = int((XXs.shape[2] - 1) * 2)

        xxs = torch.irfft(XXs_phat, signal_ndim=1, signal_sizes=[n_samples])
        xxs = xxs[:, XXs_indices]
        xxs = xxs.transpose(2, 3)

        # Setting things up
        n_fft = xxs.shape[2]

        if self.tdoa_max is None:
            self.tdoa_max = n_fft // 2

        # Splitting the GCC-PHAT values to search in the range
        slice_1 = xxs[..., 0 : self.tdoa_max, :]
        slice_2 = xxs[..., -self.tdoa_max :, :]

        xxs_sliced = torch.cat((slice_1, slice_2), 2)

        # Extracting the delays in the range
        _, delays = torch.max(xxs_sliced, 2)

        # Adjusting the delays that were affected by the slicing
        offset = n_fft - xxs_sliced.shape[2]

        idx = delays >= slice_1.shape[2]
        delays[idx] += offset

        # Centering the delays around 0
        delays[idx] -= n_fft

        # Quadratic interpolation
        tp = torch.fmod((delays - 1) + n_fft, n_fft).unsqueeze(2)
        y1 = torch.gather(xxs, 2, tp).squeeze(2)

        tp = torch.fmod(delays + n_fft, n_fft).unsqueeze(2)
        y2 = torch.gather(xxs, 2, tp).squeeze(2)

        tp = torch.fmod((delays + 1) + n_fft, n_fft).unsqueeze(2)
        y3 = torch.gather(xxs, 2, tp).squeeze(2)

        delays_frac = delays + (y1 - y3) / (2 * y1 - 4 * y2 + 2 * y3)

        return delays_frac


class SrpPhat(torch.nn.Module):
    """ Steered0-Response Power with Phase Transform (SRP-PHAT) localization
    """

    def __init__(self):

        super().__init__()

    def forward(self):

        pass


class Music(torch.nn.Module):
    """ MUltpile SIgnal Classification (MUSIC) localization
    """

    def __init__(self):

        super().__init__()

    def forward(self):

        pass
