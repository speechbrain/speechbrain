"""
Low level signal processing utilities

Authors
-------
 * Francois Grondin 2020
 * William Aris 2020
"""
import torch


def direction(doa, n_time_frames):
    """ Duplicate Direction of Arrival

    This function duplicates the DOA tensor over the number of time frames.

    Argument
    --------
    doa : tensor
        The input doa to work with. The tensor must have the following
        format: (3)

    n_time_frames : int
        The number of time frames

    Returns
    -------
    The doa repeated n_time_frames times. The result will have the following
    format: (n_time_frames, 3)

    Example
    -------
    >>> import torch
    >>> import soundfile as sf
    >>> from speechbrain.processing.features import STFT
    >>> import speechbrain.processing.processing_utils as pu
    >>> signal, fs = sf.read(
            'samples/audio_samples/multi_mic/speech_-0.98894_0_0.14834.flac'
        )
    >>> signal = torch.tensor(signal).unsqueeze(0)
    >>> compute_stft = STFT(sample_rate=fs)
    >>> xs = compute_stft(signal)
    >>> signal_pos = torch.tensor([-0.98894, 0., 0.14834])
    >>> n_time_frames = xs.shape[1]
    >>> doas = pu.direction(signal_pos, n_time_frames)
    """

    # Duplicating the DOA
    doas = doa.unsqueeze(0).repeat(n_time_frames, 1)

    return doas


def freefield(doas, mics, fs, c=343.0):
    """ Freefield

    This function converts the received directions of arrival (DOAs) into
    time-differences of arrival (TDOAs).

    Arguments
    ---------
    doas : tensor
        The input directions of arrival. The tensor must have the following
        format: (n_time_frames, 3)

    mics : tensor
        The position of each microphone. The tensor must have the following
        format: (n_channels, 3)

    fs : int
        The sampling frequency

    c : float
        The speed of the wave. By default, the value is set to 343.0 m/s which
        is the speed of sound in the air at 20 deg C.

    Returns
    -------
    The time-differences of arrival. The result will be a tensor with the
    following format: (n_time_frames, n_channels)

    Example
    -------
    >>> import torch
    >>> import soundfile as sf
    >>> from speechbrain.processing.features import STFT
    >>> import speechbrain.processing.processing_utils as pu
    >>> signal, fs = sf.read(
            'samples/audio_samples/multi_mic/speech_-0.98894_0_0.14834.flac'
        )
    >>> signal = torch.tensor(signal).unsqueeze(0)
    >>> compute_stft = STFT(sample_rate=fs)
    >>> xs = compute_stft(signal)
    >>> mics_pos = torch.tensor([
            [-0.05, -0.05, 0.00],
            [-0.05, 0.05, 0.00],
            [0.05, -0.05, 0.00],
            [0.05, 0.05, 0.00]
        ])
    >>> signal_pos = torch.tensor([-0.98894, 0., 0.14834])
    >>> n_time_frames = xs.shape[1]
    >>> doas = pu.direction(signal_pos, n_time_frames)
    >>> tdoas = pu.freefield(doas, mics_pos, fs)
    """

    # Converting the DOAs into TDOAs
    tdoas = (fs / c) * torch.matmul(doas, mics.transpose(0, 1))

    return tdoas


def steering(tdoas, n_fft):
    """ Steering Vector

    This function computes the steering vector using the tdoas and the number
    of bins (n_fft).

    Arguments
    ---------
    tdoas : tensor
        The time-differences of arrival. The tensor must have the following
        format: (n_time_frames, n_channels)

    n_fft : int
        The number of bins resulting of the STFT. It is assumed that the
        argument "onesided" was set to True for the STFT.

    Returns
    -------
    A steering vector. The result will be a tensor with the following format:
    (n_time_frames, n_fft, 2, n_channels)

    Example
    -------
    >>> import torch
    >>> import soundfile as sf
    >>> from speechbrain.processing.features import STFT
    >>> import speechbrain.processing.processing_utils as sp
    >>> signal, fs = sf.read(
            'samples/audio_samples/multi_mic/speech_-0.98894_0_0.14834.flac'
        )
    >>> signal = torch.tensor(signal).unsqueeze(0)
    >>> compute_stft = STFT(sample_rate=fs)
    >>> xs = compute_stft(signal)
    >>> mics_pos = torch.tensor([
            [-0.05, -0.05, 0.00],
            [-0.05, 0.05, 0.00],
            [0.05, -0.05, 0.00],
            [0.05, 0.05, 0.00]
        ])
    >>> signal_pos = torch.tensor([-0.98894, 0., 0.14834])
    >>> n_time_frames = xs.shape[1]
    >>> doas = pu.direction(signal_pos, n_time_frames)
    >>> tdoas = pu.freefield(doas, mics_pos, fs)
    >>> steer_vec = pu.steering(tdoas, xs.shape[2])
    """

    pi = 3.141592653589793

    n_time_frames = tdoas.shape[0]
    n_channels = tdoas.shape[1]

    N = int((n_fft - 1) * 2)

    # Computing the different parts of the steerng vector
    omegas = 2 * pi * torch.arange(0, n_fft, device=tdoas.device) / N
    omegas = omegas.unsqueeze(0).unsqueeze(-1)
    omegas = omegas.repeat(n_time_frames, 1, n_channels)

    tdoas = tdoas.unsqueeze(1).repeat(1, n_fft, 1)

    # Assembling the steering vector
    a_re = torch.cos(-1.0 * omegas * tdoas)
    a_im = -1.0 * torch.sin(-1.0 * omegas * tdoas)

    a = torch.stack((a_re, a_im), 2)

    return a
