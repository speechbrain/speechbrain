"""
Low level signal processing utilities

Author
------
Peter Plantinga 2020
"""
import torch


def compute_amplitude(waveforms, lengths):
    """Compute the average amplitude of a batch of waveforms.

    Arguments
    ---------
    waveform : tensor
        The waveforms used for computing amplitude.
    lengths : tensor
        The lengths of the waveforms excluding the padding
        added to put all waveforms in the same tensor.

    Returns
    -------
    The average amplitude of the waveforms.

    Example
    -------
    >>> signal = torch.sin(torch.arange(16000.0)).unsqueeze(0)
    >>> compute_amplitude(signal, signal.size(1))
    tensor([[0.6366]])
    """
    return torch.sum(input=torch.abs(waveforms), dim=1, keepdim=True,) / lengths


def convolve1d(
    waveform,
    kernel,
    padding=0,
    pad_type="constant",
    stride=1,
    groups=1,
    use_fft=False,
    rotation_index=0,
):
    """Use torch.nn.functional to perform 1d padding and conv.

    Arguments
    ---------
    waveform : tensor
        The tensor to perform operations on.
    kernel : tensor
        The filter to apply during convolution
    padding : int or tuple
        The padding (pad_left, pad_right) to apply.
        If an integer is passed instead, this is passed
        to the conv1d function and pad_type is ignored.
    pad_type : str
        The type of padding to use. Passed directly to
        `torch.nn.functional.pad`, see PyTorch documentation
        for available options.
    stride : int
        The number of units to move each time convolution is applied.
        Passed to conv1d. Has no effect if `use_fft` is True.
    groups : int
        This option is passed to `conv1d` to split the input into groups for
        convolution. Input channels should be divisible by number of groups.
    use_fft : bool
        When `use_fft` is passed `True`, then compute the convolution in the
        spectral domain using complex multiply. This is more efficient on CPU
        when the size of the kernel is large (e.g. reverberation). WARNING:
        Without padding, circular convolution occurs. This makes little
        difference in the case of reverberation, but may make more difference
        with different kernels.
    rotation_index : int
        This option only applies if `use_fft` is true. If so, the kernel is
        rolled by this amount before convolution to shift the output location.

    Returns
    -------
    The convolved waveform.

    Example
    -------
    >>> import soundfile as sf
    >>> signal, rate = sf.read('samples/audio_samples/example1.wav')
    >>> signal = torch.tensor(signal[None, :, None])
    >>> filter = torch.rand(1, 10, 1, dtype=signal.dtype)
    >>> signal = convolve1d(signal, filter, padding=(9, 0))
    """
    if len(waveform.shape) != 3:
        raise ValueError("Convolve1D expects a 3-dimensional tensor")

    # Move time dimension last, which pad and fft and conv expect.
    waveform = waveform.transpose(2, 1)
    kernel = kernel.transpose(2, 1)

    # Padding can be a tuple (left_pad, right_pad) or an int
    if isinstance(padding, tuple):
        waveform = torch.nn.functional.pad(
            input=waveform, pad=padding, mode=pad_type,
        )

    # This approach uses FFT, which is more efficient if the kernel is large
    if use_fft:

        # Pad kernel to same length as signal, ensuring correct alignment
        zero_length = waveform.size(-1) - kernel.size(-1)

        # Handle case where signal is shorter
        if zero_length < 0:
            kernel = kernel[..., :zero_length]
            zero_length = 0

        # Perform rotation to ensure alignment
        zeros = torch.zeros(kernel.size(0), kernel.size(1), zero_length)
        after_index = kernel[..., rotation_index:]
        before_index = kernel[..., :rotation_index]
        kernel = torch.cat((after_index, zeros, before_index), dim=-1)

        # Compute FFT for both signals
        f_signal = torch.rfft(waveform, 1)
        f_kernel = torch.rfft(kernel, 1)

        # Complex multiply
        sig_real, sig_imag = f_signal.unbind(-1)
        ker_real, ker_imag = f_kernel.unbind(-1)
        f_result = torch.stack(
            [
                sig_real * ker_real - sig_imag * ker_imag,
                sig_real * ker_imag + sig_imag * ker_real,
            ],
            dim=-1,
        )

        # Inverse FFT
        convolved = torch.irfft(f_result, 1)

        # Because we're using `onesided`, sometimes the output's length
        # is increased by one in the time dimension. Truncate to ensure
        # that the length is preserved.
        if convolved.size(-1) > waveform.size(-1):
            convolved = convolved[..., : waveform.size(-1)]

    # Use the implemenation given by torch, which should be efficient on GPU
    else:
        convolved = torch.nn.functional.conv1d(
            input=waveform,
            weight=kernel,
            stride=stride,
            groups=groups,
            padding=padding if not isinstance(padding, tuple) else 0,
        )

    # Return time dimension to the second dimension.
    return convolved.transpose(2, 1)


def dB_to_amplitude(SNR):
    """Returns the amplitude ratio, converted from decibels.

    Arguments
    ---------
    SNR : float
        The ratio in decibels to convert.

    Example
    -------
    >>> round(dB_to_amplitude(SNR=10), 3)
    3.162
    >>> dB_to_amplitude(SNR=0)
    1.0
    """
    return 10 ** (SNR / 20)


def notch_filter(notch_freq, filter_width=101, notch_width=0.05):
    """Returns a notch filter constructed from a high-pass and low-pass filter.

    (from https://tomroelandts.com/articles/
    how-to-create-simple-band-pass-and-band-reject-filters)

    Arguments
    ---------
    notch_freq : float
        frequency to put notch as a fraction of the
        sampling rate / 2. The range of possible inputs is 0 to 1.
    filter_width : int
        Filter width in samples. Longer filters have
        smaller transition bands, but are more inefficient.
    notch_width : float
        Width of the notch, as a fraction of the sampling_rate / 2.

    Example
    -------
    >>> import soundfile as sf
    >>> signal, rate = sf.read('samples/audio_samples/example1.wav')
    >>> signal = torch.tensor(signal, dtype=torch.float32)[None, :, None]
    >>> kernel = notch_filter(0.25)
    >>> notched_signal = convolve1d(signal, kernel)
    """

    # Check inputs
    assert 0 < notch_freq <= 1
    assert filter_width % 2 != 0
    pad = filter_width // 2
    inputs = torch.arange(filter_width) - pad

    # Avoid frequencies that are too low
    notch_freq += notch_width

    # Define sinc function, avoiding division by zero
    def sinc(x):
        def _sinc(x):
            return torch.sin(x) / x

        # The zero is at the middle index
        return torch.cat([_sinc(x[:pad]), torch.ones(1), _sinc(x[pad + 1 :])])

    # Compute a low-pass filter with cutoff frequency notch_freq.
    hlpf = sinc(3 * (notch_freq - notch_width) * inputs)
    hlpf *= torch.blackman_window(filter_width)
    hlpf /= torch.sum(hlpf)

    # Compute a high-pass filter with cutoff frequency notch_freq.
    hhpf = sinc(3 * (notch_freq + notch_width) * inputs)
    hhpf *= torch.blackman_window(filter_width)
    hhpf /= -torch.sum(hhpf)
    hhpf[pad] += 1

    # Adding filters creates notch filter
    return (hlpf + hhpf).view(1, -1, 1)


def cov(xs, average=True):
    """ Computes the covariance matrices of the signals.

    Arguments:
    ----------
    xs : tensor
        A batch of audio signals in the frequency domain.
        The tensor must have the following format:
        (batch, time_step, n_fft, 2, n_mics)

    average : boolean
        Informs the method if it should return an average
        (computed on the time dimension) of the covariance
        matrices. Default value is True.

    Returns
    -------
    The covariance matrices. The tensor has the following
    format: (batch, time_step, n_fft, n_mics + n_pairs)

    Example
    -------
    >>> import soundfile as sf
    >>> import torch
    >>> from speechbrain.processing.features import STFT
    >>> from speechbrain.processing.signal_processing import cov
    >>> signal, fs = sf.read('samples/audio_samples/example_multichannel.wav')
    >>> signal = torch.tensor(signal).unsqueeze(0)
    >>> compute_stft = STFT(sample_rate=fs)
    >>> xs = compute_stft(signal)
    >>> rxx = cov(xs)
    """

    # Formating the real and imaginary parts
    xs_re = xs[..., 0, :].unsqueeze(4)
    xs_im = xs[..., 1, :].unsqueeze(4)

    # Computing the covariance
    rxx_re = torch.matmul(xs_re, xs_re.transpose(3, 4)) + torch.matmul(
        xs_im, xs_im.transpose(3, 4)
    )

    rxx_im = torch.matmul(xs_im, xs_re.transpose(3, 4)) - torch.matmul(
        xs_re, xs_im.transpose(3, 4)
    )

    # Selecting the upper triangular part of the covariance matrices
    n_channels = xs.shape[4]
    indices = torch.triu_indices(n_channels, n_channels)

    rxx_re = rxx_re[..., indices[0], indices[1]]
    rxx_im = rxx_im[..., indices[0], indices[1]]

    rxx = torch.stack((rxx_re, rxx_im), 3)

    if average is True:
        n_time_frames = rxx.shape[1]
        rxx = torch.mean(rxx, 1, keepdim=True)
        rxx = rxx.repeat(1, n_time_frames, 1, 1, 1)

    return rxx


def gccphat(rxx, eps=1e-20):
    """ Generalized Cross-Correlation with Phase Transform (GCC-PHAT)

    This function locates the source of a signal by doing a cross-correlation
    and a phase transform between each pair of microphone. It is assumed
    that the argument "onesided" of the STFT was set to True.

    Arguments
    ---------
    rxx : tensor
        The covariance matrices of the input signal. The tensor must
        have the following format (batch, time_steps, n_fft/2, 2, n_pairs)

    eps : float
        A small value to avoid divisions by 0 with the phase transform. The
        default value is 1e-20.

    Returns
    -------
    The cross-correlation values for each timestamp. The tensor has the
    following format (batch, time_steps, n_fft, n_mics + n_pairs)

    Example
    -------
    >>> import soundfile as sf
    >>> import torch
    >>> from speechbrain.processing.features import STFT
    >>> from speechbrain.processing.signal_processing import cov, gccphat
    >>> signal, fs = sf.read('samples/audio_samples/example_multichannel.wav')
    >>> signal = torch.tensor(signal).unsqueeze(0)
    >>> compute_stft = STFT(sample_rate=fs)
    >>> xs = compute_stft(signal)
    >>> rxx = cov(xs)
    >>> xxs = gccphat(rxx)
    >>> _, gccphat_delays = torch.max(xxs[0, :, :, 1], 1)
    """

    # Extracting the tensors for the operations
    rxx_values, rxx_indices = torch.unique(rxx, return_inverse=True, dim=1)

    rxx_re = rxx_values[..., 0, :]
    rxx_im = rxx_values[..., 1, :]

    # Phase transform
    rxx_abs = torch.sqrt(rxx_re ** 2 + rxx_im ** 2) + eps

    rxx_re_phat = rxx_re / rxx_abs
    rxx_im_phat = rxx_im / rxx_abs

    rxx_phat = torch.stack((rxx_re_phat, rxx_im_phat), 4)

    # Returning in the temporal domain
    rxx_phat = rxx_phat.transpose(2, 3)
    n_samples = int((rxx.shape[2] - 1) * 2)

    xxs = torch.irfft(rxx_phat, signal_ndim=1, signal_sizes=[n_samples])
    xxs = xxs[:, rxx_indices]

    # Formating the output
    xxs = xxs.transpose(2, 3)

    return xxs
