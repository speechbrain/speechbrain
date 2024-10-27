"""
Functions for analyzing vocal characteristics: jitter, shimmer, and HNR.

These are typically used for analysis of dysarthric voices using more traditional approaches (i.e. not deep learning). Often useful as a baseline for e.g. pathology detection. Inspired by PRAAT.

Authors
 * Peter Plantinga, 2024
"""

import torch
import torchaudio


def vocal_characteristics(
    audio: torch.Tensor,
    min_f0_Hz: int = 75,
    max_f0_Hz: int = 300,
    step_size: float = 0.01,
    window_size: float = 0.04,
    sample_rate: int = 16000,
    harmonicity_threshold: float = 0.45,
    jitter_threshold: float = 0.05,
):
    """Estimates the vocal characteristics of a signal using auto-correlation.
    Batched estimation is hard due to removing unvoiced frames, so only accepts a single sample.

    Arguments
    ---------
    audio: torch.Tensor
        The audio signal being analyzed, shape: [time].
    min_f0_Hz: int
        The minimum allowed fundamental frequency, to reduce octave errors.
        Default is 75 Hz, based on human voice standard frequency range.
    max_f0_Hz: int
        The maximum allowed fundamental frequency, to reduce octave errors.
        Default is 400 Hz, based on human voice standard frequency range.
    step_size: float
        The time between analysis windows (in seconds).
    window_size: float
        The size of the analysis window (in seconds).
    sample_rate: int
        The number of samples in a second.
    harmonicity_threshold: float
        One of two threshold values for considering a frame as voiced. Computed
        as the ratio between lag 0 autocorrelation and t-max autocorrelation.
    jitter_threshold: float
        One of two threshold values for considering a frame as voiced. Estimated
        jitter values greater than this are conisdered unvoiced.

    Returns
    -------
    estimated_f0: torch.Tensor
        A per-frame estimate of the f0 in Hz.
    voiced_frames: torch.Tensor
        The estimate for each frame if it is voiced or unvoiced.
    jitter: torch.Tensor
        The estimate for the jitter value for each frame.
    shimmer: torch.Tensor
        The estimate for the shimmer value for each frame.
    hnr: torch.Tensor
        The estimate for the HNR value for each frame.
    """

    if audio.dim() != 1:
        raise ValueError("Takes audio tensors of only one dimension (time)")

    # Convert arguments to sample counts. Max lag corresponds to min f0 and vice versa.
    step_samples = int(step_size * sample_rate)
    window_samples = int(window_size * sample_rate)
    max_lag = int(sample_rate / min_f0_Hz)
    min_lag = int(sample_rate / max_f0_Hz)

    # Split into frames, and compute autocorrelation for each frame
    frames = audio.view(1, -1).unfold(-1, window_samples, step_samples)
    autocorrelation = autocorrelate(frames)

    # Use autocorrelation to compute harmonicity and best lags
    kbest_scores = torch.topk(autocorrelation[:, min_lag:max_lag], k=15, dim=-1)
    harmonicity = kbest_scores.values[:, 0]
    kbest_lags = kbest_scores.indices + min_lag

    # Compute period-based features, dividing frames based on median best lag
    jitter, shimmer = compute_periodic_features(audio, kbest_lags[:, 0])

    # Use both types of features to determine which frames are voiced
    voiced = compute_voiced(
        harmonicity, jitter, harmonicity_threshold, jitter_threshold
    )

    # Use neighboring frames to select best lag out of available options
    best_lags = iterative_lag_selection(kbest_lags[voiced], iterations=3)
    estimated_f0 = sample_rate / best_lags

    # Autocorrelation is the measure of harmonicity here, 1-harmonicity is noise
    # See "Harmonic to Noise Ratio Measurement - Selection of Window and Length"
    # By J. Fernandez, F. Teixeira, V. Guedes, A. Junior, and J. P. Teixeira
    hnr = 10 * torch.log10(harmonicity / torch.abs(1 - harmonicity))

    return estimated_f0, voiced, jitter, shimmer, hnr


def autocorrelate(frames):
    """Generate autocorrelation scores using circular convolution.

    Arguments
    ---------
    frames: torch.Tensor
        The audio frames to be evaluated for autocorrelation, shape [time, window]

    Returns
    -------
    autocorrelation: torch.Tensor
        The ratio of the best candidate lag's autocorrelation score against
        the theoretical maximum autocorrelation score at lag 0.
        Normalized by the autocorrelation_score of the window.
    """
    # Apply hann window to the audio to reduce edge effects
    hann = torch.hann_window(frames.size(-1)).view(1, 1, -1)
    autocorrelation = compute_cross_correlation(frames * hann, frames * hann)

    # Score should be normalized by the autocorrelation of the window
    # See 'Accurate Short-Term Analysis of the Fundamental Frequency
    # and the Harmonics-To-Noise Ratio of a Sampled Sound' by Boersma
    norm_score = compute_cross_correlation(hann, hann)
    return autocorrelation / norm_score


def iterative_lag_selection(kbest_lags, iterations=3):
    """Select the best lag out of available options by comparing
    to an average of neighboring lags to reduce jumping octaves."""
    # kbest returns sorted list, first entry should be the highest autocorrelation
    best_lag = kbest_lags[:, 0]
    for i in range(iterations):
        averaged_lag = neighbor_average(best_lag.float(), neighbors=7)
        distance = torch.abs(kbest_lags - averaged_lag.unsqueeze(1))
        best_selection = torch.argmin(distance, dim=-1)
        best_lag = select_indices(kbest_lags, best_selection)
    return best_lag


def select_indices(tensor, indices):
    """Utility function to extract a list of indices from a tensor"""
    return tensor[torch.arange(tensor.size(0)), indices]


def neighbor_average(best_values, neighbors):
    """Use convolutional kernel to average the neighbors."""
    kernel = torch.ones(1, 1, neighbors) / neighbors
    values = torch.nn.functional.pad(
        best_values[None, None, :],
        pad=(neighbors // 2, neighbors // 2),
        mode="reflect",
    )

    return torch.nn.functional.conv1d(values, kernel).squeeze()


def compute_periodic_features(audio, best_lag):
    """Function to compute periodic features: jitter, shimmer

    Compares to 5 neighboring periods, like PRAAT's ppq5 and apq5.

    Arguments
    ---------
    audio: torch.Tensor
        The audio to use for feature computation.
    best_lag: torch.Tensor
        The average period length for each frame.

    Returns
    -------
    jitter: torch.Tensor
        The average absolute deviation in period over the frame.
    shimmer: torch.Tensor
        The average absolute deviation in amplitude over the frame.
    """

    # Find median best lag to divide entire audio. We use median
    # instead of mean because the distribution may be bimodal
    # in some cases due to bouncing between octaves.
    lag = best_lag.median()

    # Divide into period-length segments
    back_offset = len(audio) - len(audio) % lag
    periods = audio[:back_offset].view(-1, lag)

    # Compare peak index of each period to its neighbors. Divide by lag to get relative.
    # Here we use min of lag and 1 - lag to avoid wraparound errors
    peak_lags = periods.argmax(dim=1).float()
    peak_lags = torch.min(peak_lags, periods.size(1) - peak_lags)
    avg_lags = neighbor_average(peak_lags, neighbors=5)
    jitter = torch.abs(peak_lags - avg_lags) / lag
    jitter = neighbor_average(jitter, neighbors=5)

    # Compare amplitude of each peak to its successor. Divide by average to get relative.
    amp = periods.max(dim=1).values
    avg_amp = neighbor_average(amp, neighbors=5)
    shimmer = torch.abs(amp - avg_amp) / amp.mean()
    shimmer = neighbor_average(shimmer, neighbors=5)

    # Reduce length of jitter and shimmer to match lags
    jitter = match_len(jitter, len(best_lag))
    shimmer = match_len(shimmer, len(best_lag))

    return jitter, shimmer


def match_len(signal, length):
    """Use interpolation to convert the length of one signal to another."""
    return torch.nn.functional.interpolate(
        signal.view(1, 1, -1), size=length, mode="linear"
    ).squeeze()


def compute_voiced(
    harmonicity: torch.Tensor,
    jitter: torch.Tensor,
    harmonicity_threshold: float = 0.45,
    jitter_threshold: float = 0.05,
    minimum_voiced: int = 5,
):
    """
    Compute which sections are voiced based on two criteria (adapted from PRAAT):
     * normalized autocorrelation above threshold
     * AND jitter below threshold

    Voicing is averaged with neighboring frames to avoid rapid changes in voicing.
    If no frames are voiced, relax thresholds until more than one frame is voiced.

    Arguments
    ---------
    harmonicity : torch.Tensor
        The normalized autocorrelation score, a number between 0 and 1 for each frame.
    jitter : torch.Tensor
        The jitter score, a number between 0 and 1 for each frame.
    harmonicity_threshold : float
        The threshold above which to consider each frame as voiced.
    jitter_threshold : float
        The threshold below which to consider each frame as voiced.
    minimum_voiced : int
        The minimum number of frames to consider voiced.

    Returns
    -------
    voiced : torch.Tensor
        A boolean value for each frame, whether to consider the frame as voiced or unvoiced.
    """
    voiced, h_tweak, j_tweak = torch.zeros(len(harmonicity)), 0.0, 0.0

    # Check on each iteration if we have more than minimum voiced frames
    while voiced.sum() < minimum_voiced:
        voiced = harmonicity > harmonicity_threshold - h_tweak
        voiced &= jitter < jitter_threshold + j_tweak
        voiced = neighbor_average(voiced.float(), neighbors=21).round().bool()

        # Relax the threshold by a bit for each iteration
        h_tweak += 0.05
        j_tweak += 0.01

    return voiced


def inverse_filter(frames, lpc_order=13):
    """Perform inverse filtering on frames to estimate glottal pulse train.

    Uses autocorrelation method and Linear Predictive Coding (LPC).
    Algorithm from https://course.ece.cmu.edu/~ece792/handouts/RS_Chap_LPC.pdf

    Arguments
    ---------
    frames : torch.Tensor
        The audio frames to filter using inverse filter.
    lpc_order : int
        The size of the filter to compute and use on the frames.

    Returns
    -------
    filtered_frames : torch.Tensor
        The frames after the inverse filter is applied
    """
    # Only lpc_order autocorrelation values are needed
    autocorrelation = compute_cross_correlation(frames, frames, width=lpc_order)

    # Construct Toeplitz matrices (one per frame)
    # This is [[p0, p1, p2...], [p1, p0, p1...], [p2, p1, p0...] ...]
    # Our sliding window should go from the end to the front, so flip
    # Also, we have one more value on each end than we need, for the target values
    R = autocorrelation[:, 1:-1].unfold(-1, lpc_order, 1).flip(dims=(1,))
    r = autocorrelation[:, lpc_order + 1 :]

    # Solve for LPC coefficients, generate inverse filter with coeffs 1, -b_1, ...
    lpc = torch.linalg.solve(R, r)
    lpc_coeffs = torch.cat((torch.ones(lpc.size(0), 1), -lpc), dim=1)
    a_coeffs = torch.zeros_like(lpc_coeffs)
    a_coeffs[:, 0] = 1

    # Perform filtering
    return torchaudio.functional.lfilter(
        frames, a_coeffs, lpc_coeffs, clamp=False
    )


def compute_hilbert_envelopes(
    frames, center_freq, bandwidth=1000, sample_rate=10000
):
    """Compute the hilbert envelope of the signal in a specific frequency band using FFT.

    Arguments
    ---------
    frames : torch.Tensor
        A set of frames from a signal for which to compute envelopes.
    center_freq : float
        The target frequency for the envelope.
    bandwidth : float
        The size of the band to use for the envelope.
    sample_rate : float
        The number of samples per second in the frame signals.

    Returns
    -------
    envelopes : torch.Tensor
        The computed envelopes.
    """

    # Step 0. Compute low/high freq for window
    low_freq = center_freq - bandwidth / 2
    high_freq = center_freq + bandwidth / 2

    # Step 1. Compute DFT for each frame
    spectra = torch.fft.fft(frames)
    freqs = torch.fft.fftfreq(spectra.size(-1), 1 / sample_rate)

    # Step 2. Mask with hann window in the frequency range (negative freqs are 0)
    mask = torch.zeros_like(spectra, dtype=torch.float)
    window_bins = (low_freq < freqs) & (freqs < high_freq)
    mask[:, :, window_bins] = torch.hann_window(window_bins.sum())

    # Step 3. Apply inverse DFT to get complex time-domain signal
    analytic_signal = torch.fft.ifft(spectra * mask)

    # Step 4. Take absolute value to get final envelopes
    return analytic_signal.abs()


def compute_cross_correlation(frames_a, frames_b, width=None):
    """Computes the correlation between two sets of frames.

    Arguments
    ---------
    frames_a : torch.Tensor
    frames_b : torch.Tensor
        The two sets of frames to compare using cross-correlation
    width : int, default is None
        The number of samples before and after 0 lag. A width of 3 returns 7 results.
        If None, 0 lag is put at the front, and the result is 1/2 the original length + 1,
        a nice default for autocorrelation as there are no repeated values.

    Returns
    -------
    The cross-correlation between frames_a and frames_b.

    Example
    -------
    >>> frames = torch.arange(10).view(1, 1, -1)
    >>> compute_cross_correlation(frames, frames, width=3)
    tensor([[0.6316, 0.7193, 0.8421, 1.0000, 0.8421, 0.7193, 0.6316]])
    >>> compute_cross_correlation(frames, frames)
    tensor([[1.0000, 0.8421, 0.7193, 0.6316, 0.5789, 0.5614]])
    """
    # Padding is used to control the number of outputs
    pad = (0, frames_a.size(-1) // 2) if width is None else (width, width)
    padded_frames_a = torch.nn.functional.pad(frames_a, pad, mode="circular")

    # Compute correlation, treating frames independently
    cross_correlation = torch.nn.functional.conv1d(
        input=padded_frames_a,
        weight=frames_b.transpose(0, 1),
        groups=frames_b.size(1),
    ).squeeze(0)

    # Normalize
    norm = torch.sqrt((frames_a**2).sum(dim=-1) * (frames_b**2).sum(dim=-1))
    cross_correlation /= norm.view(-1, 1)

    return cross_correlation


def compute_gne(audio, sample_rate=16000, bandwidth=1000, fshift=300):
    """An algorithm for GNE computation from the original paper:

    "Glottal-to-Noise Excitation Ratio - a New Measure for Describing
    Pathological Voices" by D. Michaelis, T. Oramss, and H. W. Strube.

    This algorithm divides the signal into frequency bands, and compares
    the correlation between the bands. High correlation indicates a
    relatively low amount of noise in the signal, whereas lower correlation
    could be a sign of pathology in the vocal signal.

    Godino-Llorente et al. in "The Effectiveness of the Glottal to Noise
    Excitation Ratio for the Screening of Voice Disorders." explore the
    goodness of the bandwidth and frequency shift parameters, the defaults
    here are the ones recommended in that work. They also suggest using
    log( 1 - GNE ), which they called GNE_L as the final score, as done here.

    Arguments
    ---------
    audio : torch.Tensor
        The audio signal to use for GNE computation.
    sample_rate : float
        The sample rate of the input audio.
    bandwidth : float
        The width of the frequency bands used for computing correlation.
    fshift : float
        The shift between frequency bands used for computing correlation.

    Returns
    -------
    gne_score : torch.Tensor
        The GNE_L score for each frame of the audio signal.
    """

    # Step 1. Downsample to 10 kHz since voice energy is low above 5 kHz
    old_sample_rate, sample_rate = sample_rate, 10000
    audio = torchaudio.functional.resample(audio, old_sample_rate, sample_rate)

    # Step 2. Inverse filter with 30-msec window, 10-msec hop and 13th order LPC
    frame_size, hop_size, order = 300, 100, 13
    window = torch.hann_window(frame_size).view(1, 1, -1)
    frames = audio.view(1, -1).unfold(-1, frame_size, hop_size) * window
    excitation_frames = inverse_filter(frames, order)

    # Step 3. Compute Hilbert envelopes for each frequency bin
    min_freq, max_freq = bandwidth // 2, sample_rate // 2 - bandwidth // 2
    center_freqs = range(min_freq, max_freq, fshift)
    envelopes = {
        center_freq: compute_hilbert_envelopes(
            excitation_frames, center_freq, bandwidth, sample_rate
        )
        for center_freq in center_freqs
    }

    # Step 4. Compute cross correlation between (non-neighboring) frequency bins
    correlations = [
        compute_cross_correlation(envelopes[freq_i], envelopes[freq_j], width=3)
        for freq_i in center_freqs
        for freq_j in center_freqs
        if freq_j - freq_i > bandwidth // 2
    ]

    # Step 5. The maximum cross-correlation is the GNE score
    gne = torch.stack(correlations, dim=-1).amax(dim=(1, 2))

    # Use a log scale for better differentiation
    return 10 * torch.log10(1 - gne)
