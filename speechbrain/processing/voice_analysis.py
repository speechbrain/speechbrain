"""
Functions for analyzing vocal characteristics: jitter, shimmer, HNR, and GNE.

These are typically used for analysis of dysarthric voices using more traditional approaches
(i.e. not deep learning). Often useful as a baseline for e.g. pathology detection. Inspired by PRAAT.

Authors
 * Peter Plantinga, 2024
"""

import torch
import torchaudio

# Minimum value for log measures, results in max of 30dB
EPSILON = 10**-3


@torch.no_grad()
def vocal_characteristics(
    audio: torch.Tensor,
    min_f0_Hz: int = 75,
    max_f0_Hz: int = 300,
    step_size: float = 0.01,
    window_size: float = 0.04,
    sample_rate: int = 16000,
    harmonicity_threshold: float = 0.45,
    jitter_threshold: float = 0.02,
):
    """Estimates the vocal characteristics of a signal using auto-correlation, etc.

    Arguments
    ---------
    audio: torch.Tensor
        The batched audio signal being analyzed, shape: [batch, sample].
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
        Threshold value for considering a frame as voiced. Computed
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
        The estimate for the harmonicity-to-noise ratio for each frame.
    """

    assert (
        audio.dim() == 2
    ), "Expected audio to be 2-dimensional, [batch, sample]"

    # Convert arguments to sample counts. Max lag corresponds to min f0 and vice versa.
    step_samples = int(step_size * sample_rate)
    window_samples = int(window_size * sample_rate)
    max_lag = int(sample_rate / min_f0_Hz)
    min_lag = int(sample_rate / max_f0_Hz)

    # Split into frames, and compute autocorrelation for each frame
    audio = torch.nn.functional.pad(audio, (0, window_samples))
    frames = audio.unfold(dimension=-1, size=window_samples, step=step_samples)
    autocorrelation = autocorrelate(frames)

    # Use autocorrelation to estimate harmonicity and best lags
    harmonicity, best_lags = autocorrelation[:, :, min_lag:max_lag].max(dim=-1)
    # Re-add the min_lag back in after previous step removed it
    best_lags = best_lags + min_lag
    # Frequency is 1 / period-slash-lag
    estimated_f0 = sample_rate / best_lags

    # Compute period-based features using the estimate of best lag
    jitter, shimmer = compute_periodic_features(frames, best_lags)

    # Use harmonicity features to determine which frames are voiced
    voiced = compute_voiced(
        harmonicity, jitter, harmonicity_threshold, jitter_threshold
    )

    # Autocorrelation is the measure of harmonicity here, 1-harmonicity is noise
    # See "Harmonic to Noise Ratio Measurement - Selection of Window and Length"
    # By J. Fernandez, F. Teixeira, V. Guedes, A. Junior, and J. P. Teixeira
    # Term is dominated by denominator, so just take -1 * log(noise)
    # max value for harmonicity is 25 dB, enforced by this minimum here
    noise = torch.clamp(1 - harmonicity, min=EPSILON)
    hnr = -10 * torch.log10(noise)

    return estimated_f0, voiced, jitter, shimmer, hnr


def autocorrelate(frames):
    """Generate autocorrelation scores using circular convolution.

    Arguments
    ---------
    frames: torch.Tensor
        The audio frames to be evaluated for autocorrelation, shape [batch, frame, sample]

    Returns
    -------
    autocorrelation: torch.Tensor
        The ratio of the best candidate lag's autocorrelation score against
        the theoretical maximum autocorrelation score at lag 0.
        Normalized by the autocorrelation_score of the window.
    """
    # Apply hann window to the audio to reduce edge effects
    window_size = frames.size(-1)
    hann = torch.hann_window(window_size, device=frames.device).view(1, 1, -1)
    autocorrelation = compute_cross_correlation(frames * hann, frames * hann)

    # Score should be normalized by the autocorrelation of the window
    # See 'Accurate Short-Term Analysis of the Fundamental Frequency
    # and the Harmonics-To-Noise Ratio of a Sampled Sound' by Boersma
    norm_score = compute_cross_correlation(hann, hann).clamp(min=1e-10)
    return autocorrelation / norm_score


def compute_periodic_features(frames, best_lags):
    """Function to compute periodic features: jitter, shimmer

    Arguments
    ---------
    frames: torch.Tensor
        The framed audio to use for feature computation, dims [batch, frame, sample].
    best_lags: torch.Tensor
        The estimated period length for each frame, dims [batch, frame]

    Returns
    -------
    jitter: torch.Tensor
        The average absolute deviation in period over the frame.
    shimmer: torch.Tensor
        The average absolute deviation in amplitude over the frame.
    """

    # Compute likely peaks using topk
    topk = frames.topk(dim=-1, k=7)

    # Jitter = average variation in period length, measured as lag
    lags = topk.indices.remainder(best_lags.unsqueeze(-1))
    # Compute lags as remainder, use min to avoid wraparound errors
    lags = torch.min(lags, best_lags.unsqueeze(-1) - lags)

    # Compute mean difference from mean lag, normalized by period
    jitter_frames = (lags - lags.float().mean(dim=-1, keepdims=True)).abs()
    jitter = jitter_frames.mean(dim=-1) / best_lags

    # Shimmer = average variation in amplitude, normalized by avg amplitude
    avg_amps = topk.values.mean(dim=-1, keepdims=True)
    amp_diff = (topk.values - avg_amps).abs()
    shimmer = amp_diff.mean(dim=-1) / avg_amps.squeeze(-1).clamp(min=1e-10)

    return jitter, shimmer


def compute_voiced(
    harmonicity: torch.Tensor,
    jitter: torch.Tensor,
    harmonicity_threshold: float = 0.45,
    jitter_threshold: float = 0.02,
    minimum_voiced: int = 19,
):
    """
    Compute which sections are voiced based on two criteria:
     * normalized autocorrelation above threshold
     * AND jitter below threshold

    Voicing is averaged with neighboring frames to avoid rapid changes in voicing.
    If no frames are voiced, relax threshold until more than a few frames are voiced.

    Arguments
    ---------
    harmonicity : torch.Tensor
        The normalized autocorrelation score, between 0 and 1 for each frame.
        Shape = [batch, frame]
    jitter : torch.Tensor
        The variation in period from frame to frame, between 0 and 1 for each frame.
    harmonicity_threshold : float
        The threshold above which to consider each frame as voiced.
    jitter_threshold: float
        Jitter values greater than this are conisdered unvoiced.
    minimum_voiced : int
        The minimum number of frames to consider voiced.

    Returns
    -------
    voiced : torch.Tensor
        A boolean value for each frame, whether to consider the frame as voiced or unvoiced.
    """
    h_threshold = torch.full(
        size=(jitter.size(0),),
        fill_value=harmonicity_threshold,
        device=jitter.device,
        requires_grad=False,
    )
    j_threshold = torch.full_like(h_threshold, fill_value=jitter_threshold)
    threshold_unmet = torch.ones_like(h_threshold)

    # Check on each iteration if we have more than minimum voiced frames
    while any(threshold_unmet):
        voiced = harmonicity > h_threshold
        voiced &= jitter < j_threshold

        # PRAAT uses some forward/backward search to find voicing, with a
        # penalty for on/off voicing. For speed, we just take an average.
        voiced = neighbor_average(voiced, minimum_voiced).round().bool()

        # Relax the threshold by a bit for each iteration
        threshold_unmet = voiced.sum(dim=1) < minimum_voiced
        h_threshold[threshold_unmet] -= 0.05
        j_threshold[threshold_unmet] += 0.01

    return voiced


def neighbor_average(values, neighbors):
    """Convenience function for average pooling of neighbors.

    Arguments
    ---------
    values : torch.Tensor
        The sequence of values to avg pool along last dimension.
    neighbors : int
        Should be an odd value, includes center.

    Returns
    -------
    averaged_values : torch.Tensor
        The 1-d average-pooled values across the last dimension.

    Examples
    --------
    >>> a = torch.ones(7).view(1, -1)
    >>> a[:, ::2] = 0
    >>> a
    tensor([[0., 1., 0., 1., 0., 1., 0.]])
    >>> neighbor_average(a, neighbors=3)
    tensor([[0.5000, 0.3333, 0.6667, 0.3333, 0.6667, 0.3333, 0.5000]])
    """
    return torch.nn.functional.avg_pool1d(
        input=values.float(),
        kernel_size=neighbors,
        stride=1,
        padding=neighbors // 2,
        count_include_pad=False,
    )


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

    # Collapse frame and batch into same dimension, for lfiltering
    batch, frame_count, _ = autocorrelation.shape
    autocorrelation = autocorrelation.view(batch * frame_count, -1)
    reshaped_frames = frames.view(batch * frame_count, -1)

    # An autocorrelation of all 0's -- which can happen in padding -- leads to
    # an error with the linear system solver, as the matrix is singular
    # We fix this by ensuring the zero-lag correlation is always 1
    autocorrelation[:, lpc_order] = 1.0

    # Construct Toeplitz matrices (one per frame)
    # This is [[p0, p1, p2...], [p1, p0, p1...], [p2, p1, p0...] ...]
    # Our sliding window should go from the end to the front, so flip
    # Also, we have one more value on each end than we need, for the target values
    R = autocorrelation[:, 1:-1].unfold(-1, lpc_order, 1).flip(dims=(1,))
    r = autocorrelation[:, lpc_order + 1 :]

    # Solve for LPC coefficients, generate inverse filter with coeffs 1, -b_1, ...
    lpc = torch.linalg.solve(R, r)
    lpc_coeffs = torch.nn.functional.pad(-lpc, (1, 0), value=1)
    a_coeffs = torch.zeros_like(lpc_coeffs)
    a_coeffs[:, 0] = 1

    # Perform filtering
    inverse_filtered = torchaudio.functional.lfilter(
        reshaped_frames, a_coeffs, lpc_coeffs, clamp=False
    )

    # Un-collapse batch and frames
    return inverse_filtered.view(batch, frame_count, -1)


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
    window = torch.hann_window(window_bins.sum(), device=mask.device)
    mask[:, :, window_bins] = window

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
        The two sets of frames to compare using cross-correlation,
        shape [batch, frame, sample]
    width : int, default is None
        The number of samples before and after 0 lag. A width of 3 returns 7 results.
        If None, 0 lag is put at the front, and the result is 1/2 the original length + 1,
        a nice default for autocorrelation as there are no repeated values.

    Returns
    -------
    The cross-correlation between frames_a and frames_b.

    Example
    -------
    >>> frames = torch.arange(10).view(1, 1, -1).float()
    >>> compute_cross_correlation(frames, frames, width=3)
    tensor([[[0.6316, 0.7193, 0.8421, 1.0000, 0.8421, 0.7193, 0.6316]]])
    >>> compute_cross_correlation(frames, frames)
    tensor([[[1.0000, 0.8421, 0.7193, 0.6316, 0.5789, 0.5614]]])
    """
    # Padding is used to control the number of outputs
    batch_size, frame_count, frame_size = frames_a.shape
    pad = (0, frame_size // 2) if width is None else (width, width)
    padded_frames_a = torch.nn.functional.pad(frames_a, pad, mode="circular")

    # Cross-correlation with conv1d, by keeping each frame as its own channel
    # The batch and frame channel have to be combined due to conv1d restrictions
    merged_size = batch_size * frame_count
    reshaped_a = padded_frames_a.view(1, merged_size, -1)
    reshaped_b = frames_b.view(merged_size, 1, -1)

    cross_correlation = torch.nn.functional.conv1d(
        input=reshaped_a, weight=reshaped_b, groups=merged_size
    )

    # Separate out the batch and frame dimensions again
    cross_correlation = cross_correlation.view(batch_size, frame_count, -1)

    # Normalize
    norm = torch.sqrt((frames_a**2).sum(dim=-1) * (frames_b**2).sum(dim=-1))
    cross_correlation /= norm.unsqueeze(-1).clamp(min=1e-10)

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
        The batched audio signal to use for GNE computation, [batch, sample]
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

    assert (
        audio.dim() == 2
    ), "Expected audio to be 2-dimensional, [batch, sample]"

    # Step 1. Downsample to 10 kHz since voice energy is low above 5 kHz
    old_sample_rate, sample_rate = sample_rate, 10000
    audio = torchaudio.functional.resample(audio, old_sample_rate, sample_rate)

    # Step 2. Inverse filter with 30-msec window, 10-msec hop and 13th order LPC
    frame_size, hop_size, order = 300, 100, 13
    window = torch.hann_window(frame_size, device=audio.device).view(1, 1, -1)
    audio = torch.nn.functional.pad(audio, (0, frame_size))
    frames = audio.unfold(dimension=-1, size=frame_size, step=hop_size) * window
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
    gne = torch.stack(correlations, dim=-1).amax(dim=(2, 3))

    # Use a log scale for better differentiation
    return -10 * torch.log10(torch.clamp(1 - gne, min=EPSILON))
