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
PERIODIC_NEIGHBORS = 4


@torch.no_grad()
def vocal_characteristics(
    audio: torch.Tensor,
    min_f0_Hz: int = 80,
    max_f0_Hz: int = 300,
    step_size: float = 0.01,
    window_size: float = 0.05,
    sample_rate: int = 16000,
    harmonicity_threshold: float = 0.45,
    jitter_threshold: float = 0.02,
    log_scores: bool = True,
):
    """Estimates the vocal characteristics of a signal using auto-correlation, etc.

    Arguments
    ---------
    audio: torch.Tensor
        The batched audio signal being analyzed, shape: [batch, sample].
    min_f0_Hz: int
        The minimum allowed fundamental frequency, to reduce octave errors.
        Default is 80 Hz, based on human voice standard frequency range.
    max_f0_Hz: int
        The maximum allowed fundamental frequency, to reduce octave errors.
        Default is 300 Hz, based on human voice standard frequency range.
    step_size: float
        The time between analysis windows (in seconds).
    window_size: float
        The size of the analysis window (in seconds). Must be long enough
        to contain at least 4 periods at the minimum frequency.
    sample_rate: int
        The number of samples in a second.
    harmonicity_threshold: float
        Threshold value for considering a frame as voiced. Computed
        as the ratio between lag 0 autocorrelation and t-max autocorrelation.
    jitter_threshold: float
        One of two threshold values for considering a frame as voiced. Estimated
        jitter values greater than this are conisdered unvoiced.
    log_scores: bool
        Whether to represent the jitter/shimmer/hnr on a log scale

    Returns
    -------
    features: torch.Tensor
        A [batch, frame, 12] tensor with the following features per-frame.
         * autocorr_f0: A per-frame estimate of the f0 in Hz.
         * autocorr_hnr: harmonicity-to-noise ratio for each frame.
         * periodic_jitter: Average deviation in period length.
         * periodic_shimmer: Average deviation in amplitude per period.
         * spectral_centroid: "center-of-mass" for spectral frames.
         * spectral_spread: avg distance from centroid for spectral frames.
         * spectral_skew: asymmetry of spectrum about the centroid.
         * spectral_kurtosis: tailedness of spectrum.
         * spectral_entropy: The peakiness of the spectrum.
         * spectral_flatness: The ratio of geometric mean to arithmetic mean.
         * spectral_crest: The ratio of spectral maximum to arithmetic mean.
         * spectral_flux: The 2-normed diff between successive spectral values.
    """

    assert (
        audio.dim() == 2
    ), "Expected audio to be 2-dimensional, [batch, sample]"

    # Convert arguments to sample counts. Max lag corresponds to min f0 and vice versa.
    step_samples = int(step_size * sample_rate)
    window_samples = int(window_size * sample_rate)
    max_lag = int(sample_rate / min_f0_Hz)
    min_lag = int(sample_rate / max_f0_Hz)

    assert (
        max_lag * PERIODIC_NEIGHBORS <= window_samples
    ), f"Need at least {PERIODIC_NEIGHBORS} periods in a window"

    # Split into frames, and compute autocorrelation for each frame
    frames = audio.unfold(dimension=-1, size=window_samples, step=step_samples)
    autocorrelation = autocorrelate(frames)

    # Use autocorrelation to estimate harmonicity and best lags
    harmonicity, best_lags = autocorr_feats(autocorrelation, min_lag, max_lag)
    # Frequency is 1 / period-slash-lag
    estimated_f0 = sample_rate / best_lags

    # Compute period-based features using the estimate of best lag
    jitter, shimmer = compute_periodic_features(frames, best_lags)

    # Compute spectral features
    spectral_features = compute_spectral_features(frames)

    # Autocorrelation is the measure of harmonicity here, 1-harmonicity is noise
    # See "Harmonic to Noise Ratio Measurement - Selection of Window and Length"
    # By J. Fernandez, F. Teixeira, V. Guedes, A. Junior, and J. P. Teixeira
    # Term is dominated by denominator, so just take -1 * log(noise)
    if log_scores:
        # max value for harmonicity is 30 dB, enforced by this minimum here
        noise = torch.clamp(1 - harmonicity, min=EPSILON)
        hnr = -10 * torch.log10(noise)
        jitter = -10 * torch.log10(jitter.clamp(min=EPSILON))
        shimmer = -10 * torch.log10(shimmer.clamp(min=EPSILON))
    else:
        hnr = 1 - harmonicity

    # Combine all features into a single tensor
    features = torch.stack((estimated_f0, hnr, jitter, shimmer), dim=-1)
    features = torch.cat((features, spectral_features), dim=-1)

    # Compute moving average of 3 frames (as OpenSMILE does)
    features = moving_average(features, dim=1, n=3)

    return features


def moving_average(features, dim=1, n=3):
    """Computes moving average on a given dimension.

    Arguments
    ---------
    features: torch.Tensor
        The feature tensor to smooth out.
    dim: int
        The time dimension (for smoothing).
    n: int
        The number of points in the moving average

    Returns
    -------
    smoothed_features: torch.Tensor
        The features after the moving average is applied.

    Example
    -------
    >>> feats = torch.tensor([[0., 1., 0., 1., 0., 1., 0.]])
    >>> moving_average(feats)
    tensor([[0.5000, 0.3333, 0.6667, 0.3333, 0.6667, 0.3333, 0.5000]])
    """
    features = features.transpose(dim, -1)

    pad = n // 2
    features = torch.nn.functional.avg_pool1d(
        features, kernel_size=n, padding=pad, stride=1, count_include_pad=False
    )

    return features.transpose(dim, -1)


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


def autocorr_feats(autocorrelation, min_lag, max_lag, neighbors=5):
    """Compute features based on autocorrelation"""
    # Find high values
    harmonicity, lags = autocorrelation[:, :, min_lag:max_lag].max(dim=-1)

    # Take median value of 5 neighboring cells to avoid octave errors
    lags = torch.nn.functional.pad(lags, pad=(2, 2))
    best_lags, _ = lags.unfold(-1, neighbors, 1).median(dim=-1)

    # Re-add the min_lag back in after first step removed it
    best_lags = best_lags + min_lag

    return harmonicity, best_lags


def compute_periodic_features(frames, best_lags, neighbors=PERIODIC_NEIGHBORS):
    """Function to compute periodic features: jitter, shimmer

    Arguments
    ---------
    frames: torch.Tensor
        The framed audio to use for feature computation, dims [batch, frame, sample].
    best_lags: torch.Tensor
        The estimated period length for each frame, dims [batch, frame].
    neighbors: int
        Number of neighbors to use in comparison.

    Returns
    -------
    jitter: torch.Tensor
        The average absolute deviation in period over the frame.
    shimmer: torch.Tensor
        The average absolute deviation in amplitude over the frame.
    """

    # Prepare for masking
    masked_frames = torch.clone(frames).detach()
    mask_indices = torch.arange(frames.size(-1), device=frames.device)
    mask_indices = mask_indices.view(1, 1, -1).expand(frames.shape)
    periods = best_lags.unsqueeze(-1)
    period_indices = mask_indices.remainder(periods)

    # Mask everything not within about 20% (1/5) of a period peak
    jitter_range = periods // 5
    peak, lag = torch.max(masked_frames, dim=-1, keepdim=True)

    # Handle lags close to period by checking +-1 period
    lag_indices = lag.remainder(periods)
    mask = (period_indices < lag_indices - jitter_range) & (
        period_indices > lag_indices - periods + jitter_range
    ) | (period_indices > lag_indices + jitter_range) & (
        period_indices < lag_indices + periods - jitter_range
    )
    masked_frames[mask] = 0

    # Find neighboring peaks
    peaks, lags = [], []
    for i in range(neighbors):
        peak, lag = torch.max(masked_frames, dim=-1, keepdim=True)
        mask = (mask_indices > lag - periods // 2) & (
            mask_indices < lag + periods // 2
        )
        masked_frames[mask] = 0
        peaks.append(peak.squeeze(-1))
        lags.append(lag.squeeze(-1))

    peaks = torch.stack(peaks, dim=-1)
    lags = torch.stack(lags, dim=-1)

    # Jitter = average variation in period length
    # Compute mean difference from mean lag, normalized by period
    lags = lags.remainder(periods)
    lags = torch.minimum(lags, periods - lags)
    jitter_frames = (lags - lags.float().mean(dim=-1, keepdims=True)).abs()
    jitter = jitter_frames.mean(dim=-1) / best_lags

    # Shimmer = average variation in amplitude
    # Computed as mean difference from mean amplitude, normalized by avg amplitude
    avg_amps = peaks.mean(dim=-1, keepdims=True)
    amp_diff = (peaks - avg_amps).abs()
    shimmer = amp_diff.mean(dim=-1) / avg_amps.squeeze(-1).clamp(min=1e-10)

    return jitter, shimmer


def compute_spectral_features(frames, eps=1e-10):
    """Compute statistical measures on spectral frames
    such as flux, skew, spread, flatness.

    Reference page for computing values:
    https://www.mathworks.com/help/audio/ug/spectral-descriptors.html

    Arguments
    ---------
    frames: torch.Tensor
        The framed audio to use for feature computation, dims [batch, frame, sample].
    eps: float
        A small value to avoid division by 0.

    Returns
    -------
    features: torch.Tensor
        A [batch, frame, 8] tensor of spectral features for each frame:
         * centroid: The mean of the spectrum.
         * spread: The stdev of the spectrum.
         * skew: The spectral balance.
         * kurtosis: The spectral tailedness.
         * entropy: The peakiness of the spectrum.
         * flatness: The ratio of geometric mean to arithmetic mean.
         * crest: The ratio of spectral maximum to arithmetic mean.
         * flux: The average delta-squared between one spectral value and it's successor.
    """
    window_size = frames.size(-1)
    hann = torch.hann_window(window_size, device=frames.device)
    spectrum = torch.abs(torch.fft.rfft(frames * hann.view(1, 1, -1)))

    # To keep features in a neural-network-friendly range, use normalized freq [0, 1]
    freqs = torch.fft.rfftfreq(window_size, device=frames.device)
    freqs = freqs.view(1, 1, -1) / freqs.max()

    # Mean, spread, skew, kurtosis. 1-4th standardized moments
    centroid = spec_norm(freqs, spectrum).unsqueeze(-1)
    spread = spec_norm((freqs - centroid) ** 2, spectrum).sqrt()
    skew = spec_norm((freqs - centroid) ** 3, spectrum) / (spread**3 + eps)
    kurt = spec_norm((freqs - centroid) ** 4, spectrum) / (spread**4 + eps)
    centroid = centroid.squeeze(-1)

    # Entropy measures the peakiness of the spectrum
    entropy = -(spectrum * (spectrum + eps).log()).mean(dim=-1)

    # Flatness is ratio of geometric to arithmetic means
    # Use a formulation of geometric mean that is numerically stable
    geomean = (spectrum + eps).log().mean(-1).exp()
    flatness = geomean / (spectrum.mean(dim=-1) + eps)

    # Crest measures the ratio of maximum to sum
    crest = spectrum.amax(dim=-1) / (spectrum.sum(dim=-1) + eps)

    # Flux is the root-mean-square deltas, padded to maintain same shape
    pad = spectrum[:, 0:1, :]
    flux = torch.diff(spectrum, dim=1, prepend=pad).pow(2).mean(dim=-1).sqrt()

    return torch.stack(
        (centroid, spread, skew, kurt, entropy, flatness, crest, flux),
        dim=-1,
    )


def spec_norm(value, spectrum, eps=1e-10):
    """Normalize the given value by the spectrum."""
    return (value * spectrum).sum(dim=-1) / (spectrum.sum(dim=-1) + eps)


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


@torch.no_grad()
def compute_gne(
    audio,
    sample_rate=16000,
    bandwidth=1000,
    fshift=300,
    frame_size=300,
    hop_size=100,
    log_scale=True,
):
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
    frame_size : int
        Number of samples (at 10k sampling rate) in each analysis frame.
    hop_size : int
        Number of samples (at 10k sampling rate) between the start of each analysis frame.
    log_scale : bool
        Whether to represent the output in the log scale.

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

    # Step 2. Inverse filter with 13th order LPC
    order = 13
    window = torch.hann_window(frame_size, device=audio.device).view(1, 1, -1)
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
    if log_scale:
        return -10 * torch.log10(torch.clamp(1 - gne, min=EPSILON))
    else:
        return gne
