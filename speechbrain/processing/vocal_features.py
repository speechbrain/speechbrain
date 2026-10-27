"""
Functions for analyzing vocal characteristics: jitter, shimmer, HNR, and GNE.

These are typically used for analysis of dysarthric voices using more traditional approaches
(i.e. not deep learning). Often useful as a baseline for e.g. pathology detection. Inspired by PRAAT.

Authors
 * Peter Plantinga, 2024
"""

import math

import torch
import torchaudio

PERIODIC_NEIGHBORS = 4


@torch.no_grad()
def compute_autocorr_features(frames, min_lag, max_lag, neighbors=5):
    """Compute features based on autocorrelation

    Arguments
    ---------
    frames: torch.Tensor
        The audio frames to be evaluated for autocorrelation, shape [batch, frame, sample]
    min_lag: int
        The minimum number of samples to consider for potential period length.
    max_lag: int
        The maximum number of samples to consider for potential period length.
    neighbors: int
        The number of neighbors to use for rolling median -- to avoid octave errors.

    Returns
    -------
    harmonicity: torch.Tensor
        The highest autocorrelation score relative to the 0-lag score. Used to compute HNR
    best_lags: torch.Tensor
        The lag corresponding to the highest autocorrelation score, an estimate of period length.

    Example
    -------
    >>> audio = torch.rand(1, 16000)
    >>> frames = audio.unfold(-1, 800, 200)
    >>> frames.shape
    torch.Size([1, 77, 800])
    >>> harmonicity, best_lags = compute_autocorr_features(frames, 100, 200)
    >>> harmonicity.shape
    torch.Size([1, 77])
    >>> best_lags.shape
    torch.Size([1, 77])
    """
    autocorrelation = autocorrelate(frames)

    # Find the peak, lag
    harmonicity, lags = autocorrelation[:, :, min_lag:max_lag].max(dim=-1)

    # Take median value of neighboring cells to avoid octave errors
    pad = neighbors // 2, neighbors // 2
    lags = torch.nn.functional.pad(lags, pad=pad, mode="reflect")
    best_lags, _ = lags.unfold(-1, neighbors, 1).median(dim=-1)

    # Re-add the min_lag back in after first step removed it
    best_lags = best_lags + min_lag

    return harmonicity, best_lags


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

    Example
    -------
    >>> audio = torch.rand(1, 16000)
    >>> frames = audio.unfold(-1, 800, 200)
    >>> frames.shape
    torch.Size([1, 77, 800])
    >>> autocorrelation = autocorrelate(frames)
    >>> autocorrelation.shape
    torch.Size([1, 77, 401])
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


@torch.no_grad()
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

    Example
    -------
    >>> audio = torch.rand(1, 16000)
    >>> frames = audio.unfold(-1, 800, 200)
    >>> frames.shape
    torch.Size([1, 77, 800])
    >>> harmonicity, best_lags = compute_autocorr_features(frames, 100, 200)
    >>> jitter, shimmer = compute_periodic_features(frames, best_lags)
    >>> jitter.shape
    torch.Size([1, 77])
    >>> shimmer.shape
    torch.Size([1, 77])
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


@torch.no_grad()
def compute_spectral_features(spectrum, eps=1e-10):
    """Compute statistical measures on spectral frames
    such as flux, skew, spread, flatness.

    Reference page for computing values:
    https://www.mathworks.com/help/audio/ug/spectral-descriptors.html

    Arguments
    ---------
    spectrum: torch.Tensor
        The spectrum to use for feature computation, dims [batch, frame, freq].
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

    Example
    -------
    >>> audio = torch.rand(1, 16000)
    >>> window_size = 800
    >>> frames = audio.unfold(-1, window_size, 200)
    >>> frames.shape
    torch.Size([1, 77, 800])
    >>> hann = torch.hann_window(window_size).view(1, 1, -1)
    >>> windowed_frames = frames * hann
    >>> spectrum = torch.abs(torch.fft.rfft(windowed_frames))
    >>> spectral_features = compute_spectral_features(spectrum)
    >>> spectral_features.shape
    torch.Size([1, 77, 8])
    """
    # To keep features in a neural-network-friendly range, use normalized freq [0, 1]
    nfreq = spectrum.size(-1)
    freqs = torch.linspace(0, 1, nfreq, device=spectrum.device).view(1, 1, -1)

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
        (centroid, spread, skew, kurt, entropy, flatness, crest, flux), dim=-1
    )


def spec_norm(value, spectrum, eps=1e-10):
    """Normalize the given value by the spectrum."""
    return (value * spectrum).sum(dim=-1) / (spectrum.sum(dim=-1) + eps)


@torch.no_grad()
def compute_gne(
    audio,
    sample_rate=16000,
    bandwidth=1000,
    fshift=300,
    frame_len=0.03,
    hop_len=0.01,
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
    here are the ones recommended in that work.

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
    frame_len : float
        Length of each analysis frame, in seconds.
    hop_len : float
        Length of time between the start of each analysis frame, in seconds.

    Returns
    -------
    gne : torch.Tensor
        The glottal-to-noise-excitation ratio for each frame of the audio signal.

    Example
    -------
    >>> sample_rate = 16000
    >>> audio = torch.rand(1, sample_rate)  # 1s of audio
    >>> gne = compute_gne(audio, sample_rate=sample_rate)
    >>> gne.shape
    torch.Size([1, 98])
    """

    assert audio.dim() == 2, (
        "Expected audio to be 2-dimensional, [batch, sample]"
    )

    # Step 1. Downsample to 10 kHz since voice energy is low above 5 kHz
    old_sample_rate, sample_rate = sample_rate, 10000
    audio = torchaudio.functional.resample(audio, old_sample_rate, sample_rate)

    # Step 2a. Unfold into analysis frames
    frame_size = int(sample_rate * frame_len)
    hop_size = int(sample_rate * hop_len)
    window = torch.hann_window(frame_size, device=audio.device).view(1, 1, -1)
    frames = audio.unfold(dimension=-1, size=frame_size, step=hop_size) * window

    # Step 2b. Inverse filter each frame with 13th order LPC
    excitation_frames = inverse_filter(frames, lpc_order=13)

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
    return torch.stack(correlations, dim=-1).amax(dim=(2, 3))


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

    Example
    -------
    >>> audio = torch.rand(1, 10000)
    >>> frames = audio.unfold(-1, 300, 100)
    >>> frames.shape
    torch.Size([1, 98, 300])
    >>> filtered_frames = inverse_filter(frames)
    >>> filtered_frames.shape
    torch.Size([1, 98, 300])
    """
    lpc_coeffs = compute_lpc(frames, lpc_order=lpc_order, method="autocorr")
    batch, frame_count, _ = frames.shape
    reshaped_frames = frames.view(batch * frame_count, -1)
    reshaped_lpc = lpc_coeffs.view(batch * frame_count, -1)
    a_coeffs = torch.zeros_like(reshaped_lpc)
    a_coeffs[:, 0] = 1.0

    # Perform filtering
    inverse_filtered = torchaudio.functional.lfilter(
        reshaped_frames, a_coeffs, reshaped_lpc, clamp=False
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

    Example
    -------
    >>> audio = torch.rand(1, 10000)
    >>> frames = audio.unfold(-1, 300, 100)
    >>> frames.shape
    torch.Size([1, 98, 300])
    >>> envelope = compute_hilbert_envelopes(frames, 1000)
    >>> envelope.shape
    torch.Size([1, 98, 300])
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


def compute_lpc(frames, lpc_order, method="burg"):
    """Compute Linear Predictive Coding (LPC) filter coefficients for frames.

    Supports Burg's method (the default, known for stability and sharper spectral
    resolution) and the standard autocorrelation / Yule-Walker method.
    The resulting coefficients represent the inverse filter polynomial
    A(z) = 1 + a_1 z^-1 + ... + a_p z^-p with leading coefficient 1.

    Arguments
    ---------
    frames : torch.Tensor
        The audio frames, shape [batch, frame, sample] or [..., sample].
    lpc_order : int
        The LPC prediction order (number of poles).
    method : str
        LPC estimation method, either "burg" or "autocorr". Default is "burg".

    Returns
    -------
    lpc_coeffs : torch.Tensor
        LPC coefficients of shape [..., lpc_order + 1] with leading coefficient 1.0.

    Example
    -------
    >>> frames = torch.randn(1, 10, 300)
    >>> coeffs = compute_lpc(frames, lpc_order=10)
    >>> coeffs.shape
    torch.Size([1, 10, 11])
    >>> coeffs[0, 0, 0].item()
    1.0
    """
    if method == "burg":
        device = frames.device
        dtype = frames.dtype
        lead_shape = frames.shape[:-1]

        ar_coeffs = torch.zeros(
            *lead_shape, lpc_order + 1, dtype=dtype, device=device
        )
        ar_coeffs[..., 0] = 1.0

        fwd = frames[..., 1:].clone()
        bwd = frames[..., :-1].clone()
        den = (fwd**2 + bwd**2).sum(dim=-1)
        eps = 1e-15

        for i in range(lpc_order):
            num = -2.0 * (bwd * fwd).sum(dim=-1)
            rc = num / (den + eps)

            ar_coeffs_prev = ar_coeffs.clone()
            for j in range(1, i + 2):
                ar_coeffs[..., j] = (
                    ar_coeffs_prev[..., j] + rc * ar_coeffs_prev[..., i - j + 1]
                )

            fwd_tmp = fwd.clone()
            rc_expand = rc.unsqueeze(-1)
            fwd = fwd + rc_expand * bwd
            bwd = bwd + rc_expand * fwd_tmp

            q = 1.0 - rc**2
            den = q * den - bwd[..., -1] ** 2 - fwd[..., 0] ** 2

            fwd = fwd[..., 1:]
            bwd = bwd[..., :-1]

        return ar_coeffs

    elif method == "autocorr":
        autocorrelation = compute_cross_correlation(
            frames, frames, width=lpc_order
        )
        batch, frame_count, _ = autocorrelation.shape
        autocorrelation = autocorrelation.view(batch * frame_count, -1)

        # An autocorrelation of all 0's -- which can happen in padding -- leads to
        # an error with the linear system solver, as the matrix is singular.
        # We fix this by ensuring the zero-lag correlation is always 1.
        autocorrelation[:, lpc_order] = 1.0

        # Construct Toeplitz matrices (one per frame)
        R = autocorrelation[:, 1:-1].unfold(-1, lpc_order, 1).flip(dims=(1,))
        r = autocorrelation[:, lpc_order + 1 :]

        lpc = torch.linalg.solve(R, r)
        lpc_coeffs = torch.nn.functional.pad(-lpc, (1, 0), value=1.0)
        return lpc_coeffs.view(batch, frame_count, lpc_order + 1)
    else:
        raise ValueError(
            f"Unknown LPC method: {method}. Expected 'burg' or 'autocorr'."
        )


def solve_formants_from_lpc(
    lpc_coeffs,
    sample_rate=16000,
    num_formants=4,
    max_bandwidth=800.0,
    min_freq=60.0,
):
    """Extract formant frequencies and bandwidths by polynomial root solving of LPC coefficients.

    Constructs the companion matrix of the LPC polynomial and finds roots via
    eigenvalue decomposition. Roots with positive imaginary parts are converted
    to frequencies and bandwidths, filtered, and sorted in ascending order.

    Arguments
    ---------
    lpc_coeffs : torch.Tensor
        The LPC coefficients of shape [..., lpc_order + 1] where the leading
        coefficient is 1.0.
    sample_rate : float
        Sampling frequency of the audio in Hz.
    num_formants : int
        Number of lowest formants (F1, F2, ...) to retain.
    max_bandwidth : float
        Maximum formant bandwidth in Hz. Formants exceeding this bandwidth are rejected.
    min_freq : float
        Minimum formant frequency in Hz. Frequencies below this are rejected.

    Returns
    -------
    formants : torch.Tensor
        Estimated formant frequencies in Hz, shape [..., num_formants]. Unmatched
        formants are filled with NaN.
    bandwidths : torch.Tensor
        Estimated formant bandwidths in Hz, shape [..., num_formants].

    Example
    -------
    >>> # Synthetic resonator with known pole at 500 Hz (fs=16000)
    >>> r = math.exp(-math.pi * 60.0 / 16000)
    >>> theta = 2 * math.pi * 500.0 / 16000
    >>> coeffs = torch.tensor([[[1.0, -2 * r * math.cos(theta), r**2]]])
    >>> formants, bws = solve_formants_from_lpc(
    ...     coeffs, sample_rate=16000, num_formants=1
    ... )
    >>> round(formants.item())
    500
    """
    lpc_order = lpc_coeffs.shape[-1] - 1
    lead_shape = lpc_coeffs.shape[:-1]

    # Companion matrix of monic polynomial: z^p + a_1 z^{p-1} + ... + a_p = 0
    # Top row is -a_1, -a_2, ..., -a_p
    comp = torch.zeros(
        *lead_shape,
        lpc_order,
        lpc_order,
        dtype=lpc_coeffs.dtype,
        device=lpc_coeffs.device,
    )
    comp[..., 0, :] = -lpc_coeffs[..., 1:]
    if lpc_order > 1:
        comp[..., 1:, :-1] = torch.eye(
            lpc_order - 1, dtype=lpc_coeffs.dtype, device=lpc_coeffs.device
        )

    roots = torch.linalg.eigvals(comp)
    imag = roots.imag
    angle = roots.angle()

    # Formant pole pairs have positive angle/frequency
    pos_mask = (angle > 0.0) & (imag > 0.0)
    freqs = angle * (sample_rate / (2.0 * math.pi))
    radius = roots.abs().clamp(min=1e-10)
    bws = -(sample_rate / math.pi) * torch.log(radius)

    valid = (
        pos_mask
        & (bws <= max_bandwidth)
        & (freqs >= min_freq)
        & (freqs <= sample_rate / 2.0)
    )

    # Sort formants so lowest frequencies come first; invalid candidates sort to end
    freqs_sorted = torch.where(
        valid, freqs, torch.tensor(float("inf"), device=lpc_coeffs.device)
    )
    bws_sorted = torch.where(
        valid, bws, torch.tensor(float("inf"), device=lpc_coeffs.device)
    )
    sort_idx = torch.argsort(freqs_sorted, dim=-1)
    freqs_sorted = torch.gather(freqs_sorted, dim=-1, index=sort_idx)
    bws_sorted = torch.gather(bws_sorted, dim=-1, index=sort_idx)

    # Replace inf with nan
    freqs_sorted = torch.where(
        torch.isinf(freqs_sorted),
        torch.tensor(float("nan"), device=lpc_coeffs.device),
        freqs_sorted,
    )
    bws_sorted = torch.where(
        torch.isinf(bws_sorted),
        torch.tensor(float("nan"), device=lpc_coeffs.device),
        bws_sorted,
    )

    final_freqs = freqs_sorted[..., :num_formants]
    final_bws = bws_sorted[..., :num_formants]

    # Pad if order//2 was smaller than num_formants
    if final_freqs.shape[-1] < num_formants:
        pad_amount = num_formants - final_freqs.shape[-1]
        final_freqs = torch.nn.functional.pad(
            final_freqs, (0, pad_amount), value=float("nan")
        )
        final_bws = torch.nn.functional.pad(
            final_bws, (0, pad_amount), value=float("nan")
        )

    return final_freqs, final_bws


def choose_lpc_order(audio, sample_rate=16000, frame_len=0.03, method="burg"):
    """Estimate optimal LPC order using Bayesian Information Criterion (BIC).

    Tests candidate LPC orders in [fs/1000 - 2, fs/1000 + 6] on the highest-energy
    vowel frame in the bandpass-filtered audio (120 Hz to 3000 Hz), returning
    the order that minimizes BIC. Inspired by Phonlab's choose_order.

    Arguments
    ---------
    audio : torch.Tensor
        Audio signal tensor of shape [batch, sample] or [sample].
    sample_rate : float
        Sampling frequency in Hz.
    frame_len : float
        Duration of the candidate analysis frame in seconds.
    method : str
        LPC estimation method ("burg" or "autocorr"). Default is "burg".

    Returns
    -------
    best_orders : torch.Tensor or int
        The selected LPC order(s) for the input audio. Returns int if 1D input.

    Example
    -------
    >>> audio = torch.randn(16000)
    >>> order = choose_lpc_order(audio, sample_rate=16000)
    >>> isinstance(order, int) and order >= 8
    True
    """
    was_1d = audio.dim() == 1
    if was_1d:
        audio = audio.unsqueeze(0)

    batch_size = audio.shape[0]
    frame_size = int(sample_rate * frame_len)
    step_size = max(1, frame_size // 4)

    # Bandpass filter around vowel range (120 Hz to 3000 Hz) to locate vowel peak
    nyquist = sample_rate / 2.0
    f_low = max(120.0, 10.0)
    f_high = min(3000.0, nyquist - 10.0)
    center_freq = math.sqrt(f_low * f_high)
    q = center_freq / (f_high - f_low)
    filtered = torchaudio.functional.bandpass_biquad(
        audio, sample_rate, central_freq=center_freq, Q=q
    )
    unfolded = filtered.unfold(-1, frame_size, step_size)
    energy = (unfolded**2).sum(dim=-1)
    best_frame_idx = energy.argmax(dim=-1)

    orig_unfolded = audio.unfold(-1, frame_size, step_size)
    batch_idx = torch.arange(batch_size, device=audio.device)
    peak_frames = orig_unfolded[batch_idx, best_frame_idx, :]
    window = torch.hann_window(frame_size, device=audio.device)
    peak_frames = (peak_frames * window).unsqueeze(1)

    os = max(8, int(sample_rate // 1000) - 2)
    oe = int(sample_rate // 1000) + 7
    candidate_orders = list(range(os, oe, 2))

    best_orders = torch.full(
        (batch_size,),
        candidate_orders[0],
        dtype=torch.long,
        device=audio.device,
    )
    best_bics = torch.full((batch_size,), float("inf"), device=audio.device)

    for o in candidate_orders:
        lpc_coeffs = compute_lpc(
            peak_frames, lpc_order=o, method=method
        ).squeeze(1)
        a_coeffs = torch.zeros_like(lpc_coeffs)
        a_coeffs[:, 0] = 1.0

        resid = torchaudio.functional.lfilter(
            peak_frames.squeeze(1), a_coeffs, lpc_coeffs, clamp=False
        )
        sigma2 = (resid**2).mean(dim=-1).clamp(min=1e-10)
        n = frame_size - o
        bic = n * torch.log(sigma2) + o * math.log(n)

        better = bic < best_bics
        best_bics = torch.where(better, bic, best_bics)
        best_orders = torch.where(
            better,
            torch.tensor(o, dtype=torch.long, device=audio.device),
            best_orders,
        )

    if was_1d:
        return best_orders[0].item()
    return best_orders


def _dilate(mask, kernel):
    """Binary dilation of a [batch, frame] bool mask with an odd kernel."""
    kernel = 2 * (kernel // 2) + 1
    if kernel <= 1:
        return mask
    x = mask.float().unsqueeze(1)
    x = torch.nn.functional.max_pool1d(x, kernel, stride=1, padding=kernel // 2)
    return x.squeeze(1) > 0


def _erode(mask, kernel):
    """Binary erosion of a [batch, frame] bool mask with an odd kernel."""
    return ~_dilate(~mask, kernel)


@torch.no_grad()
def compute_voicing_mask(
    audio,
    sample_rate=16000,
    frame_len=0.025,
    hop_len=0.010,
    min_f0=60.0,
    max_f0=400.0,
    harmonicity_threshold=0.45,
    energy_range_db=30.0,
    low_band_hz=1500.0,
    min_low_band_ratio=0.3,
    closing_kernel=5,
    opening_kernel=3,
):
    """Voicing detection combining periodicity, energy, and spectral balance.

    Improvements over thresholding autocorrelation harmonicity alone:

     * The periodicity window is lengthened to cover >= 3 periods of `min_f0`
       (Boersma 1993), and only true local maxima of the autocorrelation
       are accepted, so slowly decaying low-frequency noise can't "peak"
       at the edge of the lag range.
     * Frames that are very quiet (relative to the utterance, and absolutely)
       are rejected, as autocorrelation of low-level noise is unreliable.
     * Frames with little low-frequency energy (e.g. frication) are rejected.
     * Morphological closing/opening removes isolated flicker, filling short
       gaps and dropping very short voiced islands.

    Arguments
    ---------
    audio : torch.Tensor
        Input audio of shape [batch, sample].
    sample_rate : int
        Sampling frequency in Hz.
    frame_len : float
        Frame length, in seconds, of the output frames (must match any
        features the mask will be applied to).
    hop_len : float
        Hop length in seconds.
    min_f0 : float
        Lowest f0 to search for, in Hz.
    max_f0 : float
        Highest f0 to search for, in Hz.
    harmonicity_threshold : float
        Minimum harmonicity for a weak voiced candidate.
    energy_range_db : float
        Frames more than this far below the utterance's 95th-percentile
        frame energy are unvoiced.
    low_band_hz : float
        Upper edge of the "voicing" band used for the spectral-balance check.
    min_low_band_ratio : float
        Minimum fraction of frame energy below `low_band_hz`.
    closing_kernel : int
        Unvoiced gaps shorter than this many frames (odd) are filled.
    opening_kernel : int
        Voiced runs shorter than this many frames (odd) are removed.

    Returns
    -------
    voiced_mask : torch.Tensor
        Boolean mask of shape [batch, frame].

    Example
    -------
    >>> audio = torch.randn(1, 16000) * 0.01
    >>> mask = compute_voicing_mask(audio)
    >>> mask.shape
    torch.Size([1, 98])
    >>> mask.dtype
    torch.bool
    """
    frame_size = int(sample_rate * frame_len)
    hop_size = int(sample_rate * hop_len)
    frames = audio.unfold(-1, frame_size, hop_size)

    # 1. Periodicity, using a window long enough for 3 periods of min_f0.
    #    Pad symmetrically so frame count and centers match `frames`.
    win = max(frame_size, int(3 * sample_rate / min_f0))
    pad_left = (win - frame_size) // 2
    padded = torch.nn.functional.pad(
        audio, (pad_left, win - frame_size - pad_left)
    )
    pframes = padded.unfold(-1, win, hop_size)

    min_lag = max(2, int(sample_rate / max_f0))
    max_lag = min(int(sample_rate / min_f0), win // 2 - 1)
    autocorr = autocorrelate(pframes)[..., min_lag - 1 : max_lag + 2]
    center = autocorr[..., 1:-1]
    is_peak = (center >= autocorr[..., :-2]) & (center >= autocorr[..., 2:])
    harmonicity = (center * is_peak).amax(dim=-1).clamp(0.0, 1.0)
    harmonicity_ok = harmonicity >= harmonicity_threshold

    # 2. Energy relative to the utterance
    energy_db = 10 * ((frames**2).mean(dim=-1) + 1e-10).log10()
    ref_db = torch.quantile(energy_db, 0.95, dim=-1, keepdim=True)
    energy_ok = energy_db > ref_db - energy_range_db

    # 3. Spectral balance: voiced speech concentrates energy at low frequencies
    hann = torch.hann_window(frame_size, device=audio.device)
    power = torch.fft.rfft(frames * hann).abs() ** 2
    freqs = torch.fft.rfftfreq(frame_size, 1 / sample_rate).to(audio.device)
    low_energy = power[..., freqs <= low_band_hz].sum(dim=-1)
    low_ratio = low_energy / power.sum(dim=-1).clamp(min=1e-10)
    spectrum_ok = low_ratio >= min_low_band_ratio

    # 4. Clean up: fill short gaps (closing), remove short islands (opening)
    voiced = harmonicity_ok & energy_ok & spectrum_ok
    voiced = _erode(_dilate(voiced, closing_kernel), closing_kernel)
    voiced = _dilate(_erode(voiced, opening_kernel), opening_kernel)
    return voiced


@torch.no_grad()
def estimate_formants(
    audio,
    sample_rate=16000,
    lpc_order=None,
    num_formants=4,
    frame_len=0.025,
    hop_len=0.010,
    preemphasis=0.97,
    max_bandwidth=800.0,
    min_freq=60.0,
    voicing_threshold=0.45,
    voiced_only=True,
    lpc_method="burg",
    voicing_method="combined",
):
    """Estimate vowel formant frequencies (F1, F2, ...) and bandwidths in speech audio.

    Frames the audio, applies pre-emphasis, estimates or uses the specified LPC order,
    solves for formants via companion matrix polynomial root-solving, and detects
    voicing (by default combining harmonicity, energy, and spectral balance).

    Arguments
    ---------
    audio : torch.Tensor
        Input audio waveform of shape [batch, sample] or [sample].
    sample_rate : float
        Sampling frequency in Hz.
    lpc_order : int, optional
        LPC prediction order. If None or <= 0, automatically determined per
        utterance using `choose_lpc_order`.
    num_formants : int
        Number of formants to estimate per frame (default 4: F1, F2, F3, F4).
    frame_len : float
        Length of each analysis frame in seconds (default 0.025 = 25 ms).
    hop_len : float
        Hop size between frames in seconds (default 0.010 = 10 ms).
    preemphasis : float
        First-order high-frequency pre-emphasis coefficient (default 0.97).
    max_bandwidth : float
        Maximum bandwidth threshold in Hz for accepting a formant pole (default 800.0).
    min_freq : float
        Minimum frequency threshold in Hz (default 60.0).
    voicing_threshold : float
        Harmonicity threshold for classifying a frame as voiced (default 0.45).
    voiced_only : bool
        If True, unvoiced frames will have formant and bandwidth values set to NaN.
    lpc_method : str
        Algorithm to compute LPC coefficients, either "burg" or "autocorr" (default "burg").
    voicing_method : str
        "combined" (default) uses `compute_voicing_mask`; "harmonicity" uses a simpler
        harmonicity-only threshold.

    Returns
    -------
    formants : torch.Tensor
        Estimated formant frequencies in Hz, shape [batch, frame, num_formants]
        (or [frame, num_formants] if 1D input).
    bandwidths : torch.Tensor
        Estimated formant bandwidths in Hz, shape [batch, frame, num_formants]
        (or [frame, num_formants] if 1D input).
    voiced_mask : torch.Tensor
        Boolean mask indicating voiced frames, shape [batch, frame]
        (or [frame] if 1D input).

    Example
    -------
    >>> audio = torch.randn(1, 16000)
    >>> formants, bws, voiced = estimate_formants(
    ...     audio, sample_rate=16000, lpc_order=12
    ... )
    >>> formants.shape
    torch.Size([1, 98, 4])
    >>> voiced.shape
    torch.Size([1, 98])
    """
    was_1d = audio.dim() == 1
    if was_1d:
        audio = audio.unsqueeze(0)

    frame_size = int(sample_rate * frame_len)
    hop_size = int(sample_rate * hop_len)

    # Voicing detection from raw audio frames
    if voicing_method == "combined":
        voiced_mask = compute_voicing_mask(
            audio,
            sample_rate=sample_rate,
            frame_len=frame_len,
            hop_len=hop_len,
            harmonicity_threshold=voicing_threshold,
        )
    elif voicing_method == "harmonicity":
        min_lag = int(sample_rate / 400)
        max_lag = int(sample_rate / 60)
        unframed_for_voicing = audio.unfold(-1, frame_size, hop_size)
        harmonicity, _ = compute_autocorr_features(
            unframed_for_voicing, min_lag, max_lag
        )
        voiced_mask = harmonicity >= voicing_threshold
    else:
        raise ValueError(
            f"Unknown voicing method: {voicing_method}. "
            "Expected 'combined' or 'harmonicity'."
        )

    # High-frequency pre-emphasis
    if preemphasis > 0.0:
        audio_pre = torch.cat(
            [audio[:, :1], audio[:, 1:] - preemphasis * audio[:, :-1]], dim=-1
        )
    else:
        audio_pre = audio

    window = torch.hamming_window(frame_size, device=audio.device).view(
        1, 1, -1
    )
    frames = audio_pre.unfold(-1, frame_size, hop_size) * window

    if lpc_order is None or lpc_order <= 0:
        orders = choose_lpc_order(
            audio, sample_rate=sample_rate, frame_len=frame_len
        )
        if isinstance(orders, torch.Tensor):
            lpc_order = int(torch.mode(orders).values.item())
        else:
            lpc_order = int(orders)

    lpc_coeffs = compute_lpc(frames, lpc_order=lpc_order, method=lpc_method)
    formants, bandwidths = solve_formants_from_lpc(
        lpc_coeffs,
        sample_rate=sample_rate,
        num_formants=num_formants,
        max_bandwidth=max_bandwidth,
        min_freq=min_freq,
    )

    if voiced_only:
        mask = voiced_mask.unsqueeze(-1).expand_as(formants)
        formants = torch.where(
            mask, formants, torch.tensor(float("nan"), device=audio.device)
        )
        bandwidths = torch.where(
            mask, bandwidths, torch.tensor(float("nan"), device=audio.device)
        )

    if was_1d:
        return (
            formants.squeeze(0),
            bandwidths.squeeze(0),
            voiced_mask.squeeze(0),
        )
    return formants, bandwidths, voiced_mask
