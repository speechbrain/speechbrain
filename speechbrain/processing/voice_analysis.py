"""
Functions for analyzing vocal characteristics, such as jitter, shimmer, etc.

These are typically used for analysis of dysarthric voices using more traditional approaches (i.e. not deep learning). Often useful as a baseline for e.g. pathology detection. Inspired by PRAAT.

Authors
 * Peter Plantinga, 2024
"""

import torch
import torchaudio


def vocal_characteristics(
    audio: torch.Tensor,
    min_f0_Hz: int = 75,
    max_f0_Hz: int = 400,
    step_size: float = 0.01,
    window_size: float = 0.04,
    sample_rate: int = 16000,
    lowpass_prepare: bool = True,
    lowpass_frequency: int = 1000,
    voicing_threshold: float = 0.5,
):
    """Estimates the vocal characteristics of a signal using auto-correlation.
    Batched estimation is hard due to removing voiced frames, so only accepts a single sample.

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
    lowpass_prepare: bool
        Whether to prepare the signal with a lowpass filter for more accuracy.
    lowpass_frequency: int
        The upper-bound frequency of the lowpass filter, in Hz.
    voicing_threshold: float
        The threshold determining which frames are considered voiced, measured
        as the ratio of autocorrelation versus maximum possible autocorrelation.

    Returns
    -------
    estimated_f0 : torch.Tensor
        A per-frame estimate of the f0 in Hz.
    voiced_frames : torch.Tensor
        The estimate for each frame if it is voiced or unvoiced.
    jitter : torch.Tensor
        The estimate for the jitter value for each frame.
    shimmer : torch.Tensor
        The estimate for the shimmer value for each frame.
    """

    if audio.dim() != 1:
        raise ValueError("Takes audio tensors of only one dimension (time)")

    # Max lag corresponds to min f0 and vice versa
    step_samples = int(step_size * sample_rate)
    window_samples = int(window_size * sample_rate)
    max_lag_samples = int(sample_rate / min_f0_Hz)
    min_lag_samples = int(sample_rate / max_f0_Hz)

    # This preparation improves accuracy by removing unwanted peaks
    if lowpass_prepare:
        audio = torchaudio.functional.lowpass_biquad(
            audio, sample_rate, lowpass_frequency
        )

    # Use autocorrelation to compute an estimate of f0 for each frame
    windowed_audio = audio.unsqueeze(0).unfold(-1, window_samples, step_samples)
    autocorrelation = autocorrelate(windowed_audio)

    # Use autocorrelation to find voiced frames and estimate f0
    best_lags, voiced_frames = compute_lag(
        autocorrelation, min_lag_samples, max_lag_samples, voicing_threshold
    )

    # Use estimated f0 to compute jitter and shimmer
    jitter, shimmer = compute_periodic_features(
        windowed_audio[:, voiced_frames], best_lags
    )

    # Convert to Hz
    estimated_f0 = sample_rate / best_lags

    return estimated_f0, voiced_frames, jitter, shimmer


def autocorrelate(windowed_audio):
    """Generate autocorrelation scores using circular convolution.

    Arguments
    ---------
    windowed_audio : torch.Tensor
        The windowed_audio to be evaluated for autocorrelation, shape [1, time, window_samples]

    Returns
    -------
    autocorrelation : torch.Tensor
        The auto-correlation tensor from convolving each frame with itself.
    """
    window_samples = windowed_audio.size(2)
    padded_windows = torch.nn.functional.pad(
        windowed_audio, (0, window_samples // 2), mode="circular"
    )
    return torch.nn.functional.conv1d(
        input=padded_windows,
        weight=windowed_audio.transpose(0, 1),
        groups=windowed_audio.size(1),
    ).squeeze(0)


def compute_lag(
    autocorrelation, min_lag_samples, max_lag_samples, voicing_threshold
):
    """Compute the (smoothed) lag corresponding to autocorrelation peaks.

    Arguments
    ---------
    autocorrelation : torch.Tensor
        Result of convolving a signal with itself, shape [time, window]
    min_lag_samples : int
        Number of steps corresponding to the minimum allowed lag.
    max_lag_samples : int
        Number of steps corresponding to the maximum allowed lag.
    voicing_threshold : float
        The minimum ratio between the highest peak and maximum autocorrelation
        before a frame is considered unvoiced.

    Returns
    -------
    best_lag : torch.Tensor
        The number of samples between successive periods for each frame.
    voiced : torch.Tensor
        The estimation for each frame whether it is primarily voiced or unvoiced.
    """
    # Find k-best candidate lags for each window, excluding too-short lags.
    perfect_correlation = autocorrelation[:, 0]
    valid_scores = autocorrelation[:, min_lag_samples:max_lag_samples]
    kbest = torch.topk(valid_scores, k=10, dim=-1)

    # Identify which frames are (un)voiced by comparing the best candidate lag's
    # autocorrelation with the maximum possible autocorrelation.
    voiced = kbest.values[:, 0] > perfect_correlation * voicing_threshold
    kbest_lags = kbest.indices[voiced]

    # Pick whichever lag is closest to the average of 5 neighbors
    # Iterate a few times to ensure stability
    best_lag = iterative_lag_selection(kbest_lags, iterations=3)
    best_lag += min_lag_samples
    return best_lag, voiced


def iterative_lag_selection(kbest_lags, iterations):
    """Select the best lag out of available options by comparing
    to an average of neighboring lags to reduce jumping octaves."""
    # kbest returns sorted list, first entry should be the highest autocorrelation
    best_lag = kbest_lags[:, 0]
    for i in range(iterations):
        averaged_lag = neighbor_average(best_lag.float(), 5)
        distance = torch.abs(kbest_lags - averaged_lag.unsqueeze(1))
        best_selection = torch.argmin(distance, dim=-1)
        best_lag = select_indices(kbest_lags, best_selection)
    return best_lag


def neighbor_average(best_values, neighbors):
    """Use convolutional kernel to average the neighbors."""
    kernel = torch.ones(1, 1, neighbors) / neighbors
    values = torch.nn.functional.pad(
        best_values[None, None, :],
        pad=(neighbors // 2, neighbors // 2),
        mode="reflect",
    )

    # Use conv to compute average
    # TODO: Use median rather than mean to reduce outlier impact
    return torch.nn.functional.conv1d(values, kernel).squeeze()


def compute_periodic_features(voiced_windows, best_lag):
    """Function to compute periodic features: jitter, shimmer

    Arguments
    ---------
    voiced_windows : torch.Tensor
        The voiced frames of the windowed audio to use for feature computation.
    best_lag : torch.Tensor
        The average period length for each frame.

    Returns
    -------
    jitters : torch.Tensor
        The average absolute deviation in period over the frame.
    shimmers : torch.Tensor
        The average absolute deviation in amplitude over the frame.
    """

    # Compute the peak for each window as a basis for computing periods
    peak_indexes = torch.argmax(voiced_windows, dim=-1)

    # Iterating frames is slow, but I don't see an easy way around it.
    n = voiced_windows.size(1)
    jitter, shimmer = torch.empty(n), torch.empty(n)
    for i in range(n):
        frame = voiced_windows[:, i].squeeze()
        peak_index = peak_indexes[:, i].squeeze()
        lag = best_lag[i]

        # Cut to half period on either side, then unfold
        front_offset = (peak_index + lag // 2) % lag
        back_offset = (len(frame) - front_offset) % lag
        periods = torch.reshape(frame[front_offset:-back_offset], (-1, lag))

        # Compare amplitude of each period to its successor. Divide by average to get relative.
        amplitudes = periods.abs().mean(dim=1)
        shimmer[i] = torch.abs(amplitudes[:-1] - amplitudes[1:]).mean()
        shimmer[i] /= amplitudes.mean()

        # Compare average peak lag of each period to its successor. Divide by lag to get relative.
        peak_lags = periods.argmax(dim=1)
        jitter[i] = torch.abs(peak_lags[:-1] - peak_lags[1:]).float().mean()
        jitter[i] /= lag

    return jitter, shimmer


def select_indices(tensor, indices):
    """Utility function to extract a list of indices from a tensor"""
    return tensor[torch.arange(tensor.size(0)), indices]
