"""
Functions for analyzing vocal characteristics, such as jitter, shimmer, etc.

These are typically used for analysis of dysarthric voices using more traditional approaches (i.e. not deep learning). Often useful as a baseline for e.g. pathology detection. Inspired by PRAAT.

Authors
 * Peter Plantinga, 2024
"""

import torch
import torchaudio


def estimate_f0(
    audio: torch.Tensor,
    min_f0_Hz: int = 75,
    step_size: float = 0.01,
    window_size: float = 0.04,
    sample_rate: int = 16000,
    lowpass_prepare: bool = True,
    lowpass_frequency: int = 1000,
    voicing_threshold: float = 0.5,
):
    """Estimates the fundamental frequency of a signal using auto-correlation.

    Arguments
    ---------
    audio: torch.Tensor
        The audio signal being analyzed, shape: [batch, time].
    min_f0_Hz: int
        The minimum allowed fundamental frequency, to reduce octave errors.
        Default is 80 Hz, based on human voice standard frequency range.
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
    A per-frame estimate of f0
    """

    step_samples = int(step_size * sample_rate)
    window_samples = int(window_size * sample_rate)
    min_f0_steps = int(sample_rate / min_f0_Hz) // 2

    # This preparation improves accuracy by removing unwanted peaks
    if lowpass_prepare:
        audio = torchaudio.functional.lowpass_biquad(
            audio, sample_rate, lowpass_frequency
        )

    # Use autocorrelation to compute an estimate of f0 for each frame
    autocorrelation = autocorrelate(audio, step_samples, window_samples)
    best_lag = compute_lag(autocorrelation, min_f0_steps, voicing_threshold)

    # Convert to Hz
    return sample_rate / best_lag


def autocorrelate(audio, step_samples, window_samples):
    """Generate autocorrelation scores using circular convolution.

    Arguments
    ---------
    audio : torch.Tensor
        The audio to be evaluated for autocorrelation.
    step_samples : int
        The number of samples between the beginning of each frame.
    window_samples : int
        The number of samples contained in each frame.

    Returns
    -------
    autocorrelation : torch.Tensor
        The auto-correlation tensor by convolving each frame with itself.
    """
    chunks = audio.unfold(-1, window_samples, step_samples)
    padded_chunks = torch.nn.functional.pad(
        chunks, (0, window_samples // 2), mode="circular"
    )
    return torch.nn.functional.conv1d(
        input=padded_chunks,
        weight=chunks.transpose(0, 1),
        groups=chunks.size(1),
    )


def compute_lag(autocorrelation, min_f0_steps, voicing_threshold):
    """Compute the (smoothed) lag corresponding to autocorrelation peaks.

    Arguments
    ---------
    autocorrelation : torch.Tensor
        Result of convolving a signal with itself.
    min_f0_steps : int
        Number of steps corresponding to the minimum allowed lag.
    voicing_threshold : float
        The minimum ratio between the highest peak and maximum autocorrelation
        before a frame is considered unvoiced.

    Returns
    -------
    lag : torch.Tensor
        The number of steps between successive periods for each frame.
    """
    # Find k-best candidates for each chunk
    # Excluding invalid indexes
    perfect_correlation = autocorrelation[:, :, 0]
    valid_scores = autocorrelation[:, :, min_f0_steps:]
    kbest = torch.topk(valid_scores, k=10, dim=-1)

    # Identify which frames are (un)voiced
    voiced = kbest.values[:, :, 0] > perfect_correlation * voicing_threshold
    voiced_values = kbest.values[:, voiced, :]
    voiced_indices = kbest.indices[:, voiced, :]

    # Pick whichever kbest value is closest to the average of 5 neighbors
    # Iterate a few times to ensure stability
    best_selection = iterative_lag_selection(voiced_values, iterations=3)
    return min_f0_steps + voiced_indices[:, :, best_selection]


def iterative_lag_selection(voiced_values, iterations):
    """Select the best lag out of available options by comparing
    to an average of neighboring lags to reduce jumping octaves."""
    best_selection = torch.zeros(voiced_values.size(1), dtype=int)
    for i in range(iterations):
        best_values = voiced_values[:, :, best_selection]
        averaged_voiced_values = neighbor_average(best_values, 5)
        distance = torch.abs(voiced_values - averaged_voiced_values)
        best_selection = torch.argmin(distance, dim=-1)
    return best_selection


def neighbor_average(best_values, neighbors):
    """Use convolutional kernel to average the neighbors."""
    kernel = torch.ones(1, 1, neighbors) / neighbors
    values = best_values.unsqeeze(1)
    return torch.nn.conv1d(values, kernel, padding="same").squeeze(1)
