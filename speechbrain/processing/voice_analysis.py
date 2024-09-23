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
    max_f0_Hz: int = 300,
    step_size: float = 0.01,
    window_size: float = 0.04,
    sample_rate: int = 16000,
    lowpass_frequency: int = 800,
    autocorrelation_threshold: float = 0.5,
    power_threshold: float = 0.1,
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
    lowpass_frequency: int
        The upper-bound frequency of the lowpass filter, in Hz.
    autocorrelation_threshold: float
        One of two threshold values for considering a frame as voiced. Computed
        as the ratio between lag 0 autocorrelation and t-max autocorrelation.
    power_threshold: float
        One of two threshold values for considering a frame as voiced. Computed
        as the difference between the power of the lowpassed audio and the original audio.

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

    # Format arguments. Max lag corresponds to min f0 and vice versa.
    audio = audio.unsqueeze(0)
    step_samples = int(step_size * sample_rate)
    window_samples = int(window_size * sample_rate)
    max_lag_samples = int(sample_rate / min_f0_Hz)
    min_lag_samples = int(sample_rate / max_f0_Hz)

    # Use original and lowpassed audio for power ratio computation
    orig_windows = audio.unfold(-1, window_samples, step_samples)
    lowpass_audio = torchaudio.functional.lowpass_biquad(
        audio, sample_rate, lowpass_frequency
    )
    lowpass_windows = lowpass_audio.unfold(-1, window_samples, step_samples)

    # Voiced signal detection, using autocorrelation ratio and power ratio
    kbest_lags, autocorrelation_ratio = autocorrelate(
        lowpass_windows, min_lag_samples, max_lag_samples
    )
    power_ratio = compute_power_ratio(orig_windows, lowpass_windows).squeeze()
    voiced = autocorrelation_ratio > autocorrelation_threshold
    voiced &= power_ratio > power_threshold

    # Use neighboring frames to select best lag out of available options
    best_lags = iterative_lag_selection(kbest_lags[voiced], iterations=3)
    estimated_f0 = sample_rate / best_lags

    # Use estimated f0 to compute jitter and shimmer and harmonic-to-noise ratio
    voiced_windows = orig_windows[:, voiced]
    jitter, shimmer, hnr = compute_periodic_features(voiced_windows, best_lags)

    return estimated_f0, voiced, jitter, shimmer, hnr


def autocorrelate(windowed_audio, min_lag_samples, max_lag_samples):
    """Generate autocorrelation scores using circular convolution.

    Arguments
    ---------
    windowed_audio: torch.Tensor
        The windowed_audio to be evaluated for autocorrelation, shape [1, time, window_samples]
    min_lag_samples: int
        Number of steps corresponding to the minimum allowed lag.
    max_lag_samples: int
        Number of steps corresponding to the maximum allowed lag.

    Returns
    -------
    kbest_lags: torch.Tensor
        The kbest candidate lags, as measured by peaks in autocorrelation.
    autocorrelation_ratio: torch.Tensor
        The ratio of the best candidate lag's autocorrelation score against
        the theoretical maximum autocorrelation score at lag 0.
    """
    window_samples = windowed_audio.size(2)
    padded_windows = torch.nn.functional.pad(
        windowed_audio, (0, window_samples // 2), mode="circular"
    )
    autocorrelation = torch.nn.functional.conv1d(
        input=padded_windows,
        weight=windowed_audio.transpose(0, 1),
        groups=windowed_audio.size(1),
    ).squeeze(0)

    # Use autocorrelation to compute best lags and autocorrelation ratio
    valid_lags = autocorrelation[:, min_lag_samples:max_lag_samples]
    kbest = torch.topk(valid_lags, k=15, dim=-1)
    kbest_lags = kbest.indices + min_lag_samples
    autocorrelation_ratio = kbest.values[:, 0] / autocorrelation[:, 0]

    return kbest_lags, autocorrelation_ratio


def compute_power_ratio(orig_windows, lowpass_windows):
    """The power ratio calculation is used for voiced speech detection."""
    return lowpass_windows.square().mean(dim=-1) / orig_windows.square().mean()


def iterative_lag_selection(kbest_lags, iterations=3):
    """Select the best lag out of available options by comparing
    to an average of neighboring lags to reduce jumping octaves."""
    # kbest returns sorted list, first entry should be the highest autocorrelation
    best_lag = kbest_lags[:, 0]
    for i in range(iterations):
        averaged_lag = neighbor_average(best_lag.float(), neighbors=5)
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


def compute_periodic_features(voiced_windows, best_lag):
    """Function to compute periodic features: jitter, shimmer

    Arguments
    ---------
    voiced_windows: torch.Tensor
        The voiced frames of the windowed audio to use for feature computation.
    best_lag: torch.Tensor
        The average period length for each frame.

    Returns
    -------
    jitters: torch.Tensor
        The average absolute deviation in period over the frame.
    shimmers: torch.Tensor
        The average absolute deviation in amplitude over the frame.
    """

    # Compute the peak for each window as a basis for computing periods
    peak_indexes = torch.argmax(voiced_windows, dim=-1)

    # Iterating frames is slow, but I don't see an easy way around it.
    n = voiced_windows.size(1)
    jitter, shimmer, hnr = torch.empty(n), torch.empty(n), torch.empty(n)
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

        # Compare power of averaged periods vs power of individuals
        # suggested by https://doi.org/10.1121/1.387808
        individual_power = periods.square().mean()
        averaged_power = periods.mean(dim=0).square().mean()
        hnr[i] = 10 * torch.log10(
            averaged_power / (individual_power - averaged_power)
        )

    return jitter, shimmer, hnr
