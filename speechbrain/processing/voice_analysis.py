"""
Functions for analyzing vocal characteristics: jitter, shimmer, and HNR.

These are typically used for analysis of dysarthric voices using more traditional approaches (i.e. not deep learning). Often useful as a baseline for e.g. pathology detection. Inspired by PRAAT.

Authors
 * Peter Plantinga, 2024
"""

import torch


def vocal_characteristics(
    audio: torch.Tensor,
    min_f0_Hz: int = 75,
    max_f0_Hz: int = 300,
    step_size: float = 0.01,
    window_size: float = 0.04,
    sample_rate: int = 16000,
    autocorrelation_threshold: float = 0.45,
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
    autocorrelation_threshold: float
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
    max_lag_samples = int(sample_rate / min_f0_Hz)
    min_lag_samples = int(sample_rate / max_f0_Hz)

    # Compute autocorrelation-based features
    kbest_lags, autocorrelation_ratio, hnr = autocorrelate(
        audio, step_samples, window_samples, min_lag_samples, max_lag_samples
    )

    # Compute period-based features
    jitter, shimmer = compute_periodic_features(audio, kbest_lags[:, 0])

    # Use both types of features to determine which frames are voiced
    voiced = compute_voiced(
        autocorrelation_ratio,
        jitter,
        autocorrelation_threshold,
        jitter_threshold,
    )

    # Use neighboring frames to select best lag out of available options
    best_lags = iterative_lag_selection(kbest_lags[voiced], iterations=3)
    estimated_f0 = sample_rate / best_lags

    return estimated_f0, voiced, jitter, shimmer, hnr


def autocorrelate(
    audio, step_samples, window_samples, min_lag_samples, max_lag_samples
):
    """Generate autocorrelation scores using circular convolution.

    Arguments
    ---------
    audio: torch.Tensor
        The audio to be evaluated for autocorrelation, shape [time]
    step_samples: float
        The time between analysis windows (in samples).
    window_samples: float
        The size of the analysis window (in samples).
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
    hnr: torch.Tensor
        The ratio of the power of the harmonic component of the signal to the
        power of the noise component, computed using autocorrelation.
    """
    # Window the audio for frame by frame analysis
    window = torch.hann_window(window_samples).view(1, 1, -1)
    frames = audio.view(1, -1).unfold(-1, window_samples, step_samples) * window
    padded_frames = torch.nn.functional.pad(
        frames, (0, window_samples // 2), mode="circular"
    )
    autocorrelation = torch.nn.functional.conv1d(
        input=padded_frames,
        weight=frames.transpose(0, 1),
        groups=frames.size(1),
    ).squeeze(0)

    # Use autocorrelation to compute best lags and autocorrelation ratio
    valid_lags = autocorrelation[:, min_lag_samples:max_lag_samples]
    kbest = torch.topk(valid_lags, k=15, dim=-1)
    kbest_lags = kbest.indices + min_lag_samples
    autocorrelation_ratio = kbest.values[:, 0] / autocorrelation[:, 0]

    # See https://www.fon.hum.uva.nl/paul/papers/Proceedings_1993.pdf
    # First, ratio must be normalized by the autocorrelation score of the window
    padded_window = torch.nn.functional.pad(
        window, (0, window_samples // 2), mode="circular"
    )
    norm_score = torch.nn.functional.conv1d(input=padded_window, weight=window)
    norm_score = norm_score.squeeze() / norm_score.squeeze()[0]
    norm_score = norm_score.squeeze()[kbest_lags[:, 0] + 1]
    autocorrelation_ratio = autocorrelation_ratio / norm_score
    hnr = 10 * torch.log10(
        autocorrelation_ratio / torch.abs(1 - autocorrelation_ratio)
    )

    return kbest_lags, autocorrelation_ratio, hnr


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
    autocorrelation_ratio: torch.Tensor,
    jitter: torch.Tensor,
    autocorrelation_threshold: float = 0.45,
    jitter_threshold: float = 0.05,
):
    """
    Compute which sections are voiced based on two criteria (adapted from PRAAT):
     * normalized autocorrelation above threshold
     * AND jitter below threshold

    Then average to avoid rapid changes in voicing. If no frames are voiced, relax thresholds.

    Arguments
    ---------
    autocorrelation_ratio : torch.Tensor
        The normalized autocorrelation score, a number between 0 and 1 for each frame.
    jitter : torch.Tensor
        The jitter score, a number between 0 and 1 for each frame.
    autocorrelation_threshold : float
        The threshold above which to consider each frame as voiced.
    jitter_threshold : float
        The threshold below which to consider each frame as voiced.

    Returns
    -------
    voiced : torch.Tensor
        A boolean value for each frame, whether to consider the frame as voiced or unvoiced.
    """
    voiced, a_tweak, j_tweak = torch.zeros(len(autocorrelation_ratio)), 0.0, 0.0

    # Check on each iteration if we have more than 2 voiced frames
    while voiced.sum() < 2:
        voiced = autocorrelation_ratio > autocorrelation_threshold - a_tweak
        voiced &= jitter < jitter_threshold + j_tweak
        voiced = neighbor_average(voiced.float(), neighbors=21).round().bool()

        # Relax the threshold by a bit for each iteration
        a_tweak += 0.05
        j_tweak += 0.01

    return voiced
