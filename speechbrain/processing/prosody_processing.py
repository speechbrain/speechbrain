"""
Prosody processing utilities

Authors
 * Yuanchao Li 2023
"""

import torch
import librosa
import numpy as np
from scipy.stats import linregress
from scipy.signal import spectrogram, lfilter


def compute_pitch_energy(signal, sample_rate, win_length):
    """Compute pitch and energy of a batch of waveforms.

    Arguments
    ---------
    signal : torch.Tensor
        The waveforms used for computing pitch and energy.
        Shape should be `[batch, time]` or `[batch, time, channels]`.
    sample_rate : int
        The sample rate of the audio signal.
    win_length : int
        Frame length to process.

    Returns
    -------
    pitch : torch.Tensor
        The f0 envelope of the input signal.
    energy : torch.Tensor
        The energy envelope of the input signal.
    log_energy : torch.Tensor
        The log energy of the input signal.

    Example
    -------
    >>> signal = read_audio('tests/samples/single-mic/example1.wav')
    >>> signal = signal.unsqueeze(0)
    >>> sample_rate = 16000
    >>> compute_pitch_energy(signal, sample_rate, win_length)
    (tensor([290.25, 287.69, ...]), tensor([90.25, 32.58, ...]), , tensor([190.25, 232.58, ...]))  # Example output, actual values may vary
    """

    signal_np = signal.numpy()
    energy, log_energy, pitch = [], [], []

    for i in range(signal.shape[0]):
        # Compute the short-time Fourier transform
        stft = librosa.stft(signal_np[i])

        # Calculate the magnitude spectrum
        magnitude = np.abs(stft)
        
        # Compute the energy envelope
        energy.append(np.sum(magnitude, axis=0))

        log_energy.append(np.log(energy[i] + 1e-12))  # plus a small positive value to prevent log(0)

        # Compute the f0 (fundamental frequency) using the harmonic product spectrum (HPS) method
        pitch_, _, _ = librosa.pyin(signal_np[i], fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sample_rate)
        pitch.append(pitch_)

    energy = np.array(energy)
    energy = torch.tensor(energy)
    log_energy = np.array(log_energy)
    log_energy = torch.tensor(log_energy)
    pitch = np.array(pitch)
    pitch = torch.tensor(pitch)

    return energy, log_energy, pitch


def compute_CPPS(signal, sample_rate, win_length):
    """Compute Smoothed Cepstral Peak Prominence (CPPS) of an input signal.

    Arguments
    ---------
    signal : tensor
        The waveforms used for computing CPPS.
        Shape should be `[batch, time]` or `[batch, time, channels]`.
    sample_rate : int
        Sample rate of the input audio signal (e.g., 16000).
    win_length : int
        Frame length to process.

    Returns
    -------
    cpps : torch.Tensor
        The Smoothed Cepstral Peak Prominence of the input signal.

    Example
    -------
    >>> signal = read_audio('tests/samples/single-mic/example1.wav')
    >>> signal = signal.unsqueeze(0)
    >>> sample_rate = 16000
    >>> win_length = 10
    >>> compute_CPPS(signal, sample_rate, win_length)
    tensor([[2.4641, 3.0907, ..., 6.5982, 2.3857]])
    """

    win_length_s = win_length / 1000
    signal_duration_s = signal.shape[1] / sample_rate
    signal_duration_ms = signal_duration_s * 1000
    expected_CPPS_midlevel_len = int(np.floor(signal_duration_ms / win_length))

    # Window analysis settings
    win_len_s = 0.048
    win_step_s = win_length_s
    win_len = int(round(win_len_s * sample_rate))
    win_step = int(round(win_step_s * sample_rate))
    win_overlap = win_len - win_step

    # Quefrency range
    quef_bot = int(round(sample_rate / 300))
    quef_top = int(round(sample_rate / 60))
    quefs = np.arange(quef_bot, quef_top + 1)

    signal_np = signal.numpy()

    cpps_list = []
    for i in range(signal.shape[0]):
        # Pre-emphasis from 50 Hz
        alpha = np.exp(-2 * np.pi * 50 / sample_rate)
        signal_i = lfilter([1, -alpha], 1, signal_np[i])

        # Compute spectrogram
        f, t, spec = spectrogram(signal_i, fs=sample_rate, nperseg=win_len, noverlap=win_overlap)
        spec_power = np.abs(spec) ** 2  # Calculate power from the complex values
        spec_log = 10 * np.log10(spec_power)

        # Compute cepstrum
        ceps_log = 10 * np.log10(np.abs(np.fft.fft(spec_log, axis=0)) ** 2)

        # Do time- and quefrency-smoothing
        smooth_filt_samples = np.ones(2) / 2
        smooth_filt_quef = np.ones(10) / 10
        ceps_log_smoothed = lfilter(smooth_filt_samples, 1, lfilter(smooth_filt_quef, 1, ceps_log, axis=0), axis=0)

        # Find cepstral peaks in the quefrency range
        ceps_log_smoothed = ceps_log_smoothed[quefs, :]
        peak_quef = np.argmax(ceps_log_smoothed, axis=0)

        # Get the regression line and calculate its distance from the peak
        n_wins = ceps_log_smoothed.shape[1]
        ceps_norm = np.zeros(n_wins)

        for n in range(n_wins):
            p = np.polyfit(quefs, ceps_log_smoothed[:, n], 1)
            ceps_norm[n] = np.polyval(p, quefs[peak_quef[n]])

        cpps_i = np.max(ceps_log_smoothed, axis=0) - ceps_norm

        # Pad the CPPS vector and calculate means in 10-ms window
        pad_size = expected_CPPS_midlevel_len - len(cpps_i)
        prepad_size = pad_size // 2
        postpad_size = pad_size - prepad_size
        cpps_padded = np.concatenate((np.full(prepad_size, np.nan), cpps_i, np.full(postpad_size, np.nan)))
        CPPS_value = cpps_padded
        CPPS_value[np.isnan(CPPS_value)] = np.nanmedian(CPPS_value)

        cpps_list.append(CPPS_value)

    cpps = np.array(cpps_list)
    cpps = torch.from_numpy(cpps)

    return cpps

def compute_creakiness(pitch, win_length):
    """Compute creakiness of the signal for a given window size.
    A frame is, usually, 10 milliseconds worth of samples.

    Arguments
    ---------
    pitch : tensor
        Pitch values of the signal.
        Shape should be `[batch, time]`.
    win_length : int
        Frame length to process.

    Returns
    -------
    creakiness : torch.Tensor
        The creakiness values of the input pitch sequence.

    Example
    -------
    >>> pitch, energy = compute_pitch_energy(signal, sample_rate)
    >>> win_length = 10
    >>> compute_creakiness(pitch, win_length)
    tensor([[0.5, 0.5, 0., 0., 0., ..., 0., 0.5, 1.]])
    """

    ms_per_window = 10
    frames_per_window = int(win_length / ms_per_window)

    pitch_np = pitch.numpy()

    # Calculate frequency ratios between adjacent frames
    ratios = pitch_np[:, 1:] / pitch_np[:, :-1]

    # Check for octave jumps and other frame-to-frame frequency jumps
    octave_up = (ratios > 1.90) & (ratios < 2.10)
    octave_down = (ratios > 0.475) & (ratios < 0.525)
    small_up = (ratios > 1.05) & (ratios < 1.25)
    small_down = (ratios < 0.95) & (ratios > 0.80)

    # Compute creakiness for each frame
    creakiness = octave_up + octave_down + small_up + small_down

    # Calculate the integral image for efficiency
    integral_image = np.concatenate([np.zeros((pitch.shape[0], 1)), np.cumsum(creakiness, axis=1)], axis=1)
    window_values = integral_image[:, frames_per_window:] - integral_image[:, :-frames_per_window]

    # Pad in front and at the end
    padding_needed = frames_per_window - 1
    front_padding = np.zeros((pitch.shape[0], int(np.floor(padding_needed / 2))))
    tail_padding = np.zeros((pitch.shape[0], int(np.ceil(padding_needed / 2))))
    creak_array = np.concatenate([front_padding, window_values, tail_padding], axis=1)
    creak_values = creak_array / frames_per_window

    creakiness = torch.tensor(creak_values)

    return creakiness


def compute_energy_stability(energy, win_length):
    """Compute energy stability of a signal using a specified window size.

    Arguments
    ---------
    energy : 2D tensor
        Energy values of the signal.
        Shape: (batch_size, signal_length)
    win_length : int
        Frame length to process.

    Returns
    -------
    The energy stability values of the input signals.
    Shape: (batch_size, signal_length)

    Example
    -------
    >>> pitch, energy = compute_pitch_energy(signal, sample_rate)
    >>> win_length = 10
    >>> compute_energy_stability(energy, win_length)
    tensor([[9., 5.5, 3., ..., 9., 12.5],
            [8., 4.5, 2.5, ..., 8., 11.5],
            ...])

    """

    batch_size, signal_length = energy.shape
    range_count = []

    # Parameters for processing
    ms_per_window = 10
    frames_per_window = int(win_length / ms_per_window)
    relevant_span = 200  # Temporary value
    frames_per_half_span = int((relevant_span / 2) / frames_per_window)

    for batch_idx in range(batch_size):
        range_count_single = []

        for i in range(signal_length):
            start_neighbors = i - frames_per_half_span
            end_neighbors = i + frames_per_half_span

            # Ensure bounds are within the signal
            start_neighbors = max(start_neighbors, 0)
            end_neighbors = min(end_neighbors, signal_length)

            neighbors = energy[batch_idx, start_neighbors:end_neighbors]
            ratios = neighbors / energy[batch_idx, i]
            
            # Use torch functions for tensor operations
            count = torch.sum((ratios > 0.90) & (ratios < 1.10), dtype=torch.float32)
            range_count_single.append(count.item())

        # Compute integral image for efficiency
        integral_image = np.concatenate(([0], np.cumsum(range_count_single)))
        window_values = torch.tensor(integral_image[frames_per_window:] - integral_image[:-frames_per_window], dtype=torch.float32)

        # Padding for the window
        padding_needed = frames_per_window - 1
        front_padding = torch.zeros(int(np.floor(padding_needed / 2)), dtype=torch.float32)
        tail_padding = torch.zeros(int(np.ceil(padding_needed / 2)), dtype=torch.float32)
        energy_range = torch.cat((front_padding, window_values, tail_padding))
        energy_stability = energy_range / frames_per_window

        range_count.append(energy_stability)

    return torch.stack(range_count)


def compute_pitch_range(pitch, win_length, range_type):
    """Compute pitch range for a given pitch sequence and window size.

    Arguments
    ---------
    pitch : 2D tensor
        Pitch values of the signal.
        Shape: (batch_size, pitch_length)
    win_length : int
        Frame length to process.
    range_type : str
        Type of pitch range to compute. Choose between ["f", "n", "w"].
            - "f": Full pitch range
            - "n": Narrow pitch range
            - "w": Wide pitch range

    Returns
    -------
    The pitch range values of the input pitch sequence.
    Shape: (batch_size, pitch_length)

    Example
    -------
    >>> pitch, energy = compute_pitch_energy(signal, sample_rate)
    >>> win_length = 10
    >>> compute_pitch_range(pitch, win_length, range_type="n")
    tensor([[4.5, 7., 7., ..., 7., 7.5],
            [3.5, 6., 6., ..., 6., 6.5],
            ...])

    """

    batch_size, pitch_length = pitch.shape
    range_count = []

    ms_per_window = 10
    frames_per_window = int(win_length / ms_per_window)
    relevant_span = 1000
    frames_per_half_span = int(np.floor((relevant_span / 2) / frames_per_window))

    for batch_idx in range(batch_size):
        range_count_single = []

        for i in range(pitch_length):
            start_neighbors = i - frames_per_half_span
            end_neighbors = i + frames_per_half_span

            # Ensure bounds are within the pitch sequence
            start_neighbors = max(start_neighbors, 0)
            end_neighbors = min(end_neighbors, pitch_length)

            neighbors = pitch[batch_idx, start_neighbors:end_neighbors]
            ratios = neighbors / pitch[batch_idx, i]

            # Based on ratio difference to center, count points with evidence
            # for the specified pitch range
            if range_type == 'f':
                count = torch.sum((ratios > 0.99) & (ratios < 1.01), dtype=torch.float32)
                range_count_single.append(count.item())
            elif range_type == 'n':
                count = torch.sum((ratios > 0.98) & (ratios < 1.02), dtype=torch.float32)
                range_count_single.append(count.item())
            elif range_type == 'w':
                count = torch.sum(((ratios > 0.70) & (ratios < 0.90)) | ((ratios > 1.1) & (ratios < 1.3)), dtype=torch.float32)
                range_count_single.append(count.item())

        # Compute integral image for efficiency
        integral_image = np.concatenate(([0], np.cumsum(range_count_single)))
        window_values = torch.tensor(integral_image[frames_per_window:] - integral_image[:-frames_per_window], dtype=torch.float32)

        # Padding for the window
        padding_needed = frames_per_window - 1
        front_padding = torch.zeros(int(np.floor(padding_needed / 2)), dtype=torch.float32)
        tail_padding = torch.zeros(int(np.ceil(padding_needed / 2)), dtype=torch.float32)
        pitch_range_single = torch.cat((front_padding, window_values, tail_padding))
        pitch_range_single = pitch_range_single / frames_per_window

        range_count.append(pitch_range_single)

    return torch.stack(range_count)


def percentilize_pitch(pitch, max_pitch):
    """Compute percentiles for a given pitch sequence and maximum pitch value.
    Instead of using specific Hz values, it maps them to percentiles in the overall distribution.
    This non-linearly scales the input while preserving NaNs.
    It accepts either a row vector or a column vector and returns a column vector.
    The input pitch_points typically range from 50 to around 515. Any points above max_pitch are mapped to NaN.

    Arguments
    ---------
    pitch : 2D tensor
        Pitch values of the signal.
        Shape: (batch_size, pitch_length)
    max_pitch : int
        Maximum pitch value.

    Returns
    -------
    The pitch percentiles for the input pitch sequence.
    Shape: (batch_size, pitch_length)

    Example
    -------
    >>> pitch, energy = compute_pitch_energy(signal, sample_rate)
    >>> max_pitch = 300
    >>> percentilize_pitch(pitch, max_pitch)
    tensor([[1., NaN, 0.9431, ..., 0.8943, 0.8862],
            [0.9129, 0.8892, 0.8788, ..., 0.9021, 0.9054],
            ...])

    """

    batch_size, pitch_length = pitch.shape
    rounded = pitch.round()

    # Initialize a histogram of the distribution
    counts = torch.zeros((batch_size, max_pitch), dtype=torch.int64)

    # Build the histogram
    for i in range(batch_size):
        for j in range(pitch_length):
            pitch_value = rounded[i, j].item()
            pitch_value_tensor = torch.tensor(pitch_value, dtype=torch.float32)
            if 1 <= pitch_value <= max_pitch and not torch.isnan(pitch_value_tensor):
                # It's within the range and not a NaN
                counts[i, int(pitch_value) - 1] += 1

    # Compute the cumulative sum of pitches that are less than or equal to each specified value
    cumulative_sum = torch.cumsum(counts, dim=1)

    # Compute the fraction of all pitches that are less than or equal to each specified value
    mapping = cumulative_sum / cumulative_sum[:, -1].unsqueeze(1)

    percentiles = torch.zeros_like(rounded, dtype=torch.float32)

    # Map each pitch to its percentile value
    for i in range(batch_size):
        for j in range(pitch_length):
            pitch_value = rounded[i, j].item()
            pitch_value_tensor = torch.tensor(pitch_value, dtype=torch.float32)
            if 1 <= pitch_value <= max_pitch and not torch.isnan(pitch_value_tensor):
                percentiles[i, j] = mapping[i, int(pitch_value) - 1].item()
            else:
                percentiles[i, j] = float('nan')

    return percentiles


def compute_rate(log_energy, win_length):
    """Compute speaking rate of a signal using a specified window size.
    A frame is, usually, 10 milliseconds worth of samples.

    Arguments
    ---------
    log_energy : torch.Tensor
        Log energy of the input signal.
        Shape: (batch_size, signal_length)
    win_length : int
        Frame length to process.

    Returns
    -------
    The speaking rate of the signal.
    Shape: (batch_size, signal_length - 1)

    Example
    -------
    >>> signal = read_audio('example.wav')
    >>> win_length = 10
    >>> log_energy = compute_log_energy(signal, win_length)
    >>> compute_rate(log_energy, win_length)
    tensor([[-2.1875138, -2.25198406, -2.19078188, ..., -2.15308205, -2.22154751],
            [...],  # More batches
            [...]])  # More batches
    """

    ms_per_window = 10
    frames_per_window = int(win_length / ms_per_window)

    # Compute inter-frame deltas
    deltas = torch.abs(log_energy[:, 1:] - log_energy[:, :-1])
    cum_sum_deltas = torch.cat([torch.zeros_like(log_energy[:, :1]), torch.cumsum(deltas, dim=1)], dim=1)

    # Adjust the indices to ensure compatible shapes
    window_liveliness = cum_sum_deltas[:, frames_per_window:] - cum_sum_deltas[:, :-frames_per_window]

    # Normalize rate for robustness against recording volume
    silence_mean, speech_mean = find_cluster_means(log_energy)
    scaled_liveliness = (window_liveliness - silence_mean) / (speech_mean - silence_mean)

    padding_needed = frames_per_window - 1
    front_padding = torch.zeros((log_energy.shape[0], int(np.floor(padding_needed / 2))))
    tail_padding = torch.zeros((log_energy.shape[0], int(np.ceil(padding_needed / 2))))
    rate = torch.cat((front_padding, scaled_liveliness, tail_padding), dim=1)

    return rate


def average_of_near_values(values, near_mean, far_mean):
    """Compute the average of values near a specified mean.

    Arguments
    ---------
    values : torch.Tensor
        The values for which to compute the average.
        Shape: (batch_size, sequence_length)
    near_mean : float
        The mean value for selecting nearby samples.
    far_mean : float
        The mean value for selecting samples that are far away.

    Returns
    -------
    The computed subset average: a torch.Tensor with shape (batch_size,)

    Example
    -------
    >>> values = torch.tensor([[1, 2, 3, 4, 5]])
    >>> near_mean = 3
    >>> far_mean = 1
    >>> average_of_near_values(values, near_mean, far_mean)
    tensor([2.3333])
    """

    nsamples = 2000

    if values.size(1) < 2000:
        samples = values
    else:
        samples = values[:, ::round(values.size(1) / nsamples)]

    near_mean = float(near_mean)
    far_mean = float(far_mean)

    closer_samples = samples[
        torch.abs(samples - near_mean) < torch.abs(samples - far_mean)
    ]

    if closer_samples.numel() == 0:
        subset_average = 0.9 * near_mean + 0.1 * far_mean
    else:
        subset_average = torch.mean(closer_samples.float())

    return subset_average


def find_cluster_means(values):
    """Find the mean values of two clusters in the input data.

    Arguments
    ---------
    values : torch.Tensor
        The values for which to find cluster means.
        Shape: (batch_size, sequence_length)

    Returns
    -------
    Tuple containing the low and high cluster means.
    Shapes: (batch_size,), (batch_size,)

    Example
    -------
    >>> values = torch.tensor([[1, 2, 3, 4, 5]])  # example
    >>> find_cluster_means(values)
    (tensor([1.]), tensor([5.]))
    """

    max_iterations = 20
    previous_low_center = torch.min(values, dim=1).values
    previous_high_center = torch.max(values, dim=1).values
    convergence_threshold = (previous_high_center - previous_low_center) / 100

    for _ in range(max_iterations):
        high_center = average_of_near_values(values, previous_high_center, previous_low_center)
        low_center = average_of_near_values(values, previous_low_center, previous_high_center)

        # Check for convergence based on a small threshold
        if torch.all(torch.abs(high_center - previous_high_center) < convergence_threshold) and \
           torch.all(torch.abs(low_center - previous_low_center) < convergence_threshold):
            return low_center, high_center

        previous_high_center = high_center
        previous_low_center = low_center

    return previous_low_center, previous_high_center


def compute_epeakness(energy):
    """Compute the peakness of an energy signal.

    Arguments
    ---------
    energy : torch.Tensor
        The energy values for computing peakness.
        Shape: (batch_size, signal_length)

    Returns
    -------
    The peakness of the input energy signal.
    Shape: (batch_size, signal_length)

    Example
    -------
    >>> energy = torch.tensor([[0.1, 0.5, 0.8, ..., 0.3, 0.2]])
    >>> compute_epeakness(energy)
    tensor([[0.5, 2. , 3. , 3.5, 3.2]])
    """

    iSFW = 6  # in-syllable filter width, in frames
    iFFW = 15  # in-foot filter width, in frames

    energy = energy.clone()  # Clone to avoid modifying the input tensor

    # Height normalization
    height = torch.sqrt((energy - torch.min(energy)) / (torch.max(energy) - torch.min(energy)))

    # Convolve with Laplacian of Gaussian filters

    inSyllablePeakness = myconv(energy, laplacian_of_gaussian(iSFW), iSFW * 2.5)
    inFootPeakness = myconv(energy, laplacian_of_gaussian(iFFW), iFFW * 2.5)

    # Compute peakness as the product of the above values and height
    epeakness = inSyllablePeakness * inFootPeakness * height

    # Remove negative values
    epeakness[epeakness < 0] = 0

    return epeakness


def compute_ppeakness(pitch):
    """Compute the peakness of a pitch signal.

    Arguments
    ---------
    pitch : torch.Tensor
        The pitch values for computing peakness.
        Shape: (batch_size, signal_length)

    Returns
    -------
    The peakness of the input pitch signal.
    Shape: (batch_size, signal_length)

    Example
    -------
    >>> pitch, energy = compute_pitch_energy(signal, sample_rate)
    >>> compute_ppeakness(pitch)
    tensor([[0., 95.3342, 0., ..., 337.6612, 0.]])
    """

    ssFW = 10  # stressed-syllable filter width

    # Identify valid pitch values and compute local pitch amount
    valid_pitch = pitch > 0
    local_pitch_amount = myconv(1.0 * valid_pitch.float(), triangle_filter(160), 10)

    # Replace NaN values with 0
    pitch[pitch.isnan()] = 0

    # Convolve with Laplacian of Gaussian filter
    local_peakness = myconv(pitch, laplacian_of_gaussian(ssFW), 2.5 * ssFW)

    # Compute peakness as the product of the above values and pitch
    ppeakness = local_peakness * local_pitch_amount * pitch

    # Remove negative values
    ppeakness[ppeakness < 0] = 0

    return ppeakness


def compute_spectral_tilt(signal, sample_rate):
    """Compute the spectral tilt of a signal every 10ms.

    Arguments
    ---------
    signal : torch.Tensor
        The waveforms used for computing spectral tilt.
        Shape should be `[batch, time]` or `[batch, time, channels]`.
    sample_rate : int
        The sample rate of the signal.

    Returns
    -------
    tilts : torch.Tensor
        The spectral tilt values computed every 10ms for each signal in the batch.

    Example
    -------
    >>> signal = read_audio('tests/samples/single-mic/example1.wav')
    >>> signal = signal.unsqueeze(0)
    >>> compute_spectral_tilt(signal, 16000)
    tensor([-4.9795e-21, -1.0136e-20, ..., -1.0835e-20, -2.2423e-21])
    """

    # Ensure the input is a NumPy array
    signal = np.array(signal)

    start_freq = 50
    end_freq = 4000
    intervals_per_octave = 3
    octave_range = np.log2(end_freq / start_freq)
    total_intervals = int(octave_range * intervals_per_octave)
    freq_values = [50]

    for i in range(1, total_intervals + 1):
        freq_value = start_freq * 2 ** (i / intervals_per_octave)
        freq_values.append(freq_value)

    freq_values.append(4000)

    win_length = int(0.025 * sample_rate)  # 15ms windows
    overlap = int(0.015 * sample_rate)  # To get a value for every 10 ms, an overlap of 15 is needed

    tilts_list = []

    for signal in signal:
        f, t, spec = spectrogram(signal, fs=sample_rate, window='hann', nperseg=win_length, noverlap=overlap)

        # Build empty matrix for the amplitude values squaring the values
        # obtained from calling spectrogram
        signal_length = len(signal) / sample_rate
        nframes = int(np.floor(signal_length / 0.01))  # number of 10ms windows (frames)
        nfrequencies = len(freq_values)
        amplitudes = np.abs(spec) ** 2  # calculate Power Spectral Density (PSD)
        A = np.zeros((nfrequencies, nframes))
        freq_range = 10  # define the frequency range in Hz

        # Fill matrix with the average PSD at each target frequency every 10ms
        for i in range(nfrequencies):
            freq = freq_values[i]
            for j in range(amplitudes.shape[1]):
                freq_idx = (f >= (freq - freq_range)) & (f <= (freq + freq_range))

                # Check if there are enough data points to calculate the mean
                if np.sum(freq_idx) > 0:
                    A[i, j] = np.mean(amplitudes[freq_idx, j])
                else:
                    A[i, j] = 0.0  # Set to 0 if no data points are available

        print("Calculating spectral tilt every 10ms...")

        # Run linear regression in every column (every 10ms window)
        tilts = np.zeros(nframes)
        for j in range(nframes):
            slope, _ = linregress(freq_values, A[:, j])[:2]
            tilts[j] = slope

        tilts_list.append(tilts)
    
    tilts = torch.tensor(tilts_list)

    return tilts


def speaking_frames(log_energy):
    """Identify frames with speech based on log energy.

    Arguments
    ---------
    log_energy : torch.Tensor
        Log energy values with shape (batch_size, signal_length).

    Returns
    -------
    speaking_frames_tensor : torch.Tensor
        Boolean tensor indicating frames with speech for each batch.

    Example
    -------
    >>> log_energy = compute_log_energy(signal_tensor, win_length)
    >>> speaking_frames(log_energy)
    tensor([[False, False, ..., True, False],
            [False, True, ..., False, True],
            ...,
            [True, False, ..., False, True]])
    """

    # Find silence and speech mean of track using k-means
    silence_mean = log_energy.mean(dim=1)
    speech_mean = log_energy.mean(dim=1)

    # Set the speech/silence threshold closer to the silence mean
    # because the variance of silence is less than that of speech.
    # This is ad hoc; modeling with two Gaussians would probably be better
    threshold = (2 * silence_mean + speech_mean) / 3.0

    sframes = log_energy > threshold.view(-1, 1)

    return sframes


def voiced_unvoiced_ir(log_energy, pitch, win_length):
    """Compute the ratio of average voiced intensity to average unvoiced intensity.

    Arguments
    ---------
    log_energy : torch.Tensor
        Log energy values with shape (batch_size, num_frames).
    pitch : torch.Tensor
        Pitch values with shape (batch_size, num_frames).
    win_length : int
        Frame length to process.

    Returns
    -------
    ratio : torch.Tensor
        The ratio of average voiced intensity to average unvoiced intensity.

    Example
    -------
    >>> log_energy = compute_log_energy(signal, win_length)
    >>> pitch, energy = compute_pitch_energy(signal, sample_rate)
    >>> win_length = 20
    >>> voiced_unvoiced_ir(log_energy, pitch, win_length)
    tensor([0.9, 1.1, ..., 0.8, 0.5])
    """

    is_speech = speaking_frames(log_energy)

    voiced_speech = (~torch.isnan(pitch) & is_speech)
    unvoiced_speech = (torch.isnan(pitch) & is_speech)

    non_voiced_energies_zeroed = voiced_speech.float() * log_energy
    non_unvoiced_energies_zeroed = unvoiced_speech.float() * log_energy

    v_frame_cum_sum = torch.cat((torch.zeros((log_energy.shape[0], 1)), torch.cumsum(non_voiced_energies_zeroed, dim=1)), dim=1)
    u_frame_cum_sum = torch.cat((torch.zeros((log_energy.shape[0], 1)), torch.cumsum(non_unvoiced_energies_zeroed, dim=1)), dim=1)
    v_frame_cum_count = torch.cat((torch.zeros((log_energy.shape[0], 1)), torch.cumsum(voiced_speech.float(), dim=1)), dim=1)
    u_frame_cum_count = torch.cat((torch.zeros((log_energy.shape[0], 1)), torch.cumsum(unvoiced_speech.float(), dim=1)), dim=1)

    ms_per_window = 10
    frames_per_window = win_length / ms_per_window
    frames_per_half_window = frames_per_window / 2

    v_frame_win_sum = torch.zeros_like(log_energy)
    u_frame_win_sum = torch.zeros_like(log_energy)
    v_frame_count_sum = torch.zeros_like(log_energy)
    u_frame_count_sum = torch.zeros_like(log_energy)

    for i in range(log_energy.shape[1]):
        w_start = int(i - frames_per_half_window)
        w_end = int(i + frames_per_half_window)

        # Prevent out-of-bounds
        w_start = max(0, w_start)
        w_end = min(log_energy.shape[1] - 1, w_end)

        v_frame_win_sum[:, i] = v_frame_cum_sum[:, w_end] - v_frame_cum_sum[:, w_start]
        u_frame_win_sum[:, i] = u_frame_cum_sum[:, w_end] - u_frame_cum_sum[:, w_start]
        v_frame_count_sum[:, i] = v_frame_cum_count[:, w_end] - v_frame_cum_count[:, w_start]
        u_frame_count_sum[:, i] = u_frame_cum_count[:, w_end] - u_frame_cum_count[:, w_start]

    avg_voiced_intensity = torch.zeros_like(log_energy)
    avg_unvoiced_intensity = torch.zeros_like(log_energy)

    avg_voiced_intensity[v_frame_count_sum != 0] = v_frame_win_sum[v_frame_count_sum != 0] / v_frame_count_sum[v_frame_count_sum != 0]
    avg_unvoiced_intensity[u_frame_count_sum != 0] = u_frame_win_sum[u_frame_count_sum != 0] / u_frame_count_sum[u_frame_count_sum != 0]

    ratio = avg_voiced_intensity / avg_unvoiced_intensity

    # Exclude zeros, NaNs, and Infs
    average_of_valid = torch.nanmean(ratio[(~torch.isinf(ratio)) & (ratio > 0)])
    ratio = ratio - average_of_valid
    ratio[torch.isnan(ratio)] = 0
    ratio[torch.isinf(ratio)] = 0

    return ratio


def windowize(signal, pitch, win_length):
    """Compute windowed values of input signal.

    Arguments
    ---------
    signal : torch.Tensor
        The input signal. Shape: (batch_size, signal_length)
    pitch : torch.Tensor
        Pitch values. Shape: (batch_size, pitch_length)
    win_length : int
        Frame length to process.

    Returns
    -------
    window_values : torch.Tensor
        Summed values over windows of the designated size, centered at 10ms, 20ms, etc.
        Shape: (batch_size, signal_length)

    Example
    -------
    >>> signal = read_audio('tests/samples/single-mic/example1.wav')
    >>> signal = signal.unsqueeze(0)
    >>> pitch = compute_pitch(signal)
    >>> win_length = 20
    >>> windowize(signal, pitch, win_length)
    tensor([[2.5, 3.5, ..., 4.5, 5.0]])
    """

    integral_image = torch.cat([torch.zeros(signal[:, :1].shape), torch.cumsum(signal, dim=1)], dim=1)
    ms_per_window = 10
    frames_per_window = int(win_length / ms_per_window)
    window_sum = integral_image[:, frames_per_window:] - integral_image[:, :-frames_per_window]

    # Align so the first value is for the window centered at 10 ms
    padding_needed = frames_per_window - 1
    front_padding = torch.zeros((signal.shape[0], int(np.floor(padding_needed / 2))))
    tail_padding = torch.zeros((signal.shape[0], int(np.ceil(padding_needed / 2))))
    window_values = torch.cat((front_padding, window_sum, tail_padding), dim=1)

    return window_values


def window_energy(log_energy, win_length):
    """Compute window energy.

    Arguments
    ---------
    log_energy : torch.Tensor
        Log energy values. Shape: (batch_size, log_energy_length)
    win_length : int
        Frame length to process.

    Returns
    -------
    win_energy : torch.Tensor
        Window energy values. Shape: (batch_size, log_energy_length)

    Example
    -------
    >>> log_energy = compute_log_energy(signal, win_length)
    >>> win_length = 20
    >>> window_energy(log_energy, win_length)
    tensor([[2.4213e-02, -1.0180e-01, ..., -2.0930e-01, -7.4701e-02]])
    """

    # Convert to numpy array if not already
    log_energy = log_energy.numpy() if isinstance(log_energy, torch.Tensor) else np.array(log_energy)

    integral_image = np.concatenate([np.zeros((log_energy.shape[0], 1)), np.cumsum(log_energy, axis=1)], axis=1)
    ms_per_window = 10
    frames_per_window = int(win_length / ms_per_window)
    window_sum = integral_image[:, frames_per_window:] - integral_image[:, :-frames_per_window]

    # Find silence and speech mean of track using k-means, then use them
    # to normalize for robustness against recording volume
    silence_mean, speech_mean = find_cluster_means(torch.tensor(window_sum))  # Convert window_sum to torch.Tensor

    difference = speech_mean - silence_mean

    # Convert window_sum to torch.Tensor
    window_sum = torch.tensor(window_sum)

    if difference > 0:
        scaled_sum = (window_sum - silence_mean) / difference
    else:
        # Something's wrong; typically the file is mostly music or has a terribly low SNR.
        # So we just return something that at least has no NaNs, though it may or may not be useful.
        scaled_sum = (window_sum - (0.5 * silence_mean)) / silence_mean

    # Align so the first value is for the window centered at 10 ms (or 15ms if an odd number of frames)
    # Using zeros for padding is not ideal.
    padding_needed = frames_per_window - 1
    front_padding = np.zeros((log_energy.shape[0], int(np.floor(padding_needed / 2))))
    tail_padding = np.zeros((log_energy.shape[0], int(np.ceil(padding_needed / 2))))
    win_energy = np.concatenate((front_padding, scaled_sum, tail_padding), axis=1)

    # Convert back to tensor
    win_energy = torch.tensor(win_energy)

    return win_energy


def disalignment(epeakness, ppeakness):
    """Compute disalignment estimate (pitch-peak delay).

    Arguments
    ---------
    epeakness : torch.Tensor
        An array representing energy peaks. Shape: (batch_size, peak_length)
    ppeakness : torch.Tensor
        An array representing pitch peaks. Shape: (batch_size, peak_length)

    Returns
    -------
    torch.Tensor
        The disalignment estimate. Shape: (batch_size, peak_length)

    Example
    -------
    >>> epeakness = compute_epeakness(energy)
    >>> ppeakness = compute_ppeakness(pitch)
    >>> disalignment(epeakness, ppeakness)
    tensor([0.0000e+00, 0.0000e+00, ..., 1.6391e+07, 1.9512e+07])
    """

    # Find local maxima in epeaky, representing energy peaks
    local_max_epeakness = find_local_max(epeakness, 120)
    
    # Calculate the expected product of local maxima and ppeaky
    expected_product = local_max_epeakness * ppeakness
    
    # Calculate the actual product of epeaky and ppeaky
    actual_product = epeakness * ppeakness
    
    # Compute the disalignment estimate
    disalignment = (expected_product - actual_product) * ppeakness
    
    return disalignment


def find_local_max(values, win_length):
    """Find local maxima in a values input.

    Arguments
    ---------
    values : torch.Tensor
        A tensor of numeric values. Shape: (batch_size, values_length)
    win_length : int
        Frame length to process.

    Returns
    -------
    local_maxima : torch.Tensor
        Tensor with local maxima. Shape: (batch_size, values_length)

    Example
    -------
    >>> values = [0.2, 0.5, 0.8, 1.2, 1.5]
    >>> win_length = 20
    >>> find_local_max(values, win_length)
    tensor([-0.0037, -0.0036, -0.0035, ..., -0.0049])
    """

    # Calculate half the width in frames
    ms_per_window = 10
    frames_per_window = int(win_length / ms_per_window)
    local_max = torch.zeros_like(values)
    
    # Iterate through the values
    for e in range(len(values)):
        start_frame = max(0, e - frames_per_window)
        end_frame = min(e + frames_per_window, len(values))
        
        # Find the maximum value within the defined window
        local_max[e] = torch.max(values[start_frame:end_frame])
    
    return local_max


def myconv(values, kernel, filterHalfWidth):
    """Custom convolution function.

    Arguments
    ---------
    values : torch.Tensor
        A tensor to be convolved. Shape: (batch_size, signal_length)
    kernel : torch.Tensor
        The convolution kernel. Shape: (kernel_length,)
    filterHalfWidth : float
        The half-width of the filter.

    Returns
    -------
    convolved_values : torch.Tensor
        The result of the convolution. Shape: (batch_size, signal_length)

    Example
    -------
    >>> values = torch.tensor([[0.1, 0.5, 0.8, ..., 0.3, 0.2]])
    >>> kernel = torch.tensor([1, -1, 1])
    >>> filterHalfWidth = 1
    >>> myconv(values, kernel, filterHalfWidth)
    tensor([[0.5, 2. , 3. , 3.5, 3.2]])
    """

    # Convert to NumPy arrays
    values_np = values.numpy() if isinstance(values, torch.Tensor) else np.array(values)
    kernel_np = kernel.numpy() if isinstance(kernel, torch.Tensor) else np.array(kernel)

    # Convolve using np.convolve
    result_np = np.convolve(values_np[0], kernel_np, mode='same')

    # Trim the edges
    trimWidth = int(np.floor(filterHalfWidth))

    # Pad with zeros to trimwidth at beginning and end to avoid artifacts
    result_np[:trimWidth] = 0
    result_np[-trimWidth:] = 0

    # Convert back to tensor
    result = torch.tensor(result_np)

    return result


def laplacian_of_gaussian(sigma):
    """Generate Laplacian of Gaussian (LoG) kernel.

    Arguments
    ---------
    sigma : float or torch.Tensor
        Standard deviation of the Gaussian part of the filter.

    Returns
    -------
    filter : torch.Tensor
        1D Laplacian of Gaussian filter.

    Example
    -------
    >>> sigma = 2.0
    >>> laplacian_of_gaussian(sigma)
    tensor([-0.0000, -0.0040, -0.0400, -0.1800, -0.3200, -0.3600, -0.2000,  0.0000,  0.2000,  0.3600,  0.3200,  0.1800,  0.0400,  0.0040])
    """

    # Length calculation
    length = int(sigma * 5)

    sigmaSquare = sigma * sigma
    sigmaFourth = sigmaSquare * sigmaSquare

    result = torch.zeros(length)
    center = length // 2
    for i in range(length):
        x = i - center
        y = ((x * x) / sigmaFourth - 1 / sigmaSquare) * np.exp((-x * x) / (2 * sigmaSquare))
        result[i] = -y

    return result


def rectangular_filter(win_length):
    """Generate a rectangular filter.

    Arguments
    ---------
    win_length : int or torch.Tensor
        Frame length of the rectangular filter in milliseconds.

    Returns
    -------
    filter : torch.Tensor
        The rectangular filter.

    Example
    -------
    >>> win_length = 20
    >>> rectangular_filter(win_length)
    tensor([0.0333, 0.0333, 0.0333, 0.0333, 0.0333, 0.0333, 0.0333, 0.0333, 0.0333, 0.0333, 0.0333, 0.0333])
    """

    ms_per_window = 10
    duration_frames = int(np.floor(win_length / ms_per_window))
    filter_values = torch.ones(duration_frames) / duration_frames

    return filter_values


def triangle_filter(win_length):
    """Generate a triangle filter.

    Arguments
    ---------
    win_length : int or torch.Tensor
        Frame length of the triangle filter in milliseconds.

    Returns
    -------
    filter : torch.Tensor
        The triangle filter.

    Example
    -------
    >>> win_length = 20
    >>> triangle_filter(win_length)
    tensor([0.2000, 0.4000, 0.2000])
    """

    ms_per_window = 10
    duration_frames = int(np.floor(win_length / ms_per_window))
    center = int(np.floor(duration_frames / 2))
    filter_values = [center - abs(i - center) for i in range(1, duration_frames + 1)]
    filter_values = torch.tensor(filter_values, dtype=torch.float32) / torch.sum(torch.tensor(filter_values, dtype=torch.float32))  # normalize to sum to one

    return filter_values
