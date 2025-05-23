"""Time-Domain Sequential Data Augmentation Classes

This module contains classes designed for augmenting sequential data in the time domain.
It is particularly useful for enhancing the robustness of neural models during training.
The available data distortions include adding noise, applying reverberation, adjusting playback speed, and more.
All classes are implemented as `torch.nn.Module`, enabling end-to-end differentiability and gradient backpropagation.

Authors:
- Peter Plantinga (2020)
- Mirco Ravanelli (2023)
"""

# Importing libraries
import random

import torch
import torch.nn.functional as F
import torchaudio

from speechbrain.dataio.dataloader import make_dataloader
from speechbrain.dataio.legacy import ExtendedCSVDataset
from speechbrain.processing.signal_processing import (
    compute_amplitude,
    convolve1d,
    dB_to_amplitude,
    notch_filter,
    reverberate,
)


class AddNoise(torch.nn.Module):
    """This class additively combines a noise signal to the input signal.

    Arguments
    ---------
    csv_file : str
        The name of a csv file containing the location of the
        noise audio files. If none is provided, white noise will be used.
    csv_keys : list, None, optional
        Default: None . One data entry for the noise data should be specified.
        If None, the csv file is expected to have only one data entry.
    sorting : str
        The order to iterate the csv file, from one of the
        following options: random, original, ascending, and descending.
    num_workers : int
        Number of workers in the DataLoader (See PyTorch DataLoader docs).
    snr_low : int
        The low end of the mixing ratios, in decibels.
    snr_high : int
        The high end of the mixing ratios, in decibels.
    pad_noise : bool
        If True, copy noise signals that are shorter than
        their corresponding clean signals so as to cover the whole clean
        signal. Otherwise, leave the noise un-padded.
    start_index : int
        The index in the noise waveforms to start from. By default, chooses
        a random index in [0, len(noise) - len(waveforms)].
    normalize : bool
        If True, output noisy signals that exceed [-1,1] will be
        normalized to [-1,1].
    noise_funct: funct object
        function to use to draw a noisy sample. It is enabled if the csv files
        containing the noisy sequences are not provided. By default,
        torch.randn_like is used (to sample white noise). In general, it must
        be a function that takes in input the original waveform and returns
        a tensor with the corresponding noise to add (e.g., see pink_noise_like).
    replacements : dict
        A set of string replacements to carry out in the
        csv file. Each time a key is found in the text, it will be replaced
        with the corresponding value.
    noise_sample_rate : int
        The sample rate of the noise audio signals, so noise can be resampled
        to the clean sample rate if necessary.
    clean_sample_rate : int
        The sample rate of the clean audio signals, so noise can be resampled
        to the clean sample rate if necessary.

    Example
    -------
    >>> import pytest
    >>> from speechbrain.dataio.dataio import read_audio
    >>> signal = read_audio('tests/samples/single-mic/example1.wav')
    >>> clean = signal.unsqueeze(0) # [batch, time, channels]
    >>> noisifier = AddNoise('tests/samples/annotation/noise.csv',
    ...                     replacements={'noise_folder': 'tests/samples/noise'})
    >>> noisy = noisifier(clean, torch.ones(1))
    """

    def __init__(
        self,
        csv_file=None,
        csv_keys=None,
        sorting="random",
        num_workers=0,
        snr_low=0,
        snr_high=0,
        pad_noise=False,
        start_index=None,
        normalize=False,
        noise_funct=torch.randn_like,
        replacements={},
        noise_sample_rate=16000,
        clean_sample_rate=16000,
    ):
        super().__init__()

        self.csv_file = csv_file
        self.csv_keys = csv_keys
        self.sorting = sorting
        self.num_workers = num_workers
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.pad_noise = pad_noise
        self.start_index = start_index
        self.normalize = normalize
        self.replacements = replacements
        self.noise_funct = noise_funct
        self.noise_sample_rate = noise_sample_rate
        self.clean_sample_rate = clean_sample_rate

    def forward(self, waveforms, lengths):
        """
        Arguments
        ---------
        waveforms : torch.Tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : torch.Tensor
            Shape should be a single dimension, `[batch]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        # Copy clean waveform to initialize noisy waveform
        noisy_waveform = waveforms.clone()
        lengths = (lengths * waveforms.shape[1]).unsqueeze(1)

        # Compute the average amplitude of the clean waveforms
        clean_amplitude = compute_amplitude(waveforms, lengths, amp_type="rms")

        # Pick an SNR and use it to compute the mixture amplitude factors
        SNR = torch.rand(len(waveforms), 1, device=waveforms.device)
        SNR = SNR * (self.snr_high - self.snr_low) + self.snr_low
        noise_amplitude_factor = 1 / (dB_to_amplitude(SNR) + 1)

        # Support for multichannel waveforms
        if len(noisy_waveform.shape) == 3:
            noise_amplitude_factor = noise_amplitude_factor.unsqueeze(1)

        # Scale clean signal appropriately
        new_noise_amplitude = noise_amplitude_factor * clean_amplitude
        noisy_waveform *= 1 - noise_amplitude_factor

        # Loop through clean samples and create mixture
        if self.csv_file is None:
            noise_waveform = self.noise_funct(waveforms)
            if noise_waveform.shape[0] == 1:
                noise_waveform = torch.cat(
                    [noise_waveform] * waveforms.shape[0], dim=0
                )

            noise_length = lengths
        else:
            tensor_length = waveforms.shape[1]
            noise_waveform, noise_length = self._load_noise(
                lengths, tensor_length
            )

        # Rescale and add
        noise_amplitude = compute_amplitude(
            noise_waveform, noise_length, amp_type="rms"
        )
        noise_waveform *= new_noise_amplitude / (noise_amplitude + 1e-14)

        noisy_waveform += noise_waveform
        # Normalizing to prevent clipping
        if self.normalize:
            abs_max, _ = torch.max(
                torch.abs(noisy_waveform), dim=1, keepdim=True
            )
            noisy_waveform = noisy_waveform / abs_max.clamp(min=1.0)

        return noisy_waveform

    def _load_noise(self, lengths, max_length):
        """Load a batch of noises"""
        lengths = lengths.long().squeeze(1)
        batch_size = len(lengths)

        # Load a noise batch
        if not hasattr(self, "data_loader"):
            if self.noise_sample_rate != self.clean_sample_rate:
                self.resampler = Resample(
                    self.noise_sample_rate, self.clean_sample_rate
                )

            # Set parameters based on input
            self.device = lengths.device

            # Create a data loader for the noise wavforms
            if self.csv_file is not None:
                dataset = ExtendedCSVDataset(
                    csvpath=self.csv_file,
                    output_keys=self.csv_keys,
                    sorting=(
                        self.sorting if self.sorting != "random" else "original"
                    ),
                    replacements=self.replacements,
                )
                self.data_loader = make_dataloader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=self.num_workers,
                    shuffle=(self.sorting == "random"),
                )
                self.noise_data = iter(self.data_loader)

        # Load noise to correct device
        noise_batch, noise_len = self._load_noise_batch_of_size(batch_size)
        noise_batch = noise_batch.to(lengths.device)
        noise_len = noise_len.to(lengths.device)

        # Resample noise if necessary
        if hasattr(self, "resampler"):
            noise_batch = self.resampler(noise_batch)

        # Convert relative length to an index
        noise_len = (noise_len * noise_batch.shape[1]).long()

        # Ensure shortest wav can cover speech signal
        # WARNING: THIS COULD BE SLOW IF THERE ARE VERY SHORT NOISES
        if self.pad_noise:
            while torch.any(noise_len < lengths):
                min_len = torch.min(noise_len)
                prepend = noise_batch[:, :min_len]
                noise_batch = torch.cat((prepend, noise_batch), axis=1)
                noise_len += min_len

        # Ensure noise batch is long enough
        elif noise_batch.size(1) < max_length:
            padding = (0, max_length - noise_batch.size(1))
            noise_batch = torch.nn.functional.pad(noise_batch, padding)

        # Select a random starting location in the waveform
        start_index = self.start_index
        if self.start_index is None:
            start_index = 0
            max_chop = (noise_len - lengths).min().clamp(min=1)
            start_index = torch.randint(
                high=max_chop, size=(1,), device=lengths.device
            )

        # Truncate noise_batch to max_length
        noise_batch = noise_batch[:, start_index : start_index + max_length]
        noise_len = (noise_len - start_index).clamp(max=max_length).unsqueeze(1)
        return noise_batch, noise_len

    def _load_noise_batch_of_size(self, batch_size):
        """Concatenate noise batches, then chop to correct size"""

        noise_batch, noise_lens = self._load_noise_batch()

        # Expand
        while len(noise_batch) < batch_size:
            added_noise, added_lens = self._load_noise_batch()
            noise_batch, noise_lens = AddNoise._concat_batch(
                noise_batch, noise_lens, added_noise, added_lens
            )

        # Contract
        if len(noise_batch) > batch_size:
            noise_batch = noise_batch[:batch_size]
            noise_lens = noise_lens[:batch_size]

        return noise_batch, noise_lens

    @staticmethod
    def _concat_batch(noise_batch, noise_lens, added_noise, added_lens):
        """Concatenate two noise batches of potentially different lengths"""

        # pad shorter batch to correct length
        noise_tensor_len = noise_batch.shape[1]
        added_tensor_len = added_noise.shape[1]
        pad = (0, abs(noise_tensor_len - added_tensor_len))
        if noise_tensor_len > added_tensor_len:
            added_noise = torch.nn.functional.pad(added_noise, pad)
            added_lens = added_lens * added_tensor_len / noise_tensor_len
        else:
            noise_batch = torch.nn.functional.pad(noise_batch, pad)
            noise_lens = noise_lens * noise_tensor_len / added_tensor_len

        noise_batch = torch.cat((noise_batch, added_noise))
        noise_lens = torch.cat((noise_lens, added_lens))

        return noise_batch, noise_lens

    def _load_noise_batch(self):
        """Load a batch of noises, restarting iteration if necessary."""

        try:
            # Don't necessarily know the key
            noises, lens = next(self.noise_data).at_position(0)
        except StopIteration:
            self.noise_data = iter(self.data_loader)
            noises, lens = next(self.noise_data).at_position(0)
        return noises, lens


class AddReverb(torch.nn.Module):
    """This class convolves an audio signal with an impulse response.

    Arguments
    ---------
    csv_file : str
        The name of a csv file containing the location of the
        impulse response files.
    sorting : str
        The order to iterate the csv file, from one of
        the following options: random, original, ascending, and descending.
    num_workers : int
        Number of workers in the DataLoader (See PyTorch DataLoader docs).
    rir_scale_factor: float
        It compresses or dilates the given impulse response.
        If 0 < scale_factor < 1, the impulse response is compressed
        (less reverb), while if scale_factor > 1 it is dilated
        (more reverb).
    replacements : dict
        A set of string replacements to carry out in the
        csv file. Each time a key is found in the text, it will be replaced
        with the corresponding value.
    reverb_sample_rate : int
        The sample rate of the corruption signals (rirs), so that they
        can be resampled to clean sample rate if necessary.
    clean_sample_rate : int
        The sample rate of the clean signals, so that the corruption
        signals can be resampled to the clean sample rate before convolution.

    Example
    -------
    >>> import pytest
    >>> from speechbrain.dataio.dataio import read_audio
    >>> signal = read_audio('tests/samples/single-mic/example1.wav')
    >>> clean = signal.unsqueeze(0) # [batch, time, channels]
    >>> reverb = AddReverb('tests/samples/annotation/RIRs.csv',
    ...                     replacements={'rir_folder': 'tests/samples/RIRs'})
    >>> reverbed = reverb(clean)
    """

    def __init__(
        self,
        csv_file,
        sorting="random",
        num_workers=0,
        rir_scale_factor=1.0,
        replacements={},
        reverb_sample_rate=16000,
        clean_sample_rate=16000,
    ):
        super().__init__()
        self.csv_file = csv_file
        self.sorting = sorting
        self.num_workers = num_workers
        self.replacements = replacements
        self.reverb_sample_rate = reverb_sample_rate
        self.clean_sample_rate = clean_sample_rate
        self.rir_scale_factor = rir_scale_factor

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : torch.Tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        if self.reverb_sample_rate != self.clean_sample_rate:
            self.resampler = Resample(
                self.reverb_sample_rate, self.clean_sample_rate
            )

        # Add channels dimension if necessary
        channel_added = False
        if len(waveforms.shape) == 2:
            waveforms = waveforms.unsqueeze(-1)
            channel_added = True

        # Load and prepare RIR
        rir_waveform = self._load_rir(waveforms)

        # Resample to correct rate
        if hasattr(self, "resampler"):
            rir_waveform = self.resampler(rir_waveform)

        # Compress or dilate RIR
        if self.rir_scale_factor != 1:
            rir_waveform = F.interpolate(
                rir_waveform.transpose(1, -1),
                scale_factor=self.rir_scale_factor,
                mode="linear",
                align_corners=False,
            )
            rir_waveform = rir_waveform.transpose(1, -1)

        rev_waveform = reverberate(waveforms, rir_waveform, rescale_amp="avg")

        # Remove channels dimension if added
        if channel_added:
            return rev_waveform.squeeze(-1)

        return rev_waveform

    def _load_rir(self, waveforms):
        # Create a data loader for the RIR waveforms
        if not hasattr(self, "data_loader"):
            dataset = ExtendedCSVDataset(
                csvpath=self.csv_file,
                sorting=(
                    self.sorting if self.sorting != "random" else "original"
                ),
                replacements=self.replacements,
            )
            self.data_loader = make_dataloader(
                dataset,
                shuffle=(self.sorting == "random"),
                num_workers=self.num_workers,
            )
            self.rir_data = iter(self.data_loader)

        try:
            rir_waveform, length = next(self.rir_data).at_position(0)
        except StopIteration:
            self.rir_data = iter(self.data_loader)
            rir_waveform, length = next(self.rir_data).at_position(0)

        # Make sure RIR has correct channels
        if len(rir_waveform.shape) == 2:
            rir_waveform = rir_waveform.unsqueeze(-1)

        # Make sure RIR has correct type and device
        rir_waveform = rir_waveform.type(waveforms.dtype)
        return rir_waveform.to(waveforms.device)


class SpeedPerturb(torch.nn.Module):
    """Slightly speed up or slow down an audio signal.

    Resample the audio signal at a rate that is similar to the original rate,
    to achieve a slightly slower or slightly faster signal. This technique is
    outlined in the paper: "Audio Augmentation for Speech Recognition"

    Arguments
    ---------
    orig_freq : int
        The frequency of the original signal.
    speeds : list
        The speeds that the signal should be changed to, as a percentage of the
        original signal (i.e. `speeds` is divided by 100 to get a ratio).
    device : str
        The device to use for the resampling.

    Example
    -------
    >>> from speechbrain.dataio.dataio import read_audio
    >>> signal = read_audio('tests/samples/single-mic/example1.wav')
    >>> perturbator = SpeedPerturb(orig_freq=16000, speeds=[90])
    >>> clean = signal.unsqueeze(0)
    >>> perturbed = perturbator(clean)
    >>> clean.shape
    torch.Size([1, 52173])
    >>> perturbed.shape
    torch.Size([1, 46956])
    """

    def __init__(self, orig_freq, speeds=[90, 100, 110], device="cpu"):
        super().__init__()
        self.orig_freq = orig_freq
        self.speeds = speeds
        self.device = device
        # Initialize index of perturbation
        self.samp_index = 0

        # Initialize resamplers
        self.resamplers = []
        for speed in self.speeds:
            config = {
                "orig_freq": self.orig_freq,
                "new_freq": self.orig_freq * speed // 100,
            }
            self.resamplers.append(Resample(**config))

    def forward(self, waveform):
        """
        Arguments
        ---------
        waveform : torch.Tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.

        Returns
        -------
        torch.Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        # Perform a random perturbation
        self.samp_index = torch.randint(0, len(self.speeds), (1,))
        perturbed_waveform = self.resamplers[self.samp_index](
            waveform.to(self.device)
        )
        # Move back from host to original device
        return perturbed_waveform.to(waveform.device)


class Resample(torch.nn.Module):
    """This class resamples audio using the
    :class:`torchaudio resampler <torchaudio.transforms.Resample>` based on
    sinc interpolation.

    Arguments
    ---------
    orig_freq : int
        the sampling frequency of the input signal.
    new_freq : int
        the new sampling frequency after this operation is performed.
    *args
        additional arguments forwarded to the
        :class:`torchaudio.transforms.Resample` constructor
    **kwargs
        additional keyword arguments forwarded to the
        :class:`torchaudio.transforms.Resample` constructor

    Example
    -------
    >>> from speechbrain.dataio.dataio import read_audio
    >>> signal = read_audio('tests/samples/single-mic/example1.wav')
    >>> signal = signal.unsqueeze(0) # [batch, time, channels]
    >>> resampler = Resample(orig_freq=16000, new_freq=8000)
    >>> resampled = resampler(signal)
    >>> signal.shape
    torch.Size([1, 52173])
    >>> resampled.shape
    torch.Size([1, 26087])
    """

    def __init__(self, orig_freq=16000, new_freq=16000, *args, **kwargs):
        super().__init__()

        self.orig_freq = orig_freq
        self.new_freq = new_freq

        self.resampler = torchaudio.transforms.Resample(
            orig_freq=orig_freq, new_freq=new_freq, *args, **kwargs
        )

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : torch.Tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        # Don't do anything if the frequencies are the same
        if self.orig_freq == self.new_freq:
            return waveforms

        unsqueezed = False
        if len(waveforms.shape) == 2:
            waveforms = waveforms.unsqueeze(1)
            unsqueezed = True
        elif len(waveforms.shape) == 3:
            waveforms = waveforms.transpose(1, 2)
        else:
            raise ValueError("Input must be 2 or 3 dimensions")

        # If necessary, migrate the resampler to the current device, for
        # backwards compat with scripts that do not call `resampler.to()`
        # themselves.
        # Please do not reuse the sample resampler for tensors that live on
        # different devices, though.
        self.resampler.to(waveforms.device)  # in-place

        # Do resampling
        resampled_waveform = self.resampler(waveforms)

        if unsqueezed:
            resampled_waveform = resampled_waveform.squeeze(1)
        else:
            resampled_waveform = resampled_waveform.transpose(1, 2)

        return resampled_waveform


class DropFreq(torch.nn.Module):
    """This class drops a random frequency from the signal.

    The purpose of this class is to teach models to learn to rely on all parts
    of the signal, not just a few frequency bands.

    Arguments
    ---------
    drop_freq_low : float
        The low end of frequencies that can be dropped,
        as a fraction of the sampling rate / 2.
    drop_freq_high : float
        The high end of frequencies that can be
        dropped, as a fraction of the sampling rate / 2.
    drop_freq_count_low : int
        The low end of number of frequencies that could be dropped.
    drop_freq_count_high : int
        The high end of number of frequencies that could be dropped.
    drop_freq_width : float
        The width of the frequency band to drop, as
        a fraction of the sampling_rate / 2.
    epsilon : float
        A small positive value to prevent issues such as filtering 0 Hz,
        division by zero, or other numerical instabilities. This value sets
        the absolute minimum for normalized frequencies used in the filter.
        The default value is 1e-12.

    Example
    -------
    >>> from speechbrain.dataio.dataio import read_audio
    >>> dropper = DropFreq()
    >>> signal = read_audio('tests/samples/single-mic/example1.wav')
    >>> dropped_signal = dropper(signal.unsqueeze(0))
    """

    def __init__(
        self,
        drop_freq_low=1e-14,
        drop_freq_high=1,
        drop_freq_count_low=1,
        drop_freq_count_high=3,
        drop_freq_width=0.05,
        epsilon=1e-12,
    ):
        super().__init__()
        self.drop_freq_low = drop_freq_low
        self.drop_freq_high = drop_freq_high
        self.drop_freq_count_low = drop_freq_count_low
        self.drop_freq_count_high = drop_freq_count_high
        self.drop_freq_width = drop_freq_width
        self.epsilon = epsilon

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : torch.Tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        # Don't drop (return early) 1-`drop_prob` portion of the batches
        dropped_waveform = waveforms.clone()

        # Add channels dimension
        if len(waveforms.shape) == 2:
            dropped_waveform = dropped_waveform.unsqueeze(-1)

        # Pick number of frequencies to drop
        drop_count = torch.randint(
            low=self.drop_freq_count_low,
            high=self.drop_freq_count_high + 1,
            size=(1,),
        )

        # Pick a frequency to drop
        drop_range = self.drop_freq_high - self.drop_freq_low
        drop_frequency = (
            torch.rand(drop_count) * drop_range + self.drop_freq_low
        ).clamp(min=self.epsilon)
        # Filter parameters
        filter_length = 101
        pad = filter_length // 2

        # Start with delta function
        drop_filter = torch.zeros(1, filter_length, 1, device=waveforms.device)
        drop_filter[0, pad, 0] = 1

        # Subtract each frequency
        for frequency in drop_frequency:
            notch_kernel = notch_filter(
                frequency, filter_length, self.drop_freq_width
            ).to(waveforms.device)
            drop_filter = convolve1d(drop_filter, notch_kernel, pad)

        # Manage multiple channels
        if len(waveforms.shape) == 3:
            dropped_waveform = dropped_waveform.reshape(
                dropped_waveform.shape[0] * dropped_waveform.shape[2],
                dropped_waveform.shape[1],
                1,
            )

        # Apply filter
        dropped_waveform = convolve1d(dropped_waveform, drop_filter, pad)

        if len(waveforms.shape) == 3:
            dropped_waveform = dropped_waveform.reshape(
                waveforms.shape[0], waveforms.shape[1], waveforms.shape[2]
            )

        # Remove channels dimension if added
        return dropped_waveform.squeeze(-1)


class DropChunk(torch.nn.Module):
    """This class drops portions of the input signal.

    Using `DropChunk` as an augmentation strategy helps a models learn to rely
    on all parts of the signal, since it can't expect a given part to be
    present.

    Arguments
    ---------
    drop_length_low : int
        The low end of lengths for which to set the
        signal to zero, in samples.
    drop_length_high : int
        The high end of lengths for which to set the
        signal to zero, in samples.
    drop_count_low : int
        The low end of number of times that the signal
        can be dropped to zero.
    drop_count_high : int
        The high end of number of times that the signal
        can be dropped to zero.
    drop_start : int
        The first index for which dropping will be allowed.
    drop_end : int
        The last index for which dropping will be allowed.
    noise_factor : float
        The factor relative to average amplitude of an utterance
        to use for scaling the white noise inserted. 1 keeps
        the average amplitude the same, while 0 inserts all 0's.

    Example
    -------
    >>> from speechbrain.dataio.dataio import read_audio
    >>> dropper = DropChunk(drop_start=100, drop_end=200, noise_factor=0.)
    >>> signal = read_audio('tests/samples/single-mic/example1.wav')
    >>> signal = signal.unsqueeze(0) # [batch, time, channels]
    >>> length = torch.ones(1)
    >>> dropped_signal = dropper(signal, length)
    >>> float(dropped_signal[:, 150])
    0.0
    """

    def __init__(
        self,
        drop_length_low=100,
        drop_length_high=1000,
        drop_count_low=1,
        drop_count_high=3,
        drop_start=0,
        drop_end=None,
        noise_factor=0.0,
    ):
        super().__init__()
        self.drop_length_low = drop_length_low
        self.drop_length_high = drop_length_high
        self.drop_count_low = drop_count_low
        self.drop_count_high = drop_count_high
        self.drop_start = drop_start
        self.drop_end = drop_end
        self.noise_factor = noise_factor

        # Validate low < high
        if drop_length_low > drop_length_high:
            raise ValueError("Low limit must not be more than high limit")
        if drop_count_low > drop_count_high:
            raise ValueError("Low limit must not be more than high limit")

        # Make sure the length doesn't exceed end - start
        if drop_end is not None and drop_end >= 0:
            if drop_start > drop_end:
                raise ValueError("Low limit must not be more than high limit")

            drop_range = drop_end - drop_start
            self.drop_length_low = min(drop_length_low, drop_range)
            self.drop_length_high = min(drop_length_high, drop_range)

    def forward(self, waveforms, lengths):
        """
        Arguments
        ---------
        waveforms : torch.Tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : torch.Tensor
            Shape should be a single dimension, `[batch]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or
            `[batch, time, channels]`
        """

        # Reading input list
        lengths = (lengths * waveforms.size(1)).long()
        batch_size = waveforms.size(0)
        dropped_waveform = waveforms.clone()

        # Store original amplitude for computing white noise amplitude
        clean_amplitude = compute_amplitude(waveforms, lengths.unsqueeze(1))

        # Pick a number of times to drop
        drop_times = torch.randint(
            low=self.drop_count_low,
            high=self.drop_count_high + 1,
            size=(batch_size,),
        )

        # Iterate batch to set mask
        for i in range(batch_size):
            if drop_times[i] == 0:
                continue

            # Pick lengths
            length = torch.randint(
                low=self.drop_length_low,
                high=self.drop_length_high + 1,
                size=(drop_times[i],),
            )

            # Compute range of starting locations
            start_min = self.drop_start
            if start_min < 0:
                start_min += lengths[i]
            start_max = self.drop_end
            if start_max is None:
                start_max = lengths[i]
            if start_max < 0:
                start_max += lengths[i]
            start_max = max(0, start_max - length.max())

            # Pick starting locations
            start = torch.randint(
                low=start_min, high=start_max + 1, size=(drop_times[i],)
            )

            end = start + length

            # Update waveform
            if not self.noise_factor:
                for j in range(drop_times[i]):
                    dropped_waveform[i, start[j] : end[j]] = 0.0
            else:
                # Uniform distribution of -2 to +2 * avg amplitude should
                # preserve the average for normalization
                noise_max = 2 * clean_amplitude[i] * self.noise_factor
                for j in range(drop_times[i]):
                    # zero-center the noise distribution
                    noise_vec = torch.rand(length[j], device=waveforms.device)
                    noise_vec = 2 * noise_max * noise_vec - noise_max
                    dropped_waveform[i, start[j] : end[j]] = noise_vec

        return dropped_waveform


class FastDropChunk(torch.nn.Module):
    """This class drops portions of the input signal. The difference with
    DropChunk is that in this case we pre-compute the dropping masks in the
    first time the forward function is called. For all the other calls, we only
    shuffle and apply them. This makes the code faster and more suitable for
    data augmentation of large batches.

    It can be used only for fixed-length sequences.

    Arguments
    ---------
    drop_length_low : int
        The low end of lengths for which to set the
        signal to zero, in samples.
    drop_length_high : int
        The high end of lengths for which to set the
        signal to zero, in samples.
    drop_count_low : int
        The low end of number of times that the signal
        can be dropped to zero.
    drop_count_high : int
        The high end of number of times that the signal
        can be dropped to zero.
    drop_start : int
        The first index for which dropping will be allowed.
    drop_end : int
        The last index for which dropping will be allowed.
    n_masks : int
        The number of precomputed masks.

    Example
    -------
    >>> from speechbrain.dataio.dataio import read_audio
    >>> dropper = FastDropChunk(drop_start=100, drop_end=200)
    >>> signal = torch.rand(10, 250, 22)
    >>> dropped_signal = dropper(signal)
    """

    def __init__(
        self,
        drop_length_low=100,
        drop_length_high=1000,
        drop_count_low=1,
        drop_count_high=10,
        drop_start=0,
        drop_end=None,
        n_masks=1000,
    ):
        super().__init__()
        self.drop_length_low = drop_length_low
        self.drop_length_high = drop_length_high
        self.drop_count_low = drop_count_low
        self.drop_count_high = drop_count_high
        self.drop_start = drop_start
        self.drop_end = drop_end
        self.n_masks = n_masks
        self.first = True

        # Validate low < high
        if drop_length_low > drop_length_high:
            raise ValueError("Low limit must not be more than high limit")
        if drop_count_low > drop_count_high:
            raise ValueError("Low limit must not be more than high limit")

        # Make sure the length doesn't exceed end - start
        if drop_end is not None and drop_end >= 0:
            if drop_start > drop_end:
                raise ValueError("Low limit must not be more than high limit")
            drop_range = drop_end - drop_start
            self.drop_length_low = min(drop_length_low, drop_range)
            self.drop_length_high = min(drop_length_high, drop_range)

    def initialize_masks(self, waveforms):
        """
                Arguments
                ---------
                waveforms : torch.Tensor
                    Shape should be `[batch, time]` or `[batch, time, channels]`.
        `.
                Returns
                -------
                dropped_masks : torch.Tensor
                    Tensor of size `[n_masks, time]` with the dropped chunks. Dropped
                    regions are assigned to 0.
        """

        if self.n_masks < waveforms.shape[0]:
            raise ValueError("n_mask cannot be smaller than the batch size")

        # Initializing the drop mask
        dropped_masks = torch.ones(
            [self.n_masks, self.sig_len], device=waveforms.device
        )

        # Pick a number of times to drop
        drop_times = torch.randint(
            low=self.drop_count_low,
            high=self.drop_count_high + 1,
            size=(self.n_masks,),
            device=waveforms.device,
        )

        # Iterate batch to set mask
        for i in range(self.n_masks):
            if drop_times[i] == 0:
                continue

            # Pick lengths
            length = torch.randint(
                low=self.drop_length_low,
                high=self.drop_length_high + 1,
                size=(drop_times[i],),
                device=waveforms.device,
            )

            # Compute range of starting locations
            start_min = self.drop_start
            if start_min < 0:
                start_min += self.sig_len
            start_max = self.drop_end
            if start_max is None:
                start_max = self.sig_len
            if start_max < 0:
                start_max += self.sig_len
            start_max = max(0, start_max - length.max())

            # Pick starting locations
            start = torch.randint(
                low=start_min,
                high=start_max + 1,
                size=(drop_times[i],),
                device=waveforms.device,
            )

            end = start + length

            # Update waveform
            for j in range(drop_times[i]):
                dropped_masks[i, start[j] : end[j]] = 0.0

        return dropped_masks

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : torch.Tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`
        """

        dropped_waveforms = waveforms.clone()

        # Initialize the masks
        if self.first:
            self.sig_len = waveforms.shape[1]
            self.dropped_masks = self.initialize_masks(waveforms)
            self.first = False

        # Random Permutation
        rand_perm = torch.randperm(self.dropped_masks.shape[0])
        self.dropped_masks = self.dropped_masks[rand_perm, :]

        # Random shift in time
        rand_shifts = torch.randint(low=0, high=self.sig_len, size=(1,))
        self.dropped_masks = torch.roll(
            self.dropped_masks, shifts=rand_shifts.item(), dims=1
        )

        if len(waveforms.shape) == 3:
            dropped_waveforms = dropped_waveforms * self.dropped_masks[
                0 : waveforms.shape[0]
            ].unsqueeze(2)
        else:
            dropped_waveforms = (
                dropped_waveforms * self.dropped_masks[0 : waveforms.shape[0]]
            )

        return dropped_waveforms


class DoClip(torch.nn.Module):
    """This function mimics audio clipping by clamping the input tensor.
    First, it normalizes the waveforms from -1 to -1. Then, clipping is applied.
    Finally, the original amplitude is restored.

    Arguments
    ---------
    clip_low : float
        The low end of amplitudes for which to clip the signal.
    clip_high : float
        The high end of amplitudes for which to clip the signal.

    Example
    -------
    >>> from speechbrain.dataio.dataio import read_audio
    >>> clipper = DoClip(clip_low=0.01, clip_high=0.01)
    >>> signal = read_audio('tests/samples/single-mic/example1.wav')
    >>> clipped_signal = clipper(signal.unsqueeze(0))
    """

    def __init__(self, clip_low=0.5, clip_high=0.5):
        super().__init__()
        self.clip_low = clip_low
        self.clip_high = clip_high

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : torch.Tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`
        """

        # Normalize the signal
        abs_max, _ = torch.max(torch.abs(waveforms), dim=1, keepdim=True)
        waveforms = waveforms / abs_max

        # Randomly select clip value
        clipping_range = self.clip_high - self.clip_low
        clip_value = (
            torch.rand(1, device=waveforms.device)[0] * clipping_range
            + self.clip_low
        )

        # Apply clipping
        clipped_waveform = waveforms.clamp(-clip_value, clip_value)

        # Restore original amplitude
        clipped_waveform = clipped_waveform * abs_max / clip_value

        return clipped_waveform


class RandAmp(torch.nn.Module):
    """This function multiples the signal by a random amplitude. First, the
    signal is normalized to have amplitude between -1 and 1. Then it is
    multiplied with a random number.

    Arguments
    ---------
    amp_low : float
        The minimum amplitude multiplication factor.
    amp_high : float
        The maximum amplitude multiplication factor.

    Example
    -------
    >>> from speechbrain.dataio.dataio import read_audio
    >>> rand_amp = RandAmp(amp_low=0.25, amp_high=1.75)
    >>> signal = read_audio('tests/samples/single-mic/example1.wav')
    >>> output_signal = rand_amp(signal.unsqueeze(0))
    """

    def __init__(self, amp_low=0.5, amp_high=1.5):
        super().__init__()
        self.amp_low = amp_low
        self.amp_high = amp_high

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : torch.Tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`
        """

        # Normalize the signal
        abs_max, _ = torch.max(torch.abs(waveforms), dim=1, keepdim=True)
        waveforms = waveforms / abs_max

        # Pick a frequency to drop
        rand_range = self.amp_high - self.amp_low
        amp = (
            torch.rand(waveforms.shape[0], device=waveforms.device) * rand_range
            + self.amp_low
        )
        amp = amp.unsqueeze(1)
        if len(waveforms.shape) == 3:
            amp = amp.unsqueeze(2)
        waveforms = waveforms * amp

        return waveforms


class ChannelDrop(torch.nn.Module):
    """This function drops random channels in the multi-channel input waveform.

    Arguments
    ---------
    drop_rate : float
        The channel dropout factor

    Example
    -------
    >>> signal = torch.rand(4, 256, 8)
    >>> ch_drop = ChannelDrop(drop_rate=0.5)
    >>> output_signal = ch_drop(signal)
    """

    def __init__(self, drop_rate=0.1):
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : torch.Tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`
        """

        # Pick a channel to drop
        x = torch.rand(waveforms.shape[-1], device=waveforms.device)
        channel_mask = x.ge(self.drop_rate)
        waveforms = waveforms * channel_mask.unsqueeze(0).unsqueeze(1)
        return waveforms


class ChannelSwap(torch.nn.Module):
    """This function randomly swaps N channels.

    Arguments
    ---------
    min_swap : int
        The minimum number of channels to swap.
    max_swap : int
        The maximum number of channels to swap.

    Example
    -------
    >>> signal = torch.rand(4, 256, 8)
    >>> ch_swap = ChannelSwap()
    >>> output_signal = ch_swap(signal)
    """

    def __init__(self, min_swap=0, max_swap=0):
        super().__init__()
        self.min_swap = min_swap
        self.max_swap = max_swap

        # Check arguments
        if self.min_swap < 0:
            raise ValueError("min_swap must be  >= 0.")
        if self.max_swap < 0:
            raise ValueError("max_swap must be  >= 0.")
        if self.max_swap < self.min_swap:
            raise ValueError("max_swap must be  >= min_swap")

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : torch.Tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`
        """

        # Pick a frequency to drop
        rand_perm1 = torch.randperm(waveforms.shape[-1])
        rand_perm2 = torch.randperm(waveforms.shape[-1])
        N_swaps = torch.randint(
            low=self.min_swap, high=self.max_swap + 1, size=(1,)
        )

        if N_swaps < waveforms.shape[-1]:
            for i in range(N_swaps):
                store_channel = waveforms[:, :, rand_perm2[i]]
                waveforms[:, :, rand_perm2[i]] = waveforms[:, :, rand_perm1[i]]
                waveforms[:, :, rand_perm1[i]] = store_channel
        else:
            # Full swap
            waveforms = waveforms[:, :, rand_perm1]

        return waveforms


class CutCat(torch.nn.Module):
    """This function combines segments (with equal length in time) of the time series contained in the batch.
    Proposed for EEG signals in https://doi.org/10.1016/j.neunet.2021.05.032.

    Arguments
    ---------
    min_num_segments : int
        The number of segments to combine.
    max_num_segments : int
        The maximum number of segments to combine. Default is 10.

    Example
    -------
    >>> signal = torch.ones((4, 256, 22)) * torch.arange(4).reshape((4, 1, 1,))
    >>> cutcat =  CutCat()
    >>> output_signal = cutcat(signal)
    """

    def __init__(self, min_num_segments=2, max_num_segments=10):
        super().__init__()
        self.min_num_segments = min_num_segments
        self.max_num_segments = max_num_segments
        # Check arguments
        if self.max_num_segments < self.min_num_segments:
            raise ValueError("max_num_segments must be  >= min_num_segments")

    def forward(self, waveforms):
        """
        Arguments
        ---------
        waveforms : torch.Tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`
        """
        if (
            waveforms.shape[0] > 1
        ):  # only if there are at least 2 examples in batch
            # rolling waveforms to point to segments of other examples in batch
            waveforms_rolled = torch.roll(waveforms, shifts=1, dims=0)
            # picking number of segments to use
            num_segments = torch.randint(
                low=self.min_num_segments,
                high=self.max_num_segments + 1,
                size=(1,),
            )
            # index of cuts (both starts and stops)
            idx_cut = torch.linspace(
                0, waveforms.shape[1], num_segments.item() + 1, dtype=torch.int
            )
            for i in range(idx_cut.shape[0] - 1):
                # half of segments from other examples in batch
                if i % 2 == 1:
                    start = idx_cut[i]
                    stop = idx_cut[i + 1]
                    waveforms[:, start:stop, ...] = waveforms_rolled[
                        :, start:stop, ...  # noqa: W504
                    ]

        return waveforms


def pink_noise_like(waveforms, alpha_low=1.0, alpha_high=1.0, sample_rate=50):
    """Creates a sequence of pink noise (also known as 1/f). The pink noise
    is obtained by multiplying the spectrum of a white noise sequence by a
    factor (1/f^alpha).
    The alpha factor controls the decrease factor in the frequency domain
    (alpha=0 adds white noise, alpha>>0 adds low frequency noise). It is
    randomly sampled between alpha_low and alpha_high. With negative alpha this
    function generates blue noise.

    Arguments
    ---------
    waveforms : torch.Tensor
        The original waveform. It is just used to infer the shape.
    alpha_low : float
        The minimum value for the alpha spectral smoothing factor.
    alpha_high : float
        The maximum value for the alpha spectral smoothing factor.
    sample_rate : float
        The sample rate of the original signal.

    Returns
    -------
    pink_noise : torch.Tensor
        Pink noise in the shape of the input tensor.

    Example
    -------
    >>> waveforms = torch.randn(4,257,10)
    >>> noise = pink_noise_like(waveforms)
    >>> noise.shape
    torch.Size([4, 257, 10])
    """
    # Sampling white noise (flat spectrum)
    white_noise = torch.randn_like(waveforms)

    # Computing the fft of the input white noise
    white_noise_fft = torch.fft.fft(white_noise, dim=1)

    # Sampling the spectral smoothing factor
    rand_range = alpha_high - alpha_low
    alpha = (
        torch.rand(waveforms.shape[0], device=waveforms.device) * rand_range
        + alpha_low
    )

    # preparing the spectral mask (1/f^alpha)
    f = torch.linspace(
        0,
        sample_rate / 2,
        int(white_noise.shape[1] / 2),
        device=waveforms.device,
    )
    spectral_mask = 1 / torch.pow(f.unsqueeze(0), alpha.unsqueeze(1))

    # Avoid inf due to 1/0 division at f=0
    spectral_mask[:, 0] = spectral_mask[:, 1]

    # Mask for the upper part of the spectrum (f > sample_rate/2)
    spectral_mask_up = torch.flip(spectral_mask, dims=(1,))

    # Managing odd/even sequences
    if white_noise.shape[1] % 2:
        mid_element = spectral_mask[
            :, int(white_noise.shape[1] / 2) - 1
        ].unsqueeze(1)
        spectral_mask = torch.cat(
            [spectral_mask, mid_element, spectral_mask_up], dim=1
        )
    else:
        spectral_mask = torch.cat([spectral_mask, spectral_mask_up], dim=1)

    # Managing multi-channel inputs
    if len(white_noise.shape) == 3:
        spectral_mask = spectral_mask.unsqueeze(2)

    # Spectral masking
    pink_noise_fft = white_noise_fft * spectral_mask

    # Return to the time-domain
    pink_noise = torch.fft.ifft(pink_noise_fft, dim=1).real
    return pink_noise


class DropBitResolution(torch.nn.Module):
    """
    This class transforms a float32 tensor into a lower resolution one
    (e.g., int16, int8, float16) and then converts it back to a float32.
    This process loses information and can be used for data augmentation.

    Arguments:
    ---------
        target_dtype: str
            One of "int16", "int8", "float16". If "random", the bit resolution
            is randomly selected among the options listed above.

    Example:
        >>> dropper = DropBitResolution()
        >>> signal = torch.rand(4, 16000)
        >>> signal_dropped = dropper(signal)
    """

    def __init__(self, target_dtype="random"):
        super().__init__()

        self.target_dtype = target_dtype
        self.bit_depths = {
            "int16": (16, torch.int16),
            "int8": (8, torch.int8),
            "float16": (16, torch.float16),
        }

        if (
            self.target_dtype != "random"
            and self.target_dtype not in self.bit_depths
        ):
            raise ValueError(
                f"target_dtype must be one of {list(self.bit_depths.keys())}"
            )

    def forward(self, float32_tensor):
        """
        Arguments:
        ---------
            float32_tensor: torch.Tensor
                Float32 tensor with shape `[batch, time]` or `[batch, time, channels]`.

        Returns:
        ---------
            torch.Tensor
                Tensor of shape `[batch, time]` or `[batch, time, channels]` (Float32)
        """

        if self.target_dtype == "random":
            random_key = random.choice(list(self.bit_depths.keys()))
            bit, target_dtype = self.bit_depths[random_key]
        else:
            bit, target_dtype = self.bit_depths[self.target_dtype]

        # Define a scale factor to map the float32 range to the target bit depth
        if target_dtype != torch.float16:
            scale_factor = (2 ** (bit - 1) - 1) / float32_tensor.abs().max()
            quantized_tensor = (float32_tensor * scale_factor).to(target_dtype)
        else:
            quantized_tensor = float32_tensor.half()
            scale_factor = 1

        # To dequantize and recover the original float32 values
        dequantized_tensor = quantized_tensor.to(torch.float32) / scale_factor
        return dequantized_tensor


class SignFlip(torch.nn.Module):
    """Flip the sign of a signal.

    This module negates all the values in a tensor with a given probability.
    If the sign is not flipped, the original signal is returned
    unchanged. This technique is outlined in the paper:
    "CADDA: Class-wise Automatic Differentiable Data Augmentation for EEG Signals"
    https://arxiv.org/pdf/2106.13695

    Arguments
    ---------
    flip_prob : float
        The probability with which to flip the sign of the signal. Default is 0.5.

    Example
    -------
    >>> import torch
    >>> x = torch.tensor([1,2,3,4,5])
    >>> flip = SignFlip(flip_prob=1) # 100% chance to flip sign
    >>> flip(x)
    tensor([-1, -2, -3, -4, -5])
    """

    def __init__(self, flip_prob=0.5):
        super().__init__()
        self.flip_prob = flip_prob

    def forward(self, waveform):
        """
        Arguments
        ---------
        waveform : torch.Tensor
            Input tensor representaing waveform, shape does not matter.

        Returns
        -------
        torch.Tensor
            The output tensor with same shape as the input, where the
            sign of all values in the tensor has been flipped with
            probability `flip_prob`.

        """

        # Flip sign with `flip_prob` probability.
        if torch.rand(1).item() < self.flip_prob:
            return -waveform

        return waveform
