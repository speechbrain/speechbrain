"""Classes for mutating speech data for data augmentation.

This module provides classes that produce realistic distortions of speech
data for the purpose of training speech processing models. The list of
distortions includes adding noise, adding reverberation, changing speed,
and more. All the classes are of type `torch.nn.Module`. This gives the
possibility to have end-to-end differentiability and
backpropagate the gradient through them. In addition, all operations
are expected to be performed on the GPU (where available) for efficiency.
"""

# Importing libraries
import math
import torch
from speechbrain.data_io.data_io import create_dataloader
from speechbrain.utils.data_utils import (
    compute_amplitude,
    dB_to_amplitude,
    convolve1d,
    notch_filter,
)


class AddNoise(torch.nn.Module):
    """This class additively combines a noise signal to the input signal.

    Attributes:
        csv_file (str): The name of a csv file containing the location of the
            noise audio files. If none is provided, white noise will be used.
        order (str): The order to iterate the csv file, from one of the
            following options: random, original, ascending, and descending.
        batch_size (int): If an csv_file is passed, this controls the number of
            samples that are loaded at the same time, should be the same as
            or less than the size of the clean batch. If `None` is passed,
            the size of the first clean batch will be used.
        do_cache (bool): Whether or not to store noise files in the cache.
        snr_low (int): The low end of the mixing ratios, in decibels.
        snr_high (int): The high end of the mixing ratios, in decibels.
        pad_noise (bool): If True, copy noise signals that are shorter than
            their corresponding clean signals so as to cover the whole clean
            signal. Otherwise, leave the noise un-padded.
        mix_prob (float): The probability that a batch of signals will be mixed
            with a noise signal. By default, every batch is mixed with noise.
        replacements (dict): A set of string replacements to carry out in the
            csv file. Each time a key is found in the text, it will be replaced
            with the corresponding value.

    Example:
        >>> import torch
        >>> import soundfile as sf
        >>> from speechbrain.data_io.data_io import save
        >>> signal, rate = sf.read('samples/audio_samples/example1.wav')
        >>> noisifier = AddNoise('samples/noise_samples/noise.csv')
        >>> clean = torch.tensor([signal], dtype=torch.float32)
        >>> noisy = noisifier(clean, torch.ones(1))
        >>> save_signal = save(save_folder='exp/example', save_format='wav')
        >>> save_signal(noisy, ['example_add_noise'], torch.ones(1))

    Author:
        Peter Plantinga 2020
    """
    def __init__(
        self,
        csv_file=None,
        order='random',
        batch_size=None,
        do_cache=False,
        snr_low=0,
        snr_high=0,
        pad_noise=False,
        mix_prob=1.,
        replacements={},
    ):
        """"""
        super().__init__()

        self.csv_file = csv_file
        self.order = order
        self.batch_size = batch_size
        self.do_cache = do_cache
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.pad_noise = pad_noise
        self.mix_prob = mix_prob
        self.replacements = replacements

        # On first input, create dataloader with correct batch size
        def hook(self, input):

            waveforms, lengths = input

            # Set parameters based on input
            self.device = waveforms.device
            if not self.batch_size:
                self.batch_size = len(waveforms)

            # Create a data loader for the noise wavforms
            if self.csv_file is not None:
                self.data_loader = create_dataloader(
                    csv_file=self.csv_file,
                    sentence_sorting=self.order,
                    batch_size=self.batch_size,
                    cache=self.do_cache,
                    replacements=self.replacements,
                )
                self.noise_data = zip(*self.data_loader())

            # Remove this hook after it runs once
            self.hook.remove()

        self.hook = self.register_forward_pre_hook(hook)

    def forward(self, waveforms, lengths):
        """
        Parameters:
            waveforms (tensor): Shape should be `[batch, time_steps]`
                or`[batch, channels, time_steps]`.
            lengths (tensor): Shape should be a single dimension, `[batch]`.

        Returns:
            Tensor of shape `[batch, time_steps]` or
                `[batch, channels, time_steps]`
        """
        # Copy clean waveform to initialize noisy waveform
        noisy_waveform = waveforms.clone()
        lengths = (lengths * waveforms.shape[1]).unsqueeze(1)

        # current batch size is min of stored size and clean size
        batch_size = self.batch_size
        if batch_size is None or batch_size > len(waveforms):
            batch_size = len(waveforms)

        # Don't add noise (return early) 1-`mix_prob` portion of the batches
        if torch.rand(1) > self.mix_prob:
            return noisy_waveform

        # Compute the average amplitude of the clean waveforms
        clean_amplitude = compute_amplitude(waveforms, lengths)

        # Pick an SNR and use it to compute the mixture amplitude factors
        SNR = torch.rand(batch_size, 1, device=waveforms.device)
        SNR = SNR * (self.snr_high - self.snr_low) + self.snr_low
        noise_amplitude_factor = 1 / (dB_to_amplitude(SNR) + 1)
        new_noise_amplitude = noise_amplitude_factor * clean_amplitude

        # Scale clean signal appropriately
        noisy_waveform[:batch_size] *= (1 - noise_amplitude_factor)

        # Loop through clean samples and create mixture
        if self.csv_file is None:
            white_noise = torch.randn_like(waveforms)
            noisy_waveform += new_noise_amplitude * white_noise
        else:
            tensor_length = waveforms.shape[-1]
            noise_waveform, noise_length = self._load_noise(
                lengths, tensor_length, batch_size,
            )

            # Rescale and add
            noise_amplitude = compute_amplitude(noise_waveform, noise_length)
            noise_waveform *= new_noise_amplitude / noise_amplitude
            noisy_waveform[:batch_size] += noise_waveform

        return noisy_waveform

    def _load_noise(self, clean_len, tensor_len, batch_size):
        clean_len = clean_len.long().squeeze(1)

        # Load a noise batch
        try:
            wav_id, noise_batch, wav_len = next(self.noise_data)[0]
        except StopIteration:
            self.noise_data = zip(*self.data_loader())
            wav_id, noise_batch, wav_len = next(self.noise_data)[0]

        noise_batch = noise_batch.to(clean_len.device)
        wav_len = wav_len.to(clean_len.device)

        # Chop to correct size
        if len(noise_batch) > batch_size:
            noise_batch = noise_batch[:batch_size]
            wav_len = wav_len[:batch_size]

        # Convert relative length to an index
        wav_len = (wav_len * noise_batch.shape[-1]).long()

        # Ensure shortest wav can cover speech signal
        # WARNING: THIS COULD BE SLOW IF THERE ARE VERY SHORT NOISES
        if self.pad_noise:
            while torch.any(wav_len < clean_len):
                min_len = torch.min(wav_len)
                prepend = noise_batch[..., :min_len]
                noise_batch = torch.cat((prepend, noise_batch), axis=-1)
                wav_len += min_len

        # Ensure noise batch is long enough
        elif noise_batch.size(-1) < tensor_len:
            padding = (0, tensor_len - noise_batch.size(-1))
            noise_batch = torch.nn.functional.pad(noise_batch, padding)

        # Select a random starting location in the waveform
        start_index = 0
        max_chop = (wav_len - clean_len).min().clamp(min=1)
        start_index = torch.randint(high=max_chop, size=(1,))

        # Truncate noise_batch to tensor_len
        noise_batch = noise_batch[..., start_index:start_index+tensor_len]
        wav_len = (wav_len - start_index).clamp(max=tensor_len).unsqueeze(1)
        return noise_batch, wav_len


class AddReverb(torch.nn.Module):
    """This class convolves an audio signal with an impulse response.

    Attributes:
        csv_file (str): The name of a csv file containing the location of the
            impulse response files.
        order (str): The order to iterate the csv file, from one of
            the following options: random, original, ascending,
            and descending.
        do_cache (bool): Whether or not to lazily load the files to a
            cache and read the data from the cache.
        reverb_prob (float): The chance that the audio signal will be reverbed.
            By default, every batch is reverbed.
        replacements (dict): A set of string replacements to carry out in the
            csv file. Each time a key is found in the text, it will be replaced
            with the corresponding value.

    Example:
        >>> import torch
        >>> import soundfile as sf
        >>> from speechbrain.data_io.data_io import save
        >>> signal, rate = sf.read('samples/audio_samples/example1.wav')
        >>> reverb = AddReverb('samples/rir_samples/rirs.csv')
        >>> clean = torch.tensor([signal], dtype=torch.float32)
        >>> reverbed = reverb(clean, torch.ones(1))
        >>> save_signal = save(save_folder='exp/example', save_format='wav')
        >>> save_signal(reverbed, ['example_add_reverb'], torch.ones(1))

    Author:
        Peter Plantinga 2020
    """

    def __init__(
        self,
        csv_file,
        order='random',
        do_cache=False,
        reverb_prob=1.,
        replacements={},
    ):
        """"""
        super().__init__()
        self.csv_file = csv_file
        self.order = order
        self.do_cache = do_cache
        self.reverb_prob = reverb_prob
        self.replacements = replacements

        # Create a data loader for the RIR waveforms
        self.data_loader = create_dataloader(
            csv_file=self.csv_file,
            sentence_sorting=self.order,
            cache=self.do_cache,
            replacements=self.replacements,
        )
        self.rir_data = zip(*self.data_loader())

        def hook(self, input):
            self.device = input[0].device
            self.dtype = input[0].dtype
            self.hook.remove()

        self.hook = self.register_forward_pre_hook(hook)

    def forward(self, waveforms, lengthss):
        """
        Parameters:
            waveforms (tensor): Shape should be `[batch, time_steps]`
                or`[batch, channels, time_steps]`.
            lengths (tensor): Shape should be a single dimension, `[batch]`.

        Returns:
            Tensor of shape `[batch, time_steps]` or
                `[batch, channels, time_steps]`
        """
        # Don't add reverb (return early) 1-`reverb_prob` portion of the time
        if torch.rand(1) > self.reverb_prob:
            return waveforms.clone()

        # Add channels dimension if necessary
        channel_added = False
        if len(waveforms.shape) == 2:
            waveforms = clean_waveform.unsqueeze(1)
            channel_added = True

        # Convert length from ratio to number of indices
        clean_len = (lengthss * waveforms.shape[2])[:, None, None]

        # Compute the average amplitude of the clean
        clean_amplitude = compute_amplitude(waveforms, clean_len)

        # Load and prepare RIR
        rir_waveform = self._load_rir().abs()

        # Compute index of the direct signal, so we can preserve alignment
        direct_index = rir_waveform.argmax(axis=-1).median()

        # Use FFT to compute convolution, because of long reverberation filter
        reverbed_waveform = convolve1d(
            waveform=waveforms,
            kernel=rir_waveform,
            use_fft=True,
            rotation_index=direct_index,
        )

        # Rescale to the average amplitude of the clean waveform
        reverbed_amplitude = compute_amplitude(reverbed_waveform, clean_len)
        reverbed_waveform *= clean_amplitude / reverbed_amplitude

        # Remove channels dimension if added
        if channel_added:
            return reverbed_waveform.squeeze(1)

        return reverbed_waveform

    def _load_rir(self):
        try:
            wav_id, rir_waveform, length = next(self.rir_data)[0]
        except StopIteration:
            self.rir_data = zip(*self.data_loader())
            wav_id, rir_waveform, length = next(self.rir_data)[0]

        # Make sure RIR has correct channels
        if len(rir_waveform.shape) == 2:
            rir_waveform = rir_waveform.unsqueeze(1)

        # Make sure RIR has correct type and device
        rir_waveform = rir_waveform.type(self.dtype)
        return rir_waveform.to(self.device)


class SpeedPerturb(torch.nn.Module):
    """Slightly speed up or slow down an audio signal

    Resample the audio signal at a rate that is similar to the original rate,
    to achieve a slightly slower or slightly faster signal. This technique is
    outlined in the paper: "Audio Augmentation for Speech Recognition"

    Attributes:
        orig_freq (int): The frequency of the original signal.
        speeds (list): The speeds that the signal should be changed to,
            where 10 is the speed of the original signal.
        perturb_prob (float): The chance that the batch will be speed-
            perturbed. By default, every batch is perturbed.

    Example:
        >>> import torch
        >>> import soundfile as sf
        >>> from speechbrain.data_io.data_io import save
        >>> signal, rate = sf.read('samples/audio_samples/example1.wav')
        >>> perturbator = SpeedPerturb(orig_freq=rate, speeds=[9])
        >>> clean = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        >>> perturbed = perturbator(clean)
        >>> save_signal = save(save_folder='exp/example', save_format='wav')
        >>> save_signal(perturbed, ['example_perturb'], torch.ones(1))

    Author:
        Peter Plantinga 2020
    """

    def __init__(
        self,
        orig_freq,
        speeds=[9, 10, 11],
        perturb_prob=1.,
    ):
        """"""
        super().__init__()
        self.orig_freq = orig_freq
        self.speeds = speeds
        self.perturb_prob = perturb_prob

        # Initialize index of perturbation
        self.samp_index = 0

        # Initialize resamplers
        self.resamplers = []
        for speed in self.speeds:
            config = {
                'orig_freq': self.orig_freq,
                'new_freq': self.orig_freq * speed // 10,
            }
            self.resamplers.append(Resample(**config))

    def forward(self, waveform):
        """
        Parameters:
            waveforms (tensor): Shape should be `[batch, time_steps]`
                or`[batch, channels, time_steps]`.
            lengths (tensor): Shape should be a single dimension, `[batch]`.

        Returns:
            Tensor of shape `[batch, time_steps]` or
                `[batch, channels, time_steps]`
        """
        # add channels dimension
        waveform = waveform.unsqueeze(1)

        # Don't perturb (return early) 1-`perturb_prob` portion of the batches
        if torch.rand(1) > self.perturb_prob:
            return waveform.clone()

        # Perform a random perturbation
        self.samp_index = torch.randint(len(self.speeds), (1,))[0]
        perturbed_waveform = self.resamplers[self.samp_index](waveform)

        return perturbed_waveform.squeeze(1)


class Resample(torch.nn.Module):
    """This class resamples an audio signal using sinc-based interpolation.

    It is a modification of the `resample` function from torchaudio
    (https://pytorch.org/audio/transforms.html#resample)

    Attributes:
        orig_freq (int): the sampling frequency of the input signal.
        new_freq (int): the new sampling frequency after this operation
            is performed.
        lowpass_filter_width (int): Controls the sharpness of the filter,
            larger numbers result in a sharper filter, but they are less
            efficient. Values from 4 to 10 are allowed.

    Example:
        >>> import torch
        >>> import soundfile as sf
        >>> from speechbrain.data_io.data_io import save
        >>> signal, rate = sf.read('samples/audio_samples/example1.wav')
        >>> signal = torch.tensor(signal, dtype=torch.float32)[None,None,:]
        >>> resampler = Resample(orig_freq=rate, new_freq=rate//2)
        >>> resampled = resampler(signal)
        >>> save_signal = save(
        ...     save_folder='exp/example',
        ...     save_format='wav',
        ...     sampling_rate=rate//2,
        ... )
        >>> save_signal(resampled, ["example_resamp"], torch.ones(1))

    Author:
        Peter Plantinga 2020
    """
    def __init__(
        self,
        orig_freq=16000,
        new_freq=16000,
        lowpass_filter_width=6,
    ):
        """"""
        super().__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.lowpass_filter_width = lowpass_filter_width

        # Compute rate for striding
        self._compute_strides()
        assert self.orig_freq % self.conv_stride == 0
        assert self.new_freq % self.conv_transpose_stride == 0

        def hook(self, input):
            self.device = input[0].device

            # Generate and store the filter to use for resampling
            self._indices_and_weights()
            assert self.first_indices.dim() == 1

            self.hook.remove()

        self.hook = self.register_forward_pre_hook(hook)

    def _compute_strides(self):
        """Compute the phases in polyphase filter

        Author:
            (almost directly from torchaudio.compliance.kaldi)
        """
        # Compute new unit based on ratio of in/out frequencies
        base_freq = math.gcd(self.orig_freq, self.new_freq)
        input_samples_in_unit = self.orig_freq // base_freq
        self.output_samples = self.new_freq // base_freq

        # Store the appropriate stride based on the new units
        self.conv_stride = input_samples_in_unit
        self.conv_transpose_stride = self.output_samples

    def forward(self, waveform):
        """
        Parameters:
            waveforms (tensor): Shape should be `[batch, time_steps]`
                or`[batch, channels, time_steps]`.
            lengths (tensor): Shape should be a single dimension, `[batch]`.

        Returns:
            Tensor of shape `[batch, time_steps]` or
                `[batch, channels, time_steps]`
        """
        waveform = waveform.to(self.device)

        # Don't do anything if the frequencies are the same
        if self.orig_freq == self.new_freq:
            return waveform

        # Add channels dimension if necessary
        if len(waveform.shape) == 2:
            waveform = waveform.unsqueeze(1)

        # Do resampling
        resampled_waveform = self._perform_resample(waveform)

        # Remove unnecessary channels dimension
        return resampled_waveform.squeeze(1)

    def _perform_resample(self, waveform):
        """Resamples the waveform at the new frequency.

        This matches Kaldi's OfflineFeatureTpl ResampleWaveform which uses a
        LinearResample (resample a signal at linearly spaced intervals to
        up/downsample a signal). LinearResample (LR) means that the output
        signal is at linearly spaced intervals (i.e the output signal has a
        frequency of `new_freq`). It uses sinc/bandlimited interpolation to
        upsample/downsample the signal.

        https://ccrma.stanford.edu/~jos/resample/
        Theory_Ideal_Bandlimited_Interpolation.html

        https://github.com/kaldi-asr/kaldi/blob/master/src/feat/resample.h#L56

        Parameters:
            waveform (tensor): the batch of audio signals to resample

        Returns:
            The waveform at the new frequency

        Author:
            (almost directly from torchaudio.compliance.kaldi)
        """

        # Compute output size and initialize
        batch_size, num_channels, wave_len = waveform.size()
        window_size = self.weights.size(1)
        tot_output_samp = self._output_samples(wave_len)
        resampled_waveform = torch.zeros(
            (batch_size, num_channels, tot_output_samp),
            device=waveform.device,
        )
        self.weights = self.weights.to(waveform.device)

        # Check weights are on correct device
        if waveform.device != self.weights.device:
            self.weights = self.weights.to(waveform.device)

        # eye size: (num_channels, num_channels, 1)
        eye = torch.eye(num_channels, device=waveform.device).unsqueeze(2)

        # Iterate over the phases in the polyphase filter
        for i in range(self.first_indices.size(0)):
            wave_to_conv = waveform
            first_index = int(self.first_indices[i].item())
            if first_index >= 0:
                # trim the signal as the filter will not be applied
                # before the first_index
                wave_to_conv = wave_to_conv[..., first_index:]

            # pad the right of the signal to allow partial convolutions
            # meaning compute values for partial windows (e.g. end of the
            # window is outside the signal length)
            max_index = (tot_output_samp - 1) // self.output_samples
            end_index = max_index * self.conv_stride + window_size
            current_wave_len = wave_len - first_index
            right_padding = max(0, end_index + 1 - current_wave_len)
            left_padding = max(0, -first_index)
            conv_wave = convolve1d(
                waveform=wave_to_conv,
                kernel=self.weights[i].repeat(num_channels, 1, 1),
                padding=(left_padding, right_padding),
                stride=self.conv_stride,
                groups=num_channels,
            )

            # we want conv_wave[:, i] to be at
            # output[:, i + n*conv_transpose_stride]
            dilated_conv_wave = torch.nn.functional.conv_transpose1d(
                conv_wave, eye, stride=self.conv_transpose_stride)

            # pad dilated_conv_wave so it reaches the output length if needed.
            left_padding = i
            previous_padding = left_padding + dilated_conv_wave.size(-1)
            right_padding = max(0, tot_output_samp - previous_padding)
            dilated_conv_wave = torch.nn.functional.pad(
                dilated_conv_wave, (left_padding, right_padding))
            dilated_conv_wave = dilated_conv_wave[..., :tot_output_samp]

            resampled_waveform += dilated_conv_wave

        return resampled_waveform

    def _output_samples(self, input_num_samp):
        """Based on LinearResample::GetNumOutputSamples.

        LinearResample (LR) means that the output signal is at
        linearly spaced intervals (i.e the output signal has a
        frequency of ``new_freq``). It uses sinc/bandlimited
        interpolation to upsample/downsample the signal.

        Parameters:
            input_num_samp (int): The number of samples in each example in
                the batch.

        Returns:
            Number of samples in the output waveform.

        Author:
            (almost directly from torchaudio.compliance.kaldi)
        """
        # For exact computation, we measure time in "ticks" of 1.0 / tick_freq,
        # where tick_freq is the least common multiple of samp_in and
        # samp_out.
        samp_in = int(self.orig_freq)
        samp_out = int(self.new_freq)

        tick_freq = abs(samp_in * samp_out) // math.gcd(samp_in, samp_out)
        ticks_per_input_period = tick_freq // samp_in

        # work out the number of ticks in the time interval
        # [ 0, input_num_samp/samp_in ).
        interval_length = input_num_samp * ticks_per_input_period
        if interval_length <= 0:
            return 0
        ticks_per_output_period = tick_freq // samp_out

        # Get the last output-sample in the closed interval,
        # i.e. replacing [ ) with [ ]. Note: integer division rounds down.
        # See http://en.wikipedia.org/wiki/Interval_(mathematics) for an
        # explanation of the notation.
        last_output_samp = interval_length // ticks_per_output_period

        # We need the last output-sample in the open interval, so if it
        # takes us to the end of the interval exactly, subtract one.
        if last_output_samp * ticks_per_output_period == interval_length:
            last_output_samp -= 1

        # First output-sample index is zero, so the number of output samples
        # is the last output-sample plus one.
        num_output_samp = last_output_samp + 1

        return num_output_samp

    def _indices_and_weights(self):
        """Based on LinearResample::SetIndexesAndWeights

        Retrieves the weights for resampling as well as the indices in which
        they are valid. LinearResample (LR) means that the output signal is at
        linearly spaced intervals (i.e the output signal has a frequency
        of ``new_freq``). It uses sinc/bandlimited interpolation to
        upsample/downsample the signal.

        Returns:
            - the place where each filter should start being applied
            - the filters to be applied to the signal for resampling

        Author:
            (almost directly from torchaudio.compliance.kaldi)
        """
        # Lowpass filter frequency depends on smaller of two frequencies
        min_freq = min(self.orig_freq, self.new_freq)
        lowpass_cutoff = 0.99 * 0.5 * min_freq

        assert lowpass_cutoff * 2 <= min_freq
        window_width = self.lowpass_filter_width / (2.0 * lowpass_cutoff)

        assert lowpass_cutoff < min(self.orig_freq, self.new_freq) / 2
        output_t = torch.arange(0., self.output_samples, device=self.device)
        output_t /= self.new_freq
        min_t = output_t - window_width
        max_t = output_t + window_width

        min_input_index = torch.ceil(min_t * self.orig_freq)
        max_input_index = torch.floor(max_t * self.orig_freq)
        num_indices = max_input_index - min_input_index + 1

        max_weight_width = num_indices.max()
        j = torch.arange(max_weight_width, device=self.device).unsqueeze(0)
        input_index = min_input_index.unsqueeze(1) + j
        delta_t = (input_index / self.orig_freq) - output_t.unsqueeze(1)

        weights = torch.zeros_like(delta_t)
        inside_window_indices = delta_t.abs().lt(window_width)

        # raised-cosine (Hanning) window with width `window_width`
        weights[inside_window_indices] = 0.5 * (
            1 + torch.cos(2 * math.pi * lowpass_cutoff
                          / self.lowpass_filter_width
                          * delta_t[inside_window_indices])
        )

        t_eq_zero_indices = delta_t.eq(0.0)
        t_not_eq_zero_indices = ~t_eq_zero_indices

        # sinc filter function
        weights[t_not_eq_zero_indices] *= torch.sin(
            2 * math.pi * lowpass_cutoff * delta_t[t_not_eq_zero_indices]
        ) / (math.pi * delta_t[t_not_eq_zero_indices])

        # limit of the function at t = 0
        weights[t_eq_zero_indices] *= 2 * lowpass_cutoff

        # size (output_samples, max_weight_width)
        weights /= self.orig_freq

        self.first_indices = min_input_index
        self.weights = weights


class AddBabble(torch.nn.Module):
    """Simulate babble noise by mixing the signals in a batch.

    Attributes:
        speaker_count (int): The number of signals to mix with the original
            signal.
        snr_low (int): The low end of the mixing ratios, in decibels.
        snr_high (int): The high end of the mixing ratios, in decibels.
        mix_prob (float): The probability that the batch of signals will be
            mixed with babble noise. By default, every signal is mixed.

    Example:
        >>> import torch
        >>> from speechbrain.data_io.data_io import save
        >>> from speechbrain.data_io.data_io import create_dataloader
        >>> babbler = AddBabble()
        >>> dataloader = create_dataloader(
        ...     csv_file='samples/audio_samples/csv_example3.csv',
        ...     batch_size=5,
        ... )
        >>> loader = zip(*dataloader())
        >>> ids, batch, lengths = next(loader)[0]
        >>> noisy = babbler(batch, lengths)
        >>> save_signal = save(save_folder='exp/example', save_format='wav')
        >>> save_signal(noisy, ids, lengths)

    Author:
        Peter Plantinga 2020
    """

    def __init__(
        self,
        speaker_count=3,
        snr_low=0,
        snr_high=0,
        mix_prob=1,
    ):
        """"""
        super().__init__()
        self.speaker_count = speaker_count
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.mix_prob = mix_prob

    def forward(self, waveforms, lengths):
        """
        Parameters:
            waveforms (tensor): A batch of audio signals to process,
                with shape `[batch, time_steps]` or
                `[batch, channels, time_steps]`
            lengths (tensor): The length of each audio in the batch,
                with shape `[batch]`

        Returns:
            Tensor with processed waveforms.
        """
        babbled_waveform = waveforms.clone()
        clean_len = (clean_len * waveforms.shape[1]).unsqueeze(1)
        batch_size = len(waveforms)

        # Don't mix (return early) 1-`mix_prob` portion of the batches
        if torch.rand(1) > self.mix_prob:
            return babbled_waveform

        # Pick an SNR and use it to compute the mixture amplitude factors
        clean_amplitude = compute_amplitude(waveforms, clean_len)
        SNR = torch.rand(batch_size, 1, device=waveforms.device)
        SNR = SNR * (self.snr_high - self.snr_low) + self.snr_low
        noise_amplitude_factor = 1 / (dB_to_amplitude(SNR) + 1)
        new_noise_amplitude = noise_amplitude_factor * clean_amplitude

        # Scale clean signal appropriately
        babbled_waveform *= (1 - noise_amplitude_factor)

        # For each speaker in the mixture, roll and add
        babble_waveform = waveforms.roll((1,), dims=0)
        babble_len = clean_len.roll((1,), dims=0)
        for i in range(1, self.speaker_count):
            babble_waveform += waveforms.roll((1+i,), dims=0)
            babble_len = torch.max(babble_len, babble_len.roll((1,), dims=0))

        # Rescale and add to mixture
        babble_amplitude = compute_amplitude(babble_waveform, babble_len)
        babble_waveform *= new_noise_amplitude / babble_amplitude
        babbled_waveform += babble_waveform

        return babbled_waveform


class DropFreq(torch.nn.Module):
    """This class drops a random frequency from the signal.

    The purpose of this class is to teach models to learn to rely on all parts
    of the signal, not just a few frequency bands.

    Attributes:
        drop_freq_low (float): The low end of frequencies that can be dropped,
            as a fraction of the sampling rate / 2.
        drop_freq_high (float): The high end of frequencies that can be
            dropped, as a fraction of the sampling rate / 2.
        drop_count_low (int): The low end of number of frequencies that could
            be dropped.
        drop_count_high (int): The high end of number of frequencies that could
            be dropped.
        drop_width (float): The width of the frequency band to drop, as
            a fraction of the sampling_rate / 2.
        drop_prob (float): The probability that the batch of signals will
            have a frequency dropped. By default, every batch
            has frequencies dropped.

    Example:
        >>> import torch
        >>> import soundfile as sf
        >>> from speechbrain.data_io.data_io import save
        >>> dropper = DropFreq()
        >>> signal, rate = sf.read('samples/audio_samples/example1.wav')
        >>> signal = torch.tensor(signal, dtype=torch.float32)
        >>> dropped_signal = dropper(signal.unsqueeze(0))
        >>> save_signal = save(save_folder='exp/example', save_format='wav')
        >>> save_signal(dropped_signal, ['freq_drop'], torch.ones(1))

    Author:
        Peter Plantinga 2020
    """

    def __init__(
        self,
        drop_freq_low=0,
        drop_freq_high=1,
        drop_count_low=1,
        drop_count_high=2,
        drop_width=0.05,
        drop_prob=1,
    ):
        """"""
        super().__init__()
        self.drop_freq_low = drop_freq_low
        self.drop_freq_high = drop_freq_high
        self.drop_count_low = drop_count_low
        self.drop_count_high = drop_count_high
        self.drop_width = drop_width
        self.drop_prob = drop_prob

    def forward(self, waveforms):
        """
        Parameters:
            waveforms (tensor): Shape should be `[batch, time_steps]`
                or`[batch, channels, time_steps]`.

        Returns:
            Tensor of shape `[batch, time_steps]` or
                `[batch, channels, time_steps]`
        """
        # Don't drop (return early) 1-`drop_prob` portion of the batches
        dropped_waveform = waveforms.clone()
        if torch.rand(1) > self.drop_prob:
            return dropped_waveform

        # Add channels dimension
        if len(waveforms.shape) == 2:
            dropped_waveform = dropped_waveform.unsqueeze(1)

        # Pick number of frequencies to drop
        drop_count = torch.randint(
            low=self.drop_count_low,
            high=self.drop_count_high + 1,
            size=(1,),
        )

        # Pick a frequency to drop
        drop_range = self.drop_freq_high - self.drop_freq_low
        drop_frequency = torch.rand(drop_count)*drop_range + self.drop_freq_low

        # Filter parameters
        filter_length = 101
        pad = filter_length // 2

        # Start with delta function
        drop_filter = torch.zeros(1, 1, filter_length)
        drop_filter[0, 0, pad] = 1

        # Subtract each frequency
        for frequency in drop_frequency:
            notch_kernel = notch_filter(
                frequency, filter_length, self.drop_width,
            ).to(waveforms.device)
            drop_filter = convolve1d(drop_filter, notch_kernel, pad)

        # Apply filter
        dropped_waveform = convolve1d(dropped_waveform, drop_filter, pad)

        # Remove channels dimension if added
        return dropped_waveform.squeeze(1)


class DropChunk(torch.nn.Module):
    """This class drops portions of the input signal.

    Using `DropChunk` as an augmentation strategy helps a models learn to rely
    on all parts of the signal, since it can't expect a given part to be
    present.

    Attributes:
        drop_length_low (int): The low end of lengths for which to set the
            signal to zero, in samples.
        drop_length_high (int): The high end of lengths for which to set the
            signal to zero, in samples.
        drop_count_low (int): The low end of number of times that the signal
            can be dropped to zero.
        drop_count_high (int): The high end of number of times that the signal
            can be dropped to zero.
        drop_start (int): The first index for which dropping will be allowed.
        drop_end (int): The last index for which dropping will be allowed.
        drop_prob (float): The probability that the batch of signals will
            have a portion dropped. By default, every batch
            has portions dropped.

    Example:
        >>> import torch
        >>> import soundfile as sf
        >>> from speechbrain.data_io.data_io import save
        >>> dropper = DropChunk()
        >>> signal, rate = sf.read('samples/audio_samples/example1.wav')
        >>> signal = torch.tensor(signal).unsqueeze(0)
        >>> length = torch.ones(1)
        >>> dropped_signal = dropper(signal, length)
        >>> save_signal = save(save_folder='exp/example', save_format='wav')
        >>> save_signal(dropped_signal, ['drop_chunk'], length)

    Author:
        Peter Plantinga 2020
    """
    def __init__(
        self,
        drop_length_low=100,
        drop_length_high=1000,
        drop_count_low=1,
        drop_count_high=10,
        drop_start=0,
        drop_end=None,
        drop_prob=1,
    ):
        """"""
        super().__init__()
        self.drop_length_low = drop_length_low
        self.drop_length_high = drop_length_high
        self.drop_count_low = drop_count_low
        self.drop_count_high = drop_count_high
        self.drop_start = drop_start
        self.drop_end = drop_end
        self.drop_prob = drop_prob

    def forward(self, waveforms, lengths):
        """
        Parameters:
            waveforms (tensor): Shape should be `[batch, time_steps]`
                or`[batch, channels, time_steps]`.
            lengths (tensor): Shape should be a single dimension, `[batch]`.

        Returns:
            Tensor of shape `[batch, time_steps]` or
                `[batch, channels, time_steps]`
        """
        # Reading input list
        lengths = (lengths * waveforms.size(-1)).long()
        batch_size = waveforms.size(0)
        dropped_waveform = waveforms.clone()

        # Don't drop (return early) 1-`drop_prob` portion of the batches
        if torch.rand(1) > self.drop_prob:
            return dropped_waveform

        # Pick a number of times to drop
        drop_times = torch.randint(
            low=self.drop_count_low,
            high=self.drop_count_high + 1,
            size=(batch_size,),
        )

        # Iterate batch to set mask
        for i in range(batch_size):

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
            start_max -= length.max()

            # Pick starting locations
            start = torch.randint(
                low=start_min,
                high=start_max + 1,
                size=(drop_times[i],),
            )

            # Update waveform
            for j in range(drop_times[i]):
                dropped_waveform[i, ..., start[j]:start[j]+length[j]] = 0

        return dropped_waveform


class DoClip(torch.nn.Module):
    """This function mimics audio clipping by clamping the input tensor.

    Attributes:
        clip_low (float): The low end of amplitudes for which to clip
            the signal.
        clip_high (float): The high end of amplitudes for which to clip
            the signal.
        clip_prob (float): The probability that the batch of signals will
            have a portion clipped. By default, every batch
            has portions clipped.

     Example:
        >>> import torch
        >>> import soundfile as sf
        >>> from speechbrain.data_io.data_io import save
        >>> clipper = DoClip()
        >>> signal, rate = sf.read('samples/audio_samples/example1.wav')
        >>> clipped_signal = clipper(torch.tensor(signal).unsqueeze(0))
        >>> save_signal = save(save_folder='exp/example', save_format='wav')
        >>> save_signal(clipped_signal, ['clip'], torch.ones(1))

    Author:
        Peter Plantinga
    """

    def __init__(
        self,
        clip_low=0.5,
        clip_high=1,
        clip_prob=1,
    ):
        """"""
        super().__init__()
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.clip_prob = clip_prob

    def forward(self, waveforms):
        """
        Parameters:
            waveforms (tensor): Shape should be `[batch, time_steps]`
                or`[batch, channels, time_steps]`.

        Returns:
            Tensor of shape `[batch, time_steps]` or
                `[batch, channels, time_steps]`
        """
        # Don't clip (return early) 1-`clip_prob` portion of the batches
        if torch.rand(1) > self.clip_prob:
            return waveforms.clone()

        # Randomly select clip value
        clipping_range = self.clip_high - self.clip_low
        clip_value = torch.rand(1,)[0] * clipping_range + self.clip_low

        # Apply clipping
        clipped_waveform = waveforms.clamp(-clip_value, clip_value)

        return clipped_waveform
