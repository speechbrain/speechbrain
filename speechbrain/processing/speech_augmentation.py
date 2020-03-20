"""
-----------------------------------------------------------------------------
 speechbrain.processing.speech_augmentation.py

 Description: This library gathers functions that mutate batches of data
              in order to improve training of machine learning models.
              All the classes are of type nn.Module. This gives the
              possibility to have end-to-end differentiability and
              backpropagate the gradient through them.
 -----------------------------------------------------------------------------
"""

# Importing libraries
import math
import torch
import torch.nn as nn

from speechbrain.data_io.data_io import create_dataloader
from speechbrain.utils.logger import logger_write

from speechbrain.module import SpeechBrainModule

"""
-------------------------------------------
Define utility functions for augmentations
-------------------------------------------
"""


def compute_amplitude(waveform, length):
    """
    ------------------------------------------------------
    speechbrain.processing.speech_augmentation.compute_amplitude
    (author: Peter Plantinga)

    Description: Compute the average amplitude of a batch of waveforms for
                 scaling different waveforms to a given ratio.

    Input:       - waveform (type: torch.tensor, mandatory):
                     The waveform used for computing amplitude

                 - length (type: torch.tensor, mandatory):
                     The length of the (un-padded) waveform

    Output:      - average amplitude (type: torch.tensor):
                     The average amplitude of each waveform

    Example:     import torch
                 import soundfile as sf
                 from speechbrain.processing.speech_augmentation \
                     import compute_amplitude

                 signal, rate = sf.read('samples/audio_samples/example1.wav')
                 signal = torch.tensor(signal, dtype=torch.float32)

                 amplitude = compute_amplitude(signal, len(signal))

                 print('Average amplitude is: %f' % amplitude)
    ------------------------------------------------------
    """
    return torch.sum(
        input=torch.abs(waveform),
        dim=-1,
        keepdim=True,
    ) / length


def convolve1d(waveform, kernel, padding=0, pad_type='constant',
               stride=1, groups=1, use_fft=False, rotation_index=0):
    """
    ----------------------------------------------------
    speechbrain.processing.speech_augmentation.convolve1d
    (author: Peter Plantinga)

    Description: Use torch.nn.functional to perform first 1d padding
                 and then 1d convolution.

    Input:       - waveform (type: torch.tensor, mandatory):
                     The tensor to perform operations on.

                 - kernel (type: torch.tensor, mandatory):
                     The filter to apply during convolution

                 - padding (type: tuple, default: None):
                     The padding (pad_left, pad_right) to apply.
                     If an integer is passed instead, this is passed
                     to the conv1d function and pad_type is ignored.

                 - pad_type (type: string, default: 'constant'):
                     The type of padding to use. Passed directly to
                     `torch.nn.functional.pad`, see PyTorch documentation
                     for available options.

                 - stride (type: int, default: 1):
                     The number of units to move each time convolution
                     is applied. Passed to conv1d. Has no effect if
                     `use_fft` is True.

                 - groups (type: int, default: 1):
                     This option is passed to `conv1d` to split the input
                     into groups for convolution. Input channels should
                     be divisible by number of groups.

                 - use_fft (type: bool, default: False):
                     When `use_fft` is passed `True`, then compute the
                     convolution in the spectral domain using complex
                     multiply. This is more efficient on CPU when the
                     size of the kernel is large (e.g. reverberation).
                     WARNING: Without padding, circular convolution occurs.
                     This makes little difference in the case of reverberation,
                     but may make more difference with different kernels.

                 - rotation_index (type: int, default: 0):
                     This option only applies if `use_fft` is true. If so,
                     the kernel is rolled by this amount before convolution
                     to shift the output location.

    Output:      - convolved waveform (type: torch.tensor)

    Example:     import torch
                 import soundfile as sf
                 from speechbrain.data_io.data_io import save
                 from speechbrain.processing.speech_augmentation \
                     import convolve1d

                 signal, rate = sf.read('samples/audio_samples/example1.wav')
                 signal = torch.tensor(signal[None, None, :])
                 filter = torch.rand(1, 1, 10, dtype=signal.dtype)
                 signal = convolve1d(signal, filter, padding=(9, 0))

                 # save config dictionary definition
                 config = {
                    'class_name':'speechbrain.data_io.data_io.save',
                    'save_folder': 'exp/write_example',
                    'save_format': 'wav',
                 }

                 # class initialization
                 save_signal = save(config)

                 # saving
                 save_signal([signal, ['example_conv'], torch.ones(1)])

                 # signal save in exp/write_example
    ----------------------------------------------------
    """

    if isinstance(padding, tuple):
        waveform = nn.functional.pad(
            input=waveform,
            pad=padding,
            mode=pad_type,
        )

    # This approach uses FFT, which is more efficient if the kernel is large
    if use_fft:

        # Pad kernel to same length as signal, ensuring correct alignment
        zero_length = waveform.size(-1) - kernel.size(-1)

        # Handle case where signal is shorter
        if zero_length < 0:
            kernel = kernel[..., :zero_length]
            zero_length = 0

        # Perform rotation to ensure alignment
        zeros = torch.zeros(kernel.size(0), kernel.size(1), zero_length)
        after_index = kernel[..., rotation_index:]
        before_index = kernel[..., :rotation_index]
        kernel = torch.cat((after_index, zeros, before_index), dim=-1)

        # Compute FFT for both signals
        f_signal = torch.rfft(waveform, 1)
        f_kernel = torch.rfft(kernel, 1)

        # Complex multiply
        sig_real, sig_imag = f_signal.unbind(-1)
        ker_real, ker_imag = f_kernel.unbind(-1)
        f_result = torch.stack([
            sig_real*ker_real - sig_imag*ker_imag,
            sig_real*ker_imag + sig_imag*ker_real,
        ], dim=-1)

        # Inverse FFT
        return torch.irfft(f_result, 1)

    # Use the implemenation given by torch, which should be efficient on GPU
    return nn.functional.conv1d(
        input=waveform,
        weight=kernel,
        stride=stride,
        groups=groups,
        padding=padding if not isinstance(padding, tuple) else 0,
    )


def dB_to_amplitude(a):
    """
    --------------------------------------------------
    speechbrain.processing.speech_augmentation.dB_to_amplitude
    (author: Peter Plantinga)

    Description: Convert decibels to amplitude

    Inputs: - a (float, required):
                The ratio in decibels to convert

    Output: ratio between average amplitudes (float)

    Example: from speechbrain.processing.speech_augmentation \
                 import dB_to_amplitude

             SNR = 10
             amplitude = dB_to_amplitude(SNR) # Results in 3.16
    --------------------------------------------------
    """
    return 10 ** (a / 20)


def notch_filter(notch_freq, filter_width=101, notch_width=0.05):
    """
    ---------------------------------------------------------
    speechbrain.processing.speech_augmentation.notch_filter
    (from https://tomroelandts.com/articles/
    how-to-create-simple-band-pass-and-band-reject-filters)

    Description: Simple band-pass filter with small passband. Convolve
                 with an impulse response to get the notch filter.

    Inputs: - notch_freq (type: float, mandatory):
                frequency to put notch as a fraction of the sampling rate / 2.
                The range of possible inputs is 0 to 1.

            - filter_width (type: int, optional):
                Filter width in samples. Longer filters have smaller
                transition bands, but are more inefficient

            - notch_width (type: float, optional):
                Width of the notch, as a fraction of the sampling_rate / 2.

    Output: notch filter (type: torch.tensor)

    Example: import torch
             import soundfile as sf
             from speechbrain.data_io.data_io import save
             from speechbrain.processing.speech_augmentation \
                 import notch_filter

             signal, rate = sf.read('samples/audio_samples/example1.wav')
             signal = torch.tensor(signal, dtype=torch.float32)[None, None, :]

             kernel = notch_filter(0.25)
             notched_signal = torch.nn.functional.conv1d(signal, kernel)

             # save config dictionary definition
             config = {
                'class_name': 'speechbrain.data_io.data_io.save',
                'save_folder': 'exp/write_example',
                'save_format': 'wav',
             }

             # class initialization
             save_signal = save(config)

             # saving
             save_signal([notched_signal, ['freq_drop'], torch.ones(1)])

             # signal save in exp/write_example
    -----------------------------------------------------------
    """

    # Check inputs
    assert 0 < notch_freq <= 1
    assert filter_width % 2 != 0
    pad = filter_width // 2
    inputs = torch.arange(filter_width) - pad

    # Avoid frequencies that are too low
    notch_freq += notch_width

    # Define sinc function, avoiding division by zero
    def sinc(x):
        def _sinc(x):
            return torch.sin(x) / x

        # The zero is at the middle index
        return torch.cat([_sinc(x[:pad]), torch.ones(1), _sinc(x[pad+1:])])

    # Compute a low-pass filter with cutoff frequency notch_freq.
    hlpf = sinc(3 * (notch_freq - notch_width) * inputs)
    hlpf *= torch.blackman_window(filter_width)
    hlpf /= torch.sum(hlpf)

    # Compute a high-pass filter with cutoff frequency notch_freq.
    hhpf = sinc(3 * (notch_freq + notch_width) * inputs)
    hhpf *= torch.blackman_window(filter_width)
    hhpf /= -torch.sum(hhpf)
    hhpf[pad] += 1

    # Adding filters creates notch filter
    return (hlpf + hhpf).view(1, 1, -1)


"""
------------------------------------------------------
Augmentation classes
------------------------------------------------------
"""


class add_noise(SpeechBrainModule):
    """
    -------------------------------------------------------------------------
    speechbrain.processing.speech_augmentation.add_noise
    (author: Peter Plantinga)

    Description: This class additively combines a noise signal to the input
                 signal. The noise can come from a provided file or
                 from a generated white noise signal.

    Input: - csv_file (type: str, default: None):
               The csv file containing the location of the noise audio files.
               If none is provided, white noise will be used instead.

           - order (type: str, default: 'random'):
               The order to iterate the csv file, from one of the following
               options: random, original, ascending, and descending.

           - batch_size (type: int, default: None):
               If an csv_file is passed, this controls the number of samples
               that are loaded at the same time, should be the same as or less
               than the size of the clean batch. If `None` is passed, the size
               of the first clean batch will be used.

           - do_cache (type: bool, default: False):
               Whether or not to store noise files in the cache.

           - snr_low (type: float, default: 0):
               The low end of the mixing ratios, in decibels.

           - snr_high (type: float, default: 0):
               The high end of the mixing ratios, in decibels.

           - pad_noise (type: bool, default: False):
               If True, copy noise signals that are shorter than their
               corresponding clean signals so as to cover the whole clean
               signal. Otherwise, leave the noise un-padded.

           - mix_prob (type: float, default: 1.0):
               The probability that a batch of signals will be mixed with a
               noise signal. By default, every batch is mixed with noise.

    Example: import torch
             import soundfile as sf
             from speechbrain.data_io.data_io import save
             from speechbrain.processing.speech_augmentation import add_noise

             # reading an audio signal
             signal, rate = sf.read('samples/audio_samples/example1.wav')

             # Initialization of the class
             noisifier = add_noise(
                'csv_file': 'samples/noise_samples/noise.csv',
                'batch_size': 1,
             )

             # Executing computations
             clean = torch.tensor([signal], dtype=torch.float32)
             clean_len = torch.ones(1)
             noisy = noisifier(clean, clean_len)

             # class initialization
             save_signal = save(
                'save_folder': 'exp/write_example',
                'save_format': 'wav',
             )

             # saving
             save_signal(noisy, ['example_add_noise'], clean_len)

             # signal save in exp/write_example
    -------------------------------------------------------------------------
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
        **kwargs
    ):

        expected_inputs = [
            {'type': 'torch.Tensor', 'dim_count': [2, 3]},
            {'type': 'torch.Tensor', 'dim_count': [1]},
        ]

        self.csv_file = csv_file
        self.order = order
        self.batch_size = batch_size
        self.do_cache = do_cache
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.pad_noise = pad_noise
        self.mix_prob = mix_prob

        # On first input, create dataloader with correct batch size
        def hook(self, input):

            clean_waveform, clean_length = input

            # Set parameters based on input
            self.device = clean_waveform.device
            if not self.batch_size:
                self.batch_size = len(clean_waveform)

            # Create a data loader for the noise wavforms
            if self.csv_file is not None:
                self.data_loader = create_dataloader(
                    csv_file=self.csv_file,
                    sentence_sorting=self.order,
                    batch_size=self.batch_size,
                    cache=self.do_cache,
                    global_config=self.global_config,
                )
                self.noise_data = zip(*self.data_loader())

        super().__init__(expected_inputs, hook, **kwargs)

    def forward(self, clean_waveform, clean_length):
        """
        Input: - clean_waveform (type: torch.tensor, mandatory):
                   A batch of audio signals, in one of the following formats:
                   [batch, time_steps], or [batch, channels, time_steps]

               - clean_length (type: torch.tensor, mandatory):
                   The lengths of the audio signals contained in the batch.
                   Must be in the format [batch].

        Output: A batch of audio signals of the same shape as the input,
                with noise added to each signal.
        """

        # Copy clean waveform to initialize noisy waveform
        noisy_waveform = clean_waveform.clone()
        clean_length = (clean_length * clean_waveform.shape[1]).unsqueeze(1)

        # current batch size is min of stored size and clean size
        batch_size = self.batch_size
        if batch_size is None or batch_size > len(clean_waveform):
            batch_size = len(clean_waveform)

        # Don't add noise (return early) 1-`mix_prob` portion of the batches
        if torch.rand(1) > self.mix_prob:
            self.rng_state = torch.random.get_rng_state()
            return noisy_waveform

        # Compute the average amplitude of the clean waveforms
        clean_amplitude = compute_amplitude(clean_waveform, clean_length)

        # Pick an SNR and use it to compute the mixture amplitude factors
        SNR = torch.rand(batch_size, 1, device=clean_waveform.device)
        SNR = SNR * (self.snr_high - self.snr_low) + self.snr_low
        noise_amplitude_factor = 1 / (dB_to_amplitude(SNR) + 1)
        new_noise_amplitude = noise_amplitude_factor * clean_amplitude

        # Scale clean signal appropriately
        noisy_waveform[:batch_size] *= (1 - noise_amplitude_factor)

        # Loop through clean samples and create mixture
        if self.csv_file is None:
            white_noise = torch.randn_like(clean_waveform)
            noisy_waveform += new_noise_amplitude * white_noise
        else:
            tensor_length = clean_waveform.shape[-1]
            noise_waveform, noise_length = self._load_noise(
                clean_length, tensor_length, batch_size,
            )

            # Rescale and add
            noise_amplitude = compute_amplitude(noise_waveform, noise_length)
            noise_waveform *= new_noise_amplitude / noise_amplitude
            noisy_waveform[:batch_size] += noise_waveform

        return noisy_waveform

    def _load_noise(self, clean_len, tensor_len, batch_size):
        """
        -----------------------------------------------------
        speechbrain.processing.speech_augmentation.add_noise._load_noise
        (author: Peter Plantinga)

        Description: Load a section of the noise file of the appropriate
                     length. Then pad to the length of the tensor.

        Input:   - self (type: add_noise, mandatory)

                 - clean_len (type: torch.tensor, mandatory):
                     The length of the (un-padded) clean waveform

                 - tensor_len (type: torch.tensor, mandatory):
                     The length of the (padded) final tensor

        Output:  - noise segment (type: torch.tensor)
        --------------------------------------------------------
        """
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
            noise_batch = nn.functional.pad(noise_batch, padding)

        # Select a random starting location in the waveform
        start_index = 0
        max_chop = (wav_len - clean_len).min().clamp(min=1)
        start_index = torch.randint(high=max_chop, size=(1,))

        # Truncate noise_batch to tensor_len
        noise_batch = noise_batch[..., start_index:start_index+tensor_len]
        wav_len = (wav_len - start_index).clamp(max=tensor_len).unsqueeze(1)
        return noise_batch, wav_len


class add_reverb(SpeechBrainModule):
    """
    -------------------------------------------------------------------------
    speechbrain.processing.speech_augmentation.add_reverb
    (author: Peter Plantinga)

    Description: This class convolves the audio signal with an impulse
                 response. The impulse response must be provided in
                 an csv file.

    Input: - csv_file (type: str, mandatory):
               The csv file containing the location of the
               impulse response files.

           - order (type: str, default: 'random'):
               The order to iterate the csv file, from one of
               the following options: random, original, ascending,
               and descending.

           - do_cache (type: bool, default: False):
               Whether or not to lazily load the files to a
               cache and read the data from the cache.

           - reverb_prob (type: float, default: 1.0):
               The chance that the audio signal will be reverbed.
               By default, every signal is reverbed.

    Example:   import torch
               import soundfile as sf
               from speechbrain.data_io.data_io import save
               from speechbrain.processing.speech_augmentation \
                   import add_reverb

               # reading an audio signal
               signal, rate = sf.read('samples/audio_samples/example1.wav')

               # Initialization of the class
               reverb = add_reverb(csv_file='samples/rir_samples/rirs.csv')

               # Executing computations
               clean = torch.tensor([signal], dtype=torch.float32)
               clean_len = torch.ones(1)
               reverbed = reverb(clean, clean_len)

               # class initialization
               save_signal = save(
                   save_folder='exp/write_example',
                   save_format='wav',
               )

               # saving
               save_signal(reverbed, ['example_add_reverb'], clean_len)

               # signal save in exp/write_example
    -------------------------------------------------------------------------
    """

    def __init__(
        self,
        csv_file,
        order='random',
        do_cache=False,
        reverb_prob=1.,
        pad_type='zero',
        **kwargs
    ):

        self.options = {
            'csv_file': {'type': 'string', 'value': csv_file},
            'order': {
                'type': 'one_of(random,ascending,descending,original)',
                'value': order,
            },
            'do_cache': {'type': 'bool', 'value': do_cache},
            'reverb_prob': {'type': 'float(0,1)', 'value': reverb_prob},
            'random_seed': {'type': 'int(-inf, inf)', 'value': random_seed},
            'pad_type': {
                'type': 'one_of(zero,reflect,edge)',
                'value': pad_type,
            },
        }

        self.expected_inputs = [
            {'type': 'torch.Tensor', 'dim_count': [2, 3]},
            {'type': 'torch.Tensor', 'dim_count': [1]},
        ]

        super().__init__(options, expected_inputs, **kwargs)

        # Create a data loader for the RIR waveforms
        self.data_loader = create_dataloader(
            {
                'class_name': 'core.loop',
                'csv_file': self.csv_file,
                'sentence_sorting': self.order,
                'do_cache': self.do_cache,
            },
            global_config=global_config,
        )
        self.rir_data = zip(*self.data_loader.dataloader)

    def forward(self, clean_waveform, clean_lengths):
        """
        Input: - clean_waveform (type: torch.Tensor, mandatory)
                   a batch of audio signals, and the
                   tensor must be in one of the following formats:
                   [batch, time_steps], or [batch, channels, time_steps]

               - clean_lengths (type: torch.Tensor, mandatory)
                   The lengths of the audio signals contained in the tensor.
                   and must be in the format [batch].

        Output: the reverbed audio signal. The tensor is the same
                shape as the input, i.e. formatted in the following way:
                [batch, time_steps], or [batch, channels, time_steps]
        """

        # Don't add reverb (return early) 1-`reverb_prob` portion of the time
        if torch.rand(1) > self.reverb_prob:
            self.rng_state = torch.random.get_rng_state()
            return clean_waveform.clone()

        # Add channels dimension if necessary
        channel_added = False
        if len(clean_waveform.shape) == 2:
            clean_waveform = clean_waveform.unsqueeze(1)
            channel_added = True

        # Convert length from ratio to number of indices
        clean_len = (input_lst[1] * clean_waveform.shape[2])[:, None, None]

        # Compute the average amplitude of the clean
        clean_amplitude = compute_amplitude(clean_waveform, clean_len)

        # Load and prepare RIR
        rir_waveform = self._load_rir(clean_waveform).abs()

        # Compute index of the direct signal, so we can preserve alignment
        direct_index = rir_waveform.argmax(axis=-1).median()

        # Use FFT to compute convolution, because of long reverberation filter
        reverbed_waveform = convolve1d(
            waveform=clean_waveform,
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

    def _load_rir(self, clean_waveform):
        """
        ---------------------------------------------------------
        speechbrain.processing.speech_augmentation.add_reverb._load_rir
        (author: Peter Plantinga)

        Description: Internal method for loading RIR wav files.

        Input: - self (type: add_reverb, mandatory)

               - clean_wavefrom (type: torch.tensor, mandatory):
                  For collecting dtype and device info.

        Output: A single RIR waveform
        --------------------------------------------------------
        """

        try:
            wav_id, rir_waveform, length = next(self.rir_data)[0]
        except StopIteration:
            self.rir_data = zip(*self.data_loader.dataloader)
            wav_id, rir_waveform, length = next(self.rir_data)[0]

        # Make sure RIR has correct channels
        if len(rir_waveform.shape) == 2:
            rir_waveform = rir_waveform.unsqueeze(1)

        # Make sure RIR has correct type and device
        rir_waveform = rir_waveform.type(clean_waveform.dtype)
        return rir_waveform.to(clean_waveform.device)


class speed_perturb(nn.Module):
    """
    -------------------------------------------------------------------------
    speechbrain.processing.speech_augmentation.speed_perturb
    (author: Peter Plantinga)

    Description: This class resamples the original signal at a similar
                 sample rate, to achieve a slightly slower or slightly
                 faster signal. This technique is outlined in the paper:
                 "Audio Augmentation for Speech Recognition"

    Input (init):  - config (type: dict, mandatory):
                      it is a dictionary containing the keys described below.

                      - orig_freq (type: int, mandatory):
                          The frequency of the original signal.

                      - speeds (type: list, default: [9, 10, 11]):
                          The speeds that the signal should be changed to,
                          where 10 is the speed of the original signal.

                      - perturb_prob (type: float, default: 1.0):
                          The chance that the batch will be speed-perturbed.
                          By default, every batch is perturbed.

                      - random_seed (type: int, default: None):
                          The seed for randomly selecting which perturbation
                          to use. If `None` is passed, then this method will
                          cycle through the list of speeds.

                  - funct_name (type: str, default: None):
                      it is a string containing the name of the parent
                      function that has called this method.

                  - global_config (type: dict, default: None):
                      it a dictionary containing the global variables of the
                      parent config file.

                  - functions (type: dict, default: None):
                      dictionary for storing user-defined functions. Keys are
                      the function names, values are their corresponding
                      objects.

                  - logger (type: logger, default: None):
                      it the logger used to write debug and error messages.
                      If logger=None and root_cfg=True, the file is created
                      from scratch.

                  - first_input (type: list, default: None):
                      this variable allows users to analyze the first input
                      given when calling the class for the first time.

    Input (call): - inp_lst(type: list, mandatory):
                      by default the input arguments are passed with a list.
                      In this case, inp is a list containing a single
                      torch.tensor which contains an audio signal. The input
                      tensor must be in one of the following formats:
                      [batch, time_steps], or [batch, channels, time_steps]

    Output (call): - out_lst(type: list, mandatory):
                      by default the output arguments are passed with a list.
                      In this case, out is a list containing a torch.tensor
                      with the reverbed audio signal. The tensor is the same
                      shape as the input, i.e. formatted in the following way:
                      [batch, time_steps], or [batch, channels, time_steps]

    Example:   import torch
               import soundfile as sf
               from speechbrain.data_io.data_io import save
               from speechbrain.processing.speech_augmentation \
                   import speed_perturb

               # reading an audio signal
               signal, rate = sf.read('samples/audio_samples/example1.wav')

               # config dictionary definition
               config = {
                   'class_name':
                       'speechbrain.processing.speech_augmentation.speed_perturb',
                   'orig_freq': str(rate),
                   'speeds': '9,11',
               }

               # Initialization of the class
               perturbator = speed_perturb(config)

               # Executing computations
               clean = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
               perturbed = perturbator([clean])

               # save config dictionary definition
               config = {
                  'class_name': 'speechbrain.data_io.data_io.save',
                  'save_folder': 'exp/write_example',
                  'save_format': 'wav',
               }

               # class initialization
               save_signal = save(config)

               # saving
               save_signal([perturbed, ['example_perturb'], torch.ones(1)])

               # signal save in exp/write_example
    -------------------------------------------------------------------------
    """

    def __init__(
        self,
        config,
        funct_name=None,
        global_config=None,
        functions=None,
        logger=None,
        first_input=None,
    ):
        super(speed_perturb, self).__init__()

        # Logger setup
        self.logger = logger

        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "orig_freq": ("int", "mandatory"),
            "speeds": ("int_lst", "optional", "9,10,11"),
            "perturb_prob": ("float(0,1)", "optional", "1"),
            "random_seed": ("int(-inf, inf)", "optional", "None"),
        }

        # Check, cast, and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, self.logger
        )

        # Expected inputs when calling the class
        self.expected_inputs = ["torch.Tensor"]

        # Check the first input
        check_inputs(
            self.conf, self.expected_inputs, first_input, logger=self.logger
        )

        # Initialize a random number generator with the provided seed
        if self.random_seed is not None:
            torch.random.manual_seed(self.random_seed)

        # Initialize index of perturbation
        self.samp_index = 0

        # Initialize resamplers
        self.resamplers = []
        for speed in self.speeds:

            config = {
                'class_name':
                    'speechbrain.processing.speech_augmentation.resample',
                'orig_freq': str(self.orig_freq),
                'new_freq': str(self.orig_freq * speed // 10),
            }
            self.resamplers.append(resample(config))

        # Additional check on the input shapes
        if first_input is not None:
            check_input_shapes([[2, 3]], first_input, logger)

            # Use the same devices as the input vector
            self.device = str(first_input[0].device)

        # Save the state of the RNG for reproducibility
        self.rng_state = torch.random.get_rng_state()

    def forward(self, input_lst):

        # Reset the RNG state for reproducibility
        torch.random.set_rng_state(self.rng_state)

        # Reading input _list and add channels dimension
        clean_waveform = input_lst[0].unsqueeze(1)

        # Don't perturb (return early) 1-`perturb_prob` portion of the batches
        if torch.rand(1) > self.perturb_prob:
            self.rng_state = torch.random.get_rng_state()
            return clean_waveform.clone()

        # Select a perturbation to apply
        if self.random_seed is not None:
            self.samp_index = torch.randint(len(self.speeds), (1,))[0]
        else:
            self.samp_index = (self.samp_index + 1) % len(self.speeds)

        # Perform the selected perturbation
        perturbed_waveform = self.resamplers[self.samp_index]([clean_waveform])

        # Save the RNG state for reproducibility
        self.rng_state = torch.random.get_rng_state()

        return perturbed_waveform.squeeze(1)


class resample(nn.Module):
    """
    -------------------------------------------------------------------------
    speechbrain.processing.speech_augmentation.resample
    (author: Peter Plantinga)

    Description: This class resamples an audio signal using sinc-based
                 interpolation. It is a modification of the `resample`
                 function from torchaudio (https://pytorch.org/audio/
                 transforms.html#resample)

    Input (init):  - config (type: dict, mandatory):
                      a dictionary containing the keys described below.

                      - device ('cuda', 'cpu', default: None):
                          the device where to compute the resample. If None,
                          it uses the device of the input signal.

                      - orig_freq (type: int, default: 16000):
                          the sampling frequency of the input signal.

                      - new_freq (type: int, default: 16000):
                          the new sampling frequency after this operation
                          is performed.

                      - lowpass_filter_width (type: int, optional,
                              default: 6):
                          Controls the sharpness of the filter, larger
                          numbers result in a sharper filter, but they are
                          less efficient. Values from 4 to 10 are allowed.

                  - funct_name (type: str, default: None):
                      it is a string containing the name of the parent
                      function that has called this method.

                  - global_config (type: dict, default: None):
                      it a dictionary containing the global variables of the
                      parent config file.

                  - functions (type: dict, default: None):
                      dictionary for storing user-defined functions. Keys are
                      the function names, values are their corresponding
                      objects.

                  - logger (type: logger, default: None):
                      it the logger used to write debug and error messages.
                      If logger=None and root_cfg=True, the file is created
                      from scratch.

                  - first_input (type: list, default: None):
                      this variable allows users to analyze the first input
                      given when calling the class for the first time.

    Input (call): - inp_lst(type: list, mandatory):
                      by default the input arguments are passed with a list.
                      In this case, inp is a list containing a single
                      torch.tensor which contains an audio signal. The input
                      tensor must be in one of the following formats:
                      [batch, time_steps], or [batch, channels, time_steps]

    Output (call): - out_lst(type: list, mandatory):
                      by default the output arguments are passed with a list.
                      In this case, out is a list containing a torch.tensor
                      with the resampled audio signal. The tensor is
                      formatted in one of the following ways based on input:
                      [batch, time_steps * new_freq // orig_freq], or
                      [batch, channels, time_steps * new_freq // orig_freq]

    Example:   import torch
               import soundfile as sf
               from speechbrain.data_io.data_io import save
               from speechbrain.processing.speech_augmentation import resample

               # reading an audio signal
               signal, rate = sf.read('samples/audio_samples/example1.wav')
               signal = torch.tensor(signal, dtype=torch.float32)[None,None,:]

               # config dictionary definition
               config = {
                   'class_name':
                       'speechbrain.processing.speech_augmentation.resample',
                   'orig_freq': str(rate),
                   'new_freq': str(rate // 2),
               }

               # Initialization of the class
               resampler = resample(config)

               # Executing computations
               resampled = resampler(signal)

               # Save signal config
               config = {
                   'class_name': 'speechbrain.data_io.data_io.save',
                   'save_folder': 'exp/write_example',
                   'save_format': 'wav',
                   'sampling_rate': str(rate // 2),
               }

               # class initialization
               save_signal = save(config)

               # saving
               save_signal([resampled, ["example_resamp"], torch.ones(1)])

               # signal save in exp/write_example
    -------------------------------------------------------------------------
    """

    def __init__(
        self,
        config,
        funct_name=None,
        global_config=None,
        functions=None,
        logger=None,
        first_input=None,
    ):
        super(resample, self).__init__()

        # Logger setup
        self.logger = logger

        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "device": ("one_of(cpu,cuda)", "optional", "None"),
            "orig_freq": ("int(8000,48000)", "optional", "16000"),
            "new_freq": ("int(8000,48000)", "optional", "16000"),
            "lowpass_filter_width": ("int(4,10)", "optional", "6"),
        }

        # Check, cast, and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, self.logger
        )

        # Expected inputs when calling the class
        self.expected_inputs = ["torch.Tensor"]

        # Check the first input
        check_inputs(
            self.conf, self.expected_inputs, first_input, logger=self.logger
        )

        # Additional check on the input shapes
        if first_input is not None:
            check_input_shapes([[2, 3]], first_input, logger)

            # If device is not specified use the same as the input vector
            if self.device is None:
                self.device = str(first_input[0].device)

        # Compute rate for striding
        self._compute_strides()
        assert self.orig_freq % self.conv_stride == 0
        assert self.new_freq % self.conv_transpose_stride == 0

        # Generate and store the filter to use for resampling
        self._indices_and_weights()
        assert self.first_indices.dim() == 1

    def _compute_strides(self):
        """
        ---------------------------------------------------------------------
        speechbrain.processing.speech_augmentation.resample._compute_strides
        (almost directly from torchaudio.compliance.kaldi)

        Description: Compute the phases in polyphase filter

        Input:   self (type: resample, mandatory)

        Output:  None

        Example: import torch
                 import soundfile as sf
                 from speechbrain.processing.speech_augmentation \
                     import resample

                 # reading an audio signal
                 signal, rate = sf.read('samples/audio_samples/example1.wav')
                 signal = torch.tensor(signal, dtype=torch.float32)
                 signal = signal[None, None, :]

                 # config dictionary definition
                 config = {
                     'class_name':
                        'speechbrain.processing.speech_augmentation.resample',
                     'orig_freq': str(rate),
                     'new_freq': str(rate // 2),
                 }

                 # Initialization of the class
                 resampler = resample(config)

                 # Change frequency
                 resampler.new_freq = rate * 2
                 resampler._compute_strides()
                 resampler._indices_and_weights()

                 # Executing computations
                 resampled = resampler(signal)
        ---------------------------------------------------------------------
        """
        # Compute new unit based on ratio of in/out frequencies
        base_freq = math.gcd(self.orig_freq, self.new_freq)
        input_samples_in_unit = self.orig_freq // base_freq
        self.output_samples = self.new_freq // base_freq

        # Store the appropriate stride based on the new units
        self.conv_stride = input_samples_in_unit
        self.conv_transpose_stride = self.output_samples

    def forward(self, input_lst):

        # Reading input _list
        waveform = input_lst[0]
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
        """
        ---------------------------------------------------------------
        speechbrain.processing.speech_augmentation.resample._perform_resample
        (almost directly from torchaudio.compliance.kaldi)

        Description: Resamples the waveform at the new frequency. This matches
                     Kaldi's OfflineFeatureTpl ResampleWaveform which uses a
                     LinearResample (resample a signal at linearly spaced
                     intervals to up/downsample a signal). LinearResample (LR)
                     means that the output signal is at linearly spaced
                     intervals (i.e the output signal has a frequency of
                     ``new_freq``). It uses sinc/bandlimited interpolation to
                     upsample/downsample the signal.

                     https://ccrma.stanford.edu/~jos/resample/
                     Theory_Ideal_Bandlimited_Interpolation.html

                     https://github.com/kaldi-asr/kaldi/blob/master/src/feat/
                     resample.h#L56

        Inputs: waveform (type: torch.tensor, mandatory)

        Output: The waveform at the new frequency (type: torch.tensor)

        Example: import torch
                 import soundfile as sf
                 from speechbrain.processing.speech_augmentation \
                     import resample

                 # reading an audio signal
                 signal, rate = sf.read('samples/audio_samples/example1.wav')
                 signal = torch.tensor(signal, dtype=torch.float32)
                 signal = signal[None, None, :]

                 # config dictionary definition
                 config = {
                     'class_name':
                        'speechbrain.processing.speech_augmentation.resample',
                     'orig_freq': str(rate),
                     'new_freq': str(rate // 2),
                 }

                 # Initialization of the class
                 resampler = resample(config)

                 # Executing computations
                 resampled = resampler._perform_resample(signal)
        -----------------------------------------------------------------
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
        """
        ---------------------------------------------------------------------
        speechbrain.processing.speech_augmentation.resample._output_samples
        (almost directly from torchaudio.compliance.kaldi)

        Description: Based on LinearResample::GetNumOutputSamples.
                     LinearResample (LR) means that the output signal is at
                     linearly spaced intervals (i.e the output signal has a
                     frequency of ``new_freq``). It uses sinc/bandlimited
                     interpolation to upsample/downsample the signal.

        Input:       - self (type: resample, mandatory)

                     - input_num_samp (type: torch.tensor, mandatory):
                         The number of samples in each example in the batch

        Output:      Number of samples in the output waveform (type: int)

        Example: import torch
                 import soundfile as sf
                 from speechbrain.processing.speech_augmentation \
                     import resample

                 # reading an audio signal
                 signal, rate = sf.read('samples/audio_samples/example1.wav')
                 signal = torch.tensor(signal, dtype=torch.float32)
                 signal = signal[None, None, :]

                 # config dictionary definition
                 config = {
                     'class_name':
                        'speechbrain.processing.speech_augmentation.resample',
                     'orig_freq': str(rate),
                     'new_freq': str(rate // 2),
                 }

                 # Initialization of the class
                 resampler = resample(config)

                 # Executing computations
                 resampled = resampler(signal)

                 length = signal.size(-1)
                 num_samples = resampler._output_samples(length)

                 assert resampled.size(-1) == num_samples
                 assert num_samples - num_samples % 2 == length // 2
        ---------------------------------------------------------------------
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
        """
        ---------------------------------------------------------------------
        speechbrain.processing.speech_augmentation.resample._indices_and_weights
        (almost directly from torchaudio.compliance.kaldi)

        Description: Based on LinearResample::SetIndexesAndWeights where it
                     retrieves the weights for resampling as well as the
                     indices in which they are valid. LinearResample (LR)
                     means that the output signal is at linearly spaced
                     intervals (i.e the output signal has a frequency of
                     ``new_freq``). It uses sinc/bandlimited interpolation to
                     upsample/downsample the signal.

        Input:       - self (type: resample, mandatory)

        Output:      - min_index (type: torch.tensor):
                         The place where each filter should start being applied

                     - filter weights (type: torch.tensor):
                         The filter to be applied to the signal for resampling

        Example: import torch
                 import soundfile as sf
                 from speechbrain.processing.speech_augmentation \
                     import resample

                 # reading an audio signal
                 signal, rate = sf.read('samples/audio_samples/example1.wav')
                 signal = torch.tensor(signal, dtype=torch.float32)
                 signal = signal[None, None, :]

                 # config dictionary definition
                 config = {
                     'class_name':
                        'speechbrain.processing.speech_augmentation.resample',
                     'orig_freq': str(rate),
                     'new_freq': str(rate // 2),
                 }

                 # Initialization of the class
                 resampler = resample(config)

                 # Change frequency
                 resampler.new_freq = rate * 2
                 resampler._compute_strides()
                 resampler._indices_and_weights()

                 # Executing computations
                 resampled = resampler(signal)
        ---------------------------------------------------------------------
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


class add_babble(nn.Module):
    """
    -------------------------------------------------------------------------
    speechbrain.processing.speech_augmentation.add_babble
    (author: Peter Plantinga)

    Description: This class additively combines a signal with other signals
                 from the batch, to simulate babble noise.

    Input (init):  - config (type: dict, mandatory):
                      it is a dictionary containing the keys described below.

                      - speaker_count (type: int, default: 3):
                          The number of signals to mix with the original
                          signal.

                      - snr_low (type: float, default: 0):
                          The low end of the mixing ratios, in decibels.

                      - snr_high (type: float, default: 0):
                          The high end of the mixing ratios, in decibels.

                      - mix_prob (type: float, default: 1.0):
                          The probability that the batch of signals will be
                          mixed with babble noise. By default, every signal
                          is mixed.

                      - random_seed (type: int, default: None):
                          The seed for randomly selecting SNR level.

                  - funct_name (type: str, default: None):
                      it is a string containing the name of the parent
                      function that has called this method.

                  - global_config (type: dict, default: None):
                      it a dictionary containing the global variables of the
                      parent config file.

                  - functions (type: dict, default: None):
                      dictionary for storing user-defined functions. Keys are
                      the function names, values are their corresponding
                      objects.

                  - logger (type: logger, default: None):
                      it the logger used to write debug and error messages.
                      If logger=None and root_cfg=True, the file is created
                      from scratch.

                  - first_input (type: list, default: None):
                      this variable allows users to analyze the first input
                      given when calling the class for the first time.

    Input (call): - inp_lst(type: list, mandatory):
                      by default the input arguments are passed with a list.
                      In this case, inp is a list containing two tensors.
                      The first should contain an audio signal, and the
                      second should contain the lengths of the audio signals
                      contained in the first tensor. The first input
                      tensor must be in one of the following formats:
                      [batch, time_steps], or [batch, channels, time_steps]
                      and the second must be in the format [batch].

    Output (call): - out_lst(type: list, mandatory):
                      by default the output arguments are passed with a list.
                      In this case, out is a list containing a torch.tensor
                      with the noisy audio signal. The tensor is the same
                      shape as the input, i.e. formatted in the following way:
                      [batch, time_steps], or [batch, channels, time_steps]

    Example:   import torch
               from speechbrain.data_io.data_io import save
               from speechbrain.data_io.data_io import create_dataloader
               from speechbrain.processing.speech_augmentation \
                   import add_babble

               # config dictionary definition
               config = {
                   'class_name':'speechbrain.processing.speech_augmentation.add_babble',
               }

               # Initialization of the class
               babbler = add_babble(config)

               # Load batch
               config = {
                   'class_name':'speechbrain.data_io.data_io.create_dataloader',
                   'csv_file':'samples/audio_samples/csv_example3.csv',
                   'batch_size':'5',
               }

               dataloader = create_dataloader(config)
               loader = zip(*dataloader.dataloader)
               ids, batch, lengths = next(loader)[0]

               # Executing computations
               noisy = babbler([batch, lengths])

               # save config dictionary definition
               config = {
                  'class_name': 'speechbrain.data_io.data_io.save',
                  'save_folder': 'exp/write_example',
                  'save_format': 'wav',
               }

               # class initialization
               save_signal = save(config)

               # saving
               save_signal([noisy, ids, lengths])

               # signal save in exp/write_example
    -------------------------------------------------------------------------
    """

    def __init__(
        self,
        config,
        funct_name=None,
        global_config=None,
        functions=None,
        logger=None,
        first_input=None,
    ):
        super(add_babble, self).__init__()

        # Logger setup
        self.logger = logger

        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "speaker_count": ("int(1,inf)", "optional", "3"),
            "snr_low": ("float(-inf,inf)", "optional", "0"),
            "snr_high": ("float(-inf,inf)", "optional", "0"),
            "mix_prob": ("float(0,1)", "optional", "1"),
            "random_seed": ("int(-inf,inf)", "optional", "None"),
        }

        # Check, cast, and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, self.logger
        )

        # Expected inputs when calling this class:
        #    Input 1: Batch of waveforms to be processed
        #    Input 2: Length of waveforms in the batch
        self.expected_inputs = ["torch.Tensor", "torch.Tensor"]

        # Check the first input
        check_inputs(
            self.conf, self.expected_inputs, first_input, logger=self.logger
        )

        # Additional check on the input shapes
        if first_input is not None:
            check_input_shapes([[2, 3], [1]], first_input, logger)

            # We can only babble up to the batch size
            if len(first_input[0]) <= self.speaker_count:

                # Build error message
                err_msg = (
                    'Function: `add_babble` requires a `speaker_count` that '
                    'is less than the batch size, but got %d which is not '
                    'less than %d' % (self.speaker_count, len(first_input[0]))
                )

                logger_write(err_msg, logfile=logger)

        # Initialize a random number generator with the provided seed
        if self.random_seed is not None:
            torch.random.manual_seed(self.random_seed)

        # Save the state of the RNG for reproducibility
        self.rng_state = torch.random.get_rng_state()

    def forward(self, input_lst):

        # Load the state of the RNG for reproducibility
        torch.random.set_rng_state(self.rng_state)

        # Reading input list
        clean_waveform, clean_len = input_lst
        babbled_waveform = clean_waveform.clone()
        clean_len = (clean_len * clean_waveform.shape[1]).unsqueeze(1)
        batch_size = len(clean_waveform)

        # Don't mix (return early) 1-`mix_prob` portion of the batches
        if torch.rand(1) > self.mix_prob:
            self.rng_state = torch.random.get_rng_state()
            return babbled_waveform

        # Compute the average amplitude of the clean waveforms
        clean_amplitude = compute_amplitude(clean_waveform, clean_len)

        # Pick an SNR and use it to compute the mixture amplitude factors
        SNR = torch.rand(batch_size, 1, device=clean_waveform.device)
        SNR = SNR * (self.snr_high - self.snr_low) + self.snr_low
        noise_amplitude_factor = 1 / (dB_to_amplitude(SNR) + 1)
        new_noise_amplitude = noise_amplitude_factor * clean_amplitude

        # Scale clean signal appropriately
        babbled_waveform *= (1 - noise_amplitude_factor)

        # For each speaker in the mixture, roll and add
        babble_waveform = clean_waveform.roll((1,), dims=0)
        babble_len = clean_len.roll((1,), dims=0)
        for i in range(1, self.speaker_count):
            babble_waveform += clean_waveform.roll((1+i,), dims=0)
            babble_len = torch.max(babble_len, babble_len.roll((1,), dims=0))

        # Rescale and add to mixture
        babble_amplitude = compute_amplitude(babble_waveform, babble_len)
        babble_waveform *= new_noise_amplitude / babble_amplitude
        babbled_waveform += babble_waveform

        # Save the state of the RNG for reproducibility
        self.rng_state = torch.random.get_rng_state()

        return babbled_waveform


class drop_freq(nn.Module):
    """
    -------------------------------------------------------------------------
    speechbrain.processing.speech_augmentation.drop_freq
    (author: Peter Plantinga)

    Description: This class drops a random frequency from the signal, so that
                 models learn to rely on all parts of the signal, not just
                 a single frequency band.

    Input (init):  - config (type: dict, mandatory):
                      it is a dictionary containing the keys described below.

                      - drop_freq_low (type: float, default: 0):
                          The low end of frequencies that can be dropped,
                          as a fraction of the sampling rate / 2.

                      - drop_freq_high (type: float, default: 1):
                          The high end of frequencies that can be dropped,
                          as a fraction of the sampling rate / 2.

                      - drop_count_low (type: int, default: 1):
                          The low end of number of frequencies that could
                          be dropped.

                      - drop_count_high (type: int, default: 2):
                          The high end of number of frequencies that could
                          be dropped.

                      - drop_prob (type: float, default: 1.0):
                          The probability that the batch of signals will
                          have a frequency dropped. By default, every batch
                          has frequencies dropped.

                      - random_seed (type: int, default: None):
                          The seed for randomly selecting band to drop.

                  - funct_name (type: str, default: None):
                      it is a string containing the name of the parent
                      function that has called this method.

                  - global_config (type: dict, default: None):
                      it a dictionary containing the global variables of the
                      parent config file.

                  - functions (type: dict, default: None):
                      dictionary for storing user-defined functions. Keys are
                      the function names, values are their corresponding
                      objects.

                  - logger (type: logger, default: None):
                      it the logger used to write debug and error messages.
                      If logger=None and root_cfg=True, the file is created
                      from scratch.

                  - first_input (type: list, default: None):
                      this variable allows users to analyze the first input
                      given when calling the class for the first time.

    Input (call): - inp_lst(type: list, mandatory):
                      by default the input arguments are passed with a list.
                      In this case, inp is a list containing a single
                      torch.tensor which contains an audio signal. The input
                      tensor must be in one of the following formats:
                      [batch, time_steps], or [batch, channels, time_steps]

    Output (call): - out_lst(type: list, mandatory):
                      by default the output arguments are passed with a list.
                      In this case, out is a list containing a torch.tensor
                      with the noisy audio signal. The tensor is the same
                      shape as the input, i.e. formatted in the following way:
                      [batch, time_steps], or [batch, channels, time_steps]

    Example:   import torch
               import soundfile as sf
               from speechbrain.data_io.data_io import save
               from speechbrain.processing.speech_augmentation import drop_freq

               # config dictionary definition
               config = {
                   'class_name':'speechbrain.processing.speech_augmentation.drop_freq',
               }

               # Initialization of the class
               dropper = drop_freq(config)

               # Load sample
               signal, rate = sf.read('samples/audio_samples/example1.wav')
               signal = torch.tensor(signal, dtype=torch.float32)

               # Perform drop
               dropped_signal = dropper([signal.unsqueeze(0)])

               # save config dictionary definition
               config = {
                  'class_name': 'speechbrain.data_io.data_io.save',
                  'save_folder': 'exp/write_example',
                  'save_format': 'wav',
               }

               # class initialization
               save_signal = save(config)

               # saving
               save_signal([dropped_signal, ['freq_drop'], torch.ones(1)])

               # signal save in exp/write_example
    -------------------------------------------------------------------------
    """

    def __init__(
        self,
        config,
        funct_name=None,
        global_config=None,
        functions=None,
        logger=None,
        first_input=None,
    ):
        super(drop_freq, self).__init__()

        # Logger setup
        self.logger = logger

        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "drop_freq_low": ("float(0,1)", "optional", "0"),
            "drop_freq_high": ("float(0,1)", "optional", "1"),
            "drop_count_low": ("int(0,inf)", "optional", "1"),
            "drop_count_high": ("int(0,inf)", "optional", "2"),
            "drop_width": ("float(0,0.2)", "optional", "0.05"),
            "drop_prob": ("float(0,1)", "optional", "1"),
            "random_seed": ("int(-inf,inf)", "optional", "None"),
        }

        # Check, cast, and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, self.logger
        )

        # Expected inputs when calling this class:
        #    Input 1: Batch of waveforms to be processed
        self.expected_inputs = ["torch.Tensor"]

        # Check the first input
        check_inputs(
            self.conf, self.expected_inputs, first_input, logger=self.logger
        )

        # Additional check on the input shapes
        if first_input is not None:
            check_input_shapes([[2, 3]], first_input, logger)

        # Initialize a random number generator with the provided seed
        if self.random_seed is not None:
            torch.random.manual_seed(self.random_seed)

        # Save the state of the RNG for reproducibility
        self.rng_state = torch.random.get_rng_state()

    def forward(self, input_lst):

        # Load the state of the RNG for reproducibility
        torch.random.set_rng_state(self.rng_state)

        # Reading input list
        clean_waveform = input_lst[0]
        dropped_waveform = clean_waveform.clone()

        # Don't drop (return early) 1-`drop_prob` portion of the batches
        if torch.rand(1) > self.drop_prob:
            self.rng_state = torch.random.get_rng_state()
            return dropped_waveform

        # Add channels dimension
        if len(clean_waveform.shape) == 2:
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
            ).to(clean_waveform.device)
            drop_filter = convolve1d(drop_filter, notch_kernel, pad)

        # Apply filter
        dropped_waveform = convolve1d(dropped_waveform, drop_filter, pad)

        # Save the state of the RNG for reproducibility
        self.rng_state = torch.random.get_rng_state()

        # Remove channels dimension if added
        return dropped_waveform.squeeze(1)


class drop_chunk(nn.Module):
    """
    -------------------------------------------------------------------------
    speechbrain.processing.speech_augmentation.drop_chunk
    (author: Peter Plantinga)

    Description: This class drops portions of the input signal, so that
                 models learn to rely on all parts of the signal.

    Input (init):  - config (type: dict, mandatory):
                      it is a dictionary containing the keys described below.

                      - drop_length_low (type: int, default: 100):
                          The low end of lengths for which to set the signal
                          to zero, in samples.

                      - drop_length_high (type: int, default: 1000):
                          The high end of lengths for which to set the signal
                          to zero, in samples.

                      - drop_count_low (type: int, default: 1):
                          The low end of number of times that the signal
                          can be dropped to zero.

                      - drop_count_high (type: int, default: 10):
                          The high end of number of times that the signal
                          can be dropped to zero.

                      - drop_start (type: int, default: 0):
                          The first index for which dropping will be allowed.

                      - drop_end (type: int, default: None):
                          The last index for which dropping will be allowed.

                      - drop_prob (type: float, default: 1.0):
                          The probability that the batch of signals will
                          have a portion dropped. By default, every batch
                          has portions dropped.

                      - random_seed (type: int, default: None):
                          The seed for randomly selecting band to drop.

                  - funct_name (type: str, default: None):
                      it is a string containing the name of the parent
                      function that has called this method.

                  - global_config (type: dict, default: None):
                      it a dictionary containing the global variables of the
                      parent config file.

                  - functions (type: dict, default: None):
                      dictionary for storing user-defined functions. Keys are
                      the function names, values are their corresponding
                      objects.

                  - logger (type: logger, default: None):
                      it the logger used to write debug and error messages.
                      If logger=None and root_cfg=True, the file is created
                      from scratch.

                  - first_input (type: list, default: None):
                      this variable allows users to analyze the first input
                      given when calling the class for the first time.

    Input (call): - inp_lst(type: list, mandatory):
                      by default the input arguments are passed with a list.
                      In this case, inp is a list containing two torch.tensors
                      which contains a batch of signals and their lengths. The
                      input batch must be in one of the following formats:
                      [batch, time_steps], or [batch, channels, time_steps]

    Output (call): - out_lst(type: list, mandatory):
                      by default the output arguments are passed with a list.
                      In this case, out is a list containing a torch.tensor
                      with the noisy audio signal. The tensor is the same
                      shape as the input, i.e. formatted in the following way:
                      [batch, time_steps], or [batch, channels, time_steps]

    Example:   import torch
               import soundfile as sf
               from speechbrain.data_io.data_io import save
               from speechbrain.processing.speech_augmentation \
                   import drop_chunk

               # config dictionary definition
               config = {
                   'class_name':'speechbrain.processing.speech_augmentation.drop_chunk',
               }

               # Initialization of the class
               dropper = drop_chunk(config)

               # Load sample
               signal, rate = sf.read('samples/audio_samples/example1.wav')
               signal = torch.tensor(signal).unsqueeze(0)
               length = torch.ones(1)

               # Perform drop
               dropped_signal = dropper([signal, length])

               # save config dictionary definition
               config = {
                  'class_name': 'speechbrain.data_io.data_io.save',
                  'save_folder': 'exp/write_example',
                  'save_format': 'wav',
               }

               # class initialization
               save_signal = save(config)

               # saving
               save_signal([dropped_signal, ['drop_chunk'], length])

               # signal save in exp/write_example
    -------------------------------------------------------------------------
    """

    def __init__(
        self,
        config,
        funct_name=None,
        global_config=None,
        functions=None,
        logger=None,
        first_input=None,
    ):
        super(drop_chunk, self).__init__()

        # Logger setup
        self.logger = logger

        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "drop_length_low": ("int(0,inf)", "optional", "100"),
            "drop_length_high": ("int(0,inf)", "optional", "1000"),
            "drop_count_low": ("int(0,inf)", "optional", "1"),
            "drop_count_high": ("int(0,inf)", "optional", "10"),
            "drop_start": ("int(-inf,inf)", "optional", "0"),
            "drop_end": ("int(-inf,inf)", "optional", "None"),
            "drop_prob": ("float(0,1)", "optional", "1"),
            "random_seed": ("int(-inf,inf)", "optional", "None"),
        }

        # Check, cast, and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, self.logger
        )

        # Expected inputs when calling this class:
        #    Input 1: Batch of waveforms to be processed
        #    Input 2: Length of waveforms in the batch
        self.expected_inputs = ["torch.Tensor", "torch.Tensor"]

        # Check the first input
        check_inputs(
            self.conf, self.expected_inputs, first_input, logger=self.logger
        )

        # Additional check on the input shapes
        if first_input is not None:
            check_input_shapes([[2, 3], [1]], first_input, logger)

        # Initialize a random number generator with the provided seed
        if self.random_seed is not None:
            torch.random.manual_seed(self.random_seed)

        # Save the state of the RNG for reproducibility
        self.rng_state = torch.random.get_rng_state()

    def forward(self, input_lst):

        # Load the state of the RNG for reproducibility
        torch.random.set_rng_state(self.rng_state)

        # Reading input list
        clean_waveform = input_lst[0]
        clean_length = (input_lst[1] * clean_waveform.size(-1)).long()
        batch_size = clean_waveform.size(0)
        dropped_waveform = clean_waveform.clone()

        # Don't drop (return early) 1-`drop_prob` portion of the batches
        if torch.rand(1) > self.drop_prob:
            self.rng_state = torch.random.get_rng_state()
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
                start_min += clean_length[i]
            start_max = self.drop_end
            if start_max is None:
                start_max = clean_length[i]
            if start_max < 0:
                start_max += clean_length[i]
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

        # Save the state of the RNG for reproducibility
        self.rng_state = torch.random.get_rng_state()

        return dropped_waveform


class do_clip(nn.Module):
    """
    -------------------------------------------------------------------------
    speechbrain.processing.speech_augmentation.do_clip
    (author: Peter Plantinga)

    Description: This class drops a random frequency from the signal, so that
                 models learn to rely on all parts of the signal, not just
                 a single frequency band.

    Input (init):  - config (type: dict, mandatory):
                      it is a dictionary containing the keys described below.

                      - clip_low (type: float, default: 0.5):
                          The low end of amplitudes for which to clip
                          the signal.

                      - clip_high (type: float, default: 1.0):
                          The high end of amplitudes for which to clip
                          the signal.

                      - clip_prob (type: float, default: 1.0):
                          The probability that the batch of signals will
                          have a portion clipped. By default, every batch
                          has portions clipped.

                      - random_seed (type: int, default: None):
                          The seed for randomly selecting clipping range.

                  - funct_name (type: str, default: None):
                      it is a string containing the name of the parent
                      function that has called this method.

                  - global_config (type: dict, default: None):
                      it a dictionary containing the global variables of the
                      parent config file.

                  - functions (type: dict, default: None):
                      dictionary for storing user-defined functions. Keys are
                      the function names, values are their corresponding
                      objects.

                  - logger (type: logger, default: None):
                      it the logger used to write debug and error messages.
                      If logger=None and root_cfg=True, the file is created
                      from scratch.

                  - first_input (type: list, default: None):
                      this variable allows users to analyze the first input
                      given when calling the class for the first time.

    Input (call): - inp_lst(type: list, mandatory):
                      by default the input arguments are passed with a list.
                      In this case, inp is a list containing a single
                      torch.tensor which contains an audio signal. The input
                      tensor must be in one of the following formats:
                      [batch, time_steps], or [batch, channels, time_steps]

    Output (call): - out_lst(type: list, mandatory):
                      by default the output arguments are passed with a list.
                      In this case, out is a list containing a torch.tensor
                      with the noisy audio signal. The tensor is the same
                      shape as the input, i.e. formatted in the following way:
                      [batch, time_steps], or [batch, channels, time_steps]

    Example:   import torch
               import soundfile as sf
               from speechbrain.data_io.data_io import save
               from speechbrain.processing.speech_augmentation import do_clip

               # config dictionary definition
               config = {
                   'class_name':'speechbrain.processing.speech_augmentation.do_clip',
               }

               # Initialization of the class
               clipper = do_clip(config)

               # Load sample
               signal, rate = sf.read('samples/audio_samples/example1.wav')

               # Perform clipping
               clipped_signal = clipper([torch.tensor(signal).unsqueeze(0)])

               # save config dictionary definition
               config = {
                  'class_name': 'speechbrain.data_io.data_io.save',
                  'save_folder': 'exp/write_example',
                  'save_format': 'wav',
               }

               # class initialization
               save_signal = save(config)

               # saving
               save_signal([clipped_signal, ['clip'], torch.ones(1)])

               # signal save in exp/write_example
    -------------------------------------------------------------------------
    """

    def __init__(
        self,
        config,
        funct_name=None,
        global_config=None,
        functions=None,
        logger=None,
        first_input=None,
    ):
        super(do_clip, self).__init__()

        # Logger setup
        self.logger = logger

        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "clip_low": ("float(0,1)", "optional", "0.5"),
            "clip_high": ("float(0,1)", "optional", "1"),
            "clip_prob": ("float(0,1)", "optional", "1"),
            "random_seed": ("int(-inf,inf)", "optional", "None"),
        }

        # Check, cast, and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, self.logger
        )

        # Expected inputs when calling this class:
        #    Input 1: Batch of waveforms to be processed
        self.expected_inputs = ["torch.Tensor"]

        # Check the first input
        check_inputs(
            self.conf, self.expected_inputs, first_input, logger=self.logger
        )

        # Additional check on the input shapes
        if first_input is not None:
            check_input_shapes([[2, 3]], first_input, logger)

        # Initialize a random number generator with the provided seed
        if self.random_seed is not None:
            torch.random.manual_seed(self.random_seed)

        # Save the state of the RNG for reproducibility
        self.rng_state = torch.random.get_rng_state()

    def forward(self, input_lst):

        # Load the state of the RNG for reproducibility
        torch.random.set_rng_state(self.rng_state)

        # Reading input list
        clean_waveform = input_lst[0]

        # Don't clip (return early) 1-`clip_prob` portion of the batches
        if torch.rand(1) > self.clip_prob:
            self.rng_state = torch.random.get_rng_state()
            return clean_waveform.clone()

        # Randomly select clip value
        clipping_range = self.clip_high - self.clip_low
        clip_value = torch.rand(1,)[0] * clipping_range + self.clip_low

        # Apply clipping
        clipped_waveform = clean_waveform.clamp(-clip_value, clip_value)

        # Save the state of the RNG for reproducibility
        self.rng_state = torch.random.get_rng_state()

        return clipped_waveform
