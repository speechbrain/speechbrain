"""This library is used to create data on-the-fly for VAD.

Authors
 * Mirco Ravanelli 2020
"""

import random

import torch
import torchaudio

# fade-in/fade-out definition
fade_in = torchaudio.transforms.Fade(fade_in_len=1000, fade_out_len=0)
fade_out = torchaudio.transforms.Fade(fade_in_len=0, fade_out_len=1000)


def add_chunk(
    wav,
    wav_chunk,
    target,
    sample_rate=16000,
    time_resolution=0.01,
    example_length=5,
    min_len=1.0,
    max_len=2.5,
    min_begin_sample=0,
    min_amp=0.4,
    max_amp=1.0,
    chunk_shift=0,
    speech=False,
):
    """Add a new source (noise or speech) to an existing chunk of speech.
    The length of the appended signal is randomly selected within the
    min_len-max_len range. The shift is controlled by the  chunk_shift
    parameter.

    Arguments
    ---------
    wav: torch.Tensor
        The waveform to append.
    wav_chunk: torch.Tensor
        The existing waveform where to append the new source.
    target: torch.Tensor
        Old target.
    sample_rate: int
        The sample rate of the input waveforms.
    time_resolution: float
        Time resolution of the targets (in seconds)-
    example_length: float
        Duration (in seconds) of the existing chunk.
    min_len: float
        Minimum duration (in seconds) of the waveform to append.
    max_len: float
        Maximum duration (in seconds) of the waveform to append.
    min_begin_sample: int
        It allows sampling the original waveform with some shift. This might
        be useful to avoid the initial sampled of the waveform that can be
        silence samples.
    min_amp: float
        The new waveform is appended with a random amplitude sampled in the
        range min_amp-max_amp.
    max_amp: float
        See min_amp.
    chunk_shift: int
        This parameter controls where to append the new source within the
        existing one.
    speech: bool
        If True, the new waveform is assumed to be a speech signal. The targets
        will be put to 1 for all the duration on the speech signal.

    Returns
    -------
    wav_chunk: torch.Tensor
        The new waveform with the added signal.
    target: torch.Tensor
        The new targets corresponding to the output signal.
    lengths: torch.Tensor
        relative lengths of each chunk.
    end_chunk: int
        The last sample of the appended sequence. It can be used later to add
        another source that do not overlap with the current one.
    """

    # Convert from seconds to samples
    min_len_samples = int(sample_rate * min_len)
    max_len_samples = int(sample_rate * max_len)
    last_sample = int(example_length * sample_rate)

    # Randomly sampling the length of the chunk to append
    len_chunk = torch.randint(
        low=min_len_samples, high=max_len_samples, size=(1,)
    ).item()

    # Randomly sampling the start sample
    max_end_sample = min_begin_sample + (last_sample - len_chunk)
    begin_sample = torch.randint(
        low=min_begin_sample, high=max_end_sample, size=(1,)
    ).item()
    end_chunk = min(chunk_shift + len_chunk, last_sample)

    # Randomly sampling the amplitude of the chunk to append
    rand_amp = (
        torch.rand(wav.shape[0], 1, wav.shape[-1], device=wav.device)
        * (max_amp - min_amp)
        + min_amp
    )

    # Fetch the signal to append
    wav_to_paste = wav[
        :, begin_sample : begin_sample + (end_chunk - chunk_shift)
    ]

    # Random amplitude
    max_v, p = wav_to_paste.abs().max(1)
    wav_to_paste = wav_to_paste.transpose(1, 0) / max_v.unsqueeze(0)
    wav_to_paste = wav_to_paste.transpose(1, 0)
    wav_to_paste = rand_amp * wav_to_paste

    # Apply fade_in/fade_out if needed
    if chunk_shift > 0:
        wav_to_paste = fade_in(wav_to_paste.transpose(1, -1))
        wav_to_paste = wav_to_paste.transpose(1, -1)

    if end_chunk < last_sample:
        wav_to_paste = fade_out(wav_to_paste.transpose(1, -1))
        wav_to_paste = wav_to_paste.transpose(1, -1)

    # Append the signal
    wav_chunk[:, chunk_shift:end_chunk] = (
        wav_chunk[:, chunk_shift:end_chunk] + wav_to_paste
    )

    # Update targets if the appended signal is speech.
    if speech:
        beg_speech_target = int(chunk_shift / (sample_rate * time_resolution))
        end_speech_target = int(end_chunk / (sample_rate * time_resolution))
        target[:, beg_speech_target:end_speech_target] = 1

    # Length computation
    lengths = torch.ones(
        wav_chunk.shape[0], wav_chunk.shape[-1], device=wav.device
    )
    return wav_chunk, target, lengths, end_chunk


def initialize_targets(wav, sample_rate, time_resolution):
    "Initializes the targets."
    target_downsampling = sample_rate * time_resolution
    target_len = int(wav.shape[1] / (target_downsampling))
    targets = torch.zeros(
        (wav.shape[0], target_len, wav.shape[2]), device=wav.device
    )
    return targets


def get_samples_from_datasets(datasets, wav):
    """Gets samples (noise or speech) from the datasets.

    Arguments
    ---------
    datasets : list
        List containing datasets. More precisely, we expect here the pointers
        to the object used in speechbrain for data augmentation
        (e.g, speechbrain.lobes.augment.EnvCorrupt).
    wav : torch.Tensor
        The original waveform. The drawn samples will have the same
        dimensionality of the original waveform.

    Returns
    -------
    samples: torch.Tensor
        A batch of new samples drawn from the input list of datasets.
    """
    # We want a sample of the same size of the original signal
    samples = torch.zeros(
        wav.shape[0], wav.shape[1], len(datasets), device=wav.device
    )

    # Let's sample a sequence from each dataset
    for i, dataset in enumerate(datasets):
        # Initialize the signal with noise
        wav_sample = (torch.rand_like(wav) * 2) - 1
        len_sample = torch.ones(wav.shape[0], device=wav.device)

        # Sample a sequence
        wav_sample = dataset(wav_sample, len_sample)

        # Append it
        samples[:, :, i] = wav_sample

    # Random permutations of the signal
    idx = torch.randperm(samples.shape[-1], device=wav.device)
    samples[:, :] = samples[:, :, idx]
    return samples


def create_chunks(
    wav1,
    wav2,
    background,
    sample_rate=16000,
    time_resolution=0.01,
    example_length=5,
    speech1=False,
    speech2=False,
    low_background=0.05,
    high_background=0.15,
    max_pause=16000,
):
    """This method creates augmented data for training the VAD.
    It sums up two delayed sources + a noise background.

    Arguments
    ---------
    wav1 : torch.Tensor
        The waveform for source 1.
    wav2 : torch.Tensor
        The waveform for source 2.
    background : torch.Tensor
        The waveform for background noise.
    sample_rate: int
        The sample rate of the input waveforms.
    time_resolution: float
        Time resolution of the targets (in seconds)-
    example_length: float
        Duration (in seconds) of the existing chunk.
    speech1: bool
        If True, source 1 is assumed to be a speech signal. The targets
        will be put to 1 for all the duration on the speech signal.
    speech2: bool
        If True, source 2 is assumed to be a speech signal. The targets
        will be put to 1 for all the duration of the speech signal.
    low_background: float
        The amplitude of the background is randomly sampled between
        low_background and high_background.
    high_background: float
        See above.
    max_pause: int
        Max pause in samples between the two sources.

    Returns
    -------
    wavs: torch.Tensor
        The generated speech signal.
    target: torch.Tensor
        The new targets corresponding to the generated signal.
    lengths: torch.Tensor
        relative lengths of each chunk.
    """

    # Random selection of the background amplitude
    background_amp = (
        random.random() * (high_background - low_background) + low_background
    )

    # Adding the background
    wav = background_amp * (torch.rand_like(background) - 0.5)
    background = torch.roll(background, 1, dims=-1)
    wav = wav + background_amp * background

    # Adding the first source
    wav, target, lengths, end_chunk = add_chunk(
        wav1,
        wav,
        initialize_targets(wav1, sample_rate, time_resolution),
        sample_rate=sample_rate,
        time_resolution=time_resolution,
        example_length=example_length,
        speech=speech1,
    )

    # Choosing the lag of the second source
    begin_sample = torch.randint(
        low=end_chunk, high=end_chunk + max_pause, size=(1,)
    ).item()

    # Adding the second source
    wav, target, lengths, end_sample = add_chunk(
        wav2,
        wav,
        target,
        chunk_shift=begin_sample,
        sample_rate=sample_rate,
        time_resolution=time_resolution,
        example_length=example_length,
        speech=speech2,
    )

    wav = wav.transpose(1, 2).reshape(wav.shape[0] * wav.shape[2], wav.shape[1])
    target = target.transpose(1, 2).reshape(
        target.shape[0] * target.shape[2], target.shape[1]
    )
    lengths = lengths.reshape([lengths.shape[0] * lengths.shape[1]])
    return wav, target, lengths


def augment_data(noise_datasets, speech_datasets, wavs, targets, lens_targ):
    """This method creates different types of augmented that are useful to train
    a VAD system. It creates signals with different types of transitions, such
    as speech=>speech, noise=>speech, speech=>noise. The new signals are
    concatenated with the original ones (wavs, targets) such that the output
    is already an augmented batch useful to train a VAD model.


    Arguments
    ---------
    noise_datasets: list
        List containing noise datasets. More precisely, we expect here the pointers
        to the object used in speechbrain for data augmentation
        (e.g, speechbrain.lobes.augment.EnvCorrupt).
    speech_datasets: list
        List containing noise datasets. More precisely, we expect here the pointers
        to the object used in speechbrain for data augmentation
        (e.g, speechbrain.lobes.augment.EnvCorrupt).
    wavs: torch.Tensor
        The original waveform.
    targets: torch.Tensor
        The original targets.
    lens_targ: torch.Tensor
        The length of the original targets.


    Returns
    -------
    wavs: torch.Tensor
        The output batch with the augmented signals
    target: torch.Tensor
        The new targets corresponding to the augmented signals.
    lengths: torch.Tensor
        relative lengths of each element in the batch.
    """
    # Sample a noise sequence
    wav_samples_noise = get_samples_from_datasets(noise_datasets, wavs)

    # Sample a speech sequence
    wav_samples_speech = get_samples_from_datasets(speech_datasets, wavs)

    # Create chunk with noise=>speech transition
    (
        wav_noise_speech,
        target_noise_speech,
        lengths_noise_speech,
    ) = create_chunks(
        wav_samples_noise,
        wav_samples_speech,
        wav_samples_noise,
        speech1=False,
        speech2=True,
    )

    # Create chunk with speech=>noise transition
    (
        wav_speech_noise,
        target_speech_noise,
        lengths_speech_noise,
    ) = create_chunks(
        wav_samples_speech,
        wav_samples_noise,
        wav_samples_noise,
        speech1=True,
        speech2=False,
    )

    # Create chunk with speech=>speech transition
    wav_samples_speech2 = torch.roll(wav_samples_speech, 1, dims=-1)
    (
        wav_speech_speech,
        target_speech_speech,
        lengths_speech_speech,
    ) = create_chunks(
        wav_samples_speech,
        wav_samples_speech2,
        wav_samples_noise,
        speech1=True,
        speech2=True,
    )

    # Create chunk with noise=>noise transition
    wav_samples_noise2 = torch.roll(wav_samples_noise, 1, dims=-1)
    (wav_noise_noise, target_noise_noise, lengths_noise_noise) = create_chunks(
        wav_samples_noise,
        wav_samples_noise2,
        wav_samples_noise,
        speech1=False,
        speech2=False,
    )

    # Concatenate all the augmented data
    wavs = torch.cat(
        [
            wavs,
            wav_noise_speech,
            wav_speech_noise,
            wav_speech_speech,
            wav_noise_noise,
        ],
        dim=0,
    )

    # Concatenate targets
    targets = torch.cat(
        [
            targets,
            target_noise_speech,
            target_speech_noise,
            target_speech_speech,
            target_noise_noise,
        ],
        dim=0,
    )

    # Concatenate lengths
    lens = torch.cat(
        [
            lens_targ,
            lengths_noise_speech,
            lengths_speech_noise,
            lengths_speech_speech,
            lengths_noise_noise,
        ],
        dim=0,
    )

    # Assign random amplitude to the signals
    max_amp, _ = wavs.abs().max(1)
    wavs = wavs / max_amp.unsqueeze(1)
    wavs = wavs * torch.rand_like(max_amp).unsqueeze(1)

    return wavs, targets, lens
