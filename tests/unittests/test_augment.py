import os

import torch
import torchaudio

from speechbrain.dataio.dataio import write_audio


def test_add_noise(tmpdir, device):
    from speechbrain.augment.time_domain import AddNoise

    # Test concatenation of batches
    wav_a = torch.sin(torch.arange(8000.0, device=device)).unsqueeze(0)
    a_len = torch.ones(1, device=device)
    wav_b = (
        torch.cos(torch.arange(10000.0, device=device))
        .unsqueeze(0)
        .repeat(2, 1)
    )
    b_len = torch.ones(2, device=device)
    concat, lens = AddNoise._concat_batch(wav_a, a_len, wav_b, b_len)
    assert concat.shape == (3, 10000)
    assert lens.allclose(torch.Tensor([0.8, 1, 1]).to(device))
    concat, lens = AddNoise._concat_batch(wav_b, b_len, wav_a, a_len)
    assert concat.shape == (3, 10000)
    expected = torch.Tensor([1, 1, 0.8]).to(device)
    assert lens.allclose(expected)

    test_waveform = torch.sin(torch.arange(16000.0, device=device)).unsqueeze(0)
    test_noise = torch.cos(torch.arange(16000.0, device=device)).unsqueeze(0)
    wav_lens = torch.ones(1, device=device)

    # Put noise waveform into temporary file
    noisefile = os.path.join(tmpdir, "noise.wav")
    write_audio(noisefile, test_noise.transpose(0, 1).cpu(), 16000)

    csv = os.path.join(tmpdir, "noise.csv")
    with open(csv, "w", encoding="utf-8") as w:
        w.write("ID, duration, wav, wav_format, wav_opts\n")
        w.write(f"1, 1.0, {noisefile}, wav,\n")

    # Edge cases
    no_noise = AddNoise(snr_low=1000, snr_high=1000)
    assert no_noise(test_waveform, wav_lens).allclose(test_waveform)
    all_noise = AddNoise(csv_file=csv, snr_low=-1000, snr_high=-1000)
    assert all_noise(test_waveform, wav_lens).allclose(test_noise, atol=1e-4)

    # Basic 0dB case
    add_noise = AddNoise(csv_file=csv).to(device)
    expected = (test_waveform + test_noise) / 2
    assert add_noise(test_waveform, wav_lens).allclose(expected, atol=1e-4)


def test_add_reverb(tmpdir, device):
    from speechbrain.augment.time_domain import AddReverb

    test_waveform = torch.sin(torch.arange(16000.0, device=device)).unsqueeze(0)
    impulse_response = torch.zeros(1, 8000, device=device)
    impulse_response[0, 0] = 1.0

    # Put ir waveform into temporary file
    ir1 = os.path.join(tmpdir, "ir1.wav")
    ir2 = os.path.join(tmpdir, "ir2.wav")
    ir3 = os.path.join(tmpdir, "ir3.wav")
    write_audio(ir1, impulse_response.cpu().transpose(0, 1), 16000)

    impulse_response[0, 0] = 0.0
    impulse_response[0, 10] = 0.5
    write_audio(ir2, impulse_response.cpu().transpose(0, 1), 16000)

    # Check a very simple non-impulse-response case:
    impulse_response[0, 10] = 0.6
    impulse_response[0, 11] = 0.4
    # sf.write(ir3, impulse_response.squeeze(0).numpy(), 16000)
    write_audio(ir3, impulse_response.cpu().transpose(0, 1), 16000)
    ir3_result = test_waveform * 0.6 + test_waveform.roll(1, -1) * 0.4

    # write ir csv file
    csv = os.path.join(tmpdir, "ir.csv")
    with open(csv, "w", encoding="utf-8") as w:
        w.write("ID, duration, wav, wav_format, wav_opts\n")
        w.write(f"1, 0.5, {ir1}, wav,\n")
        w.write(f"2, 0.5, {ir2}, wav,\n")
        w.write(f"3, 0.5, {ir3}, wav,\n")

    # Normal cases
    add_reverb = AddReverb(csv, sorting="original")
    reverbed = add_reverb(test_waveform)[:, 0:1000]
    assert reverbed.allclose(test_waveform[:, 0:1000], atol=1e-1)
    reverbed = add_reverb(test_waveform)[:, 0:1000]
    assert reverbed.allclose(test_waveform[:, 0:1000], atol=1e-1)
    reverbed = add_reverb(test_waveform)[:, 0:1000]
    assert reverbed.allclose(ir3_result[:, 0:1000], atol=2e-1)


def test_speed_perturb(device):
    from speechbrain.augment.time_domain import SpeedPerturb

    test_waveform = torch.sin(torch.arange(16000.0, device=device)).unsqueeze(0)

    # Edge cases
    no_perturb = SpeedPerturb(16000, speeds=[100]).to(device)
    assert no_perturb(test_waveform).allclose(test_waveform)

    # Half speed
    half_speed = SpeedPerturb(16000, speeds=[50]).to(device)
    assert half_speed(test_waveform).allclose(test_waveform[:, ::2], atol=3e-1)


def test_drop_freq(device):
    from speechbrain.augment.time_domain import DropFreq

    test_waveform = torch.sin(torch.arange(16000.0, device=device)).unsqueeze(0)

    # Edge cases
    no_drop = DropFreq(drop_freq_count_low=0, drop_freq_count_high=0)
    assert no_drop(test_waveform).allclose(test_waveform)

    # Check case where frequency range *does not* include signal frequency
    drop_diff_freq = DropFreq(drop_freq_low=0.5, drop_freq_high=0.9)
    assert drop_diff_freq(test_waveform).allclose(test_waveform, atol=5e-1)

    # Check case where frequency range *does* include signal frequency
    drop_same_freq = DropFreq(drop_freq_low=0.28, drop_freq_high=0.28)
    assert drop_same_freq(test_waveform).allclose(
        torch.zeros(1, 16000, device=device), atol=4e-1
    )


def test_drop_chunk(device):
    from speechbrain.augment.time_domain import DropChunk

    test_waveform = torch.sin(torch.arange(16000.0, device=device)).unsqueeze(0)
    lengths = torch.ones(1, device=device)

    # Edge cases
    no_drop = DropChunk(drop_length_low=0, drop_length_high=0).to(device)
    assert no_drop(test_waveform, lengths).allclose(test_waveform)
    no_drop = DropChunk(drop_count_low=0, drop_count_high=0).to(device)
    assert no_drop(test_waveform, lengths).allclose(test_waveform)
    no_drop = DropChunk(drop_start=0, drop_end=0).to(device)
    assert no_drop(test_waveform, lengths).allclose(test_waveform)

    # Specify all parameters to ensure it is deterministic
    dropper = DropChunk(
        drop_length_low=100,
        drop_length_high=100,
        drop_count_low=1,
        drop_count_high=1,
        drop_start=100,
        drop_end=200,
        noise_factor=0.0,
    ).to(device)
    expected_waveform = test_waveform.clone()
    expected_waveform[:, 100:200] = 0.0

    assert dropper(test_waveform, lengths).allclose(expected_waveform)

    # Make sure amplitude is similar before and after
    dropper = DropChunk(noise_factor=1.0).to(device)
    drop_amplitude = dropper(test_waveform, lengths).abs().mean()
    orig_amplitude = test_waveform.abs().mean()
    assert drop_amplitude.allclose(orig_amplitude, atol=1e-2)


def test_fast_drop_chunk():
    from speechbrain.augment.time_domain import FastDropChunk

    test_waveform = torch.ones([8, 200, 12])

    # Edge cases
    no_drop = FastDropChunk(drop_length_low=0, drop_length_high=0)
    assert no_drop(test_waveform).allclose(test_waveform)
    no_drop = FastDropChunk(drop_count_low=0, drop_count_high=0)
    assert no_drop(test_waveform).allclose(test_waveform)
    no_drop = FastDropChunk(drop_start=0, drop_end=0)
    assert no_drop(test_waveform).allclose(test_waveform)


def test_clip(device):
    from speechbrain.augment.time_domain import DoClip

    test_waveform = torch.sin(torch.arange(16000.0, device=device)).unsqueeze(0)

    # Edge cases
    no_clip = DoClip(clip_low=1, clip_high=1).to(device)
    assert no_clip(test_waveform).allclose(test_waveform)

    # Sort of a reimplementation of clipping, but its one function call.
    expected = 2 * test_waveform.clamp(min=-0.5, max=0.5)
    half_clip = DoClip(clip_low=0.5, clip_high=0.5).to(device)
    assert half_clip(test_waveform).allclose(expected)


def test_rand_amp():
    from speechbrain.augment.time_domain import RandAmp

    rand_amp = RandAmp(amp_low=0, amp_high=0)
    signal = torch.rand(4, 500)
    output = rand_amp(signal)
    assert output.mean().mean(0) == 0


def test_channel_drop():
    from speechbrain.augment.time_domain import ChannelDrop

    signal = torch.rand(4, 256, 8)
    ch_drop = ChannelDrop(drop_rate=0.5)
    output = ch_drop(signal)
    assert signal.shape == output.shape

    signal = torch.rand(4, 256, 8)
    ch_drop = ChannelDrop(drop_rate=0.0)
    output = ch_drop(signal)
    assert torch.equal(signal, output)

    signal = torch.rand(4, 256, 8)
    ch_drop = ChannelDrop(drop_rate=1.0)
    output = ch_drop(signal)
    assert torch.sum(output) == 0


def test_channel_swap():
    from speechbrain.augment.time_domain import ChannelSwap

    signal = torch.rand(4, 256, 8)
    ch_swap = ChannelSwap(min_swap=1, max_swap=5)
    output = ch_swap(signal)
    assert signal.shape == output.shape

    signal = torch.rand(4, 256, 8)
    ch_swap = ChannelSwap(min_swap=0, max_swap=0)
    output = ch_swap(signal)
    assert torch.equal(signal, output)


def test_rand_shift():
    from speechbrain.augment.freq_domain import RandomShift

    signal = torch.rand(4, 256, 8)
    lengths = torch.tensor([0.1, 0.2, 0.9, 1.0])
    rand_shift = RandomShift(min_shift=10, max_shift=50, dim=1)
    output, lengths = rand_shift(signal, lengths)
    assert signal.shape == output.shape
    assert torch.equal(signal, output) == 0

    signal = torch.rand(4, 256, 8)
    rand_shift = RandomShift(min_shift=1, max_shift=2, dim=2)
    output, lengths = rand_shift(signal, lengths)
    assert signal.shape == output.shape
    assert torch.equal(signal, output) == 0

    signal = torch.rand(4, 256)
    rand_shift = RandomShift(min_shift=10, max_shift=50, dim=1)
    output, lengths = rand_shift(signal, lengths)
    assert signal.shape == output.shape
    assert torch.equal(signal, output) == 0

    signal = torch.rand(4, 256, 8)
    rand_shift = RandomShift(min_shift=0, max_shift=0, dim=1)
    output, lengths = rand_shift(signal, lengths)
    assert torch.equal(signal, output)

    signal = torch.Tensor([1, 0, 0])
    rand_shift = RandomShift(min_shift=1, max_shift=1, dim=0)
    output, lengths = rand_shift(signal, lengths)
    assert torch.equal(output, torch.Tensor([0, 1, 0]))


def test_pink_noise():
    from speechbrain.augment.time_domain import pink_noise_like

    signal = torch.rand(4, 256)
    noise = pink_noise_like(signal)
    assert signal.shape == noise.shape

    signal = torch.rand(4, 256, 8)
    noise = pink_noise_like(signal)
    assert signal.shape == noise.shape

    signal = torch.rand(4, 257, 8)
    noise = pink_noise_like(signal)
    assert signal.shape == noise.shape

    noise_fft = torch.fft.fft(noise, dim=1)
    mean_first_fft_points = noise_fft.abs()[:, 0:10, :].mean()
    mean_last_fft_points = noise_fft.abs()[:, 118:128, :].mean()
    assert torch.all(mean_first_fft_points > mean_last_fft_points)

    # Test blue noise
    noise = pink_noise_like(signal, alpha_low=-1.0, alpha_high=-1.0)
    noise_fft = torch.fft.fft(noise, dim=1)
    mean_first_fft_points = noise_fft.abs()[:, 0:10, :].mean()
    mean_last_fft_points = noise_fft.abs()[:, 118:128, :].mean()
    assert torch.all(mean_first_fft_points < mean_last_fft_points)


def test_sign_flip():
    from speechbrain.augment.time_domain import SignFlip

    signal = torch.rand(4, 500)
    flip_sign = SignFlip(flip_prob=0)
    assert torch.all(flip_sign(signal) > 0)

    signal = torch.rand(4, 500)
    flip_sign = SignFlip(flip_prob=1)
    assert torch.all(flip_sign(signal) < 0)

    signal = torch.rand(4, 500)
    flip_sign = SignFlip(flip_prob=0.5)
    flips = 0
    trials = 1000
    for _ in range(trials):
        flipped_sig = flip_sign(signal)
        if torch.all(flipped_sig == -signal):
            flips += 1
    test_prob = flips / trials
    # these values are 6 stds in each direction,
    # making a false negative extremely unlikely
    assert 0.405 < test_prob < 0.595


def test_SpectrogramDrop():
    from speechbrain.augment.freq_domain import SpectrogramDrop

    spectrogram = torch.rand(4, 100, 40)
    mean = spectrogram.mean()
    drop = SpectrogramDrop(
        drop_length_low=1,
        drop_length_high=15,
        drop_count_low=1,
        drop_count_high=3,
        replace="zeros",
        dim=1,
    )
    output = drop(spectrogram)
    assert mean > output.mean()
    assert spectrogram.shape == output.shape
    from speechbrain.augment.freq_domain import SpectrogramDrop

    spectrogram = torch.rand(4, 100, 40)
    mean = spectrogram.mean()
    drop = SpectrogramDrop(
        drop_length_low=0,
        drop_length_high=1,
        drop_count_low=3,
        drop_count_high=3,
        replace="zeros",
        dim=1,
    )
    output = drop(spectrogram)
    print(output)
    assert torch.allclose(mean, output.mean())
    assert spectrogram.shape == output.shape

    # NOTE: we're testing drop_length_high=1 above and drop_count_high=0 here
    # because one +1 the upper bound and the other doesn't for the high
    # exclusive range... see #2542
    spectrogram = torch.rand(4, 100, 40)
    mean = spectrogram.mean()
    drop = SpectrogramDrop(
        drop_length_low=1,
        drop_length_high=15,
        drop_count_low=0,
        drop_count_high=0,
        replace="zeros",
        dim=1,
    )
    output = drop(spectrogram)
    assert torch.allclose(mean, output.mean())
    assert spectrogram.shape == output.shape

    from speechbrain.augment.freq_domain import SpectrogramDrop

    spectrogram = torch.rand(4, 100, 40)
    mean = spectrogram.mean()
    drop = SpectrogramDrop(
        drop_length_low=1,
        drop_length_high=15,
        drop_count_low=1,
        drop_count_high=3,
        replace="zeros",
        dim=2,
    )
    output = drop(spectrogram)
    assert mean > output.mean()
    assert spectrogram.shape == output.shape

    from speechbrain.augment.freq_domain import SpectrogramDrop

    spectrogram = torch.rand(4, 100, 40)
    drop = SpectrogramDrop(
        drop_length_low=1,
        drop_length_high=15,
        drop_count_low=1,
        drop_count_high=3,
        replace="mean",
        dim=1,
    )
    output = drop(spectrogram.clone())
    assert spectrogram.shape == output.shape
    assert not torch.equal(spectrogram, output)

    from speechbrain.augment.freq_domain import SpectrogramDrop

    spectrogram = torch.rand(4, 100, 40)
    drop = SpectrogramDrop(
        drop_length_low=1,
        drop_length_high=15,
        drop_count_low=1,
        drop_count_high=3,
        replace="mean",
        dim=2,
    )
    output = drop(spectrogram.clone())
    assert spectrogram.shape == output.shape
    assert not torch.equal(spectrogram, output)

    from speechbrain.augment.freq_domain import SpectrogramDrop

    spectrogram = torch.rand(4, 100, 40)
    drop = SpectrogramDrop(
        drop_length_low=1,
        drop_length_high=15,
        drop_count_low=1,
        drop_count_high=3,
        replace="cutcat",
        dim=1,
    )
    output = drop(spectrogram.clone())
    assert spectrogram.shape == output.shape
    assert not torch.equal(spectrogram, output)

    from speechbrain.augment.freq_domain import SpectrogramDrop

    spectrogram = torch.rand(4, 100, 40)
    drop = SpectrogramDrop(
        drop_length_low=1,
        drop_length_high=15,
        drop_count_low=1,
        drop_count_high=3,
        replace="cutcat",
        dim=2,
    )
    output = drop(spectrogram.clone())
    assert spectrogram.shape == output.shape
    assert not torch.equal(spectrogram, output)

    from speechbrain.augment.freq_domain import SpectrogramDrop

    spectrogram = torch.rand(4, 100, 40)
    drop = SpectrogramDrop(
        drop_length_low=1,
        drop_length_high=15,
        drop_count_low=1,
        drop_count_high=3,
        replace="swap",
        dim=1,
    )
    output = drop(spectrogram.clone())
    assert spectrogram.shape == output.shape
    assert not torch.equal(spectrogram, output)

    from speechbrain.augment.freq_domain import SpectrogramDrop

    spectrogram = torch.rand(4, 100, 40)
    drop = SpectrogramDrop(
        drop_length_low=1,
        drop_length_high=15,
        drop_count_low=1,
        drop_count_high=3,
        replace="swap",
        dim=2,
    )
    output = drop(spectrogram.clone())
    assert spectrogram.shape == output.shape
    assert not torch.equal(spectrogram, output)
    assert torch.allclose(spectrogram.mean(), output.mean())

    # Important: understand why sometimes with random selection, spectrogram and output are the same....
    from speechbrain.augment.freq_domain import SpectrogramDrop

    spectrogram = torch.rand(4, 100, 40)
    drop = SpectrogramDrop(
        drop_length_low=1,
        drop_length_high=15,
        drop_count_low=1,
        drop_count_high=3,
        replace="random_selection",
        dim=1,
    )
    output = drop(spectrogram.clone())
    assert spectrogram.shape == output.shape
    assert not torch.equal(spectrogram, output)

    from speechbrain.augment.freq_domain import SpectrogramDrop

    spectrogram = torch.rand(4, 100, 40)
    drop = SpectrogramDrop(
        drop_length_low=1,
        drop_length_high=15,
        drop_count_low=1,
        drop_count_high=3,
        replace="random_selection",
        dim=2,
    )
    output = drop(spectrogram.clone())
    assert spectrogram.shape == output.shape
    assert not torch.equal(spectrogram, output)

    from speechbrain.augment.codec import CodecAugment

    if "ffmpeg" in torchaudio.list_audio_backends():
        waveform = torch.rand(4, 16000)
        augmenter = CodecAugment(16000)
        output_waveform = augmenter(waveform)
        assert not torch.allclose(waveform, output_waveform)

    from speechbrain.augment.time_domain import DropBitResolution

    dropper = DropBitResolution()
    signal = torch.rand(4, 16000)
    signal_dropped = dropper(signal)
    assert not torch.equal(signal, signal_dropped)
    assert signal.shape == signal_dropped.shape

    from speechbrain.augment.time_domain import DropBitResolution

    dropper = DropBitResolution(target_dtype="int8")
    signal = torch.rand(4, 16000)
    signal_dropped = dropper(signal)
    assert not torch.equal(signal, signal_dropped)
    assert signal.shape == signal_dropped.shape

    from speechbrain.augment.time_domain import DropBitResolution

    dropper = DropBitResolution(target_dtype="int16")
    signal = torch.rand(4, 16000)
    signal_dropped = dropper(signal)
    assert not torch.equal(signal, signal_dropped)
    assert signal.shape == signal_dropped.shape

    from speechbrain.augment.time_domain import DropBitResolution

    dropper = DropBitResolution(target_dtype="float16")
    signal = torch.rand(4, 16000)
    signal_dropped = dropper(signal)
    assert not torch.equal(signal, signal_dropped)
    assert signal.shape == signal_dropped.shape


def test_augment_pipeline():
    from speechbrain.augment.augmenter import Augmenter
    from speechbrain.augment.time_domain import DropChunk, DropFreq

    freq_dropper = DropFreq()
    chunk_dropper = DropChunk(drop_start=100, drop_end=16000, noise_factor=0)
    augment = Augmenter(
        parallel_augment=False,
        concat_original=False,
        min_augmentations=2,
        max_augmentations=2,
        augmentations=[freq_dropper, chunk_dropper],
    )
    signal = torch.rand([4, 16000])
    output_signal, lengths = augment(
        signal, lengths=torch.tensor([0.2, 0.5, 0.7, 1.0])
    )
    assert len(output_signal) == 4
    assert len(lengths) == 4

    freq_dropper = DropFreq()
    chunk_dropper = DropChunk(drop_start=100, drop_end=16000, noise_factor=0)
    augment = Augmenter(
        parallel_augment=True,
        concat_original=True,
        min_augmentations=1,
        max_augmentations=2,
        augment_prob=0,
        augmentations=[freq_dropper, chunk_dropper],
    )
    signal = torch.rand([4, 16000])
    output_signal, lengths = augment(
        signal, lengths=torch.tensor([0.2, 0.5, 0.7, 1.0])
    )
    assert torch.equal(signal, output_signal)

    freq_dropper = DropFreq()
    chunk_dropper = DropChunk(drop_start=100, drop_end=16000, noise_factor=0)
    augment = Augmenter(
        parallel_augment=True,
        concat_original=True,
        min_augmentations=1,
        max_augmentations=2,
        augment_prob=1.0,
        augmentations=[freq_dropper, chunk_dropper],
        enable_augmentations=[False, False],
    )
    signal = torch.rand([4, 16000])
    output_signal, lengths = augment(
        signal, lengths=torch.tensor([0.2, 0.5, 0.7, 1.0])
    )
    assert torch.equal(signal, output_signal)

    freq_dropper = DropFreq()
    chunk_dropper = DropChunk(drop_start=100, drop_end=16000, noise_factor=0)
    augment = Augmenter(
        parallel_augment=True,
        concat_original=True,
        min_augmentations=2,
        max_augmentations=2,
        augment_prob=1.0,
        augmentations=[freq_dropper, chunk_dropper],
        enable_augmentations=[True, False],
    )
    signal = torch.rand([4, 16000])
    output_signal, lengths = augment(
        signal, lengths=torch.tensor([0.2, 0.5, 0.7, 1.0])
    )
    assert output_signal.shape[0] == signal.shape[0] * 2

    augment = Augmenter(
        parallel_augment=False,
        concat_original=True,
        min_augmentations=2,
        max_augmentations=2,
        augmentations=[freq_dropper, chunk_dropper],
    )
    output_signal, lengths = augment(
        signal, lengths=torch.tensor([0.2, 0.5, 0.7, 1.0])
    )
    assert len(output_signal) == 8
    assert len(lengths) == 8
    assert torch.equal(output_signal[0:4], signal[0:4])

    augment = Augmenter(
        parallel_augment=True,
        concat_original=False,
        min_augmentations=2,
        max_augmentations=2,
        augmentations=[freq_dropper, chunk_dropper],
    )
    output_signal, lengths = augment(
        signal, lengths=torch.tensor([0.2, 0.5, 0.7, 1.0])
    )
    assert len(output_signal) == 8
    assert len(lengths) == 8

    augment = Augmenter(
        parallel_augment=True,
        concat_original=True,
        min_augmentations=2,
        max_augmentations=2,
        augmentations=[freq_dropper, chunk_dropper],
    )
    output_signal, lengths = augment(
        signal, lengths=torch.tensor([0.2, 0.5, 0.7, 1.0])
    )
    assert len(output_signal) == 12
    assert len(lengths) == 12
    assert torch.equal(output_signal[0:4], signal[0:4])

    augment = Augmenter(
        parallel_augment=True,
        concat_original=True,
        min_augmentations=2,
        max_augmentations=2,
        repeat_augment=2,
        shuffle_augmentations=True,
        augmentations=[freq_dropper, chunk_dropper],
    )

    output_signal, lengths = augment(
        signal, lengths=torch.tensor([0.2, 0.5, 0.7, 1.0])
    )
    assert len(output_signal) == 20
    assert len(lengths) == 20
    assert torch.equal(output_signal[0:4], signal[0:4])

    augment = Augmenter(
        parallel_augment=True,
        concat_original=True,
        min_augmentations=0,
        max_augmentations=0,
        repeat_augment=2,
        shuffle_augmentations=True,
        augmentations=[freq_dropper, chunk_dropper],
    )

    output_signal, lengths = augment(
        signal, lengths=torch.tensor([0.2, 0.5, 0.7, 1.0])
    )
    assert torch.equal(output_signal, signal)

    augment = Augmenter(
        parallel_augment=True,
        concat_original=True,
        min_augmentations=1,
        max_augmentations=2,
        repeat_augment=0,
        shuffle_augmentations=True,
        augmentations=[freq_dropper, chunk_dropper],
    )

    output_signal, lengths = augment(
        signal, lengths=torch.tensor([0.2, 0.5, 0.7, 1.0])
    )
    assert torch.equal(output_signal, signal)
