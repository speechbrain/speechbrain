import os
import torch
from speechbrain.dataio.dataio import write_audio


def test_add_noise(tmpdir, device):
    from speechbrain.processing.speech_augmentation import AddNoise

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
    with open(csv, "w") as w:
        w.write("ID, duration, wav, wav_format, wav_opts\n")
        w.write(f"1, 1.0, {noisefile}, wav,\n")

    # Edge cases
    no_noise = AddNoise(mix_prob=0.0).to(device)
    assert no_noise(test_waveform, wav_lens).allclose(test_waveform)
    no_noise = AddNoise(snr_low=1000, snr_high=1000)
    assert no_noise(test_waveform, wav_lens).allclose(test_waveform)
    all_noise = AddNoise(csv_file=csv, snr_low=-1000, snr_high=-1000)
    assert all_noise(test_waveform, wav_lens).allclose(test_noise, atol=1e-4)

    # Basic 0dB case
    add_noise = AddNoise(csv_file=csv).to(device)
    expected = (test_waveform + test_noise) / 2
    assert add_noise(test_waveform, wav_lens).allclose(expected, atol=1e-4)


def test_add_reverb(tmpdir, device):
    from speechbrain.processing.speech_augmentation import AddReverb

    test_waveform = torch.sin(torch.arange(16000.0, device=device)).unsqueeze(0)
    impulse_response = torch.zeros(1, 8000, device=device)
    impulse_response[0, 0] = 1.0
    wav_lens = torch.ones(1, device=device)

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
    with open(csv, "w") as w:
        w.write("ID, duration, wav, wav_format, wav_opts\n")
        w.write(f"1, 0.5, {ir1}, wav,\n")
        w.write(f"2, 0.5, {ir2}, wav,\n")
        w.write(f"3, 0.5, {ir3}, wav,\n")

    # Edge case
    no_reverb = AddReverb(csv, reverb_prob=0.0).to(device)
    assert no_reverb(test_waveform, wav_lens).allclose(test_waveform)

    # Normal cases
    add_reverb = AddReverb(csv, sorting="original")
    reverbed = add_reverb(test_waveform, wav_lens)[:, 0:1000]
    assert reverbed.allclose(test_waveform[:, 0:1000], atol=1e-1)
    reverbed = add_reverb(test_waveform, wav_lens)[:, 0:1000]
    assert reverbed.allclose(test_waveform[:, 0:1000], atol=1e-1)
    reverbed = add_reverb(test_waveform, wav_lens)[:, 0:1000]
    assert reverbed.allclose(ir3_result[:, 0:1000], atol=2e-1)


def test_speed_perturb(device):
    from speechbrain.processing.speech_augmentation import SpeedPerturb

    test_waveform = torch.sin(torch.arange(16000.0, device=device)).unsqueeze(0)

    # Edge cases
    no_perturb = SpeedPerturb(16000, perturb_prob=0.0).to(device)
    assert no_perturb(test_waveform).allclose(test_waveform)
    no_perturb = SpeedPerturb(16000, speeds=[100]).to(device)
    assert no_perturb(test_waveform).allclose(test_waveform)

    # Half speed
    half_speed = SpeedPerturb(16000, speeds=[50]).to(device)
    assert half_speed(test_waveform).allclose(test_waveform[:, ::2], atol=3e-1)


def test_babble(device):
    from speechbrain.processing.speech_augmentation import AddBabble

    test_waveform = torch.stack(
        (
            torch.sin(torch.arange(16000.0, device=device)),
            torch.cos(torch.arange(16000.0, device=device)),
        )
    )
    lengths = torch.ones(2, device=device)

    # Edge cases
    no_babble = AddBabble(mix_prob=0.0).to(device)
    assert no_babble(test_waveform, lengths).allclose(test_waveform)
    no_babble = AddBabble(speaker_count=1, snr_low=1000, snr_high=1000)
    assert no_babble(test_waveform, lengths).allclose(test_waveform)

    # One babbler just averages the two speakers
    babble = AddBabble(speaker_count=1).to(device)
    expected = (test_waveform + test_waveform.roll(1, 0)) / 2
    assert babble(test_waveform, lengths).allclose(expected, atol=1e-4)


def test_drop_freq(device):
    from speechbrain.processing.speech_augmentation import DropFreq

    test_waveform = torch.sin(torch.arange(16000.0, device=device)).unsqueeze(0)

    # Edge cases
    no_drop = DropFreq(drop_prob=0.0).to(device)
    assert no_drop(test_waveform).allclose(test_waveform)
    no_drop = DropFreq(drop_count_low=0, drop_count_high=0)
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
    from speechbrain.processing.speech_augmentation import DropChunk

    test_waveform = torch.sin(torch.arange(16000.0, device=device)).unsqueeze(0)
    lengths = torch.ones(1, device=device)

    # Edge cases
    no_drop = DropChunk(drop_prob=0.0).to(device)
    assert no_drop(test_waveform, lengths).allclose(test_waveform)
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
    from speechbrain.processing.speech_augmentation import FastDropChunk

    test_waveform = torch.ones([8, 200, 12])

    # Edge cases
    no_drop = FastDropChunk(drop_length_low=0, drop_length_high=0)
    assert no_drop(test_waveform).allclose(test_waveform)
    no_drop = FastDropChunk(drop_count_low=0, drop_count_high=0)
    assert no_drop(test_waveform).allclose(test_waveform)
    no_drop = FastDropChunk(drop_start=0, drop_end=0)
    assert no_drop(test_waveform).allclose(test_waveform)


def test_clip(device):
    from speechbrain.processing.speech_augmentation import DoClip

    test_waveform = torch.sin(torch.arange(16000.0, device=device)).unsqueeze(0)

    # Edge cases
    no_clip = DoClip(clip_prob=0.0).to(device)
    assert no_clip(test_waveform).allclose(test_waveform)
    no_clip = DoClip(clip_low=1, clip_high=1).to(device)
    assert no_clip(test_waveform).allclose(test_waveform)

    # Sort of a reimplementation of clipping, but its one function call.
    expected = test_waveform.clamp(min=-0.5, max=0.5)
    half_clip = DoClip(clip_low=0.5, clip_high=0.5).to(device)
    assert half_clip(test_waveform).allclose(expected)


def test_rand_amp():
    from speechbrain.processing.speech_augmentation import RandAmp

    rand_amp = RandAmp(amp_low=0, amp_high=0)
    signal = torch.rand(4, 500)
    output = rand_amp(signal)
    assert output.mean().mean(0) == 0

    rand_amp = RandAmp(amp_low=1, amp_high=1)
    signal = torch.rand(4, 500)
    output = rand_amp(signal)
    assert torch.equal(signal, output)

    rand_amp = RandAmp(amp_low=2, amp_high=2)
    signal = torch.rand(4, 500)
    output = rand_amp(signal)
    assert torch.equal(2 * signal, output)

    # Multi-channel waveform
    rand_amp = RandAmp(amp_low=2, amp_high=2)
    signal = torch.rand(4, 500, 3)
    output = rand_amp(signal)
    assert torch.equal(2 * signal, output)


def test_channel_drop():
    from speechbrain.processing.speech_augmentation import ChannelDrop

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
    from speechbrain.processing.speech_augmentation import ChannelSwap

    signal = torch.rand(4, 256, 8)
    ch_swap = ChannelSwap(min_swap=1, max_swap=5)
    output = ch_swap(signal)
    assert signal.shape == output.shape

    signal = torch.rand(4, 256, 8)
    ch_swap = ChannelSwap(min_swap=0, max_swap=0)
    output = ch_swap(signal)
    assert torch.equal(signal, output)


def test_rand_shift():
    from speechbrain.processing.speech_augmentation import RandomShift

    signal = torch.rand(4, 256, 8)
    rand_shift = RandomShift(min_shift=10, max_shift=50, dim=1)
    output = rand_shift(signal)
    assert signal.shape == output.shape
    assert torch.equal(signal, output) == 0

    signal = torch.rand(4, 256, 8)
    rand_shift = RandomShift(min_shift=1, max_shift=2, dim=2)
    output = rand_shift(signal)
    assert signal.shape == output.shape
    assert torch.equal(signal, output) == 0

    signal = torch.rand(4, 256)
    rand_shift = RandomShift(min_shift=10, max_shift=50, dim=1)
    output = rand_shift(signal)
    assert signal.shape == output.shape
    assert torch.equal(signal, output) == 0

    signal = torch.rand(4, 256, 8)
    rand_shift = RandomShift(min_shift=0, max_shift=0, dim=1)
    output = rand_shift(signal)
    assert torch.equal(signal, output)

    signal = torch.Tensor([1, 0, 0])
    rand_shift = RandomShift(min_shift=1, max_shift=1, dim=0)
    output = rand_shift(signal)
    assert torch.equal(output, torch.Tensor([0, 1, 0]))


def test_pink_noise():
    from speechbrain.processing.speech_augmentation import pink_noise_like

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


def muscular_noise():
    from speechbrain.processing.speech_augmentation import muscolar_noise

    signal = torch.rand(4, 256, 8)
    noise = muscolar_noise(signal)
    assert signal.shape == noise.shape

    signal = torch.rand(4, 256)
    noise = muscolar_noise(signal)
    assert signal.shape == noise.shape


def test_augment_pipeline():
    from speechbrain.processing.speech_augmentation import DropFreq, DropChunk
    from speechbrain.processing.augmentation import Augmenter

    freq_dropper = DropFreq()
    chunk_dropper = DropChunk(drop_start=100, drop_end=16000, noise_factor=0)
    augment = Augmenter(
        parallel_augment=False,
        concat_original=False,
        min_augmentations=2,
        max_augmentations=2,
        freq_drop=freq_dropper,
        chunk_dropper=chunk_dropper,
    )
    signal = torch.rand([4, 16000])
    output_signal, lenghts = augment(
        signal, lengths=torch.tensor([0.2, 0.5, 0.7, 1.0])
    )

    assert len(output_signal) == 4
    assert len(lenghts) == 4

    augment = Augmenter(
        parallel_augment=False,
        concat_original=True,
        min_augmentations=2,
        max_augmentations=2,
        freq_drop=freq_dropper,
        chunk_dropper=chunk_dropper,
    )
    output_signal, lenghts = augment(
        signal, lengths=torch.tensor([0.2, 0.5, 0.7, 1.0])
    )

    assert len(output_signal) == 8
    assert len(lenghts) == 8
    assert torch.equal(output_signal[0:4], signal[0:4])

    augment = Augmenter(
        parallel_augment=True,
        concat_original=False,
        min_augmentations=2,
        max_augmentations=2,
        freq_drop=freq_dropper,
        chunk_dropper=chunk_dropper,
    )
    output_signal, lenghts = augment(
        signal, lengths=torch.tensor([0.2, 0.5, 0.7, 1.0])
    )

    assert len(output_signal) == 8
    assert len(lenghts) == 8

    augment = Augmenter(
        parallel_augment=True,
        concat_original=True,
        min_augmentations=2,
        max_augmentations=2,
        freq_drop=freq_dropper,
        chunk_dropper=chunk_dropper,
    )
    output_signal, lenghts = augment(
        signal, lengths=torch.tensor([0.2, 0.5, 0.7, 1.0])
    )

    assert len(output_signal) == 12
    assert len(lenghts) == 12
    assert torch.equal(output_signal[0:4], signal[0:4])

    augment = Augmenter(
        parallel_augment=True,
        concat_original=True,
        min_augmentations=2,
        max_augmentations=2,
        repeat_augment=2,
        shuffle_augmentations=True,
        freq_drop=freq_dropper,
        chunk_dropper=chunk_dropper,
    )

    output_signal, lenghts = augment(
        signal, lengths=torch.tensor([0.2, 0.5, 0.7, 1.0])
    )

    assert len(output_signal) == 20
    assert len(lenghts) == 20
    assert torch.equal(output_signal[0:4], signal[0:4])

    augment = Augmenter(
        parallel_augment=True,
        concat_original=True,
        min_augmentations=0,
        max_augmentations=0,
        repeat_augment=2,
        shuffle_augmentations=True,
        freq_drop=freq_dropper,
        chunk_dropper=chunk_dropper,
    )

    output_signal, lenghts = augment(
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
        freq_drop=freq_dropper,
        chunk_dropper=chunk_dropper,
    )

    output_signal, lenghts = augment(
        signal, lengths=torch.tensor([0.2, 0.5, 0.7, 1.0])
    )

    assert torch.equal(output_signal, signal)
