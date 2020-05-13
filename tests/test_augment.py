import os
import torch
import soundfile as sf


def test_add_noise(tmpdir):
    from speechbrain.processing.speech_augmentation import AddNoise

    test_waveform = torch.sin(torch.arange(16000.0)).unsqueeze(0)
    test_noise = torch.cos(torch.arange(16000.0)).unsqueeze(0)
    equal_noisy = (test_waveform + test_noise) / 2
    wav_lens = torch.ones(1)
    assert ((test_waveform + test_noise) / 2).allclose(equal_noisy)

    # Put noise waveform into temporary file
    noisefile = os.path.join(tmpdir, "noise.wav")
    sf.write(noisefile, test_noise.squeeze(0).numpy(), 16000)
    csv = os.path.join(tmpdir, "noise.csv")
    with open(csv, "w") as w:
        w.write("ID, duration, wav, wav_format, wav_opts\n")
        w.write(f"1, 1.0, {noisefile}, wav,\n")

    # Edge cases
    add_noise = AddNoise(mix_prob=0.0)
    assert add_noise(test_waveform, wav_lens).allclose(test_waveform)
    add_noise = AddNoise(snr_low=1000, snr_high=1000)
    assert add_noise(test_waveform, wav_lens).allclose(test_waveform)
    add_noise = AddNoise(csv_file=csv, snr_low=-1000, snr_high=-1000)
    assert add_noise(test_waveform, wav_lens).allclose(test_noise, atol=1e-4)

    # Basic 0dB case
    add_noise = AddNoise(csv_file=csv)
    assert add_noise(test_waveform, wav_lens).allclose(equal_noisy, atol=1e-4)


def test_add_reverb(tmpdir):
    from speechbrain.processing.speech_augmentation import AddReverb

    test_waveform = torch.sin(torch.arange(16000.0)).unsqueeze(0)
    impulse_response = torch.zeros(1, 8000)
    impulse_response[0, 0] = 1.0
    wav_lens = torch.ones(1)

    # Put ir waveform into temporary file
    ir1 = os.path.join(tmpdir, "ir1.wav")
    ir2 = os.path.join(tmpdir, "ir2.wav")
    ir3 = os.path.join(tmpdir, "ir3.wav")
    sf.write(ir1, impulse_response.squeeze(0).numpy(), 16000)
    impulse_response[0, 0] = 0.0
    impulse_response[0, 10] = 1.0
    sf.write(ir2, impulse_response.squeeze(0).numpy(), 16000)
    impulse_response[0, 10] = 0.6
    impulse_response[0, 11] = 0.4
    sf.write(ir3, impulse_response.squeeze(0).numpy(), 16000)
    old_mean = test_waveform.mean()
    ir3_result = test_waveform * 0.6 + test_waveform.roll(1, -1) * 0.4
    new_mean = ir3_result.mean()
    ir3_result *= old_mean / new_mean

    # write ir csv file
    csv = os.path.join(tmpdir, "ir.csv")
    with open(csv, "w") as w:
        w.write("ID, duration, wav, wav_format, wav_opts\n")
        w.write(f"1, 0.5, {ir1}, wav,\n")
        w.write(f"2, 0.5, {ir2}, wav,\n")
        w.write(f"3, 0.5, {ir3}, wav,\n")

    # Edge case
    add_reverb = AddReverb(csv, reverb_prob=0.0)
    assert add_reverb(test_waveform, wav_lens).allclose(test_waveform)

    # Normal cases
    add_reverb = AddReverb(csv, order="original")
    reverbed = add_reverb(test_waveform, wav_lens)[:, 0:1000]
    assert reverbed.allclose(test_waveform[:, 0:1000], atol=1e-1)
    reverbed = add_reverb(test_waveform, wav_lens)[:, 0:1000]
    assert reverbed.allclose(test_waveform[:, 0:1000], atol=1e-1)
    reverbed = add_reverb(test_waveform, wav_lens)[:, 0:1000]
    assert reverbed.allclose(ir3_result[:, 0:1000], atol=2e-1)
