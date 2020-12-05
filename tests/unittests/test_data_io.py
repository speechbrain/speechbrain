import torch
import soundfile as sf
import os


def test_read_audio(tmpdir):
    from speechbrain.data_io.data_io import read_audio

    test_waveform = torch.rand(16000)
    wavfile = os.path.join(tmpdir, "wave.wav")
    sf.write(wavfile, test_waveform, 16000, subtype="float")

    # dummy annotation
    for i in range(3):
        start = torch.randint(0, 8000, (1,)).item()
        stop = start + torch.randint(500, 1000, (1,)).item()

        wav_obj = {"wav": {"file": wavfile, "start": start, "stop": stop}}

        loaded = read_audio(wav_obj["wav"])
        assert torch.all(torch.eq(loaded, test_waveform[start:stop]))


def test_read_audio_multichannel(tmpdir):
    from speechbrain.data_io.data_io import read_audio_multichannel

    test_waveform = torch.rand(16000, 2)
    wavfile = os.path.join(tmpdir, "wave.wav")
    sf.write(wavfile, test_waveform, 16000, subtype="float")

    # dummy annotation we save and load one multichannel file
    for i in range(2):
        start = torch.randint(0, 8000, (1,)).item()
        stop = start + torch.randint(500, 1000, (1,)).item()

        wav_obj = {"wav": {"files": [wavfile], "start": start, "stop": stop}}

        loaded = read_audio_multichannel(wav_obj["wav"])
        assert torch.all(
            torch.eq(loaded.transpose(0, 1), test_waveform[start:stop])
        )

    # we test now multiple files loading
    test_waveform_2 = torch.rand(16000, 2)
    wavfile_2 = os.path.join(tmpdir, "wave_2.wav")
    sf.write(wavfile_2, test_waveform_2, 16000, subtype="float")

    for i in range(2):
        start = torch.randint(0, 8000, (1,)).item()
        stop = start + torch.randint(500, 1000, (1,)).item()

        wav_obj = {
            "wav": {"files": [wavfile, wavfile_2], "start": start, "stop": stop}
        }

        loaded = read_audio_multichannel(wav_obj["wav"])
        assert torch.all(
            torch.eq(
                loaded.transpose(0, 1),
                torch.cat(
                    (test_waveform[start:stop], test_waveform_2[start:stop]), 1
                ),
            )
        )
