import csv
import os
from speechbrain.dataio.dataio import load_data_csv
import tempfile
import torch


def test_read_audio(tmpdir):
    from speechbrain.dataio.dataio import read_audio, write_audio

    test_waveform = torch.rand(16000)
    wavfile = os.path.join(tmpdir, "wave.wav")
    write_audio(wavfile, test_waveform, 16000)

    # dummy annotation
    for i in range(3):
        start = torch.randint(0, 8000, (1,)).item()
        stop = start + torch.randint(500, 1000, (1,)).item()
        wav_obj = {"wav": {"file": wavfile, "start": start, "stop": stop}}
        loaded = read_audio(wav_obj["wav"])
        assert loaded.allclose(test_waveform[start:stop], atol=1e-4)
        # set to equal when switching to the sox_io backend
        # assert torch.all(torch.eq(loaded, test_waveform[0, start:stop]))


def test_read_audio_multichannel(tmpdir):
    from speechbrain.dataio.dataio import read_audio_multichannel, write_audio

    test_waveform = torch.rand(16000, 2)
    wavfile = os.path.join(tmpdir, "wave.wav")
    # sf.write(wavfile, test_waveform, 16000, subtype="float")
    write_audio(wavfile, test_waveform, 16000)

    # dummy annotation we save and load one multichannel file
    for i in range(2):
        start = torch.randint(0, 8000, (1,)).item()
        stop = start + torch.randint(500, 1000, (1,)).item()

        wav_obj = {"wav": {"files": [wavfile], "start": start, "stop": stop}}

        loaded = read_audio_multichannel(wav_obj["wav"])
        assert loaded.allclose(test_waveform[start:stop, :], atol=1e-4)
        # set to equal when switching to the sox_io backend
        # assert torch.all(torch.eq(loaded, test_waveform[:,start:stop]))

    # we test now multiple files loading
    test_waveform_2 = torch.rand(16000, 2)
    wavfile_2 = os.path.join(tmpdir, "wave_2.wav")
    write_audio(wavfile_2, test_waveform_2, 16000)
    # sf.write(wavfile_2, test_waveform_2, 16000, subtype="float")

    for i in range(2):
        start = torch.randint(0, 8000, (1,)).item()
        stop = start + torch.randint(500, 1000, (1,)).item()

        wav_obj = {
            "wav": {"files": [wavfile, wavfile_2], "start": start, "stop": stop}
        }

        loaded = read_audio_multichannel(wav_obj["wav"])
        test_waveform3 = torch.cat(
            (test_waveform[start:stop, :], test_waveform_2[start:stop, :]), 1
        )
        assert loaded.allclose(test_waveform3, atol=1e-4)
        # set to equal when switching to the sox_io backend
        # assert torch.all(
        #     torch.eq(
        #         loaded,
        #         torch.cat(
        #             (test_waveform[:,start:stop], test_waveform_2[:,start:stop]), 0
        #         ),
        #     )
        # )


def test_load_csv():
    with tempfile.NamedTemporaryFile("w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["ID", "raw", "dynamic"])
        writer.writerow(["1", "It costs $100", "foo=$foo"])
        writer.writerow(["2", "1$1$1$1", "value=$foo!"])
        csv_file.flush()
        data = load_data_csv(
            csv_file.name, replacements={"foo": "bar"}, raw_keys=["raw"]
        )
        assert data["1"]["raw"] == "It costs $100"
        assert data["1"]["dynamic"] == "foo=bar"
        assert data["2"]["raw"] == "1$1$1$1"
        assert data["2"]["dynamic"] == "value=bar!"
