"""Unit tests for the VAD inference interface."""

import pytest
import torch


@pytest.mark.parametrize(
    "boundaries, expected",
    [
        pytest.param(
            torch.tensor([[0.2, 0.5]]),
            [
                "segment_001  0.00  0.20 NON_SPEECH",
                "segment_002  0.20  0.50 SPEECH",
                "segment_003  0.50  1.00 NON_SPEECH",
            ],
            id="seconds",
        ),
        pytest.param(
            torch.tensor([[3200, 8000]], dtype=torch.int),
            [
                "segment_001  0 3200NON_SPEECH",
                "segment_002  3200 8000SPEECH",
                "segment_003  8000 16000NON_SPEECH",
            ],
            id="samples",
        ),
        pytest.param(
            torch.zeros(0, 2),
            ["segment_001  0.00  1.00 NON_SPEECH"],
            id="no_speech_seconds",
        ),
        pytest.param(
            torch.zeros(0, 2, dtype=torch.int),
            ["segment_001  0 16000NON_SPEECH"],
            id="no_speech_samples",
        ),
        pytest.param(
            torch.tensor([]),
            ["segment_001  0.00  1.00 NON_SPEECH"],
            id="no_speech_get_speech_segments",
        ),
    ],
)
def test_save_boundaries(boundaries, expected, tmp_path, capsys, device):
    from speechbrain.dataio import audio_io
    from speechbrain.inference.VAD import VAD

    # save_boundaries does not run the model, so placeholder modules suffice
    vad = VAD(
        modules={
            "compute_features": torch.nn.Identity(),
            "mean_var_norm": torch.nn.Identity(),
            "model": torch.nn.Identity(),
        },
        hparams={
            "sample_rate": 16000,
            "time_resolution": 0.01,
            "device": device,
        },
    )
    audio_file = tmp_path / "audio.wav"
    audio_io.save(str(audio_file), torch.zeros(1, 16000), 16000)
    save_path = tmp_path / "boundaries.txt"

    vad.save_boundaries(
        boundaries.to(device), save_path=save_path, audio_file=audio_file
    )

    assert save_path.read_text(encoding="utf-8").splitlines() == expected
    assert capsys.readouterr().out.splitlines() == expected
