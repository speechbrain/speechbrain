"""Tests for the VAD double-check of speech segments that are very long."""

import pytest
import torch

from speechbrain.dataio import audio_io
from speechbrain.inference.VAD import VAD

SAMPLE_RATE = 16000
HOP = 160  # 10 ms, the time resolution of the fake model below
MAX_FRAMES = 300  # longest sequence the fake RNN accepts (cuDNN stand-in)


class FrameFeatures(torch.nn.Module):
    """Cuts the waveform in non-overlapping frames, [batch, frames, HOP]."""

    def forward(self, wavs):
        return wavs.unfold(-1, HOP, HOP)


class PassThroughNorm(torch.nn.Module):
    """Stands in for the normalization, which does not depend on the model."""

    def forward(self, feats, lens):
        return feats


class AddChannelAxis(torch.nn.Module):
    """Stands in for the CNN: [batch, time, feats] -> [batch, time, 1, feats]."""

    def forward(self, feats):
        return feats.unsqueeze(2)


class LimitedRNN(torch.nn.Module):
    """A stateless 'RNN' that fails on long sequences, like cuDNN does."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(HOP, 8)

    def forward(self, feats):
        if feats.shape[1] > MAX_FRAMES:
            raise RuntimeError("cuDNN error: CUDNN_STATUS_NOT_SUPPORTED")
        return torch.tanh(self.lin(feats)), None


@pytest.fixture
def vad():
    torch.manual_seed(0)
    modules = {
        "compute_features": FrameFeatures(),
        "mean_var_norm": PassThroughNorm(),
        "model": torch.nn.Identity(),
        "cnn": AddChannelAxis(),
        "rnn": LimitedRNN(),
        "dnn": torch.nn.Linear(8, 1),
    }
    hparams = {
        "sample_rate": SAMPLE_RATE,
        "time_resolution": 0.01,
        "device": "cpu",
    }
    return VAD(modules=modules, hparams=hparams)


def make_audio(tmp_path, n_frames):
    path = tmp_path / "audio.wav"
    wav = torch.rand(1, n_frames * HOP) * 2 - 1
    audio_io.save(str(path), wav, SAMPLE_RATE)
    return path


def test_bounded_prob_matches_unchunked_when_input_is_short(vad):
    wavs = torch.rand(2, MAX_FRAMES * HOP) * 2 - 1
    expected = vad.get_speech_prob_chunk(wavs)
    probs = vad._get_speech_prob_bounded(wavs, MAX_FRAMES * HOP)
    assert torch.equal(probs, expected)


def test_bounded_prob_chunks_long_input(vad):
    n_frames = 5 * MAX_FRAMES
    wavs = torch.rand(1, n_frames * HOP) * 2 - 1
    with pytest.raises(RuntimeError, match="cuDNN"):
        vad.get_speech_prob_chunk(wavs)

    probs = vad._get_speech_prob_bounded(wavs, MAX_FRAMES * HOP)

    # The fake model is stateless, so chunking must not change the result
    expected = vad.get_speech_prob_chunk(wavs[:, : MAX_FRAMES * HOP])
    assert probs.shape == (1, n_frames, 1)
    assert torch.allclose(probs[:, :MAX_FRAMES], expected)


def test_double_check_calls_model_once_on_short_segment(vad, tmp_path):
    audio = make_audio(tmp_path, n_frames=MAX_FRAMES)
    seen = []
    forward = vad.get_speech_prob_chunk
    vad.get_speech_prob_chunk = lambda w, *a: (
        seen.append(w.shape[-1]) or forward(w, *a)
    )
    boundaries = torch.tensor([[0.0, MAX_FRAMES * 0.01]])

    kept = vad.double_check_speech_segments(
        boundaries, audio, speech_th=0.0, max_chunk_size=MAX_FRAMES * 0.01
    )

    assert seen == [MAX_FRAMES * HOP]
    assert torch.equal(kept, boundaries)


def test_double_check_chunks_long_segment(vad, tmp_path):
    n_frames = 5 * MAX_FRAMES + 7
    audio = make_audio(tmp_path, n_frames=n_frames)
    seen = []
    forward = vad.get_speech_prob_chunk
    vad.get_speech_prob_chunk = lambda w, *a: (
        seen.append(w.shape[-1]) or forward(w, *a)
    )
    boundaries = torch.tensor([[0.0, n_frames * 0.01]])

    kept = vad.double_check_speech_segments(
        boundaries, audio, speech_th=0.0, max_chunk_size=MAX_FRAMES * 0.01
    )

    # Every sample is processed once, in chunks the model can handle
    assert len(seen) == 6
    assert sum(seen) == n_frames * HOP
    assert max(seen) <= MAX_FRAMES * HOP
    assert torch.equal(kept, boundaries)

    # The segment is still rejected when the threshold is not reached
    rejected = vad.double_check_speech_segments(
        boundaries, audio, speech_th=1.0, max_chunk_size=MAX_FRAMES * 0.01
    )
    assert rejected.numel() == 0
