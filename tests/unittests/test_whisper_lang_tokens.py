"""Tests of the language tokens of the Whisper searchers and WhisperASR.

Authors
 * Othman Istaiteh 2026
"""

from types import SimpleNamespace

import pytest
import torch

BOS, EN, FR = 50258, 50259, 50265


def _fake_whisper():
    """Only the attributes read by the Whisper searchers' constructors."""
    return SimpleNamespace(
        bos=BOS,
        eos=50257,
        model=SimpleNamespace(
            decoder=SimpleNamespace(config=SimpleNamespace(max_length=448))
        ),
        tokenizer=SimpleNamespace(prefix_tokens=[BOS, EN, 50359, 50363]),
    )


@pytest.mark.parametrize("beam_size", [1, 3])
def test_whisper_searcher_lang_tokens(beam_size, device):
    """The language tokens of each utterance are used for all its beams,
    and only for the next decoding."""
    from speechbrain.decoders.seq2seq import (
        S2SWhisperBeamSearcher,
        S2SWhisperGreedySearcher,
    )

    ratios = {"min_decode_ratio": 0.0, "max_decode_ratio": 1.0}
    if beam_size == 1:
        searcher = S2SWhisperGreedySearcher(_fake_whisper(), **ratios)
    else:
        searcher = S2SWhisperBeamSearcher(
            [_fake_whisper()], beam_size=beam_size, **ratios
        )

    searcher.set_lang_tokens(torch.tensor([EN, FR], device=device))
    mem = searcher.reset_mem(2 * beam_size, device=device)
    expected = torch.tensor([EN] * beam_size + [FR] * beam_size)
    assert torch.equal(mem[:, 1].cpu(), expected)
    assert searcher.lang_tokens is None

    # The next batch, of another size, falls back to the default language.
    mem = searcher.reset_mem(beam_size, device=device)
    assert (mem[:, 1] == EN).all()


@pytest.mark.parametrize("task", ["transcribe", "lang_id"])
def test_whisper_asr_detect_language(task):
    """The detected languages are given to the searcher only when they are
    used for a decoding, so that "lang_id" leaves no tokens behind."""
    from speechbrain.inference.ASR import WhisperASR

    lang_tokens = torch.tensor([EN, FR])
    probs = [{"en": 0.9, "fr": 0.1}, {"en": 0.2, "fr": 0.8}]
    calls = []
    asr = SimpleNamespace(
        mods=SimpleNamespace(
            whisper=SimpleNamespace(
                language=None,
                detect_language=lambda mel: (lang_tokens, probs),
            ),
            decoder=SimpleNamespace(set_lang_tokens=calls.append),
        )
    )
    languages, _ = WhisperASR._detect_language(
        asr, torch.zeros(2, 80, 10), task
    )
    assert languages == ["en", "fr"]
    assert len(calls) == (0 if task == "lang_id" else 1)
