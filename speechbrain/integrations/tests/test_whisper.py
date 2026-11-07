"""Tests for the Whisper integration.

Authors
 * Othman Istaiteh 2026
"""

from types import SimpleNamespace

import pytest


def test_whisper_to_language_token():
    """Language codes and English names, in any case, give the token id."""
    from speechbrain.integrations.huggingface.whisper import Whisper

    vocab = {"<|en|>": 50259, "<|fr|>": 50265}
    tokenizer = SimpleNamespace(
        unk_token_id=50257,
        convert_tokens_to_ids=lambda token: vocab.get(token, 50257),
    )
    whisper = SimpleNamespace(tokenizer=tokenizer)
    for language in ["en", "EN", "english", "English"]:
        assert Whisper.to_language_token(whisper, language) == 50259
    assert Whisper.to_language_token(whisper, "French") == 50265
    with pytest.raises(KeyError):
        Whisper.to_language_token(whisper, "klingon")
