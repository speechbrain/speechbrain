"""Tests for the SeamlessM4T integration (freezing, key-value cache, searchers).

Authors
 * Othman Istaiteh 2026
"""

import pytest
import torch

SOURCE = "facebook/hf-seamless-m4t-medium"
AUDIO = [
    "tests/samples/ASR/spk1_snt1.wav",
    "tests/samples/ASR/spk1_snt2.wav",
    "tests/samples/ASR/spk2_snt1.wav",
]


@pytest.fixture(scope="module")
def save_path(tmp_path_factory):
    """Folder where the checkpoint is downloaded, shared by the tests."""
    return str(tmp_path_factory.mktemp("seamless_m4t"))


@pytest.fixture(scope="module")
def model(save_path):
    """SeamlessM4T with frozen encoder and embeddings, returning attentions."""
    from speechbrain.integrations.huggingface.seamless_m4t import SeamlessM4T

    model = SeamlessM4T(
        SOURCE,
        save_path,
        target_lang="fra",
        output_attentions=True,
        freeze_encoder=True,
        freeze_embeddings=True,
    )
    return model.eval()


@pytest.fixture(scope="module")
def encoded(model):
    """Encoder outputs of three utterances with different lengths."""
    from speechbrain.dataio.dataio import read_audio
    from speechbrain.utils.data_utils import batch_pad_right

    wav, wav_lens = batch_pad_right([read_audio(f) for f in AUDIO])
    bos = torch.tensor([[model.bos]] * len(AUDIO))
    with torch.no_grad():
        enc_out, enc_lens, _ = model(wav, bos, wav_lens)
    return enc_out, enc_lens


def test_seamless_freezing(model):
    """Only the decoder layers are trainable when the encoder and the
    embeddings are frozen."""
    embeddings = {
        "shared.weight",
        "text_decoder.embed_tokens.weight",
        "lm_head.weight",
    }
    for name, param in model.model.named_parameters():
        frozen = name.startswith("speech_encoder.") or name in embeddings
        assert param.requires_grad != frozen, name


def test_seamless_kv_cache(model, encoded):
    """The cached decoding steps give the logits and the attention of the
    decoder run on the whole sequence."""
    enc_out, enc_lens = encoded
    abs_lens = torch.round(enc_lens * enc_out.shape[1])
    tgt = torch.tensor(
        [
            [model.bos, 100, 200, 300],
            [model.bos, 400, 500, 600],
            [model.bos, 700, 800, 900],
        ]
    )
    with torch.no_grad():
        logits, attn, cache = model.forward_decoder(enc_out, tgt, abs_lens)
        assert cache is None
        assert attn.shape == (3, 4, enc_out.shape[1])
        for t in range(1, tgt.shape[1] + 1):
            step_logits, step_attn, cache = model.forward_decoder(
                enc_out,
                tgt[:, :t],
                abs_lens,
                last_position_only=True,
                use_cache=True,
                past_key_values=cache,
            )
            assert step_logits.shape == (3, 1, model.vocab_size)
            assert torch.allclose(
                step_logits[:, 0], logits[:, t - 1], atol=1e-3
            )
            assert torch.allclose(step_attn, attn[:, t - 1], atol=1e-4)


def test_seamless_searchers(model, encoded):
    """The searchers give the same non-empty hypotheses with and without
    the key-value cache, and work with the attention-based options."""
    import transformers
    from packaging.version import Version

    from speechbrain.decoders.scorer import CoverageScorer, ScorerBuilder
    from speechbrain.decoders.seq2seq import (
        S2SSeamlessBeamSearcher,
        S2SSeamlessGreedySearcher,
    )

    enc_out, enc_lens = encoded
    kwargs = dict(min_decode_ratio=0.0, max_decode_ratio=1.0)
    # Older versions return the beam search cache as tuples, not reordered.
    kv_cache_ok = Version(transformers.__version__) >= Version("4.54")
    with torch.no_grad():
        greedy = [
            S2SSeamlessGreedySearcher(model, use_kv_cache=c, **kwargs)(
                enc_out, enc_lens
            )[0]
            for c in (False, True)
        ]
        beam = [
            S2SSeamlessBeamSearcher(
                [model], use_kv_cache=False, beam_size=3, **kwargs
            )(enc_out, enc_lens)[0]
        ]
        cached_beam = S2SSeamlessBeamSearcher(
            [model], use_kv_cache=True, beam_size=3, **kwargs
        )
        if kv_cache_ok:
            beam.append(cached_beam(enc_out, enc_lens)[0])
        else:
            with pytest.raises(RuntimeError, match="4.54"):
                cached_beam(enc_out, enc_lens)
            beam.append(beam[0])
        scorer = ScorerBuilder(
            full_scorers=[CoverageScorer(vocab_size=model.vocab_size)],
            weights={"coverage": 0.5},
        )
        S2SSeamlessBeamSearcher(
            [model],
            use_kv_cache=kv_cache_ok,
            beam_size=3,
            scorer=scorer,
            using_max_attn_shift=True,
            **kwargs,
        )(enc_out, enc_lens)
    assert all(len(h) > 0 for h in greedy[0] + beam[0])
    assert [list(h) for h in greedy[0]] == [list(h) for h in greedy[1]]
    assert [list(h) for h in beam[0]] == [list(h) for h in beam[1]]


def test_seamless_encoder_only(save_path):
    """The encoder-only model has no decoder, returns the encoder outputs
    and their relative lengths, and refuses decoder inputs."""
    from speechbrain.integrations.huggingface.seamless_m4t import SeamlessM4T

    model = SeamlessM4T(SOURCE, save_path, encoder_only=True).eval()
    assert model.model.text_decoder is None
    assert model.model.lm_head is None
    with torch.no_grad():
        enc_out, enc_lens = model(
            torch.randn(2, 16000), wav_lens=torch.tensor([1.0, 0.5])
        )
    assert enc_out.shape[0] == 2
    assert enc_lens[0] == 1.0 and enc_lens[1] < 1.0
    with pytest.raises(ValueError):
        model(torch.randn(2, 16000), torch.tensor([1.0, 0.5]))
