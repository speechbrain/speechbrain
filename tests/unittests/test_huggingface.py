import pytest
import torch


def _write_tiny_whisper_files(folder):
    from transformers import WhisperConfig, WhisperFeatureExtractor

    config = WhisperConfig(
        vocab_size=64,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_target_positions=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    )
    config.save_pretrained(str(folder))
    WhisperFeatureExtractor().save_pretrained(str(folder))
    return config


def _tiny_whisper_model(config):
    from transformers import WhisperModel

    # Values that a fresh random initialization cannot produce
    model = WhisperModel(config)
    with torch.no_grad():
        for i, param in enumerate(model.parameters()):
            param.fill_(i + 0.5)
    return model


def _load_whisper(source, save_path):
    from speechbrain.integrations.huggingface.whisper import Whisper

    return Whisper(str(source), save_path=str(save_path), encoder_only=True)


def _assert_same_state(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        assert torch.equal(actual[key], expected[key]), key


def test_whisper_loads_speechbrain_checkpoint(tmpdir):
    from speechbrain.utils.checkpoints import torch_save

    hf_dir = tmpdir / "hf"
    config = _write_tiny_whisper_files(hf_dir)
    _tiny_whisper_model(config).save_pretrained(str(hf_dir))
    original = _load_whisper(hf_dir, tmpdir / "cache")

    # Saved by the default SpeechBrain hook, so keys start with "model."
    sb_dir = tmpdir / "sb"
    _write_tiny_whisper_files(sb_dir)
    torch_save(original, str(sb_dir / "model.ckpt"))
    reloaded = _load_whisper(sb_dir, tmpdir / "cache")

    _assert_same_state(
        reloaded.model.encoder.state_dict(),
        original.model.encoder.state_dict(),
    )


def test_whisper_loads_checkpoint_with_huggingface_keys(tmpdir):
    sb_dir = tmpdir / "sb"
    config = _write_tiny_whisper_files(sb_dir)
    hf_model = _tiny_whisper_model(config)
    torch.save(hf_model.state_dict(), str(sb_dir / "model.ckpt"))
    reloaded = _load_whisper(sb_dir, tmpdir / "cache")

    _assert_same_state(
        reloaded.model.encoder.state_dict(), hf_model.encoder.state_dict()
    )


def test_whisper_checkpoint_without_model_keys_raises(tmpdir):
    sb_dir = tmpdir / "sb"
    _write_tiny_whisper_files(sb_dir)
    torch.save({"unrelated.weight": torch.zeros(1)}, str(sb_dir / "model.ckpt"))

    with pytest.raises(RuntimeError, match="None of the parameters"):
        _load_whisper(sb_dir, tmpdir / "cache")
