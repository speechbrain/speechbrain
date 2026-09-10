import pytest
import torch


@pytest.mark.parametrize("length", [0, 1, 3, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_positional_encoding_within_limit(device, length, dtype):
    from speechbrain.lobes.models.transformer.Transformer import (
        PositionalEncoding,
    )

    encoding = PositionalEncoding(input_size=2, max_len=4).to(
        device=device, dtype=dtype
    )
    inputs = torch.zeros(2, length, 2, device=device, dtype=dtype)

    result = encoding(inputs)

    positions = torch.arange(length, device=device, dtype=torch.float32)
    expected = torch.stack((positions.sin(), positions.cos()), dim=-1)
    torch.testing.assert_close(result, expected.unsqueeze(0).to(dtype))
    assert not result.requires_grad


@pytest.mark.parametrize("max_len", [4, 2500])
def test_positional_encoding_exceeds_limit(device, max_len):
    from speechbrain.lobes.models.transformer.Transformer import (
        PositionalEncoding,
    )

    encoding = PositionalEncoding(input_size=2, max_len=max_len).to(device)
    inputs = torch.zeros(1, max_len + 1, 2, device=device)

    with pytest.raises(
        ValueError,
        match=rf"Input sequence length {max_len + 1} exceeds the maximum positional encoding length {max_len}",
    ):
        encoding(inputs)


@pytest.mark.parametrize("path", ["encode", "source", "target"])
def test_transformer_asr_exceeds_positional_encoding_limit(device, path):
    from speechbrain.lobes.models.transformer.TransformerASR import (
        TransformerASR,
    )

    model = (
        TransformerASR(
            tgt_vocab=8,
            input_size=4,
            d_model=4,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            d_ffn=8,
            dropout=0,
            max_length=4,
            causal=False,
        )
        .to(device)
        .eval()
    )
    source = torch.zeros(1, 4, 4, device=device)
    target = torch.ones(1, 4, device=device, dtype=torch.long)

    with torch.no_grad():
        encoder_out, decoder_out = model(source, target)
        assert encoder_out.shape == decoder_out.shape == (1, 4, 4)
        assert torch.isfinite(encoder_out).all()
        assert torch.isfinite(decoder_out).all()

        if path == "target":
            target = torch.ones(1, 5, device=device, dtype=torch.long)
        else:
            source = torch.zeros(1, 5, 4, device=device)

        with pytest.raises(
            ValueError,
            match="Input sequence length 5 exceeds the maximum positional encoding length 4",
        ):
            if path == "encode":
                model.encode(source)
            else:
                model(source, target)
