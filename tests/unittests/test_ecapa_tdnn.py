import pytest
import torch


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16]
)
@pytest.mark.parametrize("global_context", [True, False])
@pytest.mark.parametrize("with_lengths", [True, False])
@torch.no_grad()
def test_attentive_statistics_pooling_dtype(
    dtype, global_context, with_lengths
):
    from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling

    torch.manual_seed(2544)
    pool = AttentiveStatisticsPooling(
        16, attention_channels=8, global_context=global_context
    ).to(dtype=dtype)
    pool.eval()
    inputs = torch.randn(2, 16, 20).to(dtype=dtype)
    lengths = (
        torch.tensor([1.0, 0.6], dtype=torch.float32) if with_lengths else None
    )

    output = pool(inputs, lengths)

    assert output.shape == (2, 32, 1)
    assert output.dtype == dtype
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16]
)
@pytest.mark.parametrize("with_lengths", [True, False])
@torch.no_grad()
def test_ecapa_tdnn_dtype(dtype, with_lengths):
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

    torch.manual_seed(2544)
    model = ECAPA_TDNN(
        8,
        lin_neurons=16,
        channels=[16, 16, 16, 16, 48],
        attention_channels=8,
        res2net_scale=4,
        se_channels=8,
    ).to(dtype=dtype)
    model.eval()
    inputs = torch.randn(2, 20, 8).to(dtype=dtype)
    lengths = (
        torch.tensor([1.0, 0.6], dtype=torch.float32) if with_lengths else None
    )

    output = model(inputs, lengths)

    assert output.shape == (2, 1, 16)
    assert output.dtype == dtype
    assert torch.isfinite(output).all()
