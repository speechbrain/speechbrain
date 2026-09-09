import pytest
import torch


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
@pytest.mark.parametrize("global_context", [True, False])
@pytest.mark.parametrize("use_lengths", [True, False])
def test_attentive_statistics_pooling_dtype(
    device, dtype, global_context, use_lengths
):
    from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling

    pool = AttentiveStatisticsPooling(
        channels=2, attention_channels=2, global_context=global_context
    ).to(device=device, dtype=dtype)
    # Uniform attention gives the mean and standard deviation of valid frames.
    with torch.no_grad():
        pool.conv.conv.weight.zero_()
        pool.conv.conv.bias.zero_()
    x = torch.tensor(
        [[[1, 3, 1, 3], [2, 4, 2, 4]], [[1, 3, 5, 7], [2, 4, 6, 8]]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    lengths = torch.tensor([1.0, 0.5], device=device) if use_lengths else None

    output = pool(x, lengths)

    expected = [[2, 3, 1, 1], [2, 3, 1, 1]]
    if not use_lengths:
        expected[1] = [4, 5, 2.2360679775, 2.2360679775]
    expected = torch.tensor(expected, device=device, dtype=dtype).unsqueeze(2)
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(param.grad).all() for param in pool.parameters())
