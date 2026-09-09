import pytest
import torch


@pytest.mark.parametrize("padding", [0, 1, (2, 2), (2, 0), (0, 2)])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_convolve1d_time_domain(device, padding, stride, dtype):
    import numpy as np

    from speechbrain.processing.signal_processing import convolve1d

    waveform = torch.tensor([1, 2, 3, 4], device=device, dtype=dtype)
    kernel = torch.tensor([1, 2, 4], device=device, dtype=dtype)
    pad_width = padding if isinstance(padding, tuple) else (padding, padding)
    expected = np.convolve(
        np.pad(waveform.cpu().numpy(), pad_width),
        kernel.cpu().numpy(),
        mode="valid",
    )[::stride]

    result = convolve1d(
        waveform.view(1, -1, 1),
        kernel.view(1, -1, 1),
        padding=padding,
        stride=stride,
    )

    assert result.dtype == dtype
    np.testing.assert_allclose(result[0, :, 0].cpu().numpy(), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_convolve1d_time_domain_matches_fft(device, dtype):
    from speechbrain.processing.signal_processing import convolve1d

    waveform = torch.tensor([1, 2, 3, 4], device=device, dtype=dtype).view(
        1, -1, 1
    )
    kernel = torch.tensor([1, 2, 4], device=device, dtype=dtype).view(1, -1, 1)
    time_result = convolve1d(waveform, kernel, padding=(2, 2))
    fft_result = convolve1d(waveform, kernel, padding=(0, 2), use_fft=True)
    expected = torch.tensor(
        [1, 4, 11, 18, 20, 16], device=device, dtype=dtype
    ).view(1, -1, 1)

    torch.testing.assert_close(time_result, expected)
    torch.testing.assert_close(time_result, fft_result)


@pytest.mark.parametrize("stride", [1, 2])
def test_convolve1d_grouped(device, stride):
    import numpy as np

    from speechbrain.processing.signal_processing import convolve1d

    waveform = torch.arange(24, device=device, dtype=torch.float64).view(
        2, 6, 2
    )
    kernel = torch.arange(1, 13, device=device, dtype=torch.float64).view(
        4, 3, 1
    )
    result = convolve1d(waveform, kernel, groups=2, stride=stride)

    for batch in range(2):
        for channel in range(4):
            expected = np.convolve(
                waveform[batch, :, channel // 2].cpu().numpy(),
                kernel[channel, :, 0].cpu().numpy(),
                mode="valid",
            )[::stride]
            np.testing.assert_allclose(
                result[batch, :, channel].cpu().numpy(), expected
            )


def test_convolve1d_gradients(device):
    from speechbrain.processing.signal_processing import convolve1d

    waveform = torch.randn(
        1, 4, 1, device=device, dtype=torch.float64, requires_grad=True
    )
    kernel = torch.randn(
        1, 3, 1, device=device, dtype=torch.float64, requires_grad=True
    )
    assert torch.autograd.gradcheck(convolve1d, (waveform, kernel))
