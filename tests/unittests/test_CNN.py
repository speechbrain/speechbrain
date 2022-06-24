import torch
import torch.nn


def test_SincConv(device):
    from speechbrain.nnet.CNN import SincConv

    input = torch.rand([4, 16000], device=device)
    convolve = SincConv(
        input_shape=input.shape, out_channels=8, kernel_size=65, padding="same"
    ).to(device)
    output = convolve(input)
    assert output.shape[-1] == 8

    assert torch.jit.trace(convolve, input)

    # Multichannel case
    input = torch.rand([10, 16000, 8], device=device)
    convolve = SincConv(
        input_shape=input.shape, out_channels=16, kernel_size=11, padding="same"
    ).to(device)
    output = convolve(input)
    assert output.shape[-1] == 16

    assert torch.jit.trace(convolve, input)


def test_Conv1d(device):

    from speechbrain.nnet.CNN import Conv1d

    input = (
        torch.tensor([-1, -1, -1, -1], device=device)
        .unsqueeze(0)
        .unsqueeze(2)
        .float()
    )
    convolve = Conv1d(
        out_channels=1, kernel_size=1, input_shape=input.shape, padding="same"
    ).to(device)
    output = convolve(input)
    assert input.shape == output.shape

    convolve.conv.weight = torch.nn.Parameter(
        torch.tensor([-1], device=device).float().unsqueeze(0).unsqueeze(1)
    )
    convolve.conv.bias = torch.nn.Parameter(
        torch.tensor([0], device=device).float()
    )
    output = convolve(input)
    assert torch.all(torch.eq(torch.ones(input.shape, device=device), output))

    assert torch.jit.trace(convolve, input)


def test_Conv2d(device):

    from speechbrain.nnet.CNN import Conv2d

    input = torch.rand([4, 11, 32, 1], device=device)
    convolve = Conv2d(
        out_channels=1,
        input_shape=input.shape,
        kernel_size=(1, 1),
        padding="same",
    ).to(device)
    output = convolve(input)
    assert output.shape[-1] == 1

    convolve.conv.weight = torch.nn.Parameter(
        torch.zeros(convolve.conv.weight.shape, device=device)
    )
    convolve.conv.bias = torch.nn.Parameter(
        torch.tensor([0], device=device).float()
    )
    output = convolve(input)
    assert torch.all(torch.eq(torch.zeros(input.shape, device=device), output))

    convolve.conv.weight = torch.nn.Parameter(
        torch.ones(convolve.conv.weight.shape, device=device)
    )
    convolve.conv.bias = torch.nn.Parameter(
        torch.tensor([0], device=device).float()
    )
    output = convolve(input)
    assert torch.all(torch.eq(input, output))

    assert torch.jit.trace(convolve, input)


def test_Leaf(device):
    from speechbrain.lobes.features import Leaf

    input = torch.rand([4, 16000], device=device)
    convolve = Leaf(
        input_shape=input.shape,
        window_len=25.0,
        window_stride=10.0,
        out_channels=8,
    ).to(device)
    output = convolve(input)
    assert output.shape[-1] == 8
    assert torch.jit.trace(convolve, input)
