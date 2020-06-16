import torch
import torch.nn


def test_SincConv():

    from speechbrain.nnet.CNN import SincConv

    input = torch.rand([4, 16000])
    convolve = SincConv(out_channels=8, kernel_size=65, padding=True)
    output = convolve(input, init_params=True)
    assert output.shape[-1] == 8


def test_Conv1d():

    from speechbrain.nnet.CNN import Conv1d

    input = torch.tensor([-1, -1, -1, -1]).unsqueeze(0).unsqueeze(2).float()
    convolve = Conv1d(out_channels=1, kernel_size=1, padding=True)
    output = convolve(input, init_params=True)
    assert input.shape == output.shape

    convolve.conv.weight = torch.nn.Parameter(
        torch.tensor([-1]).float().unsqueeze(0).unsqueeze(1)
    )
    convolve.conv.bias = torch.nn.Parameter(torch.tensor([0]).float())
    output = convolve(input)
    assert torch.all(torch.eq(torch.ones(input.shape), output))


def test_Conv2d():

    from speechbrain.nnet.CNN import Conv2d

    input = torch.rand([4, 11, 32, 1])
    convolve = Conv2d(out_channels=1, kernel_size=(1, 1), padding=True)
    output = convolve(input, init_params=True)
    assert output.shape[-1] == 1

    convolve.conv.weight = torch.nn.Parameter(
        torch.zeros(convolve.conv.weight.shape)
    )
    convolve.conv.bias = torch.nn.Parameter(torch.tensor([0]).float())
    output = convolve(input)
    assert torch.all(torch.eq(torch.zeros(input.shape), output))

    convolve.conv.weight = torch.nn.Parameter(
        torch.ones(convolve.conv.weight.shape)
    )
    convolve.conv.bias = torch.nn.Parameter(torch.tensor([0]).float())
    output = convolve(input)
    assert torch.all(torch.eq(input, output))
