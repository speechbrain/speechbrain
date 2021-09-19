import torch
import torch.nn


def test_SincConv():

    from speechbrain.nnet.CNN import SincConv

    input = torch.rand([4, 16000])
    convolve = SincConv(
        input_shape=input.shape, out_channels=8, kernel_size=65, padding="same"
    )
    output = convolve(input)
    assert output.shape[-1] == 8

    assert torch.jit.trace(convolve, input)


def test_Conv1d():

    from speechbrain.nnet.CNN import Conv1d

    input = torch.tensor([-1, -1, -1, -1]).unsqueeze(0).unsqueeze(2).float()
    convolve = Conv1d(
        out_channels=1, kernel_size=1, input_shape=input.shape, padding="same"
    )
    output, _ = convolve(input)
    assert input.shape == output.shape

    convolve.conv.weight = torch.nn.Parameter(
        torch.tensor([-1]).float().unsqueeze(0).unsqueeze(1)
    )
    convolve.conv.bias = torch.nn.Parameter(torch.tensor([0]).float())
    output, _ = convolve(input)
    assert torch.all(torch.eq(torch.ones(input.shape), output))

    assert torch.jit.trace(convolve, [input, input.eq(0.0)])


def test_Conv2d():

    from speechbrain.nnet.CNN import Conv2d

    input = torch.rand([4, 11, 32, 1])
    convolve = Conv2d(
        out_channels=1,
        input_shape=input.shape,
        kernel_size=(1, 1),
        padding="same",
    )
    output, _ = convolve(input)
    assert output.shape[-1] == 1

    convolve.conv.weight = torch.nn.Parameter(
        torch.zeros(convolve.conv.weight.shape)
    )
    convolve.conv.bias = torch.nn.Parameter(torch.tensor([0]).float())
    output, _ = convolve(input)
    assert torch.all(torch.eq(torch.zeros(input.shape), output))

    convolve.conv.weight = torch.nn.Parameter(
        torch.ones(convolve.conv.weight.shape)
    )
    convolve.conv.bias = torch.nn.Parameter(torch.tensor([0]).float())
    output, _ = convolve(input)
    assert torch.all(torch.eq(input, output))

    assert torch.jit.trace(convolve, [input, input.eq(0.0)])


def test_Conv1d_with_padding():

    from speechbrain.nnet.CNN import Conv1d

    sample_length = 31
    # Padded input with the shape (batch, time, fea)
    input_padded = torch.rand([2, 65, 40])
    # The first sample is padded
    input_padded[0, sample_length:] = 0.0
    input_nonpadded = input_padded[0, :sample_length].unsqueeze(0)
    # Create mask
    input_mask = input_padded.eq(0.0)

    convolve = Conv1d(
        out_channels=4,
        input_shape=input_padded.shape,
        kernel_size=3,
        stride=2,
        padding="valid",
    )
    convolve.eval()
    output_padded, mask = convolve(input_padded, mask=input_mask)
    output_nonpadded, _ = convolve(input_nonpadded)
    output, mask = convolve(input_padded)

    assert torch.allclose(
        torch.sum(output_nonpadded[0]), torch.sum(output_padded[0])
    )

    assert torch.jit.trace(convolve, [input_padded, input_mask])


def test_Conv2d_with_padding():

    from speechbrain.nnet.CNN import Conv2d

    sample_length = 32
    # Padded input with the shape (batch, time, fea)
    input_padded = torch.rand([2, 64, 40])
    # The first sample is padded
    input_padded[0, sample_length:] = 0.0
    input_nonpadded = input_padded[0, :sample_length].unsqueeze(0)
    # Create mask
    input_mask = input_padded.eq(0.0)

    convolve = Conv2d(
        out_channels=4,
        input_shape=input_padded.shape,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding="valid",
    )
    convolve.eval()
    output_padded, mask = convolve(input_padded, mask=input_mask)
    output_nonpadded, _ = convolve(input_nonpadded)

    assert torch.allclose(
        torch.sum(output_nonpadded[0]), torch.sum(output_padded[0])
    )

    assert torch.jit.trace(convolve, [input_padded, input_mask])
