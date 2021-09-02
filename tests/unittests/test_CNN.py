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
    output = convolve(input)
    assert input.shape == output.shape

    convolve.conv.weight = torch.nn.Parameter(
        torch.tensor([-1]).float().unsqueeze(0).unsqueeze(1)
    )
    convolve.conv.bias = torch.nn.Parameter(torch.tensor([0]).float())
    output = convolve(input)
    assert torch.all(torch.eq(torch.ones(input.shape), output))

    assert torch.jit.trace(convolve, input)


def test_Conv2d():

    from speechbrain.nnet.CNN import Conv2d

    input = torch.rand([4, 11, 32, 1])
    convolve = Conv2d(
        out_channels=1,
        input_shape=input.shape,
        kernel_size=(1, 1),
        padding="same",
    )
    output = convolve(input)
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

    assert torch.jit.trace(convolve, input)


def test_Conv2dMask():

    from speechbrain.nnet.CNN import Conv2d, Mask2d

    sample_length = 32
    # Padded input with the shape (batch, time, fea)
    input_padded = torch.rand([2, 66, 40])
    # The first sample is padded
    input_padded[0, sample_length:] = 0.0
    input_nonpadded = input_padded[0, :sample_length].unsqueeze(0)
    # Create mask
    input_mask = input_padded.eq(0.0)

    convolve = Conv2d(
        out_channels=4,
        input_shape=input_padded.shape,
        kernel_size=(1, 1),
        stride=(2, 2),
        padding="same",
    )
    convolve.eval()
    conv_mask = Mask2d(kernel_size=(1, 1), stride=(2, 2), padding="same",)
    output_padded = convolve(input_padded)
    conv_mask = conv_mask(input_mask)
    # Mask conv output
    output_padded.masked_fill_(conv_mask, 0.0)

    output_nonpadded = convolve(input_nonpadded)
    output_length = output_nonpadded.size(1)

    assert torch.allclose(output_nonpadded[0], output_padded[0][:output_length])

    assert torch.jit.trace(convolve, input_padded)
