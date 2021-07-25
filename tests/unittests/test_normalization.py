import torch
import torch.nn


def test_BatchNorm1d():

    from speechbrain.nnet.normalization import BatchNorm1d

    input = torch.randn(100, 10) + 2.0
    norm = BatchNorm1d(input_shape=input.shape)
    output = norm(input)
    assert input.shape == output.shape

    current_mean = output.mean(dim=0).mean()
    assert torch.abs(current_mean) < 1e-07

    current_std = output.std(dim=0).mean()
    assert torch.abs(1.0 - current_std) < 0.01

    input = torch.randn(100, 20, 10) + 2.0
    output = norm(input)
    assert input.shape == output.shape

    current_mean = output.mean(dim=0).mean()
    assert torch.abs(current_mean) < 1e-07

    current_std = output.std(dim=0).mean()
    assert torch.abs(1.0 - current_std) < 0.01

    # Test with combined dimensions
    input = torch.randn(100, 10, 20) + 2.0
    norm = BatchNorm1d(input_shape=input.shape, combine_batch_time=True)
    output = norm(input)
    assert input.shape == output.shape

    current_mean = output.mean(dim=0).mean()
    assert torch.abs(current_mean) < 1e-07

    current_std = output.std(dim=0).mean()
    assert torch.abs(1.0 - current_std) < 0.01

    input = torch.randn(100, 40, 20, 30) + 2.0
    norm = BatchNorm1d(input_shape=input.shape, combine_batch_time=True)
    output = norm(input)
    assert input.shape == output.shape

    current_mean = output.mean(dim=0).mean()
    assert torch.abs(current_mean) < 5e-06

    current_std = output.std(dim=0).mean()
    assert torch.abs(1.0 - current_std) < 0.01

    assert torch.jit.trace(norm, input)


def test_BatchNorm2d():

    from speechbrain.nnet.normalization import BatchNorm2d

    input = torch.randn(100, 10, 4, 20) + 2.0
    norm = BatchNorm2d(input_shape=input.shape)
    output = norm(input)
    assert input.shape == output.shape

    current_mean = output.mean(dim=0).mean()
    assert torch.abs(current_mean) < 1e-07

    current_std = output.std(dim=0).mean()
    assert torch.abs(1.0 - current_std) < 0.01

    assert torch.jit.trace(norm, input)


def test_LayerNorm():

    from speechbrain.nnet.normalization import LayerNorm

    input = torch.randn(4, 101, 256) + 2.0
    norm = LayerNorm(input_shape=input.shape)
    output = norm(input)
    assert input.shape == output.shape

    current_mean = output.mean(dim=2).mean()
    assert torch.abs(current_mean) < 1e-07

    current_std = output.std(dim=2).mean()
    assert torch.abs(1.0 - current_std) < 0.01

    input = torch.randn(100, 101, 16, 32) + 2.0
    norm = LayerNorm(input_shape=input.shape)
    output = norm(input)
    assert input.shape == output.shape

    current_mean = output.mean(dim=[2, 3]).mean()
    assert torch.abs(current_mean) < 1e-07

    current_std = output.std(dim=[2, 3]).mean()
    assert torch.abs(1.0 - current_std) < 0.01

    assert torch.jit.trace(norm, input)


def test_InstanceNorm1d():

    from speechbrain.nnet.normalization import InstanceNorm1d

    input = torch.randn(100, 10, 128) + 2.0
    norm = InstanceNorm1d(input_shape=input.shape)
    output = norm(input)
    assert input.shape == output.shape

    current_mean = output.mean(dim=2).mean()
    assert torch.abs(current_mean) < 1e-07

    current_std = output.std(dim=2).mean()
    assert torch.abs(1.0 - current_std) < 0.01

    assert torch.jit.trace(norm, input)


def test_InstanceNorm2d():

    from speechbrain.nnet.normalization import InstanceNorm2d

    input = torch.randn(100, 10, 20, 2) + 2.0
    norm = InstanceNorm2d(input_shape=input.shape)
    output = norm(input)
    assert input.shape == output.shape

    current_mean = output.mean(dim=[2, 3]).mean()
    assert torch.abs(current_mean) < 1e-07

    current_std = output.std(dim=[2, 3]).mean()
    assert torch.abs(1.0 - current_std) < 0.01

    assert torch.jit.trace(norm, input)
