import torch
import torch.nn


def test_BatchNorm1d():

    from speechbrain.nnet.normalization import BatchNorm1d

    input = torch.randn(100, 10) + 2.0
    norm = BatchNorm1d()
    output = norm(input, init_params=True)
    assert input.shape == output.shape

    current_mean = output.mean(dim=0).mean()
    assert torch.abs(current_mean) < 1e-07

    current_std = output.std(dim=0).mean()
    assert torch.abs(1.0 - current_std) < 0.01

    input = torch.randn(100, 10, 20) + 2.0
    output = norm(input, init_params=True)
    assert input.shape == output.shape

    current_mean = output.mean(dim=0).mean()
    assert torch.abs(current_mean) < 1e-07

    current_std = output.std(dim=0).mean()
    assert torch.abs(1.0 - current_std) < 0.01

    # Test with combined dimensions
    input = torch.randn(100, 10, 20) + 2.0
    norm = BatchNorm1d(combine_batch_time=True)
    output = norm(input, init_params=True)
    assert input.shape == output.shape

    current_mean = output.mean(dim=0).mean()
    assert torch.abs(current_mean) < 1e-07

    current_std = output.std(dim=0).mean()
    assert torch.abs(1.0 - current_std) < 0.01

    input = torch.randn(100, 10, 20, 4) + 2.0
    norm = BatchNorm1d(combine_batch_time=True)
    output = norm(input, init_params=True)
    assert input.shape == output.shape

    current_mean = output.mean(dim=0).mean()
    assert torch.abs(current_mean) < 1e-07

    current_std = output.std(dim=0).mean()
    assert torch.abs(1.0 - current_std) < 0.01


def test_BatchNorm2d():

    from speechbrain.nnet.normalization import BatchNorm2d

    input = torch.randn(100, 10, 4, 20) + 2.0
    norm = BatchNorm2d()
    output = norm(input, init_params=True)
    assert input.shape == output.shape

    current_mean = output.mean(dim=0).mean()
    assert torch.abs(current_mean) < 1e-07

    current_std = output.std(dim=0).mean()
    assert torch.abs(1.0 - current_std) < 0.01


def test_LayerNorm():

    from speechbrain.nnet.normalization import LayerNorm

    input = torch.randn(4, 101, 256) + 2.0
    norm = LayerNorm()
    output = norm(input, init_params=True)
    assert input.shape == output.shape

    current_mean = output.mean(dim=2).mean()
    assert torch.abs(current_mean) < 1e-07

    current_std = output.std(dim=2).mean()
    assert torch.abs(1.0 - current_std) < 0.01

    input = torch.randn(100, 101, 16, 32) + 2.0
    norm = LayerNorm()
    output = norm(input, init_params=True)
    assert input.shape == output.shape

    current_mean = output.mean(dim=[2, 3]).mean()
    assert torch.abs(current_mean) < 1e-07

    current_std = output.std(dim=[2, 3]).mean()
    assert torch.abs(1.0 - current_std) < 0.01


def test_InstanceNorm1d():

    from speechbrain.nnet.normalization import InstanceNorm1d

    input = torch.randn(100, 10, 128) + 2.0
    norm = InstanceNorm1d()
    output = norm(input, init_params=True)
    assert input.shape == output.shape

    current_mean = output.mean(dim=2).mean()
    assert torch.abs(current_mean) < 1e-07

    current_std = output.std(dim=2).mean()
    assert torch.abs(1.0 - current_std) < 0.01


def test_InstanceNorm2d():

    from speechbrain.nnet.normalization import InstanceNorm2d

    input = torch.randn(100, 10, 20, 2) + 2.0
    norm = InstanceNorm2d()
    output = norm(input, init_params=True)
    assert input.shape == output.shape

    current_mean = output.mean(dim=[2, 3]).mean()
    assert torch.abs(current_mean) < 1e-07

    current_std = output.std(dim=[2, 3]).mean()
    assert torch.abs(1.0 - current_std) < 0.01
