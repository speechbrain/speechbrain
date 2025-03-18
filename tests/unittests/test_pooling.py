import torch
import torch.nn


def test_pooling1d(device):
    from speechbrain.nnet.pooling import Pooling1d

    input = torch.tensor([1, 3, 2], device=device).unsqueeze(0).unsqueeze(-1).float()
    pool = Pooling1d("max", 3).to(device)
    output = pool(input)
    assert output == 3

    pool = Pooling1d("avg", 3).to(device)
    output = pool(input)
    assert output == 2

    assert torch.jit.trace(pool, input)


def test_pooling2d(device):
    from speechbrain.nnet.pooling import Pooling2d

    input = torch.tensor([[1, 3, 2], [4, 6, 5]], device=device).float().unsqueeze(0)
    pool = Pooling2d("max", (2, 3)).to(device)
    output = pool(input)
    assert output == 6

    input = torch.tensor([[1, 3, 2], [4, 6, 5]], device=device).float().unsqueeze(0)
    pool = Pooling2d("max", (1, 3)).to(device)
    output = pool(input)
    assert output[0][0] == 3
    assert output[0][1] == 6

    input = torch.tensor([[1, 3, 2], [4, 6, 5]], device=device).float().unsqueeze(0)
    pool = Pooling2d("avg", (2, 3)).to(device)
    output = pool(input)
    assert output == 3.5

    input = torch.tensor([[1, 3, 2], [4, 6, 5]], device=device).float().unsqueeze(0)
    pool = Pooling2d("avg", (1, 3)).to(device)
    output = pool(input)
    assert output[0][0] == 2
    assert output[0][1] == 5

    assert torch.jit.trace(pool, input)
