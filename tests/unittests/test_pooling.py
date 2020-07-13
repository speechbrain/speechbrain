import torch
import torch.nn


def test_pooling1d():

    from speechbrain.nnet.pooling import Pooling1d

    input = torch.tensor([1, 3, 2]).unsqueeze(0).unsqueeze(-1).float()
    pool = Pooling1d("max", 3)
    output = pool(input)
    assert output == 3

    pool = Pooling1d("avg", 3)
    output = pool(input)
    assert output == 2


def test_pooling2d():

    from speechbrain.nnet.pooling import Pooling2d

    input = torch.tensor([[1, 3, 2], [4, 6, 5]]).float().unsqueeze(0)
    pool = Pooling2d("max", (2, 3))
    output = pool(input)
    assert output == 6

    input = torch.tensor([[1, 3, 2], [4, 6, 5]]).float().unsqueeze(0)
    pool = Pooling2d("max", (1, 3))
    output = pool(input)
    assert output[0][0] == 3
    assert output[0][1] == 6

    input = torch.tensor([[1, 3, 2], [4, 6, 5]]).float().unsqueeze(0)
    pool = Pooling2d("avg", (2, 3))
    output = pool(input)
    assert output == 3.5

    input = torch.tensor([[1, 3, 2], [4, 6, 5]]).float().unsqueeze(0)
    pool = Pooling2d("avg", (1, 3))
    output = pool(input)
    assert output[0][0] == 2
    assert output[0][1] == 5
