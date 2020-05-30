import torch
import torch.nn


def test_dropout():

    from speechbrain.nnet.dropout import Dropout2d

    inputs = torch.rand([4, 10, 32])
    drop = Dropout2d(drop_rate=0.0)
    outputs = drop(inputs)
    assert torch.all(torch.eq(inputs, outputs))

    drop = Dropout2d(drop_rate=1.0)
    outputs = drop(inputs)
    assert torch.all(torch.eq(torch.zeros(inputs.shape), outputs))
