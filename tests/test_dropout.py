import torch
import torch.nn


def test_dropout():

    from speechbrain.nnet.dropout import Dropout

    inputs = torch.rand(1, 2, 4)
    drop = Dropout(drop_rate=0.0)
    outputs = drop(inputs)
    assert torch.all(torch.eq(inputs, outputs))

    drop = Dropout(drop_rate=1.0)
    drop.init_params(inputs)
    outputs = drop(inputs)
    assert torch.all(torch.eq(torch.zeros(inputs.shape), outputs))
