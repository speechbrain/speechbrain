import torch
import torch.nn


def test_dropout(device):

    from speechbrain.nnet.dropout import Dropout2d

    inputs = torch.rand([4, 10, 32], device=device)
    drop = Dropout2d(drop_rate=0.0).to(device)
    outputs = drop(inputs)
    assert torch.all(torch.eq(inputs, outputs))

    drop = Dropout2d(drop_rate=1.0).to(device)
    outputs = drop(inputs)
    assert torch.all(
        torch.eq(torch.zeros(inputs.shape, device=device), outputs)
    )

    assert torch.jit.trace(drop, inputs)
