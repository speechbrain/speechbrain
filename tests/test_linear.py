import torch
import torch.nn


def test_linear():

    from speechbrain.nnet.linear import Linear

    inputs = torch.rand(1, 2, 4)
    lin_t = Linear(inputs.shape[-1], False)
    lin_t.init_params(inputs)
    lin_t.w.weight = torch.nn.Parameter(torch.eye(inputs.shape[-1]))
    outputs = lin_t(inputs)
    assert torch.all(torch.eq(inputs, outputs))
