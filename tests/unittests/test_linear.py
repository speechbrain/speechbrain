import torch
import torch.nn


def test_linear():

    from speechbrain.nnet.linear import Linear

    inputs = torch.rand(1, 2, 4)
    lin_t = Linear(n_neurons=4, input_size=inputs.shape[-1], bias=False)
    lin_t.w.weight = torch.nn.Parameter(torch.eye(inputs.shape[-1]))
    outputs = lin_t(inputs)
    assert torch.all(torch.eq(inputs, outputs))

    assert torch.jit.trace(lin_t, inputs)
