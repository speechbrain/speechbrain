import torch
import torch.nn


def test_linear(device):
    from speechbrain.nnet.linear import Linear

    inputs = torch.rand(1, 2, 4, device=device)
    lin_t = Linear(n_neurons=4, input_size=inputs.shape[-1], bias=False)
    lin_t.w.weight = torch.nn.Parameter(torch.eye(inputs.shape[-1], device=device))
    outputs = lin_t(inputs)
    assert torch.all(torch.eq(inputs, outputs))

    assert torch.jit.trace(lin_t, inputs)
