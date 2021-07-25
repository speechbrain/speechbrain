import torch
import torch.nn


def test_softmax():

    from speechbrain.nnet.activations import Softmax

    inputs = torch.tensor([1, 2, 3]).float()
    act = Softmax(apply_log=False)
    outputs = act(inputs)
    assert torch.argmax(outputs) == 2

    assert torch.jit.trace(act, inputs)
