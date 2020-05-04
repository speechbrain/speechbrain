import torch
import torch.nn


def test_softmax():

    from speechbrain.nnet.activations import softmax

    inputs = torch.tensor([1, 2, 3]).float()
    act = softmax(apply_log=False)
    outputs = act(inputs)
    assert torch.argmax(outputs) == 2
