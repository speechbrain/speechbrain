import torch
import torch.nn


def test_pooling():

    from speechbrain.nnet.pooling import Pooling

    inputs = torch.tensor([1, 3, 2]).unsqueeze(0).unsqueeze(0).float()
    pool = Pooling("max", len(inputs.shape))
    pool.init_params(inputs)
    outputs = pool(inputs)
    assert outputs == 3
