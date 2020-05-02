import torch
import torch.nn


def test_pooling():

    from speechbrain.nnet.pooling import pooling

    inputs = torch.tensor([1, 3, 2]).unsqueeze(0).unsqueeze(0).float()
    pool = pooling("max", len(inputs.shape))
    pool.init_params(inputs)
    outputs = pool(inputs)
    assert outputs == 3
