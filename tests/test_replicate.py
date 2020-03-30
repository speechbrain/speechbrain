import pytest


def test_replicate():
    from speechbrain.core import Replicate

    module_list = [
        {'class': 'torch.nn.Linear', 'kwargs': {}},
        {'class': 'torch.nn.Linear', 'kwargs': {}},
    ]
    module_list[0]['kwargs']['in_features'] = 100
    module_list[0]['kwargs']['out_features'] = 200
    module_list[0]['kwargs']['bias'] = False
    module_list[1]['kwargs']['in_features'] = 200
    module_list[1]['kwargs']['out_features'] = 100
    module_list[1]['kwargs']['bias'] = False

    model = Replicate(number_of_copies=2, module_list=module_list)

    assert len(list(model.parameters())) == 4
