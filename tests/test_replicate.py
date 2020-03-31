import pytest


def test_replicate():
    from speechbrain.core import Replicate

    module_list = [
        {'class_name': 'torch.nn.Linear', 'kwargs': {}},
        {'class_name': 'torch.nn.Linear', 'kwargs': {}},
    ]
    module_list[0]['kwargs']['in_features'] = 100
    module_list[0]['kwargs']['out_features'] = 200
    module_list[0]['kwargs']['bias'] = False
    module_list[1]['kwargs']['in_features'] = 200
    module_list[1]['kwargs']['out_features'] = 100
    module_list[1]['kwargs']['bias'] = False

    model = Replicate(number_of_copies=2, module_list=module_list)

    assert len(list(model.parameters())) == 4
    assert model.block_list[0][0].out_features == 200

    # Vary parameter based on block position in the stack
    override_list = [
        {0: {'out_features': 100}, 1: {'in_features': 100}},
        {0: {'out_features': 300}, 1: {'in_features': 300}},
    ]
    model = Replicate(2, module_list, override_list)

    assert len(list(model.parameters())) == 4
    assert model.block_list[0][0].out_features == 100
    assert model.block_list[1][0].out_features == 300
