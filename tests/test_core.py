import pytest


def test_nest():
    from speechbrain.core import nest

    params = {"a": {"b": "c"}}
    nest(params, ["a", "b"], "d")
    assert params == {"a": {"b": "d"}}
    nest(params, ["a", "c"], "e")
    assert params == {"a": {"b": "d", "c": "e"}}
    nest(params, ["a", "b"], {"d": "f"})
    assert params == {"a": {"b": {"d": "f"}, "c": "e"}}


def test_parse_arguments():
    from speechbrain.core import parse_arguments

    args = parse_arguments(["--seed", "3", "--data_folder", "TIMIT"])
    assert args == {"seed": 3, "data_folder": "TIMIT"}


def test_parse_overrides():
    from speechbrain.core import parse_overrides

    overrides = parse_overrides("{model.arg1: 1, model.arg1: 2}")
    assert overrides == {"model": {"arg1": 2}}


def test_experiment():
    from speechbrain.core import Experiment

    yaml = """
    constants:
        output_folder: exp
    """
    sb = Experiment(yaml)
    assert sb.output_folder == "exp"
    sb = Experiment(yaml, ["--output_folder", "exp/example"])
    assert sb.output_folder == "exp/example"

    yaml = """
    constants:
        output_folder: exp
    functions:
        output_folder: y
    """
    with pytest.raises(KeyError):
        sb = Experiment(yaml)


def test_brain():
    import torch
    from speechbrain.core import Brain
    from speechbrain.nnet.optimizers import optimize

    model = torch.nn.Linear(in_features=10, out_features=10)

    class SimpleBrain(Brain):
        def forward(self, x, init_params=False):
            return model(x)

        def compute_objectives(self, predictions, targets, train=True):
            return torch.nn.functional.l1_loss(predictions, targets)

    brain = SimpleBrain([model], optimize("sgd", 0.1))

    inputs = torch.rand(10, 10)
    targets = torch.rand(10, 10)
    train_set = ([inputs], [targets])

    start_loss = brain.compute_objectives(brain(inputs), targets)
    brain.learn(
        epoch_counter=range(10), train_set=train_set,
    )
    end_loss = brain.compute_objectives(brain(inputs), targets)
    assert end_loss < start_loss
