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

    filename, args = parse_arguments(
        ["params.yaml", "--seed", "3", "--data_folder", "TIMIT"]
    )
    assert filename == "params.yaml"
    assert args == {"seed": 3, "data_folder": "TIMIT"}


def test_parse_overrides():
    from speechbrain.core import parse_overrides

    overrides = parse_overrides("{model.arg1: 1, model.arg2: 2}")
    assert overrides == {"model": {"arg1": 1, "arg2": 2}}


def test_brain():
    import torch
    from speechbrain.core import Brain
    from speechbrain.nnet.optimizers import Optimize

    model = torch.nn.Linear(in_features=10, out_features=10)

    class SimpleBrain(Brain):
        def compute_forward(self, x, train_mode=True, init_params=False):
            return model(x)

        def compute_objectives(self, predictions, targets, train_mode=True):
            return torch.nn.functional.l1_loss(predictions, targets)

    inputs = torch.rand(10, 10)
    brain = SimpleBrain(
        modules=[model], optimizer=Optimize("sgd", 0.1), first_inputs=[inputs],
    )

    targets = torch.rand(10, 10)
    train_set = ([inputs], [targets])

    start_output = brain.compute_forward(inputs)
    start_loss = brain.compute_objectives(start_output, targets)
    brain.fit(epoch_counter=range(10), train_set=train_set)
    end_output = brain.compute_forward(inputs)
    end_loss = brain.compute_objectives(end_output, targets)
    assert end_loss < start_loss
