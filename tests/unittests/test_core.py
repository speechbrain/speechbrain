def test_parse_arguments():
    from speechbrain.core import parse_arguments

    filename, overrides = parse_arguments(
        ["params.yaml", "--seed", "3", "--data_folder", "TIMIT"]
    )
    assert filename == "params.yaml"
    assert overrides == "data_folder: TIMIT\nseed: 3\n"


def test_brain():
    import torch
    from speechbrain.core import Brain
    from speechbrain.nnet.optimizers import SGD_Optimizer

    model = torch.nn.Linear(in_features=10, out_features=10)

    class SimpleBrain(Brain):
        def compute_forward(self, x, train_mode=True, init_params=False):
            return model(x)

        def compute_objectives(self, predictions, targets, train_mode=True):
            return torch.nn.functional.l1_loss(predictions, targets), {}

    inputs = torch.rand(10, 10)
    brain = SimpleBrain(
        modules=[model], optimizer=SGD_Optimizer(0.1), first_inputs=[inputs],
    )

    targets = torch.rand(10, 10)
    train_set = ([inputs, targets],)

    start_output = brain.compute_forward(inputs)
    start_loss = brain.compute_objectives(start_output, targets)
    brain.fit(epoch_counter=range(10), train_set=train_set)
    end_output = brain.compute_forward(inputs)
    end_loss = brain.compute_objectives(end_output, targets)
    assert end_loss < start_loss
