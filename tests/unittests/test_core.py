def test_parse_arguments():
    from speechbrain.utils.run_opts import RunOptions

    run_opts = RunOptions()

    filename, run_opts, overrides = run_opts.from_command_line_args(
        ["params.yaml", "--device=cpu", "--seed=3", "--data_folder", "TIMIT"]
    )
    assert filename == "params.yaml"
    assert run_opts["device"] == "cpu"
    assert overrides == "seed: 3\ndata_folder: TIMIT"


def test_brain(device):
    import torch
    from torch.optim import SGD

    from speechbrain.core import Brain, Stage

    model = torch.nn.Linear(in_features=10, out_features=10, device=device)

    class SimpleBrain(Brain):
        def compute_forward(self, batch, stage):
            return self.modules.model(batch[0])

        def compute_objectives(self, predictions, batch, stage):
            return torch.nn.functional.l1_loss(predictions, batch[1])

    brain = SimpleBrain(
        {"model": model}, lambda x: SGD(x, 0.1), run_opts={"device": device}
    )

    inputs = torch.rand(10, 10, device=device)
    targets = torch.rand(10, 10, device=device)
    train_set = ([inputs, targets],)
    valid_set = ([inputs, targets],)

    start_output = brain.compute_forward(inputs, Stage.VALID)
    start_loss = brain.compute_objectives(start_output, targets, Stage.VALID)
    brain.fit(epoch_counter=range(10), train_set=train_set, valid_set=valid_set)
    end_output = brain.compute_forward(inputs, Stage.VALID)
    end_loss = brain.compute_objectives(end_output, targets, Stage.VALID)
    assert end_loss < start_loss
