import torch


def test_losses():

    from speechbrain.nnet.losses import ComputeCost

    cost = ComputeCost(cost_type="nll")
    predictions = torch.zeros(4, 10, 8)
    targets = torch.zeros(4, 10)
    lengths = torch.ones(4)
    out_cost = cost(predictions, targets, lengths)
    assert torch.all(torch.eq(out_cost, 0))

    cost = ComputeCost(cost_type="mse")
    predictions = torch.ones(4, 10, 8)
    targets = torch.ones(4, 10, 8)
    lengths = torch.ones(4)
    out_cost = cost(predictions, targets, lengths)
    assert torch.all(torch.eq(out_cost, 0))

    predictions = torch.zeros(4, 10, 8)
    out_cost = cost(predictions, targets, lengths)
    assert torch.all(torch.eq(out_cost, 1))

    cost = ComputeCost(cost_type="l1")
    predictions = torch.ones(4, 10, 8)
    targets = torch.ones(4, 10, 8)
    lengths = torch.ones(4)
    out_cost = cost(predictions, targets, lengths)
    assert torch.all(torch.eq(out_cost, 0))

    predictions = torch.zeros(4, 10, 8)
    out_cost = cost(predictions, targets, lengths)
    assert torch.all(torch.eq(out_cost, 1))
