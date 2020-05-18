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

    if torch.cuda.device_count() > 0:
        device = torch.device("cuda")
        cost = ComputeCost(cost_type="transducer", blank_index=0)
        log_probs = (
            torch.Tensor(
                [
                    [
                        [
                            [0.1, 0.6, 0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.6, 0.1, 0.1],
                            [0.1, 0.1, 0.2, 0.8, 0.1],
                        ],
                        [
                            [0.1, 0.6, 0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.2, 0.1, 0.1],
                            [0.7, 0.1, 0.2, 0.1, 0.1],
                        ],
                    ]
                ]
            )
            .cuda()
            .requires_grad_()
            .log_softmax(dim=-1)
        )
        targets = torch.Tensor([[1, 2]]).to(device).int()
        probs_length = torch.Tensor([1.0]).to(device)
        target_length = torch.Tensor([1.0]).to(device)
        out_cost = cost(log_probs, targets, [probs_length, target_length])
        assert out_cost.item() == 4.49566650390625
