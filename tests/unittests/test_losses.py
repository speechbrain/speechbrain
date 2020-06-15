import torch
import pytest


def test_nll():
    from speechbrain.nnet.losses import nll_loss

    predictions = torch.zeros(4, 10, 8)
    targets = torch.zeros(4, 10)
    lengths = torch.ones(4)
    out_cost = nll_loss(predictions, targets, lengths)
    assert torch.all(torch.eq(out_cost, 0))


def test_mse():
    from speechbrain.nnet.losses import mse_loss

    predictions = torch.ones(4, 10, 8)
    targets = torch.ones(4, 10, 8)
    lengths = torch.ones(4)
    out_cost = mse_loss(predictions, targets, lengths)
    assert torch.all(torch.eq(out_cost, 0))

    predictions = torch.zeros(4, 10, 8)
    out_cost = mse_loss(predictions, targets, lengths)
    assert torch.all(torch.eq(out_cost, 1))


def test_l1():
    from speechbrain.nnet.losses import l1_loss

    predictions = torch.ones(4, 10, 8)
    targets = torch.ones(4, 10, 8)
    lengths = torch.ones(4)
    out_cost = l1_loss(predictions, targets, lengths)
    assert torch.all(torch.eq(out_cost, 0))


def test_classification_error():
    from speechbrain.nnet.losses import classification_error

    predictions = torch.zeros(4, 10, 8)
    predictions[:, :, 0] += 1.0
    targets = torch.zeros(4, 10)
    lengths = torch.ones(4)
    out_cost = classification_error(predictions, targets, lengths)
    assert torch.all(torch.eq(out_cost, 0))


def test_transducer_loss():
    # Make this its own test since it can only be run
    # if numba is installed and a GPU is available
    pytest.importorskip("numba")
    from speechbrain.nnet.losses import transducer_loss

    if torch.cuda.device_count() > 0:
        pytest.skip("This test can only be run if a GPU is available")
    device = torch.device("cuda")
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
    out_cost = transducer_loss(
        log_probs, targets, probs_length, target_length, blank_index=0
    )
    assert out_cost.item() == 4.49566650390625
