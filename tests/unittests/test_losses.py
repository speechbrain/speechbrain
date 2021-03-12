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


def test_bce_loss():
    from speechbrain.nnet.losses import bce_loss

    # Ensure this works both with and without singleton dimension
    predictions_singleton = torch.zeros(4, 10, 1)
    predictions_match = torch.zeros(4, 10)
    targets = torch.ones(4, 10)
    lengths = torch.ones(4)
    out_cost_singleton = bce_loss(predictions_singleton, targets, lengths)
    out_cost_match = bce_loss(predictions_match, targets, lengths)
    assert torch.allclose(torch.exp(out_cost_singleton), torch.tensor(2.0))
    assert torch.allclose(torch.exp(out_cost_match), torch.tensor(2.0))

    # How about one dimensional inputs
    predictions = torch.zeros(5, 1)
    targets = torch.ones(5)
    out_cost = bce_loss(predictions, targets)
    assert torch.allclose(torch.exp(out_cost), torch.tensor(2.0))

    # Can't pass lengths in 1D case
    with pytest.raises(ValueError):
        bce_loss(predictions, targets, length=torch.ones(5))


def test_classification_error():
    from speechbrain.nnet.losses import classification_error

    predictions = torch.zeros(4, 10, 8)
    predictions[:, :, 0] += 1.0
    targets = torch.zeros(4, 10)
    lengths = torch.ones(4)
    out_cost = classification_error(predictions, targets, lengths)
    assert torch.all(torch.eq(out_cost, 0))


def test_pitwrapper():
    from speechbrain.nnet.losses import PitWrapper
    import torch
    from torch import nn

    base_loss = nn.MSELoss(reduction="none")
    pit = PitWrapper(base_loss)
    predictions = torch.rand((2, 32, 4))  # batch, frames, sources
    p = (3, 0, 2, 1)
    # same but we invert the ordering to check if permutation invariant
    targets = predictions[..., p]
    loss, opt_p = pit(predictions, targets)
    assert [x == p for x in opt_p] == [True for i in range(len(opt_p))]
    predictions = pit.reorder_tensor(predictions, opt_p)
    assert torch.all(torch.eq(base_loss(predictions, targets), 0))

    predictions = torch.rand((3, 32, 32, 32, 5))  # batch, ..., sources
    p = (3, 0, 2, 1, 4)
    targets = predictions[
        ..., p
    ]  # same but we invert the ordering to check if permutation invariant
    loss, opt_p = pit(predictions, targets)
    assert [x == p for x in opt_p] == [True for i in range(len(opt_p))]
    predictions = pit.reorder_tensor(predictions, opt_p)
    assert torch.all(torch.eq(base_loss(predictions, targets), 0))


def test_transducer_loss():
    # Make this its own test since it can only be run
    # if numba is installed and a GPU is available
    pytest.importorskip("numba")
    if torch.cuda.device_count() == 0:
        pytest.skip("This test can only be run if a GPU is available")

    from speechbrain.nnet.losses import transducer_loss

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
        .to(device)
        .requires_grad_()
        .log_softmax(dim=-1)
    )
    targets = torch.Tensor([[1, 2]]).to(device).int()
    probs_length = torch.Tensor([1.0]).to(device)
    target_length = torch.Tensor([1.0]).to(device)
    out_cost = transducer_loss(
        log_probs, targets, probs_length, target_length, blank_index=0
    )
    out_cost.backward()
    assert out_cost.item() == 2.247833251953125
