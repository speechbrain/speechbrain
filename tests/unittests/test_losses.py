import pytest
import torch


def test_nll(device):
    from speechbrain.nnet.losses import nll_loss

    predictions = torch.zeros(4, 10, 8, device=device)
    targets = torch.zeros(4, 10, device=device)
    lengths = torch.ones(4, device=device)
    out_cost = nll_loss(predictions, targets, lengths)
    assert torch.all(torch.eq(out_cost, 0))


def test_mse(device):
    from speechbrain.nnet.losses import mse_loss

    predictions = torch.ones(4, 10, 8, device=device)
    targets = torch.ones(4, 10, 8, device=device)
    lengths = torch.ones(4, device=device)
    out_cost = mse_loss(predictions, targets, lengths)
    assert torch.all(torch.eq(out_cost, 0))

    predictions = torch.zeros(4, 10, 8, device=device)
    out_cost = mse_loss(predictions, targets, lengths)
    assert torch.all(torch.eq(out_cost, 1))


def test_l1(device):
    from speechbrain.nnet.losses import l1_loss

    predictions = torch.ones(4, 10, 8, device=device)
    targets = torch.ones(4, 10, 8, device=device)
    lengths = torch.ones(4, device=device)
    out_cost = l1_loss(predictions, targets, lengths)
    assert torch.all(torch.eq(out_cost, 0))


def test_bce_loss(device):
    from speechbrain.nnet.losses import bce_loss

    # Ensure this works both with and without singleton dimension
    predictions_singleton = torch.zeros(4, 10, 1, device=device)
    predictions_match = torch.zeros(4, 10, device=device)
    targets = torch.ones(4, 10, device=device)
    lengths = torch.ones(4, device=device)
    out_cost_singleton = bce_loss(predictions_singleton, targets, lengths)
    out_cost_match = bce_loss(predictions_match, targets, lengths)
    assert torch.allclose(
        torch.exp(out_cost_singleton), torch.tensor(2.0, device=device)
    )
    assert torch.allclose(
        torch.exp(out_cost_match), torch.tensor(2.0, device=device)
    )

    # How about one dimensional inputs
    predictions = torch.zeros(5, 1, device=device)
    targets = torch.ones(5, device=device)
    out_cost = bce_loss(predictions, targets)
    assert torch.allclose(torch.exp(out_cost), torch.tensor(2.0, device=device))

    # Can't pass lengths in 1D case
    with pytest.raises(ValueError):
        bce_loss(predictions, targets, length=torch.ones(5, device=device))


def test_classification_error(device):
    from speechbrain.nnet.losses import classification_error

    predictions = torch.zeros(4, 10, 8, device=device)
    predictions[:, :, 0] += 1.0
    targets = torch.zeros(4, 10, device=device)
    lengths = torch.ones(4, device=device)
    out_cost = classification_error(predictions, targets, lengths)
    assert torch.all(torch.eq(out_cost, 0))


def test_pitwrapper(device):
    import torch
    from torch import nn

    from speechbrain.nnet.losses import PitWrapper

    base_loss = nn.MSELoss(reduction="none")
    pit = PitWrapper(base_loss)
    predictions = torch.rand(
        (2, 32, 4), device=device
    )  # batch, frames, sources
    p = (3, 0, 2, 1)
    # same but we invert the ordering to check if permutation invariant
    targets = predictions[..., p]
    loss, opt_p = pit(predictions, targets)
    assert [x == p for x in opt_p] == [True for i in range(len(opt_p))]
    predictions = pit.reorder_tensor(predictions, opt_p)
    assert torch.all(torch.eq(base_loss(predictions, targets), 0))

    predictions = torch.rand(
        (3, 32, 32, 32, 5), device=device
    )  # batch, ..., sources
    p = (3, 0, 2, 1, 4)
    targets = predictions[
        ..., p
    ]  # same but we invert the ordering to check if permutation invariant
    loss, opt_p = pit(predictions, targets)
    assert [x == p for x in opt_p] == [True for i in range(len(opt_p))]
    predictions = pit.reorder_tensor(predictions, opt_p)
    assert torch.all(torch.eq(base_loss(predictions, targets), 0))


def test_transducer_loss(device):
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
        log_probs,
        targets,
        probs_length,
        target_length,
        blank_index=0,
        use_torchaudio=False,
    )
    out_cost.backward()
    assert out_cost.item() == pytest.approx(2.2478, 0.0001)


def test_guided_attention_loss_mask(device):
    from speechbrain.nnet.loss.guidedattn_loss import GuidedAttentionLoss

    loss = GuidedAttentionLoss().to(device)
    input_lengths = torch.tensor([3, 2, 6], device=device)
    output_lengths = torch.tensor([4, 3, 5], device=device)
    soft_mask = loss.guided_attentions(input_lengths, output_lengths)
    ref_soft_mask = torch.tensor(
        [
            [
                [0.0, 0.54216665, 0.9560631, 0.9991162, 0.0],
                [0.7506478, 0.08314464, 0.2933517, 0.8858382, 0.0],
                [0.9961341, 0.8858382, 0.2933517, 0.08314464, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.7506478, 0.9961341, 0.0, 0.0],
                [0.9560631, 0.2933517, 0.2933517, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.39346933, 0.86466473, 0.988891, 0.99966455],
                [0.2933517, 0.01379288, 0.49366438, 0.90436554, 0.993355],
                [0.7506478, 0.1992626, 0.05404053, 0.5888877, 0.93427145],
                [0.9560631, 0.6753475, 0.1175031, 0.1175031, 0.6753475],
                [0.9961341, 0.93427145, 0.5888877, 0.05404053, 0.1992626],
                [0.9998301, 0.993355, 0.90436554, 0.49366438, 0.01379288],
            ],
        ],
        device=device,
    )
    assert torch.allclose(soft_mask, ref_soft_mask)


def test_guided_attention_loss_value(device):
    from speechbrain.nnet.loss.guidedattn_loss import GuidedAttentionLoss

    loss = GuidedAttentionLoss().to(device)
    input_lengths = torch.tensor([2, 3], device=device)
    target_lengths = torch.tensor([3, 4], device=device)
    alignments = torch.tensor(
        [
            [
                [0.8, 0.2, 0.0],
                [0.4, 0.6, 0.0],
                [0.2, 0.8, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.6, 0.2, 0.2],
                [0.1, 0.7, 0.2],
                [0.3, 0.4, 0.3],
                [0.2, 0.3, 0.5],
            ],
        ],
        device=device,
    )
    loss_value = loss(alignments, input_lengths, target_lengths)
    ref_loss_value = torch.tensor(0.1142)
    assert torch.isclose(loss_value, ref_loss_value, 0.0001, 0.0001).item()


def test_guided_attention_loss_shapes(device):
    from speechbrain.nnet.loss.guidedattn_loss import GuidedAttentionLoss

    loss = GuidedAttentionLoss().to(device)
    input_lengths = torch.tensor([3, 2, 6], device=device)
    output_lengths = torch.tensor([4, 3, 5], device=device)
    soft_mask = loss.guided_attentions(input_lengths, output_lengths)
    assert soft_mask.shape == (3, 6, 5)
    soft_mask = loss.guided_attentions(
        input_lengths, output_lengths, max_input_len=10
    )
    assert soft_mask.shape == (3, 10, 5)
    soft_mask = loss.guided_attentions(
        input_lengths, output_lengths, max_target_len=12
    )
    assert soft_mask.shape == (3, 6, 12)
    soft_mask = loss.guided_attentions(
        input_lengths, output_lengths, max_input_len=10, max_target_len=12
    )
    assert soft_mask.shape == (3, 10, 12)
