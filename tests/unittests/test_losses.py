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

    # Try with weight
    weight = torch.full((5,), 0.5)
    out_cost = bce_loss(predictions, targets, weight=weight)
    assert torch.allclose(
        torch.exp(out_cost), torch.tensor(2.0, device=device).sqrt()
    )

    # Try with smoothing
    out_cost = bce_loss(predictions, targets, label_smoothing=0.5)
    assert torch.allclose(
        torch.exp(out_cost), torch.tensor(2.0, device=device).sqrt()
    )


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


def test_jeffreys_loss_without_regularization(device):
    import torch.nn.functional as F

    from speechbrain.nnet.losses import JeffreysLoss

    torch.manual_seed(0)
    n_classes = 5
    predictions = torch.randn(8, n_classes, device=device)
    targets = torch.randint(0, n_classes, (8,), device=device)

    # Without smoothing nor regularization, it is the cross entropy.
    loss = JeffreysLoss()(predictions, targets)
    assert torch.allclose(loss, F.cross_entropy(predictions, targets))

    # coeff1 alone is label smoothing over the non-target classes, which is
    # PyTorch label smoothing over all classes with a rescaled value.
    coeff1 = 0.1
    loss = JeffreysLoss(coeff1=coeff1)(predictions, targets)
    ref = F.cross_entropy(
        predictions,
        targets,
        label_smoothing=coeff1 * n_classes / (n_classes - 1),
    )
    assert torch.allclose(loss, ref, atol=1e-6)


def test_jeffreys_loss_value(device):
    import torch.nn.functional as F

    from speechbrain.nnet.losses import JeffreysLoss

    torch.manual_seed(0)
    n_classes = 6
    coeff1, coeff2 = 0.1, 0.05
    predictions = torch.randn(4, n_classes, device=device)
    targets = torch.randint(0, n_classes, (4,), device=device)

    probs = F.softmax(predictions, dim=-1)
    log_probs = F.log_softmax(predictions, dim=-1)
    one_hot = F.one_hot(targets, n_classes).float()
    smoothed = one_hot * (1 - coeff1 - coeff2) + (1 - one_hot) * coeff1 / (
        n_classes - 1
    )
    cross_entropy = -(smoothed * log_probs).sum(-1)
    p_target = (one_hot * probs).sum(-1)
    jeffreys = ((1 - one_hot) * probs * log_probs).sum(-1) / (1 - p_target)
    ref = (cross_entropy + coeff2 * jeffreys).mean()

    loss = JeffreysLoss(coeff1=coeff1, coeff2=coeff2)(predictions, targets)
    assert torch.allclose(loss, ref, atol=1e-5)


def test_jeffreys_loss_shapes_and_reductions(device):
    from speechbrain.nnet.losses import JeffreysLoss

    torch.manual_seed(0)
    predictions = torch.randn(3, 4, device=device)
    targets = torch.tensor([0, 2, 3], device=device)
    criterion = JeffreysLoss(coeff1=0.1, coeff2=0.05)

    # Pooled embeddings come as [batch, 1, classes] with [batch, 1] targets.
    loss = criterion(predictions, targets)
    loss_3d = criterion(predictions.unsqueeze(1), targets.unsqueeze(1))
    assert torch.allclose(loss, loss_3d)

    per_utterance = JeffreysLoss(coeff1=0.1, coeff2=0.05, reduction="batch")(
        predictions, targets
    )
    assert per_utterance.shape == (3,)
    assert torch.allclose(per_utterance.mean(), loss)

    batchmean = JeffreysLoss(coeff1=0.1, coeff2=0.05, reduction="batchmean")(
        predictions, targets
    )
    assert torch.allclose(batchmean, loss)


def test_jeffreys_loss_weight_and_gradient(device):
    from speechbrain.nnet.losses import JeffreysLoss

    torch.manual_seed(0)
    predictions = torch.randn(5, 4, device=device, requires_grad=True)
    targets = torch.randint(0, 4, (5,), device=device)

    loss = JeffreysLoss(coeff1=0.1, coeff2=0.05)(predictions, targets)
    weighted = JeffreysLoss(
        weight=torch.ones(4, device=device), coeff1=0.1, coeff2=0.05
    )(predictions, targets)
    assert torch.allclose(loss, weighted)

    loss.backward()
    assert torch.isfinite(predictions.grad).all()
    assert predictions.grad.abs().sum() > 0


def test_jeffreys_loss_invalid_coefficients(device):
    from speechbrain.nnet.losses import JeffreysLoss

    predictions = torch.randn(2, 3, device=device)
    targets = torch.tensor([0, 1], device=device)
    with pytest.raises(AssertionError):
        JeffreysLoss(coeff1=1.0)(predictions, targets)
    with pytest.raises(AssertionError):
        JeffreysLoss(coeff2=1.0)(predictions, targets)
