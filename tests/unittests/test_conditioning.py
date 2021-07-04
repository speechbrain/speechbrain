import torch
import pytest

from math import dist
from numpy.lib.function_base import flip
from torch import nn
from speechbrain.nnet.conditioning import GarbageConditioner, ConditioningFeature

class FakeModel(nn.Module):
    def forward(self, x1, x2):
        return torch.stack([
                torch.sum(x1 * torch.arange(1, x1.size(-1)+1), dim=1),
                torch.prod(x2, dim=1)
            ],
            dim=1
        )


class FakeLoss(nn.Module):
    def forward(self, predicted, actual):
        return torch.sum(torch.abs(predicted - actual))


def get_sample_batch():
    x1 = torch.tensor(
        [[1., 2.],
         [3., 4.]]
    )

    x2 = torch.tensor(
        [[2., 3.],
         [4., 5.]]
    )
    return x1, x2

def x1_flip_sequence(batch):
    x1, x2 = batch
    return torch.flip(x1, dims=(-1,)), x2

def x2_to_zero(batch):
    x1, x2 = batch
    return x1, torch.zeros_like(x2)

def get_conditioner():
    model = FakeModel()
    loss = FakeLoss()
    return GarbageConditioner(
        model=lambda batch: model(*batch),
        criterion=loss,
        features=[
            ConditioningFeature(
                name="x1", weight=0.2, distortion=x1_flip_sequence),
            ConditioningFeature(
                name="x2", weight=0.1, distortion=x2_to_zero)
        ]
    )


def test_conditioning_predictions():
    conditioner = get_conditioner()
    batch = get_sample_batch()
    predictions = conditioner.compute_forward(batch)
    assert predictions is not None
    ref_actual_prediction = torch.tensor(
        [[5., 6.],
         [11., 20.]]
    )
    assert torch.allclose(predictions.actual , ref_actual_prediction)
    x1_preds = predictions.distorted['x1']
    ref_x1_preds = torch.tensor(
        [[4., 6.],
         [10., 20.]]
    )
    assert torch.allclose(x1_preds, ref_x1_preds)
    ref_x2_preds = torch.tensor(
        [[5., 0.],
         [11., 0.]]
    )
    x2_preds = predictions.distorted['x2']
    assert torch.allclose(x2_preds, ref_x2_preds)


def test_compute_objectives():
    conditioner = get_conditioner()
    batch = get_sample_batch()
    predictions = conditioner.compute_forward(batch)
    targets = torch.tensor(
        [[ 6.,  7.],
         [12., 21.]])
    loss = conditioner.compute_objectives_detailed(
        predictions, targets)
    assert loss.raw_loss == 4
    assert loss.feature_loss['x1'].raw_loss == pytest.approx(6.)
    assert loss.feature_loss['x1'].difference_loss == pytest.approx(-2.)
    assert loss.feature_loss['x1'].weighted_difference_loss == (
        pytest.approx(-0.4))
    assert loss.feature_loss['x2'].raw_loss == pytest.approx(30.)
    assert loss.feature_loss['x2'].difference_loss == pytest.approx(-26.)
    assert loss.feature_loss['x2'].weighted_difference_loss == (
        pytest.approx(-2.6))

    assert loss.effective_loss == pytest.approx(1.0)
    assert loss.garbage_loss == pytest.approx(-3.0)

    loss = conditioner.compute_objectives(predictions, targets)
    assert loss == pytest.approx(1.0)
