import pytest


def test_scalarize():
    import torch
    from collections import namedtuple
    from speechbrain.utils.data_utils import scalarize

    values = {"foo": torch.tensor(1.2), "bar": torch.tensor(2)}
    scalarized = scalarize(values)
    assert isinstance(scalarized["foo"], float)
    assert isinstance(scalarized["bar"], int)
    assert scalarized["foo"] == pytest.approx(1.2)
    assert scalarized["bar"] == 2
    Loss = namedtuple("Loss", "foo bar")
    values = Loss(foo=torch.tensor(1.2), bar=torch.tensor(2))
    scalarized = scalarize(values)
    assert scalarized["foo"] == pytest.approx(1.2)
    assert scalarized["bar"] == 2


def test_detach():
    import torch
    from speechbrain.utils.data_utils import detach

    x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    x_det = detach(x)
    assert not x_det.requires_grad
    batch = {
        "x": torch.tensor([1.0, 2.0], requires_grad=True),
        "y": torch.tensor([1.0, 2.0, 3.0], requires_grad=True),
        "sub": {
            "x": torch.tensor([3.0, 2.0], requires_grad=True),
            "y": torch.tensor([5.0, 2.0, 3.0], requires_grad=True),
        },
    }
    batch_det = detach(batch)
    assert not batch_det["x"].requires_grad
    assert not batch_det["y"].requires_grad
    assert not batch_det["sub"]["x"].requires_grad
    assert not batch_det["sub"]["y"].requires_grad
