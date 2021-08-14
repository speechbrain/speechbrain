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
