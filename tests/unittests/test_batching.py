import numpy as np
import pytest
import torch


def test_batch_pad_right_to(device):
    import random

    from speechbrain.utils.data_utils import batch_pad_right

    n_channels = 40
    batch_lens = [1, 5]

    for b in batch_lens:
        rand_lens = [random.randint(10, 53) for x in range(b)]
        tensors = [
            torch.ones((rand_lens[x], n_channels), device=device)
            for x in range(b)
        ]
        batched, lens = batch_pad_right(tensors)
        assert batched.shape[0] == b
        np.testing.assert_almost_equal(
            lens, [x / max(rand_lens) for x in rand_lens], decimal=3
        )

    for b in batch_lens:
        rand_lens = [random.randint(10, 53) for x in range(b)]
        tensors = [torch.ones(rand_lens[x], device=device) for x in range(b)]
        batched, lens = batch_pad_right(tensors)
        assert batched.shape[0] == b
        np.testing.assert_almost_equal(
            lens, [x / max(rand_lens) for x in rand_lens], decimal=3
        )


def test_paddedbatch(device):
    from speechbrain.dataio.batch import PaddedBatch

    batch = PaddedBatch(
        [
            {
                "id": "ex1",
                "foo": torch.Tensor([1.0]).to(device),
                "bar": torch.Tensor([1.0, 2.0, 3.0]).to(device),
            },
            {
                "id": "ex2",
                "foo": torch.Tensor([2.0, 1.0]).to(device),
                "bar": torch.Tensor([2.0]).to(device),
            },
        ]
    )
    batch.to(dtype=torch.half)
    assert batch.foo.data.dtype == torch.half
    assert batch["foo"][1].dtype == torch.half
    assert batch.bar.lengths.dtype == torch.half
    assert batch.foo.data.shape == torch.Size([2, 2])
    assert batch.bar.data.shape == torch.Size([2, 3])
    ids, foos, bars = batch
    assert ids == ["ex1", "ex2"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_pin_memory():
    from speechbrain.dataio.batch import PaddedBatch

    batch = PaddedBatch(
        [
            {
                "id": "ex1",
                "foo": torch.Tensor([1.0]),
                "bar": torch.Tensor([1.0, 2.0, 3.0]),
            },
            {
                "id": "ex2",
                "foo": torch.Tensor([2.0, 1.0]),
                "bar": torch.Tensor([2.0]),
            },
        ]
    )
    batch.pin_memory()
    assert batch.foo.data.is_pinned()
