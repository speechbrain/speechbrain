import pytest
import torch


def test_batch_pad_right_to():
    from speechbrain.utils.data_utils import batch_pad_right
    import random

    n_channels = 40
    batch_lens = [1, 5]

    for b in batch_lens:
        tensors = [
            torch.ones(n_channels, random.randint(10, 53),) for x in range(b)
        ]
        batched, lens = batch_pad_right(tensors)
        assert batched.shape[0] == b

    for b in batch_lens:
        tensors = [torch.ones(random.randint(10, 53),) for x in range(b)]
        batched, lens = batch_pad_right(tensors)
        assert batched.shape[0] == b


def test_paddedbatch():
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
