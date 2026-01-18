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


def test_paddedbatch_per_key_padding(device):
    """Test per-key padding configuration functionality."""
    from speechbrain.dataio.batch import PaddedBatch

    examples = [
        {
            "wav": torch.tensor([1, 2, 3]).to(device),
            "labels": torch.tensor([1, 2]).to(device),
        },
        {
            "wav": torch.tensor([4, 5]).to(device),
            "labels": torch.tensor([3]).to(device),
        },
    ]

    # Configure different padding values for different keys
    per_key_padding_kwargs = {
        "wav": {"value": 0},  # Pad wav with 0
        "labels": {"value": -100},  # Pad labels with -100
    }

    batch = PaddedBatch(examples, per_key_padding_kwargs=per_key_padding_kwargs)

    # Check that wav is padded with 0
    assert torch.all(batch.wav.data[1, 2:] == 0)
    assert torch.all(
        batch.wav.data[0, :3] == torch.tensor([1, 2, 3]).to(device)
    )

    # Check that labels is padded with -100
    assert torch.all(batch.labels.data[1, 1:] == -100)
    assert torch.all(
        batch.labels.data[0, :2] == torch.tensor([1, 2]).to(device)
    )


def test_paddedbatch_mixed_padding_config(device):
    """Test mixed configuration where some keys use global config and others use per-key config."""
    from speechbrain.dataio.batch import PaddedBatch

    examples = [
        {
            "wav": torch.tensor([1, 2, 3]).to(device),
            "labels": torch.tensor([1, 2]).to(device),
            "features": torch.tensor([0.1, 0.2]).to(device),
        },
        {
            "wav": torch.tensor([4, 5]).to(device),
            "labels": torch.tensor([3]).to(device),
            "features": torch.tensor([0.3]).to(device),
        },
    ]

    # Global padding config (default)
    padding_kwargs = {"value": 0}

    # Per-key config (overrides global for specific keys)
    per_key_padding_kwargs = {
        "labels": {"value": -100}  # Only labels get special padding
    }

    batch = PaddedBatch(
        examples,
        padding_kwargs=padding_kwargs,
        per_key_padding_kwargs=per_key_padding_kwargs,
    )

    # Check that wav uses global padding (0)
    assert torch.all(batch.wav.data[1, 2:] == 0)

    # Check that labels uses per-key padding (-100)
    assert torch.all(batch.labels.data[1, 1:] == -100)

    # Check that features uses global padding (0)
    assert torch.all(batch.features.data[1, 1:] == 0)


def test_paddedbatch_numpy_arrays():
    """Test with numpy arrays to ensure conversion works with per-key padding."""
    from speechbrain.dataio.batch import PaddedBatch

    examples = [
        {"wav": np.array([1, 2, 3]), "labels": np.array([1, 2])},
        {"wav": np.array([4, 5]), "labels": np.array([3])},
    ]

    per_key_padding_kwargs = {"wav": {"value": 0}, "labels": {"value": -100}}

    batch = PaddedBatch(examples, per_key_padding_kwargs=per_key_padding_kwargs)

    # Check that numpy arrays are converted to torch tensors and padded correctly
    assert isinstance(batch.wav.data, torch.Tensor)
    assert isinstance(batch.labels.data, torch.Tensor)

    # Check padding values
    assert torch.all(batch.wav.data[1, 2:] == 0)
    assert torch.all(batch.labels.data[1, 1:] == -100)


def test_paddedbatch_backward_compatibility(device):
    """Test that the new functionality maintains backward compatibility."""
    from speechbrain.dataio.batch import PaddedBatch

    examples = [
        {
            "wav": torch.tensor([1, 2, 3]).to(device),
            "labels": torch.tensor([1, 2]).to(device),
        },
        {
            "wav": torch.tensor([4, 5]).to(device),
            "labels": torch.tensor([3]).to(device),
        },
    ]

    # Test with only padding_kwargs (old behavior)
    batch_old = PaddedBatch(examples, padding_kwargs={"value": 0})

    # Test with only per_key_padding_kwargs (new behavior)
    batch_new = PaddedBatch(
        examples,
        per_key_padding_kwargs={"wav": {"value": 0}, "labels": {"value": 0}},
    )

    # Both should produce the same result
    assert torch.allclose(batch_old.wav.data, batch_new.wav.data)
    assert torch.allclose(batch_old.labels.data, batch_new.labels.data)
    assert torch.allclose(batch_old.wav.lengths, batch_new.wav.lengths)
    assert torch.allclose(batch_old.labels.lengths, batch_new.labels.lengths)
