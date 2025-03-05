import numpy
import pytest
import torch

from speechbrain.processing.features import InputNormalization


class TestInputNormalization:
    """Test suite for the InputNormalization class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Create a batch of 3 sequences with 4 features and variable lengths
        x = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            dtype=torch.float32,
        )

        # Relative lengths: first has 100%, second 75%, third 50%
        lengths = torch.tensor([1.0, 0.75, 0.5], dtype=torch.float32)

        # Mask of the input tensor based on the lengths
        mask = torch.tensor(
            [
                [True, True, True, True],
                [True, True, True, False],
                [True, True, False, False],
            ]
        )

        return x, lengths, mask

    def test_constructor_defaults(self):
        """Test constructor with default parameters."""
        norm = InputNormalization()
        assert norm.mean_norm is True
        assert norm.std_norm is True
        assert norm.norm_type == "global"
        assert norm.update_until_epoch == 2
        assert norm.avoid_padding_norm is False
        assert norm.epsilon == 1e-10

    def test_invalid_norm_type(self):
        """Test that invalid norm_type raises an error."""
        with pytest.raises(ValueError):
            InputNormalization(norm_type="invalid")

    def test_sentence_normalization(self, sample_data):
        """Test sentence-level normalization."""
        x, lengths, mask = sample_data
        norm = InputNormalization(norm_type="sentence", avoid_padding_norm=True)

        # Manual calculation for comparison, using corrected variance
        expected_means = torch.tensor([2.5, 6.0, 9.5]).view(-1, 1)
        expected_stds = torch.tensor([5.0 / 3.0, 1.0, 0.5]).sqrt().view(-1, 1)

        # Apply sentence normalization
        output = norm(x, lengths)

        # Calculate expected output manually
        expected = torch.zeros_like(x)
        for i in range(3):
            expected[i, mask[i]] = (
                x[i, mask[i]] - expected_means[i]
            ) / expected_stds[i]

        # Check if output matches expected values where not padding
        assert torch.allclose(output[mask], expected[mask], atol=1e-5)

        # Check if padding values are preserved properly
        assert torch.allclose(output[~mask], x[~mask], atol=1e-5)

    def test_batch_normalization(self, sample_data):
        """Test batch-level normalization."""
        x, lengths, mask = sample_data
        norm = InputNormalization(norm_type="batch", avoid_padding_norm=True)

        # Apply normalization
        output = norm(x, lengths)

        # Manual calculation for comparison. See code for note about
        # how this algorithm is not correct, kept only for backwards compat.
        mean_mean = torch.tensor([2.5, 6.0, 9.5]).mean()
        expected_mean = (x - mean_mean)[mask].mean()
        std_mean = torch.tensor([5.0 / 3.0, 1.0, 0.5]).sqrt().mean()
        expected_std = ((x - mean_mean)[mask] / std_mean).std()

        # Check normalization was applied correctly to non-padding values
        assert torch.allclose(output[mask].mean(), expected_mean, atol=1e-3)
        assert torch.allclose(output[mask].std(), expected_std, atol=1e-2)

        # Check if padding values are preserved properly
        assert torch.allclose(output[~mask], x[~mask], atol=1e-5)

    def test_global_normalization_updates(self, sample_data):
        """Test global normalization with statistics updates."""
        x, lengths, mask = sample_data
        norm = InputNormalization(norm_type="global")

        # Apply normalization multiple times to accumulate statistics
        # First call (epoch 0)
        output1 = norm(x, lengths, epoch=0)

        # Second call (epoch 0) with different data
        x2 = x + 1.0  # Shift data by 1
        _ = norm(x2, lengths, epoch=0)

        # Third call (epoch 0) with original data
        output3 = norm(x, lengths, epoch=0)

        # Check that mean and variance have been updated
        # The running stats should be somewhere between the original x and x2
        valid_values1 = output1[mask]
        valid_values3 = output3[mask]

        # Since we've seen the same pattern twice and the shifted pattern once,
        # the normalized output should be different for the same input
        assert not torch.allclose(valid_values1, valid_values3, atol=0.1)

        # Check that after many trials, the overall stats match the tensor stats
        for i in range(1000):
            _ = norm(x, lengths)

        assert torch.allclose(norm.glob_mean, x[mask].mean(), atol=1e-3)
        assert torch.allclose(norm.glob_std, x[mask].std(), atol=1e-2)

    def test_global_normalization_stops_updates(self, sample_data):
        """Test that global normalization stops updates after specified epoch."""
        x, lengths, mask = sample_data
        norm = InputNormalization(norm_type="global", update_until_epoch=2)

        # First call (epoch 0) - should update statistics
        _ = norm(x, lengths, epoch=0)

        # Save statistics after first epoch
        saved_mean = norm.glob_mean.clone()
        saved_std = norm.glob_std.clone()
        saved_count = norm.count

        # Second call (epoch 1) - should update statistics
        x2 = x + 2.0  # Shift data by 2
        _ = norm(x2, lengths, epoch=1)

        # Check that statistics have been updated
        assert not torch.allclose(saved_mean, norm.glob_mean, atol=1e-5)
        assert not torch.allclose(saved_std, norm.glob_std, atol=1e-5)

        # Save statistics after second input
        saved_mean = norm.glob_mean.clone()
        saved_std = norm.glob_std.clone()
        saved_count = norm.count

        # Third call (epoch 2) - should not update statistics
        x3 = x + 4.0  # Shift data by 4
        _ = norm(x3, lengths, epoch=2)

        # Check that statistics have not been updated
        assert torch.allclose(saved_mean, norm.glob_mean, atol=1e-5)
        assert torch.allclose(saved_std, norm.glob_std, atol=1e-5)
        assert saved_count == norm.count

    def test_no_std_normalization(self, sample_data):
        """Test normalization with std_norm=False."""
        x, lengths, mask = sample_data
        norm = InputNormalization(norm_type="global", std_norm=False)

        # Apply normalization
        output = norm(x, lengths)

        # Check that mean normalization was applied but std normalization wasn't
        assert torch.allclose(output[mask].mean(), torch.zeros(1), atol=1e-5)
        assert not torch.allclose(output[mask].std(), torch.ones(1), atol=0.1)

    def test_save_load(self, tmp_path, sample_data):
        """Test save and load functionality."""
        x, lengths, mask = sample_data
        norm = InputNormalization(norm_type="global")

        # Train the normalizer with multiple batches
        _ = norm(x, lengths, epoch=0)
        _ = norm(x + 1.0, lengths, epoch=0)

        # Save the statistics
        save_path = tmp_path / "norm_stats.pt"
        norm._save(save_path)

        # Create a new normalizer and load the statistics
        new_norm = InputNormalization(norm_type="global")
        new_norm._load(save_path)

        # Check that the loaded statistics match
        assert torch.allclose(norm.glob_mean, new_norm.glob_mean)
        assert torch.allclose(norm.glob_std, new_norm.glob_std)
        assert norm.count == new_norm.count

        # Ensure both normalizers produce the same output
        output1 = norm(
            x, lengths, epoch=3
        )  # Beyond update epoch to avoid stats change
        output2 = new_norm(x, lengths, epoch=3)
        assert torch.allclose(output1, output2)

    def test_to_device(self, sample_data):
        """Test moving the normalizer to a different device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        x, lengths, mask = sample_data
        norm = InputNormalization(norm_type="global")

        # Update statistics
        _ = norm(x, lengths)

        # Move to GPU
        norm = norm.to("cuda")

        # Check that tensors are on the right device
        assert norm.glob_mean.device.type == "cuda"
        assert norm.glob_std.device.type == "cuda"

        # Test on GPU
        x_cuda = x.to("cuda")
        lengths_cuda = lengths.to("cuda")
        output = norm(x_cuda, lengths_cuda)

        # Check output is on the right device
        assert output.device.type == "cuda"


# Utility tests for the helper functions


def test_get_mask():
    """Test the get_mask function."""
    from speechbrain.processing.features import get_mask

    # Create a batch of sequences
    x = torch.ones(3, 4, 2)  # Batch size 3, seq len 4, feature dim 2

    # Relative lengths: 100%, 75%, 50%
    lengths = torch.tensor([1.0, 0.75, 0.5])

    # Get mask
    mask = get_mask(x, lengths, length_dim=1)

    # Expected mask:
    # First sequence: all True
    # Second sequence: first 3 True, last False
    # Third sequence: first 2 True, last 2 False
    expected_mask = torch.tensor(
        [
            [[True, True], [True, True], [True, True], [True, True]],
            [[True, True], [True, True], [True, True], [False, False]],
            [[True, True], [True, True], [False, False], [False, False]],
        ]
    )

    assert torch.equal(mask, expected_mask)


def parallel_mean_var_update(rank, world_size, tmpdir):
    """Test that the mean_var_norm works in ddp."""

    import os

    from speechbrain.processing.features import mean_std_update

    # The masked tensors are [1., 2.] and [2., 3.] for ranks 0 and 1
    tensor = torch.tensor([[1.0, 2.0, 3.0]]) + rank
    mask = torch.tensor([[True, True, False]])
    dims = (1,)

    # Running values should be the same between processes
    run_count = 2
    run_mean = torch.tensor([2.0])
    run_std = torch.tensor([1.0])

    # initialize the process group
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    sync_file = f"file://{tmpdir}/sync"
    torch.distributed.init_process_group(
        "gloo", rank=rank, world_size=world_size, init_method=sync_file
    )

    # Run mean_var_norm
    new_count, new_mean, new_std = mean_std_update(
        tensor, mask, dims, run_count, run_mean, run_std
    )

    # Expected values
    expected_count = run_count + 2 + 2
    expected_mean = run_mean + (1.0 - 1.0) / expected_count
    expected_std = ((2.0 + 1.0 + 1.0) / (expected_count - 1)) ** 0.5

    # Same values on all processes
    assert numpy.isclose(new_count, expected_count)
    assert numpy.isclose(new_mean, expected_mean)
    assert numpy.isclose(new_std, expected_std)


def test_mean_var_update_parallel(tmpdir):
    """Test the mean_var_update function in parallel."""
    world_size = 2
    torch.multiprocessing.spawn(
        parallel_mean_var_update,
        args=(world_size, tmpdir),
        nprocs=world_size,
        join=True,
    )
