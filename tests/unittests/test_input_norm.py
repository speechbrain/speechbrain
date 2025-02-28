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
        assert norm.avg_factor is None
        assert norm.update_until_epoch == 2
        assert norm.epsilon == 1e-10

    def test_invalid_norm_type(self):
        """Test that invalid norm_type raises an error."""
        with pytest.raises(ValueError):
            InputNormalization(norm_type="invalid")

    def test_sentence_normalization(self, sample_data):
        """Test sentence-level normalization."""
        x, lengths, mask = sample_data
        norm = InputNormalization(norm_type="sentence")

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
        norm = InputNormalization(norm_type="batch")

        # Apply normalization
        output = norm(x, lengths)

        # Check normalization was applied correctly to non-padding values
        assert torch.allclose(output[mask].mean(), torch.zeros(1), atol=1e-5)
        assert torch.allclose(output[mask].std(), torch.ones(1), atol=1e-5)

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
        assert not torch.allclose(valid_values1, valid_values3, atol=1e-5)

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

    def test_normalization_with_avg_factor(self, sample_data):
        """Test normalization with custom avg_factor."""
        x, lengths, mask = sample_data
        avg_factor = 0.1  # Small weight to new samples
        norm = InputNormalization(norm_type="global", avg_factor=avg_factor)

        # First call - should initialize with this data
        _ = norm(x, lengths)

        # Save statistics after first batch
        saved_mean = norm.glob_mean.clone()

        # Second call with shifted data
        shift = 10.0
        x2 = x + shift
        _ = norm(x2, lengths)

        expected_mean = saved_mean + shift * avg_factor

        # Check that mean has been updated with the expected weight
        assert torch.allclose(norm.glob_mean, expected_mean, atol=1e-4)

        # Check that after many trials, the overall stats match the tensor stats
        for i in range(100):
            _ = norm(x, lengths)

        assert torch.allclose(norm.glob_mean, x[mask].mean(), atol=1e-4)
        assert torch.allclose(norm.glob_std, x[mask].std(), atol=1e-3)

    def test_no_std_normalization(self, sample_data):
        """Test normalization with std_norm=False."""
        x, lengths, mask = sample_data
        norm = InputNormalization(norm_type="batch", std_norm=False)

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


def test_global_norm_update():
    """Test the global_norm_update function."""

    # Setup
    tensor = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, True, True]])
    dims = (1,)
    weight = 1.0
    run_weight = 2.0
    run_mean = torch.tensor([2.0])
    run_std = torch.tensor([1.0])

    # Mock the distributed functions
    def mock_ddp_all_reduce(tensor, op):
        return tensor  # Just return the tensor, simulating no distribution

    # Patch the ddp_all_reduce function
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "speechbrain.processing.features":
            module = real_import(name, *args, **kwargs)
            module.ddp_all_reduce = mock_ddp_all_reduce
            return module
        return real_import(name, *args, **kwargs)

    builtins.__import__ = mock_import

    # Call the function
    try:
        from speechbrain.processing.features import mean_std_update

        updated_weight, updated_mean, updated_std = mean_std_update(
            tensor,
            mask=mask,
            dim=dims,
            weight=weight,
            run_sum=run_weight,
            run_mean=run_mean,
            run_std=run_std,
        )
    finally:
        builtins.__import__ = real_import

    # Check results
    assert updated_weight == 3.0  # weight + new_weight

    # Check mean update: old_mean + (new_mean - old_mean) * new_weight / (old_weight + new_weight)
    # new_mean = 2.0, old_mean = 2.0, so updated_mean should be 2.0
    assert torch.isclose(updated_mean, torch.tensor(2.0))

    # Check variance update
    # The formula is complex but can be verified
    assert updated_std <= run_std  # In this case, should be smaller
