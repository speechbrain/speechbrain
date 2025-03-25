"""Test input normalization in processing/features.py.
"""

import functools
from typing import List, Tuple, Union

import numpy as np
import pytest
import torch

from speechbrain.processing.features import (
    InputNormalization,
    combine_gaussian_statistics,
    combine_gaussian_statistics_distributed,
    gaussian_statistics,
)


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
        assert torch.allclose(
            norm.glob_std, x[mask].std(unbiased=False), atol=1e-2
        )

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


def reference_gaussian_statistics(
    x: np.ndarray, dimensions: Union[int, tuple, None]
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Compute reference count, mean, variance with Numpy, in the simplest way
    possible.
    """
    # Ensure dimensions object is a tuple.
    if isinstance(dimensions, int):
        dimensions = (dimensions,)
    elif dimensions is None:
        # All dimensions
        dimensions = tuple(range(len(x.shape)))
    assert isinstance(dimensions, tuple)

    # Start by pretending that dimensions=() and then roll them up one by one.
    count = 1
    mean = x
    variance_statistics = np.square(x)

    for dimension in sorted(dimensions, reverse=True):
        count *= x.shape[dimension]
        mean = np.mean(mean, axis=dimension)
        variance_statistics = np.mean(variance_statistics, axis=dimension)

    variance = variance_statistics - np.square(mean)

    return count, mean, variance


@pytest.mark.parametrize(
    "size", [(), (1,), (5,), (4, 2), (7, 8, 9), (2, 3, 4, 5)]
)
@pytest.mark.parametrize(
    "dimensions",
    [
        None,
        0,
        1,
        2,
        3,
        (),
        (0,),
        (1,),
        (2,),
        (3,),
        (0, 1),
        (0, 2),
        (0, 1, 2),
        (0, 1, 3),
    ],
)
def test_gaussian_statistics(size, dimensions):
    if isinstance(dimensions, tuple):
        if any(dimension >= len(size) for dimension in dimensions):
            return
    elif isinstance(dimensions, int):
        if dimensions >= len(size):
            return
    generator = np.random.default_rng(20250304)

    x = generator.uniform(low=-5, high=+5, size=size)

    reference_count, reference_mean, reference_variance = (
        reference_gaussian_statistics(x, dimensions=dimensions)
    )

    count, mean, variance = gaussian_statistics(torch.tensor(x), dim=dimensions)

    assert count == reference_count
    assert mean.shape == reference_mean.shape
    assert variance.shape == reference_variance.shape
    assert np.allclose(mean.cpu().numpy(), reference_mean)
    assert np.allclose(variance.cpu().numpy(), reference_variance)


# For this test, assume that we compute the statistics across all dimensions
# except the last.
# Note that only the last dimension needs to match.
@pytest.mark.parametrize(
    "size_left, size_right",
    [
        ((2,), (2,)),
        ((1, 4), (3, 4)),
        ((5, 2, 5), (7, 6, 4, 5)),
        ((2, 5, 3), (4, 3)),
    ],
)
def test_combine_gaussian_statistics(size_left, size_right):
    generator = np.random.default_rng(20250304)

    last_size = size_left[-1]
    assert size_right[-1] == last_size

    left = generator.uniform(low=-5, high=+5, size=size_left)
    right = generator.uniform(low=-7, high=+3, size=size_right)

    # Concatenate left and right into one tensor, since the mean and variance on
    # this tensor is what we should be computing.
    flat_left = np.reshape(left, (-1, last_size))
    flat_right = np.reshape(right, (-1, last_size))
    combined = np.concatenate([flat_left, flat_right], axis=0)

    reference_count, reference_mean, reference_variance = gaussian_statistics(
        torch.tensor(combined)
    )

    count, mean, variance = combine_gaussian_statistics(
        gaussian_statistics(torch.tensor(left)),
        gaussian_statistics(torch.tensor(right)),
    )

    assert count == reference_count
    assert torch.allclose(mean, reference_mean)
    assert torch.allclose(variance, reference_variance)


def initialise_process_group(rank: int, world_size: int, tmpdir):
    import os

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    sync_file = f"file://{tmpdir}/sync"
    torch.distributed.init_process_group(
        "gloo", rank=rank, world_size=world_size, init_method=sync_file
    )


def parallel_combine_gaussian_statistics_distributed(
    rank, world_size, tmpdir, sizes: List[tuple], dimensions
):
    """
    Test for combine_gaussian_statistics_distributed, to be run on "world_size"
    processes at the same time.
    """
    assert world_size == len(sizes)
    initialise_process_group(rank, world_size, tmpdir=tmpdir)

    generator = torch.Generator()
    generator.manual_seed(20240311)

    data = [
        5 - 15 * torch.rand(size=process_size, generator=generator)
        for process_size in sizes
    ]

    all_statistics = [gaussian_statistics(d, dim=dimensions) for d in data]

    reference_count, reference_mean, reference_variance = functools.reduce(
        combine_gaussian_statistics, all_statistics
    )

    count, mean, variance = combine_gaussian_statistics_distributed(
        all_statistics[rank]
    )

    assert count == reference_count
    assert torch.allclose(mean, reference_mean)
    assert torch.allclose(variance, reference_variance)


@pytest.mark.parametrize(
    "sizes, dimensions",
    [
        ([(2,), (2,)], ()),
        ([(1, 4), (3, 4), (2, 4)], 0),
        ([(1, 6, 2, 5), (7, 6, 4, 5)], (0, 2)),
        ([(2, 5, 3), (3, 4, 3)], (0, 1)),
    ],
)
def test_combine_gaussian_statistics_distributed(tmpdir, sizes, dimensions):
    """Test the mean_var_update function in parallel."""
    world_size = len(sizes)
    torch.multiprocessing.spawn(
        parallel_combine_gaussian_statistics_distributed,
        args=(world_size, tmpdir, sizes, dimensions),
        nprocs=world_size,
        join=True,
    )


def parallel_mean_var_update(rank, world_size, tmpdir):
    """Test that the mean_var_norm works in ddp."""

    from speechbrain.processing.features import mean_std_update

    initialise_process_group(rank, world_size, tmpdir)

    generator = torch.Generator()
    generator.manual_seed(20240307)

    feature_length = 10
    num_rounds = 3

    batch_size = 4
    utterance_length = 3
    main_shape = (batch_size, utterance_length)

    inputs = [
        [
            10
            - 5
            * torch.rand(
                size=main_shape + (feature_length,), generator=generator
            )
            for _ in range(num_rounds)
        ]
        for _ in range(world_size)
    ]

    # TODO different masks
    mask = torch.full(main_shape + (1,), fill_value=True)

    # TODO test other dimensions
    dimensions = (0, 1)

    # Running values should be the same between processes
    running_count = torch.tensor(0)
    running_mean = torch.zeros((feature_length,))
    # TODO test running_std is None
    running_std = torch.zeros((feature_length,))

    for round in range(num_rounds):
        running_count, running_mean, running_std = mean_std_update(
            x=inputs[rank][round],
            mask=mask,
            dim=dimensions,
            run_count=running_count,
            run_mean=running_mean,
            run_std=running_std,
        )

    # Flatten all inputs.
    input_list = []
    for tensors in inputs:
        input_list.extend(tensors)
    flat_inputs = torch.cat(
        [input.reshape((-1, feature_length)) for input in input_list]
    )

    # Expected values
    expected_count = torch.tensor(flat_inputs.shape[0])
    assert (
        expected_count
        == world_size * num_rounds * batch_size * utterance_length
    )
    expected_mean = torch.mean(flat_inputs, dim=0)
    expected_variance = torch.mean(
        torch.square(flat_inputs - expected_mean), dim=0
    )
    expected_std = torch.sqrt(expected_variance)

    # Same values on all processes
    assert torch.allclose(running_count, expected_count)
    assert torch.allclose(running_mean, expected_mean)
    assert torch.allclose(running_std, expected_std)


def test_mean_var_update_parallel(tmpdir):
    """Test the mean_var_update function in parallel."""
    world_size = 3
    torch.multiprocessing.spawn(
        parallel_mean_var_update,
        args=(world_size, tmpdir),
        nprocs=world_size,
        join=True,
    )
