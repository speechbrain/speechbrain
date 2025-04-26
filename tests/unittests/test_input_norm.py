"""Test input normalization in processing/features.py.
"""

import functools
from typing import List, Optional, Tuple, Union

import numpy as np
import pytest
import torch

from speechbrain.processing.features import (
    InputNormalization,
    combine_gaussian_statistics,
    combine_gaussian_statistics_distributed,
    gaussian_statistics,
    make_padding_mask,
)


class TestInputNormalization:
    """Test suite for the InputNormalization class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Create a batch of 2 sequences with 3 features and variable lengths
        x = torch.arange(4 * 4 * 3, dtype=torch.float32).view(4, 4, 3) + 1

        # Relative lengths: first has 100%, second 75%, third 50%, last 25%
        lengths = torch.tensor([0.25, 0.5, 0.75, 1.0], dtype=torch.float32)

        # Mask of the input tensor based on the lengths
        mask = torch.triu(torch.ones(4, 4)).transpose(0, 1).bool()
        mask = mask.unsqueeze(-1).expand_as(x)

        return x, lengths, mask

    def test_constructor_defaults(self):
        """Test constructor with default parameters."""
        norm = InputNormalization()
        assert norm.std_norm is True
        assert norm.norm_type == "global"
        assert norm.update_until_epoch == 2
        assert norm.avoid_padding_norm is False
        assert norm.epsilon == 1e-10
        assert norm.length_dim == 1

    def test_invalid_norm_type(self):
        """Test that invalid norm_type raises an error."""
        with pytest.raises(ValueError):
            InputNormalization(norm_type="invalid")

    def test_sentence_normalization(self, sample_data):
        """Test sentence-level normalization."""
        x, lengths, mask = sample_data
        norm = InputNormalization(norm_type="sentence", avoid_padding_norm=True)

        # Manual calculation for comparison
        mean, std = reference_sentence_norm(x, mask, avoid_padding_norm=True)
        expected = (x - mean) / std

        # Apply sentence normalization
        output = norm(x, lengths)

        # Check if output matches expected values where not padding
        assert torch.allclose(output[mask], expected[mask], atol=1e-5)

        # Check if padding values are preserved properly
        assert torch.allclose(output[~mask], x[~mask], atol=1e-5)

    def test_batch_normalization(self, sample_data):
        """Test batch-level normalization."""
        x, lengths, mask = sample_data
        norm = InputNormalization(norm_type="batch", avoid_padding_norm=True)

        # Padding Mask is slightly different format (singletons for non-dims)
        padding_mask = make_padding_mask(x, lengths)
        _, mean, var = reference_gaussian_statistics(
            x.numpy(), (0, 1), padding_mask.numpy()
        )
        mean = torch.FloatTensor(mean).expand_as(x).masked_fill(~mask, 0.0)
        var = torch.FloatTensor(var).expand_as(x).masked_fill(~mask, 1.0)
        expected = (x - mean) / var.clamp(min=1e-8).sqrt()

        # Apply normalization
        output = norm(x, lengths)

        # print(output)
        # print(expected)

        # Check normalization was applied correctly to non-padding values
        assert torch.allclose(output[mask], expected[mask], atol=1e-3)

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
        assert not torch.allclose(valid_values1, valid_values3, atol=0.01)

        # Check that after many trials, the overall stats match the tensor stats
        for i in range(1000):
            _ = norm(x, lengths)

        padding_mask = make_padding_mask(x, lengths)
        _, mean, var = reference_gaussian_statistics(
            x.numpy(), (0, 1), padding_mask.numpy()
        )
        assert torch.allclose(
            norm.glob_mean, torch.FloatTensor(mean), atol=1e-3
        )
        assert torch.allclose(
            norm.glob_std, torch.FloatTensor(var).sqrt(), atol=1e-2
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


def reference_sentence_norm(x, mask, avoid_padding_norm=True):
    """Compute reference sentence norm"""
    n = mask.sum(dim=1, keepdim=True)
    mean = (x * mask).sum(dim=1, keepdim=True) / n
    var = ((x - mean) * mask).square().sum(dim=1, keepdim=True) / n
    if avoid_padding_norm:
        var = var.masked_fill(~mask, 1.0)
        mean = mean.masked_fill(~mask, 0.0)

    return mean, var.clamp(min=1e-8).sqrt()


# Utility tests for the helper functions
def test_make_padding_mask():
    """Test the make_padding_mask function."""

    # Create a batch of sequences
    x = torch.ones(3, 4, 2)  # Batch size 3, seq len 4, feature dim 2

    # Relative lengths: 100%, 75%, 50%
    lengths = torch.tensor([1.0, 0.75, 0.5])

    # Get mask
    mask = make_padding_mask(x, lengths, length_dim=1)

    # Expected mask:
    # First sequence: all True
    # Second sequence: first 3 True, last False
    # Third sequence: first 2 True, last 2 False
    # Last dimension is singleton, can be broadcast to apply mask
    expected_mask = torch.tensor(
        [
            [[True], [True], [True], [True]],
            [[True], [True], [True], [False]],
            [[True], [True], [False], [False]],
        ]
    )

    assert torch.equal(mask, expected_mask)

    # Test for potential rounding error
    x = torch.ones(22, 22)
    lengths = (torch.arange(22) + 1) / 22
    mask = make_padding_mask(x, lengths, length_dim=1)
    expected_mask = torch.triu(torch.ones(22, 22)).transpose(0, 1).bool()

    assert torch.equal(mask, expected_mask)


def normalise_dimensions(
    dimensions: Union[int, tuple, None], num_dimensions: int
):
    """Ensure dimensions object is a tuple."""
    if isinstance(dimensions, int):
        return (dimensions,)
    elif dimensions is None or dimensions == ():
        # All dimensions
        return tuple(range(num_dimensions))
    assert isinstance(dimensions, tuple)
    return dimensions


def random_mask_numpy(
    generator: np.random.Generator,
    data_shape: tuple,
    dimensions: Union[int, tuple, None],
):
    dimensions_set = set(normalise_dimensions(dimensions, len(data_shape)))

    mask_shape = tuple(
        (data_shape[d] if d in dimensions_set else 1)
        for d in range(len(data_shape))
    )

    mask = generator.integers(0, 2, size=mask_shape, dtype=bool)

    if np.count_nonzero(mask) == 0:
        return None
    return mask


def reference_gaussian_statistics(
    x: np.ndarray,
    dimensions: Union[int, tuple, None],
    mask: Optional[np.ndarray],
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Compute reference count, mean, variance with Numpy, in the simplest way
    possible.
    """
    # Ensure dimensions object is a tuple.
    dimensions = normalise_dimensions(dimensions, len(x.shape))

    # Start by pretending that dimensions=() and then roll them up one by one.
    all_count = 1
    masked_data = x if mask is None else mask * x
    sum = masked_data
    sum_squares = np.square(masked_data)

    for dimension in sorted(dimensions, reverse=True):
        all_count *= x.shape[dimension]
        sum = np.sum(sum, axis=dimension)
        sum_squares = np.sum(sum_squares, axis=dimension)

    count = all_count if mask is None else np.sum(mask)

    mean = sum / count
    variance = (sum_squares / count) - np.square(mean)

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
@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("random_seed", [20250304, 20250326, 20250327])
def test_gaussian_statistics(
    size, dimensions, use_mask: bool, random_seed: int
):
    if isinstance(dimensions, tuple):
        if any(dimension >= len(size) for dimension in dimensions):
            return
    elif isinstance(dimensions, int):
        if dimensions >= len(size):
            return
    generator = np.random.default_rng(random_seed)

    x = generator.uniform(low=-5, high=+5, size=size)

    if use_mask:
        mask = random_mask_numpy(generator, size, dimensions)
    else:
        mask = None

    reference_count, reference_mean, reference_variance = (
        reference_gaussian_statistics(x, dimensions=dimensions, mask=mask)
    )

    count, mean, variance = gaussian_statistics(
        torch.tensor(x),
        dim=dimensions,
        mask=None if mask is None else torch.tensor(mask),
    )

    assert count == reference_count
    assert mean.shape == reference_mean.shape
    assert variance.shape == reference_variance.shape
    if not np.all(np.isnan(reference_mean)):
        assert np.allclose(mean.cpu().numpy(), reference_mean)
        assert np.allclose(variance.cpu().numpy(), reference_variance)
    else:
        assert np.all(np.isnan(mean.cpu().numpy()))
        assert np.all(np.isnan(variance.cpu().numpy()))


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


def parallel_mean_var_update(rank, world_size, tmpdir, random_seed):
    """Test that the mean_var_norm works in ddp."""

    from speechbrain.processing.features import mean_std_update

    initialise_process_group(rank, world_size, tmpdir)

    generator = torch.Generator()
    generator.manual_seed(random_seed)

    feature_length = 10
    num_rounds = 3

    batch_size = 4
    utterance_length = 3
    main_shape = (batch_size, utterance_length)
    dimensions = (0, 1)

    def random_input(generator):
        """Return a random input"""
        return 10 - 5 * torch.rand(
            size=main_shape + (feature_length,), generator=generator
        )

    def random_mask(generator):
        """Return a random mask"""
        # Sometimes produce None.
        if float(torch.rand(size=(), generator=generator)) < 0.3:
            return None
        else:
            mask = torch.randint(
                high=2,
                size=main_shape + (1,),
                generator=generator,
                dtype=torch.bool,
            )
            if mask.count_nonzero() == 0:
                return None
            return mask

    inputs = [
        [random_input(generator) for _ in range(num_rounds)]
        for _ in range(world_size)
    ]

    full_mask = torch.full(main_shape + (1,), fill_value=True)
    masks = [
        [random_mask(generator) for _ in range(num_rounds)]
        for _ in range(world_size)
    ]

    # Running values should be the same between processes
    running_count = torch.tensor(0)
    running_mean = torch.zeros((feature_length,))
    running_std = torch.zeros((feature_length,))

    for round in range(num_rounds):
        running_count, running_mean, running_std = mean_std_update(
            x=inputs[rank][round],
            mask=masks[rank][round],
            dim=dimensions,
            run_count=running_count,
            run_mean=running_mean,
            run_std=running_std,
        )

    def flatten(tensor_list_list: List[List[torch.Tensor]]):
        flat_list = []
        for tensors in tensor_list_list:
            flat_list.extend(tensors)
        # Replace masks "None" by a tensor with only True.
        flat_list = [(t if t is not None else full_mask) for t in flat_list]
        last_dimension = flat_list[0].size(-1)
        return torch.cat(
            [tensor.reshape((-1, last_dimension)) for tensor in flat_list]
        )

    # Flatten all inputs.
    flat_inputs = flatten(inputs)
    flat_masks = flatten(masks)

    # Expected values
    expected_count = torch.sum(flat_masks)

    expected_mean = torch.sum(flat_masks * flat_inputs, dim=0) / expected_count
    expected_variance = (
        torch.sum(flat_masks * torch.square(flat_inputs - expected_mean), dim=0)
        / expected_count
    )
    expected_std = torch.sqrt(expected_variance)

    # Same values on all processes
    assert torch.allclose(running_count, expected_count)
    assert torch.allclose(running_mean, expected_mean)
    assert torch.allclose(running_std, expected_std)


@pytest.mark.parametrize("random_seed", [20250307, 20250326, 20250327])
def test_mean_var_update_parallel(tmpdir, random_seed):
    """Test the mean_var_update function in parallel."""
    world_size = 3
    torch.multiprocessing.spawn(
        parallel_mean_var_update,
        args=(world_size, tmpdir, random_seed),
        nprocs=world_size,
        join=True,
    )
