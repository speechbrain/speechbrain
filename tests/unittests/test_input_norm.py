"""Test input normalization in processing/features.py.
"""

import functools
from typing import List, Optional, Tuple, Union

import numpy as np
import pytest
import torch

from speechbrain.processing.features import (
    combine_gaussian_statistics,
    combine_gaussian_statistics_distributed,
    gaussian_statistics,
)


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
