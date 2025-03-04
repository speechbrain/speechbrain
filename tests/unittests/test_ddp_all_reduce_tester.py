"""
Test mock_ddp_all_reduce by testing operation on two example use cases.

Authors
 * Rogier van Dalen 2025
"""

import pytest
import torch

from speechbrain.utils import distributed
from tests.utils.mock_ddp_all_reduce import run_with_mocked_ddp_all_reduce


# Example use case 1.
def make_sum_distributed(thread_index: int):
    def sum_distributed():
        value = torch.tensor([thread_index + 3])
        # Note that ddp_all_reduce changes the value in-place!
        distributed.ddp_all_reduce(value, torch.distributed.ReduceOp.SUM)

        return value

    return sum_distributed


# Example use case 2.
def make_compute_mean_variance(thread_index: int):
    def compute_variance():
        """
        Compute the variance across contributions from each thread.
        """
        count = torch.tensor([1])
        # Note that ddp_all_reduce changes the value in-place!
        distributed.ddp_all_reduce(count, torch.distributed.ReduceOp.SUM)

        weight = 1 / count

        # Each thread has a deterministic contribution.
        value = torch.tensor([1.0, 2.0, 3.0]) * thread_index

        mean = weight * value
        distributed.ddp_all_reduce(mean, torch.distributed.ReduceOp.SUM)

        # Around the global mean.
        variance = weight * (torch.square(value) - torch.square(mean))
        distributed.ddp_all_reduce(variance, torch.distributed.ReduceOp.SUM)

        return variance

    return compute_variance


@pytest.mark.parametrize(
    "make_function, num_threads, expected",
    [
        (make_sum_distributed, 3, [3 + 4 + 5]),
        (make_compute_mean_variance, 3, [2 / 3, 2 / 3 * 4.0, 2 / 3 * 9.0]),
    ],
)
def test_run_with_mocked_ddp_all_reduce(make_function, num_threads, expected):
    functions = [
        make_function(thread_index) for thread_index in range(num_threads)
    ]
    results = run_with_mocked_ddp_all_reduce(functions)

    assert len(results) == num_threads
    for thread_result in results:
        assert torch.allclose(thread_result, torch.tensor(expected))
