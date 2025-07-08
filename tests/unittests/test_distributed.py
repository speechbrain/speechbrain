"""
Test for distributed.py.
"""

import operator

import torch

from speechbrain.utils import distributed


def mock_initialise_process_group(rank: int, world_size: int, tmpdir):
    """
    Pretend to run on under "torchrun".
    """
    import os

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    sync_file = f"file://{tmpdir}/sync"
    torch.distributed.init_process_group(
        "gloo", rank=rank, world_size=world_size, init_method=sync_file
    )


# Test @main_process_only a single call with a return value.


@distributed.main_process_only
def return_rank_345(rank, world_size):
    # This should run on the main process only.
    assert rank == 0
    return 345 + rank


def return_ranks_345_in_mock_process(rank, world_size, tmpdir):
    mock_initialise_process_group(rank, world_size, tmpdir)
    result = return_rank_345(rank, world_size)
    assert result == 345


def test_main_process_only(tmpdir):
    world_size = 2
    torch.multiprocessing.spawn(
        return_ranks_345_in_mock_process,
        (world_size, tmpdir),
        world_size,
        join=True,
    )


# Test @main_process_only a recursive call.


@distributed.main_process_only
def fibonacci(n):
    if n == 0 or n == 1:
        return 1
    return fibonacci(n - 2) + fibonacci(n - 1)


def check_fibonacci(rank, world_size, tmpdir):
    mock_initialise_process_group(rank, world_size, tmpdir)
    assert fibonacci(0) == 1
    assert fibonacci(1) == 1
    assert fibonacci(2) == 2
    assert fibonacci(3) == 3
    assert fibonacci(4) == 5


def test_main_process_only_nested(tmpdir):
    world_size = 2
    torch.multiprocessing.spawn(
        check_fibonacci, (world_size, tmpdir), world_size, join=True
    )


# Test run_on_main.


def check_add_in_mock_process(rank, world_size, tmpdir, i, j):
    mock_initialise_process_group(rank, world_size, tmpdir)
    assert distributed.run_on_main(operator.add, args=(i, j)) == i + j


def test_run_on_main(tmpdir):
    world_size = 2
    torch.multiprocessing.spawn(
        check_add_in_mock_process,
        (world_size, tmpdir, 23, 54),
        world_size,
        join=True,
    )
