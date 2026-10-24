"""
Test for distributed.py.
"""

import operator

import pytest
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


# Test infer_device.


def test_infer_device():
    inferred_device = distributed.infer_device()
    current_device = str(torch.accelerator.current_accelerator())
    if torch.accelerator.is_available():
        assert current_device.startswith(inferred_device)
    else:
        assert inferred_device == "cpu"


def test_infer_device_local_rank(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "1")
    device = distributed.infer_device()
    if torch.accelerator.is_available():
        assert device == f"{torch.accelerator.current_accelerator()}:1"
    else:
        assert device == "cpu"


# Test the accelerator handling of ddp_init_group.


class _FakeAccelerator:
    """Minimal stand-in for ``torch.accelerator``: an accelerator that is not CUDA alone."""

    def __init__(self, device_type="cuda", device_count=2):
        self._device_type = device_type
        self._device_count = device_count

    def is_available(self):
        return True

    def current_accelerator(self):
        return torch.device(self._device_type)

    def device_count(self):
        return self._device_count


def test_ddp_init_group_accepts_a_non_cuda_backend(monkeypatch):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    recorded = {}
    bound = []

    monkeypatch.setattr(distributed.torch, "accelerator", _FakeAccelerator())
    monkeypatch.setattr(distributed.torch.cuda, "set_device", bound.append)
    monkeypatch.setattr(
        distributed.torch.distributed,
        "is_backend_available",
        lambda name: name == "hccl",
        raising=False,
    )
    monkeypatch.setattr(
        distributed.torch.distributed,
        "init_process_group",
        lambda **kwargs: recorded.update(kwargs),
    )

    distributed.ddp_init_group({"distributed_backend": "hccl"})

    assert recorded["backend"] == "hccl"
    assert recorded["rank"] == 1
    # The rank is bound to its own device even when the backend is not nccl.
    assert [str(device) for device in bound] == ["cuda:1"]


def test_ddp_init_group_counts_devices_through_the_accelerator(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(
        distributed.torch, "accelerator", _FakeAccelerator(device_count=2)
    )

    with pytest.raises(ValueError, match="Not enough GPUs"):
        distributed.ddp_init_group({"distributed_backend": "hccl"})


def test_ddp_init_group_rejects_an_unknown_backend(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(distributed.torch, "accelerator", _FakeAccelerator())
    monkeypatch.setattr(
        distributed.torch.distributed,
        "is_backend_available",
        lambda name: False,
        raising=False,
    )

    with pytest.raises(ValueError, match="communication protocol"):
        distributed.ddp_init_group({"distributed_backend": "not-a-backend"})
