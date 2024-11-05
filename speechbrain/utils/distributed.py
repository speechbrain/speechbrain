"""Guard for running certain operations on main process only

Authors:
 * Abdel Heba 2020
 * Aku Rouhe 2020
 * Peter Plantinga 2023
 * Adel Moumen 2024
"""

import datetime
import os
from functools import wraps
from typing import Optional

import torch

MAIN_PROC_ONLY: int = 0


def rank_prefixed_message(message: str) -> str:
    r"""Prefix a message with the rank of the process.

    Arguments
    ---------
    message : str
        The message to prefix.

    Returns
    -------
    str
        The message prefixed with the rank, if known.
    """
    rank = get_rank()
    if rank is not None:
        return f"[rank: {rank}] {message}"
    return message


def get_rank() -> Optional[int]:
    r"""Get the rank of the current process.

    This code is taken from the Pytorch Lightning library:
    https://github.com/Lightning-AI/pytorch-lightning/blob/bc3c9c536dc88bfa9a46f63fbce22b382a86a9cb/src/lightning/fabric/utilities/rank_zero.py#L39-L48

    Returns
    -------
    int or None
        The rank of the current process, or None if the rank could not be determined.
    """
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    # None to differentiate whether an environment variable was set at all
    return None


def run_on_main(
    func,
    args=None,
    kwargs=None,
    post_func=None,
    post_args=None,
    post_kwargs=None,
    run_post_on_main=False,
):
    r"""Runs a function with DPP (multi-gpu) support.

    The main function is only run on the main process.
    A post_function can be specified, to be on non-main processes after the main
    func completes. This way whatever the main func produces can be loaded on
    the other processes.

    Arguments
    ---------
    func : callable
        Function to run on the main process.
    args : list, None
        Positional args to pass to func.
    kwargs : dict, None
        Keyword args to pass to func.
    post_func : callable, None
        Function to run after func has finished on main. By default only run on
        non-main processes.
    post_args : list, None
        Positional args to pass to post_func.
    post_kwargs : dict, None
        Keyword args to pass to post_func.
    run_post_on_main : bool
        Whether to run post_func on main process as well. (default: False)
    """
    # Handle the mutable data types' default args:
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    if post_args is None:
        post_args = []
    if post_kwargs is None:
        post_kwargs = {}

    main_process_only(func)(*args, **kwargs)
    ddp_barrier()

    if post_func is not None:
        if run_post_on_main:
            # Just run on every process without any barrier.
            post_func(*post_args, **post_kwargs)
        else:
            # Do the opposite of `run_on_main`
            if not if_main_process():
                post_func(*post_args, **post_kwargs)
            ddp_barrier()


def is_distributed_initialized() -> bool:
    r"Returns whether the current system is distributed."
    # `is_initialized` is only defined conditionally
    # https://github.com/pytorch/pytorch/blob/v2.1.0/torch/distributed/__init__.py#L25
    # this might happen to MacOS builds from source (default) or any build from source that sets `USE_DISTRIBUTED=0`
    return (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )


def if_main_process() -> bool:
    r"Returns whether the current process is the main process."
    if is_distributed_initialized():
        return torch.distributed.get_rank() == 0
    else:
        return True


class MainProcessContext:
    r"""
    Context manager to ensure code runs only on the main process.
    This is useful to make sure that `MAIN_PROC_ONLY` global variable
    is decreased even if there's an exception raised inside of
    `main_proc_wrapped_func` fn.
    """

    def __enter__(self):
        r"""Enter the context. Increase the counter."""
        global MAIN_PROC_ONLY
        MAIN_PROC_ONLY += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        r"""Exit the context. Decrease the counter."""
        global MAIN_PROC_ONLY
        MAIN_PROC_ONLY -= 1


def main_process_only(function):
    r"""Function decorator to ensure the function runs only on the main process.
    This is useful for things like saving to the filesystem or logging
    to a web address where you only want it to happen on a single process.
    """

    @wraps(function)
    def main_proc_wrapped_func(*args, **kwargs):
        """This decorated function runs only if this is the main process."""
        with MainProcessContext():
            if if_main_process():
                return function(*args, **kwargs)
            else:
                return None

    return main_proc_wrapped_func


def ddp_barrier():
    r"""
    Synchronize all processes in distributed data parallel (DDP) mode.

    This function blocks the execution of the current process until all
    processes in the distributed group have reached the same point. It ensures
    that no process moves ahead until every other process has also reached this
    barrier. If DDP is not being used (i.e., only one process is running),
    this function has no effect and immediately returns.

    Returns
    -------
    None


    Example
    -------
    >>> ddp_barrier()
    >>> print("hello world")
    hello world
    """
    if MAIN_PROC_ONLY >= 1 or not is_distributed_initialized():
        return

    if torch.distributed.get_backend() == torch.distributed.Backend.NCCL:
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    else:
        torch.distributed.barrier()


def ddp_broadcast(communication_object, src=0):
    r"""In DDP mode, this function will broadcast an object to all
    processes.

    Arguments
    ---------
    communication_object: Any
        The object to be communicated to all processes. Must be picklable.
        See docs for ``torch.distributed.broadcast_object_list()``
    src: int
        The rank which holds the object to be communicated.

    Returns
    -------
    The communication_object passed on rank src.
    """
    if MAIN_PROC_ONLY >= 1 or not is_distributed_initialized():
        return communication_object

    # Wrapping object in a list is required for preventing
    # a copy of the object, maintaining a pointer instead
    communication_list = [communication_object]
    torch.distributed.broadcast_object_list(communication_list, src=src)
    return communication_list[0]


def ddp_init_group(run_opts):
    r"""This function will initialize the ddp group if
    distributed_launch bool is given in the python command line.

    The ddp group will use distributed_backend arg for setting the
    DDP communication protocol. `RANK` Unix variable will be used for
    registering the subprocess to the ddp group.

    Arguments
    ---------
    run_opts: list
        A list of arguments to parse, most often from `sys.argv[1:]`.

    Returns
    -------
    None
    """
    rank = os.environ.get("RANK")
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None or rank is None:
        return

    local_rank = int(local_rank)
    if not run_opts["distributed_backend"] == "gloo":
        if local_rank + 1 > torch.cuda.device_count():
            raise ValueError(
                "Killing process " + str() + "\n" "Not enough GPUs available!"
            )
    rank = int(rank)

    if run_opts["distributed_backend"] == "nccl":
        if not torch.distributed.is_nccl_available():
            raise ValueError("NCCL is not supported in your machine.")
    elif run_opts["distributed_backend"] == "gloo":
        if not torch.distributed.is_gloo_available():
            raise ValueError("GLOO is not supported in your machine.")
    elif run_opts["distributed_backend"] == "mpi":
        if not torch.distributed.is_mpi_available():
            raise ValueError("MPI is not supported in your machine.")
    else:
        raise ValueError(
            run_opts["distributed_backend"]
            + " communication protocol doesn't exist."
        )

    if run_opts["distributed_backend"] == "nccl":
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

    # rank arg is used to set the right rank of the current process for ddp.
    # if you have 2 servers with 2 gpu:
    # server1:
    #   GPU0: local_rank=device=0, rank=0
    #   GPU1: local_rank=device=1, rank=1
    # server2:
    #   GPU0: local_rank=device=0, rank=2
    #   GPU1: local_rank=device=1, rank=3
    torch.distributed.init_process_group(
        backend=run_opts["distributed_backend"],
        rank=rank,
        timeout=datetime.timedelta(seconds=7200),
    )
