"""Guard for running certain operations on main process only

Authors:
 * Abdel Heba 2020
 * Aku Rouhe 2020
 * Peter Plantinga 2023
"""
import datetime
import os
import torch
from functools import wraps

MAIN_PROC_ONLY = 0


def run_on_main(
    func,
    args=None,
    kwargs=None,
    post_func=None,
    post_args=None,
    post_kwargs=None,
    run_post_on_main=False,
):
    """Runs a function with DPP (multi-gpu) support.

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


def if_main_process():
    """Checks if the current process is the main local process and authorized to run
    I/O commands. In DDP mode, the main local process is the one with LOCAL_RANK == 0.
    In standard mode, the process will not have `LOCAL_RANK` Unix var and will be
    authorized to run the I/O commands.
    """
    if "LOCAL_RANK" in os.environ:
        if os.environ["LOCAL_RANK"] == "":
            return False
        else:
            if int(os.environ["LOCAL_RANK"]) == 0:
                return True
            return False
    return True


def main_process_only(function):
    """Function decorator to ensure the function runs only on the main process.
    This is useful for things like saving to the filesystem or logging
    to a web address where you only want it to happen on a single process.
    """

    @wraps(function)
    def main_proc_wrapped_func(*args, **kwargs):
        """This decorated function runs only if this is the main process."""
        global MAIN_PROC_ONLY
        MAIN_PROC_ONLY += 1
        if if_main_process():
            result = function(*args, **kwargs)
        else:
            result = None
        MAIN_PROC_ONLY -= 1
        return result

    return main_proc_wrapped_func


def ddp_barrier():
    """In DDP mode, this function will synchronize all processes.
    torch.distributed.barrier() will block processes until the whole
    group enters this function.
    """
    # Check if we're in a single-threaded section, skip barrier
    if MAIN_PROC_ONLY >= 1:
        return
    elif torch.distributed.is_initialized():
        torch.distributed.barrier()


def ddp_broadcast(communication_object, src=0):
    """In DDP mode, this function will broadcast an object to all
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
    if MAIN_PROC_ONLY >= 1 or not torch.distributed.is_initialized():
        return communication_object

    # Wrapping object in a list is required for preventing
    # a copy of the object, maintaining a pointer instead
    communication_list = [communication_object]
    torch.distributed.broadcast_object_list(communication_list, src=src)
    return communication_list[0]


def ddp_init_group(run_opts):
    """This function will initialize the ddp group if
    distributed_launch bool is given in the python command line.

    The ddp group will use distributed_backend arg for setting the
    DDP communication protocol. `RANK` Unix variable will be used for
    registering the subprocess to the ddp group.

    Arguments
    ---------
    run_opts: list
        A list of arguments to parse, most often from `sys.argv[1:]`.
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
            + " communcation protocol doesn't exist."
        )

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
