"""Guard for running certain operations on main process only

Authors:
 * Abdel Heba 2020
 * Aku Rouhe 2020
"""
import speechbrain as sb


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
        Function to run on main process.
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
        Whether to run post_func on main process as well. By default False.
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

    if sb.if_main_process():
        # Main comes here
        try:
            func(*args, **kwargs)
        finally:
            sb.ddp_barrier()
    else:
        # Others go here
        sb.ddp_barrier()
    if post_func is not None:
        if run_post_on_main:
            # Just run on every process without any barrier.
            post_func(*post_args, **post_kwargs)
        elif not sb.if_main_process():
            # Others go here
            try:
                post_func(*post_args, **post_kwargs)
            finally:
                sb.ddp_barrier()
        else:
            # But main comes here
            sb.ddp_barrier()
