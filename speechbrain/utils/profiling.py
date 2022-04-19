"""Decoraters to handle PyTorch profiling and benchmarking.

Author:
    * Andreas Nautsch 2022
"""
from torch import profiler
from functools import wraps
from torch.autograd.profiler_util import EventList
from typing import Any, Callable, Iterable, Optional


def scheduler(func, wait: int = 2, warmup: int = 2, active: int = 2, repeat: int = 1, skip_first: int = 0):
    scheduler = profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat, skip_first=skip_first)

    @wraps(func)
    def wrapper(*args, **kwargs):
        kwargs["schedule"] = scheduler
        return func(*args, **kwargs)
    return wrapper


def trace_handler(func):
    """
            def trace_handler(prof):
            print(prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=-1))
            # prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")

            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log') -> more rigour at chrome trace

            # p.export_stacks("/tmp/profiler_stacks_scheduler.txt", "self_cuda_time_total") -> FlameGraph
    """
    pass


def profile(func: callable,
            activities: Optional[Iterable[profiler.ProfilerActivity]] = None,
            schedule: Optional[Callable[[int], profiler.ProfilerAction]] = None,
            on_trace_ready: Optional[Callable[..., Any]] = None,
            record_shapes: bool = False,
            profile_memory: bool = False,
            with_stack: bool = False,
            with_flops: bool = False,
            with_modules: bool = False,
            ) -> object:
    """Wrapper to create a PyTorch profiler to benchmark training/inference of speechbrain.core.Brain instances.
    See ``torch.profiler.profile`` documentation for details (brief summary below).

    func : object
        ``speechbrain.core.Brain``:s or a (train/eval) function to be profiled.
    activities : iterable
        List of activity groups.
        Default: ProfilerActivity.CPU and (when available) ProfilerActivity.CUDA.
        (Default value should be ok for most cases.)
    schedule : callable
        Waits a specified amount of steps for PyTorch to warm-up; see the above ``schedule`` decorator.
        Default: ``ProfilerAction.RECORD`` (immediately starts recording).
    on_trace_ready : callable
        Specifies what benchmark record should be saved (after each scheduled step);
        see above ``trace_handler`` decorator.
        Default: ``None`` (pick up collected reporting once profiling ended, but not details per step).
    record_shapes : bool
        Save input shapes of operations (enables to group benchmark data by after profiling).
        Default: ``False``.
    profile_memory : bool
        Track tensor memory allocation/deallocation.
        Default: ``False``.
    with_stack : bool
        Record source information (file and line number).
        Default: ``False``.
    with_flops: bool
        Estimate the number of FLOPs.
        Default: ``False``.
    with_modules: bool
        Record module hierarchy (including function names)
        Default: ``False``

    Example
    -------
    >>> import torch
    ... @profile
    ... def run(x : torch.Tensor):
    ...     y = x ** 2
    ...     z = y ** 3
    ...     return y.backward()
    >>> data = torch.randn((1, 1), requires_grad=True)
    >>> run(data)
    ['this', 'example', 'gets', 'tokenized']
    :rtype: object
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        with profiler.profile(
                activities=activities,
                schedule=schedule,
                on_trace_ready=on_trace_ready,
                record_shapes=record_shapes,
                profile_memory=profile_memory,
                with_stack=with_stack,
                with_flops=with_flops,
                with_modules=with_modules,
        ) as prof:
            # Adds profiler to attributes if func is not a function (implies: speechbrain.core.Brain); preserving prof
            if "__call__" not in dir(func):
                kwargs["profiler"] = prof
                # TODO hook fit method -> copy of on_trace_ready but different file name pattern
                # TODO hook eval method -> copy of on_trace_ready but different file name pattern

            # Benchmark
            result = func(*args, **kwargs)

            # Direct profiler post-processing if func is a function (implies: result is None); prof is lost at return
            if "__call__" in dir(func):
                return prof

            return result

    return wrapper


def profile_details(func):
    """Pre-configured profiling for a detailed benchmark.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return profile(schedule=scheduler(...),
                       on_trace_ready=trace_handler(...),
                       record_shapes=True,
                       profile_memory=True,
                       with_stack=True,
                       with_flops=True,
                       with_modules=True,
                       )(func)(*args, **kwargs)

    return wrapper


def events_diff(a: EventList, b: EventList, sort_by: str = "cpu_time_total"):
    # TODO abstract count -> let user choose a parameter to filter by

    # Maps: function name -> (call count, position) // the position helps to remove alike call numbers later on.
    a_counts = dict([(i.key, (i.count, p)) for p, i in enumerate(a)])
    b_counts = dict([(i.key, (i.count, p)) for p, i in enumerate(b)])

    # Figuring our which ones to delete.
    a_to_remove = list([])
    b_to_remove = list([])
    for key in a_counts.keys():
        # Equal values are filtered.
        if a_counts[key][0] == b_counts[key][0]:
            # Enlist position to be removed.
            a_to_remove.append(a_counts[key][1])
            b_to_remove.append(b_counts[key][1])

    # Since EventLists are lists: removing items from the back.
    a_to_remove.reverse()
    b_to_remove.reverse()
    for k in a_to_remove:
        a.remove(a[k])
    for k in b_to_remove:
        b.remove(b[k])

    # Print the diffs.
    print(a.table(sort_by=sort_by))
    print(b.table(sort_by=sort_by))
