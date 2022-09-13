"""Polymorphic decorators to handle PyTorch profiling and benchmarking.

Author:
    * Andreas Nautsch 2022
"""
import numpy as np
from copy import deepcopy
from torch import profiler
from functools import wraps
from typing import Any, Callable, Iterable, Optional

# from typing import List
# from itertools import chain

"""
from torch.autograd.profiler_util import (  # pytorch v1.10.1
    EventList,
    FunctionEvent,
    _format_time,
    _format_memory,
)
"""


def set_profiler_attr(func: object, set_attr: str, handler: Callable):
    """Sets handler for profiler: scheduler or trace export.
    """
    assert set_attr in [
        "on_trace_ready",
        "schedule",
    ], "Needs to be a callable profiler attribute."

    if (
        func is None
    ):  # Polymorph: not used as decorator; func is used as e.g.: trace_export()
        return handler
    elif callable(
        func
    ):  # Polymorph: decorates a decorator of function/class constructor

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper implementation."""
            if "__call__" not in dir(
                func
            ):  # Decorator for class constructor (directly)
                result = func(*args, **kwargs)
                setattr(result.profiler, set_attr, handler)
                return result  # not tested
            else:  # Return as additional argument.
                kwargs[set_attr] = handler
                return func(*args, **kwargs)

        return wrapper
    else:  # Polymorph: func is assumed to be an instance of speechbrain.core.Brain
        # No return: in-place edit
        if hasattr(func, "profiler"):
            if func.profiler is profiler.profile:
                setattr(func.profiler, set_attr, handler)


def schedule(
    func: Optional[object] = None,
    wait: int = 2,
    warmup: int = 2,
    active: int = 2,
    repeat: int = 1,
    skip_first: int = 0,
):
    """Wrapper to create a ```torch.profiler.schedule``` (sets default parameters for warm-up).
    """
    torch_scheduler = profiler.schedule(
        wait=wait,
        warmup=warmup,
        active=active,
        repeat=repeat,
        skip_first=skip_first,
    )
    """
    Curious which action a default scheduler suggests at which profiler.step() ?
        [torch_scheduler(x) for x in range(10)]

        00 = {ProfilerAction} ProfilerAction.NONE
        01 = {ProfilerAction} ProfilerAction.NONE
        02 = {ProfilerAction} ProfilerAction.WARMUP
        03 = {ProfilerAction} ProfilerAction.WARMUP
        04 = {ProfilerAction} ProfilerAction.RECORD
        05 = {ProfilerAction} ProfilerAction.RECORD_AND_SAVE
        06 = {ProfilerAction} ProfilerAction.NONE
        07 = {ProfilerAction} ProfilerAction.NONE
        08 = {ProfilerAction} ProfilerAction.NONE
        09 = {ProfilerAction} ProfilerAction.NONE
    """

    return set_profiler_attr(
        func=func, set_attr="schedule", handler=torch_scheduler
    )


def export(
    func: Optional[object] = None,
    dir_name: str = "./log/",
    worker_name: Optional[str] = None,
    use_gzip: bool = False,
):
    """Exports current and aggregated traces for:
    - Chrome tensorboard
    - FlameGraph
    (and sets default parameters for log file folder/filenames).
    """
    import os
    import socket
    import time

    # Chrome export (default handler); inspired the log_file() function below.
    tensorboard_handler = profiler.tensorboard_trace_handler(
        dir_name=dir_name, worker_name=worker_name, use_gzip=use_gzip
    )

    def trace_handler(prof: profiler.profile):
        """trace_handler implementation."""

        def log_file(export_chrome: bool = False, info: str = ""):
            """Implementation of logging file."""
            nonlocal worker_name
            if not worker_name:
                worker_name = "{}_{}".format(
                    socket.gethostname(), str(os.getpid())
                )
            if export_chrome:
                ext = "pt.trace.json"
            else:
                ext = "txt"
            if info:
                pattern = "{{}}.{{}}_{}.{{}}".format(info)
            else:
                pattern = "{}.{}.{}"
            file_name = pattern.format(
                worker_name, int(time.time() * 1000), ext
            )
            if use_gzip:
                file_name = file_name + ".gz"
            return os.path.join(dir_name, file_name)

        def export_stacks(log_path: str, metric: str):
            """Implementation of export_stacks."""
            prof.export_stacks(log_file(), metric)

        def export_traces(aggregated_traces: bool = False):
            """Implementation of export_traces."""
            if not aggregated_traces:
                # Chrome export (also checks for dir_name existing).
                tensorboard_handler(prof)

            # FlameGraph exports.
            if prof.with_stack or aggregated_traces:
                log_path = (
                    log_file(info="aggregated")
                    if aggregated_traces
                    else log_file()
                )
                export_stacks(log_path=log_path, metric="self_cpu_time_total")
                if prof.profiler is not None:
                    if prof.profiler.use_cuda:
                        export_stacks(
                            log_path=log_path, metric="self_cuda_time_total"
                        )

        # export last logged trace - skip if events are empty (e.g., profiler created w/o any torch.nn call)
        if prof.events():
            export_traces()

    return set_profiler_attr(
        func=func, set_attr="on_trace_ready", handler=trace_handler
    )


def prepare_profiler_for_brain(prof: profiler.profile):
    """Sets up a ``torch.profiler.profile`` to also (a) aggregate traces issued from various interactions
    with ``speechbrain.core.Brain``:s and (b) hooks a method to ``merge_traces``.
    """
    # Brain functions will be called independently -> traces will be segregated, so we aggregate them.
    prof.speechbrain_event_traces = list()

    # Preparing the profiler to be re-used during Brain:s' lifecycles.
    def hook_profiler_stop(stop: Callable):
        """Implementation of hook_profiler_stop."""

        @wraps(stop)
        def stop_wrapper():
            """Implementation of stop_wrapper."""
            kineto_profiler = prof.profiler
            if kineto_profiler is not None:
                stop_result = stop()
                if (
                    prof.events()
                ):  # kineto events are not aggregatable (sticking with parsed kineto events)
                    # see: torch.autograd.profiler.__exit__
                    kineto_events = kineto_profiler._parse_kineto_results(
                        kineto_profiler.kineto_results
                    )
                    # add to trace record
                    prof.speechbrain_event_traces.append(
                        deepcopy(kineto_events)
                    )
                    # set flag to disable the profiler
                    kineto_profiler.enabled = False
                return stop_result
            else:
                return stop()  # will be: None

        return stop_wrapper

    # Preparing the profiler to be re-started during Brain:s' lifecycles.
    def hook_profiler_start(start: Callable):
        """Implementation of hook_profiler_start."""

        @wraps(start)
        def start_wrapper():
            """Implementation of start_wrapper."""
            prof.step_num = 0
            prof.current_action = prof.schedule(prof.step_num)
            kineto_profiler = prof.profiler
            if kineto_profiler is not None:
                # check flag if profiler is disabled (i.e. as of stop_wrapper); prevents entering its __init__ twice
                if not kineto_profiler.enabled:
                    # reset kineto profiler (otherwise, one obtains the same traces over & over again)
                    kineto_profiler.enabled = True
            return start()

        return start_wrapper

    """
    # It's currently designed as hiding an Easter Egg.
    def merge_traces():
        " ""Implementation of merge_traces." ""
        # Alternative re-design quirks: make trace aggregator a GLOBAL -or- create another profiler class.
        trace_aggregator = "speechbrain_event_traces"
        if prof.profiler is not None:
            if trace_aggregator in dir(prof) and prof.events():
                # clear all assigned parents/children (from previous mergers & trees)
                for trace in getattr(prof, trace_aggregator):
                    for event in trace:
                        event.cpu_parent = None
                        event.cpu_children: List[FunctionEvent] = []
                # assemble new list
                merged_events = EventList(
                    list(chain.from_iterable(getattr(prof, trace_aggregator))),
                    use_cuda=prof.profiler.use_cuda,
                    profile_memory=prof.profiler.profile_memory,
                    with_flops=prof.profiler.with_flops,
                )
                merged_events._build_tree()
                return merged_events
            else:  # not tested
                return prof.events()
        else:
            return []
    """

    # Augment torch's profiler.
    setattr(prof, "start", hook_profiler_start(getattr(prof, "start")))
    setattr(prof, "stop", hook_profiler_stop(getattr(prof, "stop")))
    # setattr(prof, "merge_traces", merge_traces)

    # Return so it can be readily assigned elsewhere :)
    return prof


def hook_brain_methods(
    func: object,
    prof: profiler.profile,
    class_hooks: Optional[Iterable[str]] = None,
):
    """For instances of ``speechbrain.core.Brain``, critical functions are hooked to profiler start/stop methods.
    """
    # Prepare additional hook decorators for methods of Brain:s.
    def hook_brain(f: Callable):
        """Implementation of hook_brain."""

        @wraps(f)
        def hook(*f_args, **f_kwargs):
            """Implementation of hook."""
            # The profiler stopped after __init__ so we need to get it up again and stop it manually also.
            prof.start()
            r = f(*f_args, **f_kwargs)
            prof.stop()
            return r

        return hook

    # Hook the crucial Brain methods.
    if class_hooks is None:
        class_hooks = ["fit", "evaluate"]
    for method in class_hooks:
        if method in dir(func):  # func is an instance of Brain
            setattr(func, method, hook_brain(getattr(func, method)))


def profile(
    func: Optional[object] = None,
    class_hooks: Optional[Iterable[str]] = None,
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

    Arguments
    ---------
    func : object
        ``speechbrain.core.Brain``:s or a (train/eval) function to be profiled.
    class_hooks : iterable
        List of method/function names of ``speechbrain.core.Brain``:s that should be profiled also.
        Otherwise, only the __init__ constructor will be profiled when decorating a Brain class.
        Default: ``['fit', 'evaluate']`` for classes, and ``None`` for functions.
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
    >>> @profile
    ... def run(x : torch.Tensor):
    ...     y = x ** 2
    ...     z = y ** 3
    ...     return y.backward()  # y.backward() returns None --> return value is substituted with profiler
    >>> data = torch.randn((1, 1), requires_grad=True)
    >>> prof = run(data)
    >>> out = [len(prof.events()), len(prof.key_averages()), prof.profiler.total_average().count]
    """
    if func is None:  # return a profiler; not tested
        return prepare_profiler_for_brain(
            profiler.profile(
                activities=activities,
                schedule=schedule,
                on_trace_ready=on_trace_ready,
                record_shapes=record_shapes,
                profile_memory=profile_memory,
                with_stack=with_stack,
                with_flops=with_flops,
                with_modules=with_modules,
            )
        )
    # Polymorph: func is pretrained or an instance of Brain (assumed case)
    if hasattr(func, "HPARAMS_NEEDED") or not callable(func):
        with profiler.profile(
            activities=activities,
            schedule=schedule,  # scheduler needs to be set directly (fetching is here not possible as for wrappers)
            on_trace_ready=on_trace_ready,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
            with_modules=with_modules,
        ) as prof:
            func.profiler = prepare_profiler_for_brain(prof)
            hook_brain_methods(func=func, class_hooks=class_hooks, prof=prof)
            return func  # no need to return anything; all done in-place; but if needs to be readily assigned elsewhere
    else:
        # callable(func) - polymorph: __init__ Brain constructor -or- function to be wrapped
        @wraps(func)
        def wrapper(*args, **kwargs):
            """Implementation of the wrapper."""
            # Binding variables.
            nonlocal class_hooks
            nonlocal schedule
            nonlocal on_trace_ready
            # Check if there's a nested decorators.
            if schedule is None:
                if "schedule" in kwargs:
                    schedule = kwargs.pop("schedule")
            if on_trace_ready is None:
                if "on_trace_ready" in kwargs:
                    on_trace_ready = kwargs.pop("on_trace_ready")
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
                # Preserves profiler as class attribute if func is not a function (implies: speechbrain.core.Brain).
                if "__call__" not in dir(func):
                    # Passing the profiler to Bain:s' __init__ constructor as an additional argument.
                    kwargs["profiler"] = prepare_profiler_for_brain(prof)
                    hook_brain_methods(
                        func=func, class_hooks=class_hooks, prof=prof
                    )

                # Run & trace to benchmark.
                result = func(*args, **kwargs)

                # Prof is about to be lost at return.
                if "__call__" in dir(func):
                    if result is None:
                        return prof  # for void function, simply return profiling data
                    else:  # not tested - returns both
                        return result, prof

                return result

        return wrapper


def profile_analyst(
    func: Optional[object] = None, class_hooks: Optional[Iterable[str]] = None,
):  # to diverge, define parameters from scratch: @schedule; @export & @profile
    """Pre-configured profiling for a fully detailed benchmark - analyst perspective.

    Creating this analyst view will create overheads (disabling some PyTorch optimisations);
    use @profile_optimiser to take benefits of optimisations and further optimise your modules, accordingly.
    """
    profiler_kwargs = {
        "schedule": schedule(),
        "on_trace_ready": None,
        "record_shapes": True,
        "profile_memory": True,
        "with_stack": True,
        "with_flops": True,  # only for: matrix multiplication & 2D conv; see: torch.autograd.profiler.profile
        "with_modules": True,
        "class_hooks": class_hooks,
    }
    wrapped_func = profile(func, **profiler_kwargs)
    # Polymorph: func is pretrained or an instance of Brain (assumed case)
    if hasattr(func, "HPARAMS_NEEDED") or not callable(func):
        return wrapped_func
    else:  # callable(func) - polymorph: __init__ Brain constructor -or- function to be wrapped

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Implementation of the wrapper."""
            return wrapped_func(*args, **kwargs)

        return wrapper


def profile_optimiser(
    func: Optional[object] = None, class_hooks: Optional[Iterable[str]] = None,
):  # to diverge, define parameters from scratch: @schedule; @export & @profile
    """Pre-configured profiling for a detailed benchmark (better suitable for speed-optimisation than @profile_analyst).
    """
    profiler_kwargs = {
        "schedule": schedule(),
        "on_trace_ready": None,
        "record_shapes": False,  # avoid: overheads
        "profile_memory": True,
        "with_stack": False,  # avoid: overheads
        "with_flops": False,  # only for: matrix multiplication & 2D conv; see: torch.autograd.profiler.profile
        "with_modules": True,
        "class_hooks": class_hooks,
    }
    wrapped_func = profile(func, **profiler_kwargs)
    # Polymorph: func is pretrained or an instance of Brain (assumed case)
    if hasattr(func, "HPARAMS_NEEDED") or not callable(func):
        return wrapped_func
    else:  # callable(func) - polymorph: __init__ Brain constructor -or- function to be wrapped

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Implementation of the wrapper."""
            return wrapped_func(*args, **kwargs)

        return wrapper


def profile_report(  # not part of unittests
    func: Optional[object] = None, class_hooks: Optional[Iterable[str]] = None,
):
    """Pre-configured profiling for a reporting benchmark (changed scheduler to @profile_optimiser).
    """
    profiler_kwargs = {
        "schedule": schedule(
            wait=1, warmup=2, active=7, repeat=1, skip_first=0,
        ),  # gives #active, avg:ed of #repeat
        "on_trace_ready": None,
        "record_shapes": False,  # avoid: overheads
        "profile_memory": True,
        "with_stack": False,  # avoid: overheads
        "with_flops": False,  # only for: matrix multiplication & 2D conv; see: torch.autograd.profiler.profile
        "with_modules": True,
        "class_hooks": class_hooks,
    }
    wrapped_func = profile(func, **profiler_kwargs)
    # Polymorph: func is pretrained or an instance of Brain (assumed case)
    if hasattr(func, "HPARAMS_NEEDED") or not callable(func):
        return wrapped_func
    else:  # callable(func) - polymorph: __init__ Brain constructor -or- function to be wrapped

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Implementation of the wrapper."""
            return wrapped_func(*args, **kwargs)

        return wrapper


"""
def events_diff(
    a: EventList, b: EventList, filter_by: str = "count",
):
    " ""Takes two ``EventList``:s in, filters events of equal value (default: by the count of events).

    The purpose of the results of this diff are for visualisation only (to see the difference between implementations).
    " ""
    # Making copies from the originals instead of simply adding the diff directly might be slower (preserves structure).
    aa = deepcopy(a)
    bb = deepcopy(b)

    # Maps: function name -> (call count, position) // the position helps to remove alike call numbers later on.
    a_filter = dict(
        [(i.key, (getattr(i, filter_by), p)) for p, i in enumerate(aa)]
    )
    b_filter = dict(
        [(i.key, (getattr(i, filter_by), p)) for p, i in enumerate(bb)]
    )

    # Figuring our which ones to delete.
    a_to_remove = list([])
    b_to_remove = list([])
    for key in a_filter.keys():
        if key in b_filter.keys():
            # Equal values are filtered.
            if a_filter[key][0] == b_filter[key][0]:
                # Enlist position to be removed.
                a_to_remove.append(a_filter[key][1])
                b_to_remove.append(b_filter[key][1])

    # Since EventLists are lists: removing items from the back.
    if a_to_remove:
        a_to_remove.sort(reverse=True)
        for k in a_to_remove:
            aa.remove(aa[k])

    if b_to_remove:
        b_to_remove.sort(reverse=True)
        for k in b_to_remove:
            bb.remove(bb[k])

    return aa, bb
"""


def report_time(events: object, verbose=False, upper_control_limit=False):
    """Summary reporting of total time - see: torch.autograd.profiler_util
    """
    # Aggregate CPU & CUDA time.
    """
    if isinstance(events, FunctionEvent):
        function_events = events
    elif
    """
    if isinstance(events, profiler.profile):
        function_events = events.events()
    elif hasattr(events, "profiler"):  # assumes speechbrain.core.Brain
        function_events = events.profiler.events()
    else:
        raise TypeError(
            "Expected a FunctionEvent; profiler.profile, or a SpeechBrain."
        )

    if upper_control_limit:
        # discerns top-level event (among others) aten:zeros which is in the avg range of 10-20ms on laptop CPU
        cpu_data = np.array(
            [e.cpu_time for e in function_events if e.key == "ProfilerStep*"]
        )
        cuda_data = np.array(
            [e.cuda_time for e in function_events if e.key == "ProfilerStep*"]
        )
        cpu_time = cpu_data.mean() + 3 * cpu_data.std()
        cuda_time = cuda_data.mean() + 3 * cuda_data.std()
    else:
        total = function_events.total_average()
        cpu_time = total.self_cpu_time_total
        cuda_time = total.self_cuda_time_total

    """
    if verbose:
        print("CPU time: {}".format(_format_time(cpu_time)))
        if cuda_time > 0:
            print("CUDA time: {}".format(_format_time(cuda_time)))
    """

    return cpu_time, cuda_time


def report_memory(handler: object, verbose=False):
    """Summary reporting of total time - see: torch.autograd.profiler_util
    """
    # Aggregate CPU & CUDA time.
    """
    if isinstance(handler, FunctionEvent):
        events = handler
    elif
    """
    if isinstance(handler, profiler.profile):
        events = handler.events()
    elif hasattr(handler, "profiler"):  # assumes speechbrain.core.Brain
        events = handler.profiler.events()
    else:
        raise TypeError(
            "Expected a FunctionEvent; profiler.profile, or a SpeechBrain."
        )

    """memory allocation during each time step is of relevance, e.g. for visualisation - time intensive for lots events
    mem_times = np.unique(
        [[x.time_range.start, x.time_range.end] for x in events]
    )
    cpu_memory = np.zeros_like(mem_times)
    cuda_memory = np.zeros_like(mem_times)
    for x in events:
        idx = (x.time_range.start <= mem_times) & (
            x.time_range.end >= mem_times
        )
        cpu_memory[idx] += x.cpu_memory_usage
        cuda_memory[idx] += x.cuda_memory_usage

    # variable names instead of labeling pandas' columns
    cpu_mem = np.max(cpu_memory)
    cuda_mem = np.max(cuda_memory)
    """

    cpu_mem = cuda_mem = 0
    for e in events:
        if len(e.cpu_children) == 0:
            leaf_cpu_mem = e.cpu_memory_usage
            leaf_cuda_mem = e.cuda_memory_usage
            parent = e.cpu_parent
            while parent is not None:
                leaf_cpu_mem += parent.cpu_memory_usage
                leaf_cuda_mem += parent.cuda_memory_usage
                parent = parent.cpu_parent
            if leaf_cpu_mem > cpu_mem:
                cpu_mem = leaf_cpu_mem
            if leaf_cuda_mem > cuda_mem:
                cuda_mem = leaf_cuda_mem

    """
    if verbose:
        print("Peak CPU Mem: {}".format(_format_memory(cpu_mem)))
        if cuda_mem > 0:
            print("Peak CUDA Mem: {}".format(_format_memory(cuda_mem)))
    """

    return cpu_mem, cuda_mem
