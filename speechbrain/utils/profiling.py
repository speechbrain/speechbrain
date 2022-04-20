"""Decorators to handle PyTorch profiling and benchmarking.

Author:
    * Andreas Nautsch 2022
"""
from copy import deepcopy
from torch import profiler
from itertools import chain
from functools import wraps
from torch.autograd.profiler_util import EventList
from typing import Any, Callable, Iterable, Optional


def scheduler(func, wait: int = 2, warmup: int = 2, active: int = 2, repeat: int = 1, skip_first: int = 0):
    """Wrapper to create a ```torch.profiler.schedule``` (sets default parameters for warm-up).
    """
    torch_scheduler = profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat, skip_first=skip_first)

    @wraps(func)
    def wrapper(*args, **kwargs):
        kwargs["schedule"] = torch_scheduler
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


def profile(func: object,
            sub_methods: Optional[Iterable[str]] = None,
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
    sub_methods : iterable
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
    ...     return y.backward()
    >>> data = torch.randn((1, 1), requires_grad=True)
    >>> prof = run(data)
    >>> [len(prof.events()), len(prof.key_averages()), prof.key_averages().total_average().count]
    [25, 15, 25]
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        class_hooks = sub_methods  # only called when func decorates a class
        # check if there's a nested schedule decorated
        scheduler = schedule
        if scheduler is None:
            if 'schedule' in kwargs:
                scheduler = kwargs.pop('schedule')
        with profiler.profile(
                activities=activities,
                schedule=scheduler,
                on_trace_ready=on_trace_ready,
                record_shapes=record_shapes,
                profile_memory=profile_memory,
                with_stack=with_stack,
                with_flops=with_flops,
                with_modules=with_modules,
        ) as prof:
            # Adds profiler to attributes if func is not a function (implies: speechbrain.core.Brain); preserving prof.
            if ("__call__" not in dir(func)) or (not callable(func)):  # __init__ -or- instance of Brain class
                # Brain functions will be called independently -> traces will be segregated, so we aggregate them.
                prof.speechbrain_parsed_kineto_traces = list()

                # Preparing the profiler to be re-used during Brain:s' lifecycles.
                def hook_profiler_stop(stop: Callable):
                    @wraps(stop)
                    def stop_wrapper():
                        if prof.profiler is not None:
                            stop_result = stop()
                            prof.speechbrain_parsed_kineto_traces.append(
                                deepcopy(prof.profiler._parse_kineto_results(prof.profiler.kineto_results))
                            )
                            return stop_result
                        else:
                            return stop()  # will be: None
                    return stop_wrapper

                # It's currently designed as hiding an Easter Egg.
                def merge_traces():
                    # Alternative re-design quirks: make trace aggregator a GLOBAL -or- create another profiler class.
                    trace_aggregator = "speechbrain_parsed_kineto_traces"
                    if trace_aggregator in dir(prof):
                        merged_events = EventList(list(chain(*getattr(prof, trace_aggregator))),
                                                  use_cuda=prof.profiler.use_cuda,
                                                  profile_memory=prof.profiler.profile_memory,
                                                  with_flops=prof.profiler.with_flops)
                        merged_events._build_tree()
                        return merged_events
                    else:  # not tested
                        return prof.events()

                # Augment torch's profiler.
                setattr(prof, "stop", hook_profiler_stop(getattr(prof, "stop")))
                setattr(prof, "merge_traces", merge_traces)

                # Passing the profiler to Bain:s' __init__ constructor as an additional argument.
                kwargs["profiler"] = prof

                # Prepare additional hook decorators for methods of Brain:s.
                def hook_brain(f: Callable):
                    @wraps(f)
                    def hook(*f_args, **f_kwargs):
                        # The profiler stopped after __init__ so we need to get it up again and stop it manually also.
                        prof.start()
                        r = f(*f_args, **f_kwargs)
                        prof.stop()
                        return r
                    return hook

                # Hook the crucial Brain methods.
                if class_hooks is None:
                    class_hooks = ['fit', 'evaluate']
                for method in class_hooks:
                    if method in dir(func):  # func is an instance of Brain
                        setattr(func, method, hook_brain(getattr(func, method)))

            if callable(func):
                # Run & trace to benchmark.
                result = func(*args, **kwargs)

                # Direct profiler post-processing if func is a function (implies: result is None); prof is lost at return.
                if "__call__" in dir(func):
                    if result is None:
                        return prof
                    else:  # not tested
                        pass

                return result
            else:  # if it's an instance of Brain class
                return func

    return wrapper


def profile_details(func):
    """Pre-configured profiling for a detailed (standard) benchmark.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapped_func = profile(schedule=scheduler(...),
                       on_trace_ready=trace_handler(...),
                       record_shapes=True,
                       profile_memory=True,
                       with_stack=True,
                       with_flops=True,
                       with_modules=True,
                       )(func)
        return wrapped_func(*args, **kwargs)

    return wrapper


def events_diff(a: EventList,
                b: EventList,
                filter_by: str = 'count',
                ):
    """Takes two ``EventList``:s in, filters events of equal value (default: by the count of events).

    The purpose of the results of this diff are for visualisation only (to see the difference between implementations).
    """
    # Making copies from the originals instead of simply adding the diff directly might be slower (preserves structure).
    aa = deepcopy(a)
    bb = deepcopy(b)

    # Maps: function name -> (call count, position) // the position helps to remove alike call numbers later on.
    a_filter = dict([(i.key, (getattr(i, filter_by), p)) for p, i in enumerate(aa)])
    b_filter = dict([(i.key, (getattr(i, filter_by), p)) for p, i in enumerate(bb)])

    # Figuring our which ones to delete.
    a_to_remove = list([])
    b_to_remove = list([])
    for key in a_filter.keys():
        # Equal values are filtered.
        if a_filter[key][0] == b_filter[key][0]:
            # Enlist position to be removed.
            a_to_remove.append(a_filter[key][1])
            b_to_remove.append(b_filter[key][1])

    # Since EventLists are lists: removing items from the back.
    a_to_remove.reverse()
    b_to_remove.reverse()
    for k in a_to_remove:
        aa.remove(aa[k])
    for k in b_to_remove:
        bb.remove(bb[k])

    return aa, bb
