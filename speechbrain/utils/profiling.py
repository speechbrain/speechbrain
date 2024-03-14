"""Wrapper to handle PyTorch profiling and benchmarking.

Author:
    * Titouan Parcollet 2024
"""

from torch import profiler
from typing import Optional
import os


def prepare_profiler(
    profile_warmup: Optional[int] = 5,
    profile_steps: Optional[int] = 5,
    logdir: Optional[str] = "tensorboard_logs",
) -> object:
    """Wrapper to create a PyTorch profiler to benchmark training of speechbrain.core.Brain instances.
    See ``torch.profiler.profile`` documentation for details (brief summary below).

    Arguments
    ---------
    profile_warmup: int
        Number of warmup step before starting to log.
    profile_steps: int
        Number of steps to log after warmup.
    logdir: str
        Path to the output folder of the logs.
    """

    logdir = os.path.join(logdir, "profiler_logs")

    return profiler.profile(
        schedule=profiler.schedule(
            wait=0, warmup=profile_warmup, active=profile_steps, repeat=1
        ),
        on_trace_ready=profiler.tensorboard_trace_handler(logdir),
        record_shapes=True,
        with_stack=True,
    )
