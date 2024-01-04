"""This module implements utilities and abstractions for use with
`torch.autocast`, i.e. Automatic Mixed Precision.

Authors
 * Sylvain de Langen 2023
"""
import functools
from typing import Callable, Optional
import torch


def fwd_default_precision(
    fwd: Optional[Callable] = None,
    cast_inputs: Optional[torch.dtype] = torch.float32,
):
    """Decorator for forward methods which, by default, *disables* autocast
    and casts any floating-point tensor parameters into the specified dtype
    (much like `torch.cuda.amp.custom_fwd`).

    The *wrapped forward* will gain an additional `force_allow_autocast` keyword
    parameter.
    When set to `True`, the function will ignore `cast_inputs` and will not
    disable autocast, as if this decorator was not specified.
    (Thus, modules can specify a default recommended precision, and users can
    override that behavior when desired.)

    Note that as of PyTorch 2.1.1, this will **only** affect **CUDA** AMP.
    Non-CUDA AMP will be unaffected and no input tensors will be cast!
    This usecase may be supported by this function in the future.

    When autocast is *not* active, this decorator does not change any behavior.

    Arguments
    ---------
    fwd: Optional[Callable]
        The function to wrap. If omitted, returns a partial application of the
        decorator, e.g. allowing
        `new_decorator = fwd_default_precision(cast_inputs=torch.float32)`.

        Reminder: If you are decorating a function directly, this argument is
        already specified implicitly.

    cast_inputs: Optional[torch.dtype]
        If not `None` (the default being `torch.float32`), then any
        floating-point inputs to the wrapped function will be cast to the
        specified type.

        Note: When autocasting is enabled, output tensors of autocast-compatible
        operations may be of the autocast data type.
        Disabling autocast *without* casting inputs will not change this fact,
        so lower precision operations can happen even inside of an
        autocast-disabled region, which this argument helps avoid if desired.
    """

    if fwd is None:
        return functools.partial(fwd_default_precision, cast_inputs=cast_inputs)

    # NOTE: torch.cuda.amp.custom_fwd is written with the assumption of CUDA
    # autocast. There does not seem to be a generic equivalent.
    # Detecting CUDA AMP specifically also seems difficult or impossible, so we
    # cannot even reliably warn about the issue. For now, we just document the
    # problem.
    wrapped_fwd = torch.cuda.amp.custom_fwd(fwd, cast_inputs=cast_inputs)

    @functools.wraps(fwd)
    def wrapper(*args, force_allow_autocast: bool = False, **kwargs):
        """Wrapped forward function from fwd_default_precision.

        Arguments
        ---------
        force_allow_autocast: bool
            When `True`, the wrapped function will be executed directly with no
            change to the autocast context and no input casting."""
        if force_allow_autocast:
            return fwd(*args, **kwargs)
        else:
            return wrapped_fwd(*args, **kwargs)

    return wrapper
