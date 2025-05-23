"""This module implements utilities and abstractions for use with
`torch.autocast`, i.e. Automatic Mixed Precision.

Authors
 * Sylvain de Langen 2023
 * Adel Moumen 2025
"""

import functools
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class AMPConfig:
    """Configuration for automatic mixed precision (AMP).

    Arguments
    ---------
    dtype : torch.dtype
        The dtype to use for AMP.
    """

    dtype: torch.dtype

    @classmethod
    def from_name(self, name):
        """Create an AMPConfig from a string name.

        Arguments
        ---------
        name : str
            The name of the AMPConfig to create.  Must be one of `fp32`,
            `fp16`, or `bf16`.

        Returns
        -------
        AMPConfig
            The AMPConfig corresponding to the name.
        """
        if name is None or name == "fp32":
            return AMPConfig(torch.float32)
        elif name == "fp16":
            return AMPConfig(torch.float16)
        elif name == "bf16":
            return AMPConfig(torch.bfloat16)
        else:
            raise ValueError(
                f"Specified autocast mode ({name}) incorrect, expected one of `fp32`, `fp16`, `bf16`."
            )


class TorchAutocast:
    """
    A context manager that conditionally enables ``torch.autocast`` for GPU operations.

    This manager wraps around ``torch.autocast`` to automatically enable autocasting when
    running on a GPU and a data type other than float32 is specified. If the desired
    data type is float32, autocasting is bypassed and the context manager behaves as a
    no-op.

    Parameters
    ----------
    *args : tuple
        Positional arguments forwarded to `torch.autocast`.
        See the PyTorch documentation: https://pytorch.org/docs/stable/amp.html#torch.autocast
    **kwargs : dict
        Keyword arguments forwarded to `torch.autocast`.
        Typically includes the `dtype` argument to specify the desired precision.
        See the PyTorch documentation for more details.
    """

    def __init__(self, *args, **kwargs):
        enabled = kwargs.get("dtype", torch.float32) != torch.float32
        if enabled:
            self.context = torch.autocast(*args, **kwargs)
        else:
            self.context = nullcontext()  # no-op context manager

    def __enter__(self):
        """
        Enter the autocast context.

        Returns
        -------
        context
            The result of entering the underlying autocast context manager.

        Raises
        ------
        RuntimeError
            If an error occurs while entering the autocast context and the context
            provides 'device' and 'fast_dtype' attributes, a RuntimeError is raised
            with additional diagnostic information.
        """
        try:
            return self.context.__enter__()
        except RuntimeError as e:
            if hasattr(self.context, "device") and hasattr(
                self.context, "fast_dtype"
            ):
                device = self.context.device
                dtype = self.context.fast_dtype
                raise RuntimeError(
                    f"Error during autocasting with dtype={dtype} on device={device}.\n"
                ) from e
            else:
                raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the autocast context.

        Parameters
        ----------
        exc_type : type
            Exception type if an exception occurred, otherwise None.
        exc_val : Exception
            Exception instance if an exception occurred, otherwise None.
        exc_tb : traceback
            Traceback object if an exception occurred, otherwise None.

        Returns
        -------
        bool or None
            The result of exiting the underlying autocast context manager.
        """
        return self.context.__exit__(exc_type, exc_val, exc_tb)


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

    Returns
    -------
    The wrapped function
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
        *args: tuple
            Arguments to be forwarded to the unwrapped function.
        force_allow_autocast: bool
            When `True`, the wrapped function will be executed directly with no
            change to the autocast context and no input casting.
        **kwargs: dict
            Arguments to be forwarded to the unwrapped function.

        Returns
        -------
        The wrapped function if force_allow_autocast, else the original
        """
        if force_allow_autocast:
            return fwd(*args, **kwargs)
        else:
            return wrapped_fwd(*args, **kwargs)

    return wrapper
