"""Global changes and platform/GPU-specific quirks, i.e. workarounds and saner
defaults, sometimes due to platform-specific issues.

Author:
    * Sylvain de Langen 2024
"""

import os

import torch

import speechbrain.kernels.common
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


def disable_cudnn_benchmarking():
    """Disables CuDNN benchmarking. no-op on platforms where it is already off
    by default.

    Benchmarking, when enabled, theoretically improves convolution performance
    by automatically comparing different kernels for some operations.

    However, benchmarking has to be re-run for every unique input shape, which
    makes it unsuitable for highly dynamic shapes.
    Since SpeechBrain does tend to use very varied shapes without attempting to
    pad the differences out, leaving benchmarking on can severely degrade
    training performance.

    This function disables it as we deem no-benchmarking to be a saner default
    to avoid performance bugs at the moment.

    As of PyTorch 2.3.0, the default is `False` for CUDA GPUs, but `True`
    for HIP GPUs.

    The HIP equivalent to CuDNN is MIOpen, but it is controlled through the same
    PyTorch API.
    """

    torch.backends.cudnn.benchmark = False


def disable_jit_profiling():
    """Disables JIT profiling to avoid performance issues on highly dynamic
    shapes."""

    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)


def allow_tf32():
    """On CUDA backends (potentially including ROCm), enables TensorFloat32
    support for CuDNN and the matmul operator.

    This allows performing certain operations transparently at a lower
    precision, even in fp32 math when AMP is not in use, when otherwise tensor
    cores would not be used. TF32 supports accumulation into fp32, so the
    concern for overflowing is somewhat mitigated.

    On NVIDIA GPUs, this is available since Ampere (e.g. A100).

    See `PyTorch documentation <https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices>`__ for more
    details."""

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def enable_triton_kernels():
    """Enables SpeechBrain to make use of opt-in high-performance GPU kernels
    to speed up certain operations."""

    speechbrain.kernels.common.triton_enabled = True

    try:
        import triton  # noqa
        import triton.language  # noqa
    except ImportError as e:
        raise ImportError(
            "Failed to import Triton. You can sometimes disable our custom "
            "Triton kernels by disabling the `allow_triton_kernels` quirk, but "
            "it may significantly degrade performance."
        ) from e


KNOWN_QUIRKS = {
    "disable_cudnn_benchmarking": disable_cudnn_benchmarking,
    "disable_jit_profiling": disable_jit_profiling,
    "allow_tf32": allow_tf32,
    "allow_triton_kernels": enable_triton_kernels,
}

"""Applied quirk list. Populated by `apply_quirks`."""
applied_quirks = set()

"""Excluded quirk list. Populated by `apply_quirks` from the `SB_DISABLE_QUIRKS`
environment variable, which is a comma-separated list of quirks to disable."""
excluded_quirks = set()


def apply_quirks():
    """Apply quirks depending on the platform. Also populates `applied_quirks`."""

    global applied_quirks, excluded_quirks

    # global quirks
    applied_quirks.add("disable_jit_profiling")
    applied_quirks.add("allow_tf32")
    applied_quirks.add("allow_triton_kernels")

    # AMD HIP?
    if torch.cuda.is_available() and torch.version.hip:
        applied_quirks.add("disable_cudnn_benchmarking")

    if "SB_DISABLE_QUIRKS" in os.environ:
        for quirk_to_exclude in os.environ["SB_DISABLE_QUIRKS"].split(","):
            if quirk_to_exclude != "":
                if quirk_to_exclude not in KNOWN_QUIRKS.keys():
                    raise ValueError(
                        f'SB_DISABLE_QUIRKS environment variable includes unknown quirk name "{quirk_to_exclude}". Supported quirks: [{", ".join(KNOWN_QUIRKS.keys())}]'
                    )
                excluded_quirks.add(quirk_to_exclude)

    applied_quirks = applied_quirks - excluded_quirks

    # finally, apply quirks
    for quirk in applied_quirks:
        KNOWN_QUIRKS[quirk]()

    log_applied_quirks()


def log_applied_quirks():
    """Logs whichever quirks have been applied by `apply_quirks`."""
    logger.info(
        "Applied quirks (see `speechbrain.utils.quirks`): [%s]",
        ", ".join(applied_quirks),
    )

    logger.info(
        "Excluded quirks specified by the `SB_DISABLE_QUIRKS` environment (comma-separated list): [%s]",
        ", ".join(excluded_quirks),
    )
