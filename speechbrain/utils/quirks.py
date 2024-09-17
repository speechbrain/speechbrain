"""Global changes and platform/GPU-specific quirks, i.e. workarounds and saner
defaults, sometimes due to platform-specific issues.

Author:
    * Sylvain de Langen 2024
"""

import logging

import torch

logger = logging.getLogger(__name__)


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


KNOWN_QUIRKS = {
    "disable_cudnn_benchmarking": disable_cudnn_benchmarking,
    "disable_jit_profiling": disable_jit_profiling,
    "allow_tf32": allow_tf32,
}

applied_quirks = set()


def apply_quirks():
    """Apply quirks depending on the platform. Also populates `applied_quirks`."""

    # global quirks
    applied_quirks.add("disable_jit_profiling")
    applied_quirks.add("allow_tf32")

    # AMD HIP?
    if torch.cuda.is_available() and torch.version.hip:
        applied_quirks.add("disable_cudnn_benchmarking")

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
