
from speechbrain.utils.logger import get_logger
import logging
import os

logger = get_logger(__name__)

triton_enabled = False

def _should_verbose_log():
    return "SB_TRITON_VERBOSE" in os.environ and os.environ["SB_TRITON_VERBOSE"]

# FIXME: lean way to add traceback here? + combining kernel logs?

def log_deselect_reason(kernel_name, fmt_string, *args, **kwargs):
    if _should_verbose_log():
        logging.warning(f"Performance: {kernel_name} optimized kernel deselected: {fmt_string}", *args, **kwargs)

def log_perf_warning(kernel_name, fmt_string, *args, **kwargs):
    if _should_verbose_log():
        logging.warning(f"Performance: {kernel_name} may work suboptimally: {fmt_string}", *args, **kwargs)

def log_picked_kernel(kernel_name):
    if _should_verbose_log():
        logging.info(f"Performance: candidate custom kernel {kernel_name} picked!")
