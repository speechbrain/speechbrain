"""Managing the logger, utilities

Author
 * Fang-Pen Lin 2012 https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/
 * Peter Plantinga 2020
 * Aku Rouhe 2020
"""

import functools
import logging
import logging.config
import math
import os
import sys

import torch
import tqdm
import yaml

from speechbrain.utils.data_utils import recursive_update
from speechbrain.utils.distributed import if_main_process
from speechbrain.utils.superpowers import run_shell

ORDERS_ABBREV = {
    -24: "y",
    -21: "z",
    -18: "a",
    -15: "f",
    -12: "p",
    -9: "n",
    -6: "µ",
    -3: "m",
    0: "",
    3: "k",
    6: "M",
    9: "G",
    12: "T",
    15: "P",
    18: "E",
    21: "Z",
    24: "Y",
}

# Short scale
# Negative powers of ten in lowercase, positive in uppercase
ORDERS_WORDS = {
    -24: "septillionths",
    -21: "sextillionths",
    -18: "quintillionths",
    -15: "quadrillionths",
    -12: "trillionths",
    -9: "billionths",
    -6: "millionths",
    -3: "thousandths",
    0: "",
    3: "Thousand",
    6: "Million",
    9: "Billion",
    12: "Trillion",
    15: "Quadrillion",
    18: "Quintillion",
    21: "Sextillion",
    24: "Septillion",
}


class MultiProcessLoggerAdapter(logging.LoggerAdapter):
    r"""
    Logger adapter that handles multi-process logging, ensuring logs are written
    only on the main process if specified. This class extends `logging.LoggerAdapter`
    and provides additional functionality for controlling logging in multi-process
    environments, with the option to limit logs to the main process only.

    This class is heavily inspired by HuggingFace Accelerate toolkit:
    https://github.com/huggingface/accelerate/blob/85b1a03552cf8d58e036634e004220c189bfb247/src/accelerate/logging.py#L22
    """

    @staticmethod
    def _should_log(main_process_only: bool) -> bool:
        r"""
        Determines if logging should occur based on whether the code is running
        on the main process or not.

        Arguments
        ---------
        main_process_only : bool
            A flag indicating if logging should be restricted to the main process.

        Returns
        -------
        bool
            True if logging should be performed (based on the process and the flag),
            False otherwise.
        """
        return not main_process_only or (
            main_process_only and if_main_process()
        )

    def log(self, level: int, msg: str, *args: tuple, **kwargs: dict):
        r"""
        Logs a message with the specified log level, respecting the `main_process_only`
        flag to decide whether to log based on the current process.

        Arguments
        ---------
        level : int
            Logging level (e.g., logging.INFO, logging.WARNING).
        msg : str
            The message to log.
        *args : tuple
            Additional positional arguments passed to the logger.
        **kwargs : dict
            Additional keyword arguments passed to the logger, including:
            - main_process_only (bool): If True, log only from the main process (default: True).
            - stacklevel (int): The stack level to use when logging (default: 2).

        Notes
        -----
        If `main_process_only` is True, the log will only be written if the current process
        is the main process, as determined by `if_main_process()`.
        """
        main_process_only = kwargs.pop("main_process_only", True)
        kwargs.setdefault("stacklevel", 2)

        if self.isEnabledFor(level):
            if self._should_log(main_process_only):
                msg, kwargs = self.process(msg, kwargs)
                self.logger.log(level, msg, *args, **kwargs)

    @functools.lru_cache(None)
    def warning_once(self, *args: tuple, **kwargs: dict):
        r"""
        Logs a warning message only once by using caching to prevent duplicate warnings.

        Arguments
        ---------
        *args : tuple
            Positional arguments passed to the warning log.
        **kwargs : dict
            Keyword arguments passed to the warning log.

        Notes
        -----
        This method is decorated with `functools.lru_cache(None)`, ensuring that the warning
        message is logged only once regardless of how many times the method is called.
        """
        self.warning(*args, **kwargs)


def get_logger(name: str) -> MultiProcessLoggerAdapter:
    """
    Retrieves a logger with the specified name, applying a log level from the environment variable
    `SB_LOG_LEVEL` if set, or defaults to `INFO` level.

    If the environment variable `SB_LOG_LEVEL` is not defined, it defaults to `INFO` level and sets
    this level in the environment for future use. The environment variable can be set manually or
    automatically in `Brain` class following `setup_logging`.

    Arguments
    ---------
    name : str
        The name of the logger to retrieve.

    Returns
    -------
    MultiProcessLoggerAdapter
        An instance of `MultiProcessLoggerAdapter` wrapping the logger with the specified name.
    """

    logger = logging.getLogger(name)
    log_level = os.environ.get("SB_LOG_LEVEL", None)
    if log_level is None:
        log_level = logging.INFO
        os.environ["SB_LOG_LEVEL"] = str(log_level)
    logging.basicConfig(level=int(log_level))
    return MultiProcessLoggerAdapter(logger, {})


def setup_logging(
    config_path="log-config.yaml",
    overrides={},
    default_level=logging.INFO,
):
    """Setup logging configuration.

    Arguments
    ---------
    config_path : str
        The path to a logging config file.
    overrides : dict
        A dictionary of the same structure as the config dict
        with any updated values that need to be applied.
    default_level : int
        The level to use if the config file is not found.
    """
    if os.path.exists(config_path):
        with open(config_path, "rt", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        recursive_update(config, overrides)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
    os.environ["SB_LOG_LEVEL"] = str(default_level)


class TqdmCompatibleStreamHandler(logging.StreamHandler):
    """TQDM compatible StreamHandler.

    Writes and prints should be passed through tqdm.tqdm.write
    so that the tqdm progressbar doesn't get messed up.
    """

    def emit(self, record):
        """TQDM compatible StreamHandler."""
        try:
            msg = self.format(record)
            stream = self.stream
            tqdm.tqdm.write(msg, end=self.terminator, file=stream)
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


def format_order_of_magnitude(number, abbreviate=True):
    """Formats number to the appropriate order of magnitude for printing.

    Arguments
    ---------
    number : int, float
        The number to format.
    abbreviate : bool
        Whether to use abbreviations (k,M,G) or words (Thousand, Million,
        Billion). Numbers will be either like: "123.5k" or "123.5 Thousand".

    Returns
    -------
    str
        The formatted number. Note that the order of magnitude token is part
        of the string.

    Example
    -------
    >>> print(format_order_of_magnitude(123456))
    123.5k
    >>> print(format_order_of_magnitude(0.00000123, abbreviate=False))
    1.2 millionths
    >>> print(format_order_of_magnitude(5, abbreviate=False))
    5
    """
    style = ORDERS_ABBREV if abbreviate else ORDERS_WORDS
    precision = "{num:3.1f}"
    order = 3 * math.floor(math.log(math.fabs(number), 1000))
    # Fallback for very large numbers:
    while order not in style and order != 0:
        order = order - math.copysign(3, order)  # Bring 3 units towards 0
    order_token = style[order]
    if order != 0:
        formatted_number = precision.format(num=number / 10**order)
    else:
        if isinstance(number, int):
            formatted_number = str(number)
        else:
            formatted_number = precision.format(num=number)
    if abbreviate or not order_token:
        return formatted_number + order_token
    else:
        return formatted_number + " " + order_token


def get_environment_description():
    """Returns a string describing the current Python / SpeechBrain environment.

    Useful for making experiments as replicable as possible.

    Returns
    -------
    str
        The string is formatted ready to be written to a file.

    Example
    -------
    >>> get_environment_description().splitlines()[0]
    'SpeechBrain system description'
    """
    python_version_str = "Python version:\n" + sys.version + "\n"
    try:
        freezed, _, _ = run_shell("pip freeze")
        python_packages_str = "Installed Python packages:\n"
        python_packages_str += freezed.decode(errors="replace")
    except OSError:
        python_packages_str = "Could not list python packages with pip freeze"
    try:
        git_hash, _, _ = run_shell("git rev-parse --short HEAD")
        git_str = "Git revision:\n" + git_hash.decode(errors="replace")
    except OSError:
        git_str = "Could not get git revision"
    if torch.cuda.is_available():
        if torch.version.cuda is None:
            cuda_str = "ROCm version:\n" + torch.version.hip
        else:
            cuda_str = "CUDA version:\n" + torch.version.cuda
    else:
        cuda_str = "CUDA not available"
    result = "SpeechBrain system description\n"
    result += "==============================\n"
    result += python_version_str
    result += "==============================\n"
    result += python_packages_str
    result += "==============================\n"
    result += git_str
    result += "==============================\n"
    result += cuda_str
    return result
