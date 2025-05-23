"""Seed utilities for reproducibility.

Authors
 * Adel Moumen 2024
"""

import os
import random

import torch

from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)

max_seed_value = 4294967295  # 2^32 - 1 (uint32)
min_seed_value = 0


def seed_everything(
    seed: int = 0, verbose: bool = True, deterministic: bool = False
) -> int:
    r"""Function that sets the seed for pseudo-random number generators in: torch, numpy, and Python's random module. Important note on DDP: all DDP
    process have the same seed. This is important to ensure that parameters
    without a require_grad set to True are the same across processes. This
    must be taken into account if one wants to build a custom data sampler as
    the processes would pick the same samples... SpeechBrain takes care of that
    internally.

    Arguments
    ---------
    seed: int
        the integer value seed for global random state.
    verbose: bool
        Whether to print a message on each rank with the seed being set.
    deterministic: bool
        Whether to set the seed for deterministic operations.

    Returns
    -------
    int
        The seed that was set.
    """

    if not (min_seed_value <= seed <= max_seed_value):
        logger.info(
            f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}",
        )
        seed = min_seed_value

    if verbose:
        logger.info(f"Setting seed to {seed}")

    os.environ["SB_GLOBAL_SEED"] = str(seed)
    random.seed(seed)

    # if numpy is available, seed it
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    torch.manual_seed(seed)
    # safe to call this function even if cuda is not available
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
    return seed
