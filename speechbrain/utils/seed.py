"""Seed utilities for reproducibility.

Authors
 * Adel Moumen 2024
"""

import os
import random

import torch

from speechbrain.utils.distributed import get_rank, rank_prefixed_message
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)

max_seed_value = 4294967295  # 2^32 - 1 (uint32)
min_seed_value = 0


def seed_everything(
    seed: int = 0, verbose: bool = True, deterministic: bool = False
) -> int:
    r"""Function that sets the seed for pseudo-random number generators in: torch, numpy, and Python's random module.

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

    # if DDP, we need to offset the seed by the rank
    # to avoid having the same seed on all processes
    seed_offset = 0 if get_rank() is None else get_rank()

    if not (min_seed_value <= seed <= max_seed_value):
        logger.info(
            f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}",
        )
        seed = seed_offset
    else:
        seed += seed_offset

    if verbose:
        logger.info(
            rank_prefixed_message(f"Setting seed to {seed}"),
            main_process_only=False,
        )

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
