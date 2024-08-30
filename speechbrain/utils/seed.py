"""Seed utilities for reproducibility.

Authors
 * Adel Moumen 2024
"""

import logging
import random

import torch

from speechbrain.utils.distributed import (
    distributed_is_initialized,
    get_rank,
    rank_prefixed_message,
)

logger = logging.getLogger(__name__)

max_seed_value = 4294967295  # 2^32 - 1 (uint32)
min_seed_value = 0


def seed_everything(seed: int, verbose: bool = True) -> None:
    r"""Function that sets the seed for pseudo-random number generators in: torch, numpy, and Python's random module.

    Arguments
    ---------
    seed: the integer value seed for global random state.
    verbose: Whether to print a message on each rank with the seed being set.
    """
    # if DDP, we need to offset the seed by the rank to avoid having the same seed on all processes
    if distributed_is_initialized():
        seed_offset = get_rank()
    else:
        seed_offset = 0

    if not (min_seed_value <= seed <= max_seed_value):
        logger.info(
            f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}"
        )
        seed = 0 + seed_offset
    else:
        seed += seed_offset

    if verbose:
        logger.info(
            rank_prefixed_message(f"Setting seed to {seed}", get_rank())
        )

    random.seed(seed)

    # if numpy is available, seed it
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
