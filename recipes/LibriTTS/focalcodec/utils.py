"""Common utilities.

Authors
 * Luca Della Libera 2025
"""

import os

import torch

__all__ = ["download_wavlm6"]


def download_wavlm6(cache_dir: "str") -> "str":
    """Download WavLM6 checkpoint to cache and return the path.

    Arguments
    ---------
    cache_dir:
        Cache directory where the checkpoint will be saved.

    Returns
    -------
        Path to the saved checkpoint.

    """
    os.makedirs(cache_dir, exist_ok=True)
    checkpoint_path = os.path.join(cache_dir, "wavlm6.pt")

    # If already cached, return immediately
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    # Load FocalCodec model
    codec = torch.hub.load(
        repo_or_dir="lucadellalib/focalcodec",
        model="focalcodec",
        config="lucadellalib/focalcodec_50hz",
    )

    # Save WavLM6 checkpoint
    torch.save(codec.encoder.state_dict(), checkpoint_path)

    return checkpoint_path
