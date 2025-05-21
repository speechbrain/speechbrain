"""Library for checking the torchaudio backend.

Authors
-------
 * Mirco Ravanelli 2021
 * Adel Moumen 2025
"""

import platform
from typing import Optional

import torchaudio

from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


def try_parse_torchaudio_major_version() -> Optional[int]:
    """Tries parsing the torchaudio major version.

    Returns
    -------
    The parsed major version, otherwise ``None``.
    """
    if not hasattr(torchaudio, "__version__"):
        return None

    version_split = torchaudio.__version__.split(".")

    # expect in format x.y.z whatever; we care only about x

    if len(version_split) <= 2:
        # not sure how to parse this
        return None

    try:
        major_version = int(version_split[0])
        minor_version = int(version_split[1])
    except Exception:
        return None

    return major_version, minor_version


def check_torchaudio_backend():
    """Checks the torchaudio backend and sets it to soundfile if
    windows is detected.
    """
    torchaudio_major, torchaudio_minor = try_parse_torchaudio_major_version()

    if torchaudio_major is None:
        logger.warning(
            "Failed to detect torchaudio major version; unsure how to check your setup. We recommend that you keep torchaudio up-to-date."
        )
    elif torchaudio_major >= 2 and torchaudio_minor >= 1:
        available_backends = torchaudio.list_audio_backends()

        if len(available_backends) == 0:
            logger.warning(
                "SpeechBrain could not find any working torchaudio backend. Audio files may fail to load. Follow this link for instructions and troubleshooting: https://speechbrain.readthedocs.io/en/latest/audioloading.html"
            )
    else:
        logger.warning(
            "This version of torchaudio is old. SpeechBrain no longer tries using the torchaudio global backend mechanism in recipes, so if you encounter issues, update torchaudio to >=2.1.0."
        )
        current_system = platform.system()
        if current_system == "Windows":
            logger.warning(
                'Switched audio backend to "soundfile" because you are running Windows and you are running an old torchaudio version.'
            )
            torchaudio.set_audio_backend("soundfile")


def validate_backend(backend):
    """
    Validates the specified audio backend.

    Parameters
    ----------
    backend : str or None
        The name of the backend to validate. Must be one of [None, 'ffmpeg', 'sox', 'soundfile'].

    Raises
    ------
    ValueError
        If the `backend` is not one of the allowed values.
    """
    allowed_backends = [None, "ffmpeg", "sox", "soundfile"]
    if backend not in allowed_backends:
        raise ValueError(
            f"backend must be one of {allowed_backends}",
            "Available backends on your system: ",
            torchaudio.list_audio_backends(),
        )
