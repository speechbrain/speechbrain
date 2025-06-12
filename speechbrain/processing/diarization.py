"""This file ensures old links to diarization continue to work while providing a Deprecation warning"""

import warnings

from speechbrain.integrations.alignment.diarization import *  # noqa: F401, F403

warnings.warn(
    message="speechbrain.processing.diarization has moved to speechbrain.integrations.alignment.diarization",
    category=DeprecationWarning,
    stacklevel=2,
)
