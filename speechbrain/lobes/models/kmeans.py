"""This file ensures old links to kmeans continue to work while providing a Deprecation warning"""

import warnings

from speechbrain.integrations.audio_tokenizers.kmeans import *  # noqa: F401, F403

warnings.warn(
    message="speechbrain.lobes.models.kmeans has moved to speechbrain.integrations.audio_tokenizers.kmeans",
    category=DeprecationWarning,
    stacklevel=2,
)
