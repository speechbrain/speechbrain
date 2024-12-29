"""This file ensures old links to kmeans continue to work while providing a Deprecation warning"""

import warnings

from speechbrain.integrations.discrete.kmeans import *  # noqa: F401, F403

warnings.warn(
    message="speechbrain.lobes.models.kmeans has moved to speechbrain.integrations.discrete.kmeans",
    category=DeprecationWarning,
    stacklevel=2,
)
