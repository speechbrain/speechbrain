"""This file ensures old links to bleu continue to work while providing a Deprecation warning"""

import warnings

from speechbrain.integrations.metrics.bleu import *  # noqa: F401, F403

warnings.warn(
    message="speechbrain.util.bleu has moved to speechbrain.integrations.metrics.bleu",
    category=DeprecationWarning,
    stacklevel=2,
)
