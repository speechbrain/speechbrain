"""This file ensures old links to this file continue to work while providing a Deprecation warning"""

import warnings

from speechbrain.integrations.decoders.kenlm_scorer import *  # noqa: F401, F403

warnings.warn(
    message="speechbrain.decoders.language_model has moved to speechbrain.integrations.decoders.kenlm_scorer",
    category=DeprecationWarning,
    stacklevel=2,
)
