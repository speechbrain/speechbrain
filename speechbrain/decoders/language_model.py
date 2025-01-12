"""This file ensures old links to this file continue to work while providing a Deprecation warning"""

import warnings

from speechbrain.integrations.lm.ken import *  # noqa: F401, F403

warnings.warn(
    message="speechbrain.decoders.language_model has moved to speechbrain.integrations.lm.ken",
    category=DeprecationWarning,
    stacklevel=2,
)
