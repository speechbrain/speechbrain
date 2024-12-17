"""This file ensures old links to this file continue to work while providing a Deprecation warning"""

import warnings

from speechbrain.integrations.lm.ctc import *  # noqa: F401, F403

warnings.warn(
    message="speechbrain.decoders.language_model has moved to speechbrain.integrations.lm.ctc",
    category=DeprecationWarning,
    stacklevel=2,
)
