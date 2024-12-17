"""This file ensures old links to speechtokenizer continue to work while providing a Deprecation warning"""

import warnings

from speechbrain.integrations.alignment.ctc_segmentation import *  # noqa: F401, F403

warnings.warn(
    message="speechbrain.alignment.ctc_segmentation has moved to speechbrain.integrations.alignment.ctc_segmentation",
    category=DeprecationWarning,
    stacklevel=2,
)
