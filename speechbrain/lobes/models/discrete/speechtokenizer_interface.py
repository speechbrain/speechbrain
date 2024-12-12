import warnings

from speechbrain.integrations.discrete.speechtokenizer import *  # noqa: F401, F403

warnings.warn(
    message="speechbrain.lobes.models.discrete.speechtokenizer_interface has moved to speechbrain.integrations.discrete.speechtokenizer",
    category=DeprecationWarning,
    stacklevel=2,
)
