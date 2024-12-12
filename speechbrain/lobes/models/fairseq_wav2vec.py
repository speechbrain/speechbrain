import warnings

from speechbrain.integrations.fairseq.wav2vec import *  # noqa: F401, F403

warnings.warn(
    message="speechbrain.lobes.models.fairseq_wav2vec has moved to speechbrain.integrations.fairseq.wav2vec",
    category=DeprecationWarning,
    stacklevel=2,
)
