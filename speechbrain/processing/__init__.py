"""Data augmentation and low level feature extraction
"""

from .features import (
    STFT,
    ISTFT,
    spectral_magnitude,
    Filterbank,
    DCT,
    Deltas,
    ContextWindow,
    InputNormalization,
)
from .PLDA_LDA import LDA, PLDA
from .signal_processing import (
    compute_amplitude,
    normalize,
    rescale,
    convolve1d,
    reverberate,
    dB_to_amplitude,
    notch_filter,
)
from .speech_augmentation import (
    AddNoise,
    AddReverb,
    SpeedPerturb,
    Resample,
    AddBabble,
    DropFreq,
    DropChunk,
    DoClip,
)

__all__ = [
    "STFT",
    "ISTFT",
    "spectral_magnitude",
    "Filterbank",
    "DCT",
    "Deltas",
    "ContextWindow",
    "InputNormalization",
    "LDA",
    "PLDA",
    "compute_amplitude",
    "normalize",
    "rescale",
    "convolve1d",
    "reverberate",
    "dB_to_amplitude",
    "notch_filter",
    "AddNoise",
    "AddReverb",
    "SpeedPerturb",
    "Resample",
    "AddBabble",
    "DropFreq",
    "DropChunk",
    "DoClip",
]
