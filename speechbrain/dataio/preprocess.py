"""Preprocessors for audio"""
import torch
import functools
from speechbrain.processing.speech_augmentation import Resample


class AudioNormalizer:
    """Normalizes audio into a standard format

    Example
    -------
    >>> import torchaudio
    >>> example_file = 'samples/audio_samples/example_multichannel.wav'
    >>> signal, sr = torchaudio.load(example_file, channels_first = False)
    >>> normalizer = AudioNormalizer(sample_rate=8000)
    >>> normalized = normalizer(signal, sr)
    >>> signal.shape
    torch.Size([33882, 2])
    >>> normalized.shape
    torch.Size([16941])

    """

    def __init__(self, sample_rate=16000, mix="avg-to-mono"):
        self.sample_rate = sample_rate
        if mix not in ["avg-to-mono", "keep"]:
            raise ValueError(f"Unexpected mixing configuration {mix}")
        self.mix = mix
        self._cached_resample = functools.lru_cache(maxsize=12)(Resample)

    def __call__(self, audio, sample_rate):
        """Perform normalization

        Arguments
        ---------
        audio : tensor
            The input waveform torch tensor. Assuming [time, channels],
            or [time].
        """
        resampler = self._cached_resample(sample_rate, self.sample_rate)
        resampled = resampler(audio.unsqueeze(0)).squeeze(0)
        return self._mix(resampled)

    def _mix(self, audio):
        flat_input = audio.dim() == 1
        if self.mix == "avg-to-mono":
            if flat_input:
                return audio
            return torch.mean(audio, 1)
        if self.mix == "keep":
            return audio
