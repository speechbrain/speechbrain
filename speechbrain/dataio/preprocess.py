"""Preprocessors for audio"""
import torch
from speechbrain.processing.speech_augmentation import Resample


class AudioNormalizer:
    """Normalizes audio into a standard format

    Arguments
    ---------
    sample_rate : int
        The sampling rate to which the incoming signals should be converted.
    mix : {"avg-to-mono", "keep"}
        "avg-to-mono" - add all channels together and normalize by number of
        channels. This also removes the channel dimension, resulting in [time]
        format tensor.
        "keep" - don't normalize channel information

    Example
    -------
    >>> import torchaudio
    >>> example_file = 'tests/samples/multi-mic/speech_-0.82918_0.55279_-0.082918.flac'
    >>> signal, sr = torchaudio.load(example_file, channels_first = False)
    >>> normalizer = AudioNormalizer(sample_rate=8000)
    >>> normalized = normalizer(signal, sr)
    >>> signal.shape
    torch.Size([160000, 4])
    >>> normalized.shape
    torch.Size([80000])

    NOTE
    ----
    This will also upsample audio. However, upsampling cannot produce meaningful
    information in the bandwidth which it adds. Generally models will not work
    well for upsampled data if they have not specifically been trained to do so.
    """

    def __init__(self, sample_rate=16000, mix="avg-to-mono"):
        self.sample_rate = sample_rate
        if mix not in ["avg-to-mono", "keep"]:
            raise ValueError(f"Unexpected mixing configuration {mix}")
        self.mix = mix
        self._cached_resamplers = {}

    def __call__(self, audio, sample_rate):
        """Perform normalization
        Arguments
        ---------
        audio : tensor
            The input waveform torch tensor. Assuming [time, channels],
            or [time].
        """
        if sample_rate not in self._cached_resamplers:
            # Create a Resample instance from this newly seen SR to internal SR
            self._cached_resamplers[sample_rate] = Resample(
                sample_rate, self.sample_rate
            )
        resampler = self._cached_resamplers[sample_rate]
        resampled = resampler(audio.unsqueeze(0)).squeeze(0)
        return self._mix(resampled)

    def _mix(self, audio):
        """Handle channel mixing"""
        flat_input = audio.dim() == 1
        if self.mix == "avg-to-mono":
            if flat_input:
                return audio
            return torch.mean(audio, 1)
        if self.mix == "keep":
            return audio
