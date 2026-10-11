"""
Codec Augmentation via soundfile

This library provides codec augmentation techniques for enhanced audio data
processing. Waveforms are encoded to an in-memory buffer with an audio codec
and decoded back, simulating the artefacts introduced by lossy compression.

Authors
 * Mirco Ravanelli 2023
 * Peter Plantinga 2026
"""

import io
import random

import soundfile as sf
import torch


def _match_length(waveform, length):
    """Pad or trim a waveform so that its last dimension equals ``length``."""
    if waveform.shape[-1] < length:
        waveform = torch.nn.functional.pad(
            waveform, (0, length - waveform.shape[-1])
        )
    return waveform[..., :length]


class CodecAugment(torch.nn.Module):
    """
    Apply random audio codecs to input waveforms using soundfile.

    This class provides an interface for applying codec augmentation techniques to audio data.

    Arguments
    ---------
    sample_rate: int
        The sample rate of the input waveform.

    Example
    -------
    >>> waveform = torch.rand(4, 16000)
    >>> augmenter = CodecAugment(16000)
    >>> output_waveform = augmenter(waveform)
    >>> output_waveform.shape
    torch.Size([4, 16000])
    """

    def __init__(self, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.available_format_encoders = [
            ("WAV", "ULAW"),
            ("MP3", "MPEG_LAYER_III"),
            ("OGG", "OPUS"),
        ]

    def _encode_decode(self, waveform, format=None, encoder=None):
        """Round-trip a single mono waveform through the given codec."""
        buffer = io.BytesIO()
        sf.write(
            buffer,
            waveform.numpy(),
            self.sample_rate,
            format=format,
            subtype=encoder,
        )
        buffer.seek(0)
        decoded, _ = sf.read(buffer, dtype="float32")
        return torch.from_numpy(decoded)

    def apply_codec(self, waveform, format=None, encoder=None):
        """
        Apply the selected audio codec.

        Arguments
        ----------
        waveform: torch.Tensor
            Input waveform of shape `[batch, time]`.
        format: str
            The soundfile format to use (e.g., "WAV", "MP3"). Default is None.
        encoder: str
            The soundfile subtype to use for the format (e.g., "ULAW", "OPUS").
            Default is None, letting soundfile pick the default subtype.

        Returns
        ---------
        torch.Tensor:
            Coded version of the input waveform of shape `[batch, time]`.
        """
        device = waveform.device
        waveform = waveform.detach().cpu().float()
        squeeze = waveform.dim() == 1
        if squeeze:
            waveform = waveform.unsqueeze(0)
        length = waveform.shape[-1]

        # Encode each batch element as a mono stream: some codecs (e.g. MP3)
        # only support one or two channels per file.
        coded = [
            _match_length(self._encode_decode(sample, format, encoder), length)
            for sample in waveform
        ]
        waveform_aug = torch.stack(coded)
        if squeeze:
            waveform_aug = waveform_aug.squeeze(0)
        return waveform_aug.to(device)

    def forward(self, waveform):
        """
        Apply a random audio codec from the available list.

        Arguments
        ---------
        waveform: torch.Tensor
            Input waveform of shape `[batch, time]`.

        Returns
        -------
        torch.Tensor
            Coded version of the input waveform of shape `[batch, time]`.
        """
        format, encoder = random.choice(self.available_format_encoders)
        return self.apply_codec(waveform, format=format, encoder=encoder)
