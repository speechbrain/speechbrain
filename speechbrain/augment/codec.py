"""
Codec Augmentation via torchaudio

This library provides codec augmentation techniques in torchaudio for enhanced
audio data processing.

For detailed guidance and usage examples, refer to the tutorial at:
https://pytorch.org/audio/stable/tutorials/audio_data_augmentation_tutorial.html

Note: This code is compatible with FFmpeg as the torchaudio backend.
When using FFmpeg2, the maximum number of samples for processing is limited to 16.

Authors
 * Mirco Ravanelli 2023
"""

import random
import torch
import torchaudio


class CodecAugment(torch.nn.Module):
    """
    Apply random audio codecs to input waveforms using torchaudio.

    This class provides an interface for applying codec augmentation techniques to audio data.

    Arguments
    ---------
    sample_rate: int
        The sample rate of the input waveform.

    Example
    -------
    >>> waveform = torch.rand(4, 16000)
    >>> if torchaudio.list_audio_backends()[0] == 'ffmpeg':
    ...     augmenter = CodecAugment(16000)
    ...     output_waveform = augmenter(waveform)
    """

    def __init__(self, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.available_format_encoders = [
            ("wav", "pcm_mulaw"),
            ("mp3", None),
            ("g722", None),
        ]

    def apply_codec(self, waveform, format=None, encoder=None):
        """
        Apply the selected audio codec.

        Arguments
        ----------
        waveform: torch.Tensor
            Input waveform of shape `[batch, time]`.
        format: str
            The audio format to use (e.g., "wav", "mp3"). Default is None.
        encoder: str
            The encoder to use for the format (e.g., "opus", "vorbis"). Default is None.

        Returns
        ---------
        torch.Tensor:
            Coded version of the input waveform of shape `[batch, time]`.
        """
        audio_effector = torchaudio.io.AudioEffector(
            format=format, encoder=encoder
        )
        waveform_aug = audio_effector.apply(
            waveform.transpose(0, 1).to("cpu"), self.sample_rate
        )
        return waveform_aug.transpose(0, 1).to(waveform.device)

    def forward(self, waveform):
        """
        Apply a random audio codec from the available list.

        Arguments
        ---------
        waveform: torch.Tensor
            Input waveform of shape `[batch, time]`.

        Returns
        ---------
        torch.Tensor
            Coded version of the input waveform of shape `[batch, time]`.
        """
        format, encoder = random.choice(self.available_format_encoders)
        return self.apply_codec(waveform, format=format, encoder=encoder)
