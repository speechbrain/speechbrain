"""Adapters for specific TTS system

Authors
* Artem Ploujnikov, 2024
"""

from torch import nn


class MelAdapter(nn.Module):
    """An adapter for TTSes that output a MEL spectrogram
    and require a vocoder to synthesize an
    audio wave

    Arguments
    ---------
    vocoder : torch.nn.Module | speechbrain.inference.Pretrained
        the vocoder to be used
    vocoder_run_opts : dict
        run options for the vocoder
    """

    def __init__(self, vocoder, vocoder_run_opts=None):
        super().__init__()
        self.vocoder_fn = vocoder
        self.vocoder_run_opts = vocoder_run_opts or {}
        self.vocoder = None
        self.device = None

    def _get_vocoder(self):
        """Instantiates the vocoder, if not already instantiated"""
        if self.vocoder is None:
            run_opts = dict(self.vocoder_run_opts)
            if self.device is not None:
                run_opts["device"] = self.device
            self.vocoder = self.vocoder_fn(run_opts=run_opts)
        return self.vocoder

    def forward(self, tts_out):
        """Applies a vocoder to the waveform

        Arguments
        ---------
        tts_out : tuple
            a (tensor, tensor) tuple with a MEL spectrogram
            of shape (batch x mel x length)
            and absolute lengths (as in the output of Tacotron2
            or similar models)

        Returns
        -------
        wav : torch.Tensor
            The waveform
        lengths : torch.Tensor
            The lengths
        """
        mel_outputs, mel_lengths, _ = tts_out
        vocoder = self._get_vocoder()
        max_len = mel_lengths.max()
        mel_outputs = mel_outputs[:, :, :max_len]
        wav = vocoder(mel_outputs)
        lengths = mel_lengths / max_len
        return wav, lengths

    def to(self, device):
        """Transfers the adapter (and the underlying model) to the
        specified device

        Arguments
        ---------
        device : str | torch.Device
            The device

        Returns
        -------
        result : MelAdapter
            the adapter (i.e. returns itself)
        """
        self.device = device
        return super().to(device)
