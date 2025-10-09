"""Specifies the inference interfaces for speech enhancement modules.

Authors:
 * Aku Rouhe 2021
 * Peter Plantinga 2021
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
 * Titouan Parcollet 2021
 * Abdel Heba 2021
 * Andreas Nautsch 2022, 2023
 * Pooneh Mousavi 2023
 * Sylvain de Langen 2023
 * Adel Moumen 2023
 * Pradnya Kandarkar 2023
"""

import torch
import torchaudio

from speechbrain.inference.interfaces import Pretrained
from speechbrain.utils.callchains import lengths_arg_exists


class SpectralMaskEnhancement(Pretrained):
    """A ready-to-use model for speech enhancement.

    Arguments
    ---------
    See ``Pretrained``.

    Example
    -------
    >>> import torch
    >>> from speechbrain.inference.enhancement import SpectralMaskEnhancement
    >>> # Model is downloaded from the speechbrain HuggingFace repo
    >>> tmpdir = getfixture("tmpdir")
    >>> enhancer = SpectralMaskEnhancement.from_hparams(
    ...     source="speechbrain/metricgan-plus-voicebank",
    ...     savedir=tmpdir,
    ... )
    >>> enhanced = enhancer.enhance_file(
    ...     "speechbrain/metricgan-plus-voicebank/example.wav"
    ... )
    """

    HPARAMS_NEEDED = ["compute_stft", "spectral_magnitude", "resynth"]
    MODULES_NEEDED = ["enhance_model"]

    def compute_features(self, wavs):
        """Compute the log spectral magnitude features for masking.

        Arguments
        ---------
        wavs : torch.Tensor
            A batch of waveforms to convert to log spectral mags.

        Returns
        -------
        feats : torch.Tensor
            The log spectral magnitude features.
        """
        feats = self.hparams.compute_stft(wavs)
        feats = self.hparams.spectral_magnitude(feats)
        return torch.log1p(feats)

    def enhance_batch(self, noisy, lengths=None):
        """Enhance a batch of noisy waveforms.

        Arguments
        ---------
        noisy : torch.Tensor
            A batch of waveforms to perform enhancement on.
        lengths : torch.Tensor
            The lengths of the waveforms if the enhancement model handles them.

        Returns
        -------
        wavs : torch.Tensor
            A batch of enhanced waveforms of the same shape as input.
        """
        noisy = noisy.to(self.device)
        noisy_features = self.compute_features(noisy)

        # Perform masking-based enhancement, multiplying output with input.
        if lengths is not None:
            mask = self.mods.enhance_model(noisy_features, lengths=lengths)
        else:
            mask = self.mods.enhance_model(noisy_features)
        enhanced = torch.mul(mask, noisy_features)

        # Return resynthesized waveforms
        return self.hparams.resynth(torch.expm1(enhanced), noisy)

    def enhance_file(self, filename, output_filename=None, **kwargs):
        """Enhance a wav file.

        Arguments
        ---------
        filename : str
            Location on disk to load file for enhancement.
        output_filename : str
            If provided, writes enhanced data to this file.
        **kwargs : dict
            Arguments forwarded to ``load_audio``.

        Returns
        -------
        wav : torch.Tensor
            The enhanced waveform.
        """
        noisy = self.load_audio(filename, **kwargs)
        noisy = noisy.to(self.device)

        # Fake a batch:
        batch = noisy.unsqueeze(0)
        if lengths_arg_exists(self.enhance_batch):
            enhanced = self.enhance_batch(batch, lengths=torch.tensor([1.0]))
        else:
            enhanced = self.enhance_batch(batch)

        if output_filename is not None:
            torchaudio.save(
                uri=output_filename,
                src=enhanced,
                sample_rate=self.hparams.compute_stft.sample_rate,
            )

        return enhanced.squeeze(0)


class WaveformEnhancement(Pretrained):
    """A ready-to-use model for speech enhancement.

    Arguments
    ---------
    See ``Pretrained``.

    Example
    -------
    >>> from speechbrain.inference.enhancement import WaveformEnhancement
    >>> # Model is downloaded from the speechbrain HuggingFace repo
    >>> tmpdir = getfixture("tmpdir")
    >>> enhancer = WaveformEnhancement.from_hparams(
    ...     source="speechbrain/mtl-mimic-voicebank",
    ...     savedir=tmpdir,
    ... )
    >>> enhanced = enhancer.enhance_file(
    ...     "speechbrain/mtl-mimic-voicebank/example.wav"
    ... )
    """

    MODULES_NEEDED = ["enhance_model"]

    def enhance_batch(self, noisy, lengths=None):
        """Enhance a batch of noisy waveforms.

        Arguments
        ---------
        noisy : torch.Tensor
            A batch of waveforms to perform enhancement on.
        lengths : torch.Tensor
            The lengths of the waveforms if the enhancement model handles them.

        Returns
        -------
        torch.Tensor
            A batch of enhanced waveforms of the same shape as input.
        """
        noisy = noisy.to(self.device)
        enhanced_wav, _ = self.mods.enhance_model(noisy)
        return enhanced_wav

    def enhance_file(self, filename, output_filename=None, **kwargs):
        """Enhance a wav file.

        Arguments
        ---------
        filename : str
            Location on disk to load file for enhancement.
        output_filename : str
            If provided, writes enhanced data to this file.
        **kwargs : dict
            Arguments forwarded to ``load_audio``

        Returns
        -------
        enhanced : torch.Tensor
            The enhanced waveform.
        """
        noisy = self.load_audio(filename, **kwargs)

        # Fake a batch:
        batch = noisy.unsqueeze(0)
        enhanced = self.enhance_batch(batch)

        if output_filename is not None:
            torchaudio.save(
                uri=output_filename,
                src=enhanced,
                sample_rate=self.audio_normalizer.sample_rate,
            )

        return enhanced.squeeze(0)

    def forward(self, noisy, lengths=None):
        """Runs enhancement on the noisy input"""
        return self.enhance_batch(noisy, lengths)
