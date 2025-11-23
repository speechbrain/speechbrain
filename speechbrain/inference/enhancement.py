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
 * Jonas Rochdi 2025
"""

import torch

from speechbrain.dataio import audio_io
from speechbrain.inference.interfaces import Pretrained
from speechbrain.utils.callchains import lengths_arg_exists


def pad_spec(Y, mode="zero_pad"):
    """Pad tensor `Y` along axis 3 to 64 with the given algorithm."""
    T = Y.size(3)
    if T % 64 != 0:
        num_pad = 64 - T % 64
    else:
        num_pad = 0
    if mode == "zero_pad":
        pad2d = torch.nn.ZeroPad2d((0, num_pad, 0, 0))
    elif mode == "reflection":
        pad2d = torch.nn.ReflectionPad2d((0, num_pad, 0, 0))
    elif mode == "replication":
        pad2d = torch.nn.ReplicationPad2d((0, num_pad, 0, 0))
    else:
        raise NotImplementedError("This function hasn't been implemented yet.")
    return pad2d(Y)


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
            audio_io.save(
                path=output_filename,
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
            audio_io.save(
                path=output_filename,
                src=enhanced,
                sample_rate=self.audio_normalizer.sample_rate,
            )

        return enhanced.squeeze(0)

    def forward(self, noisy, lengths=None):
        """Runs enhancement on the noisy input"""
        return self.enhance_batch(noisy, lengths)


class SGMSEEnhancement(Pretrained):
    """Ready-to-use SGMSE speech enhancement.

    Arguments
    ---------
    See ``Pretrained``.

    Example
    -------
    >>> from speechbrain.inference.enhancement import SGMSEEnhancement
    >>> tmpdir = getfixture("tmpdir")
    >>> enh = SGMSEEnhancement.from_hparams(
    ...     source="speechbrain/sgmse-voicebank", savedir=tmpdir
    ... )  # doctest: +SKIP
    >>> out = enh.enhance_file(
    ...     "speechbrain/sgmse-voicebank/example.wav"
    ... )  # doctest: +SKIP
    """

    MODULES_NEEDED = ["score_model"]
    HPARAMS_NEEDED = [
        "sample_rate",
        "n_fft",
        "hop_length",
        "window_type",
        "transform_type",
        "spec_factor",
        "sampling",
    ]

    def _ensure_stft_setup(self):
        if getattr(self, "_stft_ready", False):
            return
        n_fft = self.hparams.n_fft
        self._window = self._get_window(self.hparams.window_type, n_fft).to(
            self.device
        )
        self._stft_kwargs = dict(
            n_fft=n_fft,
            hop_length=self.hparams.hop_length,
            center=True,
            return_complex=True,
        )
        self._stft_ready = True

    def enhance_batch(self, noisy, lengths=None):
        """Enhance a batch of noisy waveforms (B, T) → (B, T)."""
        self._ensure_stft_setup()

        noisy = noisy.to(self.device)
        # scale to [-1,1] by max abs per item (like the Brain inference)
        norms = torch.clamp(noisy.abs().amax(dim=1, keepdim=True), min=1e-8)
        y = noisy / norms

        # STFT + forward spec transform + channel dim
        Y = self._spec_fwd(self._stft(y)).unsqueeze(1)  # (B,1,F,T)
        F_orig, T_orig_spec = Y.shape[-2:]

        # pad for U-Net constraints
        Yp = pad_spec(Y, mode="reflection")

        # Call the SGMSE sampler on spectrograms
        smp = self.hparams.sampling
        x_hat = self.mods.score_model.enhance(
            Yp,
            sampler_type=smp.get("sampler_type", "pc"),
            predictor=smp.get("predictor", "reverse_diffusion"),
            corrector=smp.get("corrector", "ald"),
            N=smp.get("N", 30),
            corrector_steps=smp.get("corrector_steps", 1),
            snr=smp.get("snr", 0.5),
        )  # (B,1,F,T)

        # Trim padding, drop channel, inverse spec transform, iSTFT
        Xh = x_hat[:, :, :F_orig, :T_orig_spec].squeeze(1)  # (B,F,T)
        Xh = self._spec_back(Xh)
        enh = self._istft(Xh, length=y.size(1)) * norms  # (B,T)
        return enh

    def enhance_file(self, filename, output_filename=None, **kwargs):
        """Enhance a wav file; optionally write to disk."""
        noisy = self.load_audio(filename, **kwargs).to(self.device)
        enhanced = self.enhance_batch(noisy.unsqueeze(0)).squeeze(0)

        if output_filename is not None:
            audio_io.save(
                output_filename,
                src=enhanced.unsqueeze(0).cpu(),
                sample_rate=self.hparams.sample_rate,
            )
        return enhanced

    def forward(self, noisy, lengths=None):
        """Alias to enable nn.Module-style calls."""
        return self.enhance_batch(noisy, lengths)

    # HELPERS
    def _stft(self, sig):
        return torch.stft(sig, **{**self._stft_kwargs, "window": self._window})

    def _istft(self, spec, length=None):
        kw = dict(self._stft_kwargs)
        kw.pop("return_complex", None)
        kw["window"] = self._window
        kw["length"] = length
        return torch.istft(spec, **kw)

    def _spec_fwd(self, S):
        ttype = self.hparams.transform_type
        factor = self.hparams.spec_factor
        e = getattr(self.hparams, "spec_abs_exponent", 1.0)

        if ttype == "exponent":
            if e != 1.0:
                mag, ph = S.abs() ** e, S.angle()
                S = mag * torch.exp(1j * ph)
            S = S * factor
        elif ttype == "log":
            mag, ph = torch.log1p(S.abs()), S.angle()
            S = mag * torch.exp(1j * ph)
            S = S * factor
        return S

    def _spec_back(self, S):
        ttype = self.hparams.transform_type
        factor = self.hparams.spec_factor
        e = getattr(self.hparams, "spec_abs_exponent", 1.0)

        if ttype == "exponent":
            S = S / factor
            if e != 1.0:
                mag, ph = S.abs() ** (1.0 / e), S.angle()
                S = mag * torch.exp(1j * ph)
        elif ttype == "log":
            S = S / factor
            mag, ph = torch.expm1(S.abs()), S.angle()
            S = mag * torch.exp(1j * ph)
        return S

    def _get_window(self, window_type, n_fft):
        if window_type == "sqrthann":
            return torch.sqrt(torch.hann_window(n_fft, periodic=True))
        elif window_type == "hann":
            return torch.hann_window(n_fft, periodic=True)
        raise NotImplementedError(f"Window type {window_type} not implemented!")
