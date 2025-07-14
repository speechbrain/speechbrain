"""Specifies the inference interfaces for interpretability modules.

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
import torch.nn.functional as F
import torchaudio

import speechbrain
from speechbrain.inference.interfaces import Pretrained
from speechbrain.processing.NMF import spectral_phase
from speechbrain.utils.data_utils import split_path
from speechbrain.utils.fetching import LocalStrategy, fetch


class PIQAudioInterpreter(Pretrained):
    """
    This class implements the interface for the PIQ posthoc interpreter for an audio classifier.

    Arguments
    ---------
    *args : tuple
    **kwargs : dict
        Arguments are forwarded to ``Pretrained`` parent class.

    Example
    -------
    >>> from speechbrain.inference.interpretability import PIQAudioInterpreter
    >>> tmpdir = getfixture("tmpdir")
    >>> interpreter = PIQAudioInterpreter.from_hparams(
    ...     source="speechbrain/PIQ-ESC50",
    ...     savedir=tmpdir,
    ... )
    >>> signal = torch.randn(1, 16000)
    >>> interpretation, _ = interpreter.interpret_batch(signal)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(self, wavs):
        """Pre-process wavs to calculate STFTs"""
        X_stft = self.mods.compute_stft(wavs)
        X_stft_power = speechbrain.processing.features.spectral_magnitude(
            X_stft, power=self.hparams.spec_mag_power
        )
        X_stft_logpower = torch.log1p(X_stft_power)

        return X_stft_logpower, X_stft, X_stft_power

    def classifier_forward(self, X_stft_logpower):
        """the forward pass for the classifier"""
        hcat = self.mods.embedding_model(X_stft_logpower)
        embeddings = hcat.mean((-1, -2))
        predictions = self.mods.classifier(embeddings).squeeze(1)
        class_pred = predictions.argmax(1)
        return hcat, embeddings, predictions, class_pred

    def invert_stft_with_phase(self, X_int, X_stft_phase):
        """Inverts STFT spectra given phase."""
        X_stft_phase_sb = torch.cat(
            (
                torch.cos(X_stft_phase).unsqueeze(-1),
                torch.sin(X_stft_phase).unsqueeze(-1),
            ),
            dim=-1,
        )

        X_stft_phase_sb = X_stft_phase_sb[:, : X_int.shape[1], :, :]
        if X_int.ndim == 3:
            X_int = X_int.unsqueeze(-1)
        X_wpsb = X_int * X_stft_phase_sb
        x_int_sb = self.mods.compute_istft(X_wpsb)
        return x_int_sb

    def interpret_batch(self, wavs):
        """Classifies the given audio into the given set of labels.
        It also provides the interpretation in the audio domain.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model. Make sure the sample rate is fs=16000 Hz.

        Returns
        -------
        x_int_sound_domain : torch.Tensor
            The interpretation in the waveform domain
        text_lab : str
            The text label for the classification
        """
        wavs = wavs.to(self.device)
        X_stft_logpower, X_stft, X_stft_power = self.preprocess(wavs)
        X_stft_phase = spectral_phase(X_stft)

        # Embeddings + sound classifier
        hcat, embeddings, predictions, class_pred = self.classifier_forward(
            X_stft_logpower
        )

        if self.hparams.use_vq:
            xhat, hcat, z_q_x = self.mods.psi(hcat, class_pred)
        else:
            xhat = self.mods.psi.decoder(hcat)
        xhat = xhat.squeeze(1)
        Tmax = xhat.shape[1]
        if self.hparams.use_mask_output:
            xhat = F.sigmoid(xhat)
            X_int = xhat * X_stft_logpower[:, :Tmax, :]
        else:
            xhat = F.softplus(xhat)
            th = xhat.max() * self.hparams.mask_th
            X_int = (xhat > th) * X_stft_logpower[:, :Tmax, :]
        X_int = torch.expm1(X_int)
        x_int_sound_domain = self.invert_stft_with_phase(X_int, X_stft_phase)
        text_lab = self.hparams.label_encoder.decode_torch(
            class_pred.unsqueeze(0)
        )

        return x_int_sound_domain, text_lab

    def interpret_file(self, path, savedir=None):
        """Classifies the given audiofile into the given set of labels.
        It also provides the interpretation in the audio domain.

        Arguments
        ---------
        path : str
            Path to audio file to classify.
        savedir : str
            Path to cache directory.

        Returns
        -------
        x_int_sound_domain : torch.Tensor
            The interpretation in the waveform domain
        text_lab : str
            The text label for the classification
        fs_model : int
            The sampling frequency of the model. Useful to save the audio.
        """
        source, fl = split_path(path)
        path = fetch(
            fl,
            source=source,
            savedir=savedir,
            local_strategy=LocalStrategy.SYMLINK,
        )

        batch, fs_file = torchaudio.load(path)
        batch = batch.to(self.device)
        fs_model = self.hparams.sample_rate

        # resample the data if needed
        if fs_file != fs_model:
            print(f"Resampling the audio from {fs_file} Hz to {fs_model} Hz")
            tf = torchaudio.transforms.Resample(
                orig_freq=fs_file, new_freq=fs_model
            ).to(self.device)
            batch = batch.mean(dim=0, keepdim=True)
            batch = tf(batch)

        x_int_sound_domain, text_lab = self.interpret_batch(batch)
        return x_int_sound_domain, text_lab, fs_model

    def forward(self, wavs, wav_lens=None):
        """Runs the classification"""
        return self.interpret_batch(wavs, wav_lens)
