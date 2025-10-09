"""Specifies the inference interfaces for speech and audio encoders.

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

from speechbrain.inference.interfaces import Pretrained


class WaveformEncoder(Pretrained):
    """A ready-to-use waveformEncoder model

    It can be used to wrap different embedding models such as SSL ones (wav2vec2)
    or speaker ones (Xvector) etc. Two functions are available: encode_batch and
    encode_file. They can be used to obtain the embeddings directly from an audio
    file or from a batch of audio tensors respectively.

    The given YAML must contain the fields specified in the *_NEEDED[] lists.

    Arguments
    ---------
    See ``Pretrained``

    Example
    -------
    >>> from speechbrain.inference.encoders import WaveformEncoder
    >>> tmpdir = getfixture("tmpdir")
    >>> ssl_model = WaveformEncoder.from_hparams(
    ...     source="speechbrain/ssl-wav2vec2-base-libri",
    ...     savedir=tmpdir,
    ... )  # doctest: +SKIP
    >>> ssl_model.encode_file(
    ...     "samples/audio_samples/example_fr.wav"
    ... )  # doctest: +SKIP
    """

    MODULES_NEEDED = ["encoder"]

    def encode_file(self, path, **kwargs):
        """Encode the given audiofile into a sequence of embeddings.

        Arguments
        ---------
        path : str
            Path to audio file which to encode.
        **kwargs : dict
            Arguments forwarded to ``load_audio``

        Returns
        -------
        torch.Tensor
            The audiofile embeddings produced by this system.
        """
        waveform = self.load_audio(path, **kwargs)
        # Fake a batch:
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        results = self.encode_batch(batch, rel_length)
        return results["embeddings"]

    def encode_batch(self, wavs, wav_lens):
        """Encodes the input audio into a sequence of hidden states

        The waveforms should already be in the model's desired format.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        torch.Tensor
            The encoded batch
        """
        wavs = wavs.float()
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        encoder_out = self.mods.encoder(wavs, wav_lens)
        return encoder_out

    def forward(self, wavs, wav_lens):
        """Runs the encoder"""
        return self.encode_batch(wavs, wav_lens)


class MelSpectrogramEncoder(Pretrained):
    """A MelSpectrogramEncoder class created for the Zero-Shot Multi-Speaker TTS models.

    This is for speaker encoder models using the PyTorch MelSpectrogram transform for compatibility with the
    current TTS pipeline.

    This class can be used to encode a single waveform, a single mel-spectrogram, or a batch of mel-spectrograms.

    Arguments
    ---------
    See ``Pretrained``

    Example
    -------
    >>> import torchaudio
    >>> from speechbrain.inference.encoders import MelSpectrogramEncoder
    >>> # Model is downloaded from the speechbrain HuggingFace repo
    >>> tmpdir = getfixture("tmpdir")
    >>> encoder = MelSpectrogramEncoder.from_hparams(
    ...     source="speechbrain/tts-ecapa-voxceleb",
    ...     savedir=tmpdir,
    ... )  # doctest: +SKIP

    >>> # Compute embedding from a waveform (sample_rate must match the sample rate of the encoder)
    >>> signal, fs = torchaudio.load(
    ...     "tests/samples/single-mic/example1.wav"
    ... )  # doctest: +SKIP
    >>> spk_emb = encoder.encode_waveform(signal)  # doctest: +SKIP

    >>> # Compute embedding from a mel-spectrogram (sample_rate must match the sample rate of the ecoder)
    >>> mel_spec = encoder.mel_spectogram(audio=signal)  # doctest: +SKIP
    >>> spk_emb = encoder.encode_mel_spectrogram(mel_spec)  # doctest: +SKIP

    >>> # Compute embeddings for a batch of mel-spectrograms
    >>> spk_embs = encoder.encode_mel_spectrogram_batch(
    ...     mel_spec
    ... )  # doctest: +SKIP
    """

    MODULES_NEEDED = ["normalizer", "embedding_model"]

    def dynamic_range_compression(self, x, C=1, clip_val=1e-5):
        """Dynamic range compression for audio signals"""
        return torch.log(torch.clamp(x, min=clip_val) * C)

    def mel_spectogram(self, audio):
        """calculates MelSpectrogram for a raw audio signal

        Arguments
        ---------
        audio : torch.tensor
            input audio signal

        Returns
        -------
        mel : torch.Tensor
            Mel-spectrogram
        """
        from torchaudio import transforms

        audio_to_mel = transforms.MelSpectrogram(
            sample_rate=self.hparams.sample_rate,
            hop_length=self.hparams.hop_length,
            win_length=self.hparams.win_length,
            n_fft=self.hparams.n_fft,
            n_mels=self.hparams.n_mel_channels,
            f_min=self.hparams.mel_fmin,
            f_max=self.hparams.mel_fmax,
            power=self.hparams.power,
            normalized=self.hparams.mel_normalized,
            norm=self.hparams.norm,
            mel_scale=self.hparams.mel_scale,
        ).to(audio.device)

        mel = audio_to_mel(audio)

        if self.hparams.dynamic_range_compression:
            mel = self.dynamic_range_compression(mel)

        return mel

    def encode_waveform(self, wav):
        """
        Encodes a single waveform

        Arguments
        ---------

        wav : torch.Tensor
            waveform

        Returns
        -------
        encoder_out : torch.Tensor
            Speaker embedding for the input waveform
        """

        # Moves tensor to the appropriate device
        wav = wav.to(self.device)

        # Computes mel-spectrogram
        mel_spec = self.mel_spectogram(audio=wav)

        # Calls encode_mel_spectrogram to compute the speaker embedding
        return self.encode_mel_spectrogram(mel_spec)

    def encode_mel_spectrogram(self, mel_spec):
        """
        Encodes a single mel-spectrograms

        Arguments
        ---------

        mel_spec : torch.Tensor
            Mel-spectrograms

        Returns
        -------
        encoder_out : torch.Tensor
            Speaker embedding for the input mel-spectrogram
        """

        # Fakes a batch
        batch = mel_spec
        if len(mel_spec.shape) == 2:
            batch = mel_spec.unsqueeze(0)
        rel_length = torch.tensor([1.0])

        # Calls encode_mel_spectrogram_batch to compute speaker embeddings
        results = self.encode_mel_spectrogram_batch(batch, rel_length)

        return results

    def encode_mel_spectrogram_batch(self, mel_specs, lens=None):
        """
        Encodes a batch of mel-spectrograms

        Arguments
        ---------

        mel_specs : torch.Tensor
            Mel-spectrograms
        lens : torch.Tensor
            Relative lengths of the mel-spectrograms

        Returns
        -------
        encoder_out : torch.Tensor
            Speaker embedding for the input mel-spectrogram batch
        """

        # Assigns full length if lens is not assigned
        if lens is None:
            lens = torch.ones(mel_specs.shape[0], device=self.device)

        # Moves the tensors to the appropriate device
        mel_specs, lens = mel_specs.to(self.device), lens.to(self.device)

        # Computes speaker embeddings
        mel_specs = torch.transpose(mel_specs, 1, 2)
        feats = self.hparams.normalizer(mel_specs, lens)
        encoder_out = self.hparams.embedding_model(feats)

        return encoder_out

    def __forward(self, mel_specs, lens):
        """Runs the encoder"""
        return self.encode_batch(mel_specs, lens)
