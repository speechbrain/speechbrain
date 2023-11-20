""" Specifies the inference interfaces for interpretability modules.

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
import torch.nn.functional as F
import speechbrain
from speechbrain.utils.fetching import fetch
from speechbrain.utils.data_utils import split_path
from speechbrain.processing.NMF import spectral_phase
from speechbrain.inference.interfaces import Pretrained


class PIQAudioInterpreter(Pretrained):
    """
    This class implements the interface for the PIQ posthoc interpreter for an audio classifier.

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
        x_int_sound_domain
            The interpretation in the waveform domain
        text_lab:
            The text label for the classification
        fs_model:
            The sampling frequency of the model. Useful to save the audio.
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

    def interpret_file(self, path, savedir="audio_cache"):
        """Classifies the given audiofile into the given set of labels.
        It also provides the interpretation in the audio domain.

        Arguments
        ---------
        path : str
            Path to audio file to classify.

        Returns
        -------
        x_int_sound_domain
            The interpretation in the waveform domain
        text_lab:
            The text label for the classification
        fs_model:
            The sampling frequency of the model. Useful to save the audio.
        """
        source, fl = split_path(path)
        path = fetch(fl, source=source, savedir=savedir)

        batch, fs_file = torchaudio.load(path)
        batch = batch.to(self.device)
        fs_model = self.hparams.sample_rate

        # resample the data if needed
        if fs_file != fs_model:
            print(
                "Resampling the audio from {} Hz to {} Hz".format(
                    fs_file, fs_model
                )
            )
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


class EncoderDecoderS2UT(Pretrained):
    """A ready-to-use Encoder Decoder for speech-to-unit translation model

    The class can be used  to  run the entire encoder-decoder S2UT model
    (translate_file()) to translate speech. The given YAML must contains the fields
    specified in the *_NEEDED[] lists.

    Example
    -------
    >>> from speechbrain.inference import EncoderDecoderS2UT
    >>> tmpdir = getfixture("tmpdir")
    >>> s2ut_model = EncoderDecoderS2UT.from_hparams(source="speechbrain/s2st-transformer-fr-en-hubert-l6-k100-cvss", savedir=tmpdir) # doctest: +SKIP
    >>> s2ut_model.translate_file("speechbrain/s2st-transformer-fr-en-hubert-l6-k100-cvss/example-fr.wav") # doctest: +SKIP
    """

    HPARAMS_NEEDED = ["sample_rate"]
    MODULES_NEEDED = ["encoder", "decoder"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_rate = self.hparams.sample_rate

    def translate_file(self, path):
        """Translates the given audiofile into a sequence speech unit.

        Arguments
        ---------
        path : str
            Path to audio file which to translate.

        Returns
        -------
        int[]
            The audiofile translation produced by this speech-to-unit translationmodel.
        """

        audio = self.load_audio(path)
        audio = audio.to(self.device)
        # Fake a batch:
        batch = audio.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        predicted_tokens = self.translate_batch(batch, rel_length)
        return predicted_tokens[0]

    def encode_batch(self, wavs, wav_lens):
        """Encodes the input audio into a sequence of hidden states

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = EncoderDecoderS2UT.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.tensor
            Batch of waveforms [batch, time, channels].
        wav_lens : torch.tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        torch.tensor
            The encoded batch
        """
        wavs = wavs.float()
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        encoder_out = self.mods.encoder(wavs, wav_lens)
        return encoder_out

    def translate_batch(self, wavs, wav_lens):
        """Translates the input audio into a sequence of words

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = EncoderDecoderS2UT.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.tensor
            Batch of waveforms [batch, time, channels].
        wav_lens : torch.tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        list
            Each waveform in the batch translated.
        tensor
            Each predicted token id.
        """
        with torch.no_grad():
            wav_lens = wav_lens.to(self.device)
            encoder_out = self.encode_batch(wavs, wav_lens)
            predicted_tokens, _ = self.mods.decoder(encoder_out, wav_lens)
        return predicted_tokens

    def forward(self, wavs, wav_lens):
        """Runs full translation"""
        return self.encode_batch(wavs, wav_lens)
