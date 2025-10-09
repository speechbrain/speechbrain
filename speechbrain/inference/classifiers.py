"""Specifies the inference interfaces for Audio Classification modules.

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

import speechbrain
from speechbrain.inference.interfaces import Pretrained
from speechbrain.utils.data_utils import split_path
from speechbrain.utils.fetching import LocalStrategy, fetch


class EncoderClassifier(Pretrained):
    """A ready-to-use class for utterance-level classification (e.g, speaker-id,
    language-id, emotion recognition, keyword spotting, etc).

    The class assumes that an encoder called "embedding_model" and a model
    called "classifier" are defined in the yaml file. If you want to
    convert the predicted index into a corresponding text label, please
    provide the path of the label_encoder in a variable called 'lab_encoder_file'
    within the yaml.

    The class can be used either to run only the encoder (encode_batch()) to
    extract embeddings or to run a classification step (classify_batch()).

    Arguments
    ---------
    See ``Pretrained``

    Example
    -------
    >>> import torchaudio
    >>> from speechbrain.inference.classifiers import EncoderClassifier
    >>> # Model is downloaded from the speechbrain HuggingFace repo
    >>> tmpdir = getfixture("tmpdir")
    >>> classifier = EncoderClassifier.from_hparams(
    ...     source="speechbrain/spkrec-ecapa-voxceleb",
    ...     savedir=tmpdir,
    ... )
    >>> classifier.hparams.label_encoder.ignore_len()

    >>> # Compute embeddings
    >>> signal, fs = torchaudio.load("tests/samples/single-mic/example1.wav")
    >>> embeddings = classifier.encode_batch(signal)

    >>> # Classification
    >>> prediction = classifier.classify_batch(signal)
    """

    MODULES_NEEDED = [
        "compute_features",
        "mean_var_norm",
        "embedding_model",
        "classifier",
    ]

    def encode_batch(self, wavs, wav_lens=None, normalize=False):
        """Encodes the input audio into a single vector embedding.

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = <this>.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model. Make sure the sample rate is fs=16000 Hz.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.
        normalize : bool
            If True, it normalizes the embeddings with the statistics
            contained in mean_var_norm_emb.

        Returns
        -------
        torch.Tensor
            The encoded batch
        """
        # Manage single waveforms in input
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)

        # Storing waveform in the specified device
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = wavs.float()

        # Computing features and embeddings
        feats = self.mods.compute_features(wavs)
        feats = self.mods.mean_var_norm(feats, wav_lens)
        embeddings = self.mods.embedding_model(feats, wav_lens)
        if normalize:
            embeddings = self.hparams.mean_var_norm_emb(
                embeddings, torch.ones(embeddings.shape[0], device=self.device)
            )
        return embeddings

    def classify_batch(self, wavs, wav_lens=None):
        """Performs classification on the top of the encoded features.

        It returns the posterior probabilities, the index and, if the label
        encoder is specified it also the text label.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model. Make sure the sample rate is fs=16000 Hz.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        out_prob
            The log posterior probabilities of each class ([batch, N_class])
        score:
            It is the value of the log-posterior for the best class ([batch,])
        index
            The indexes of the best class ([batch,])
        text_lab:
            List with the text labels corresponding to the indexes.
            (label encoder should be provided).
        """
        emb = self.encode_batch(wavs, wav_lens)
        out_prob = self.mods.classifier(emb).squeeze(1)
        score, index = torch.max(out_prob, dim=-1)
        text_lab = self.hparams.label_encoder.decode_torch(index)
        return out_prob, score, index, text_lab

    def classify_file(self, path, **kwargs):
        """Classifies the given audiofile into the given set of labels.

        Arguments
        ---------
        path : str
            Path to audio file to classify.
        **kwargs : dict
            Arguments forwarded to ``load_audio``.

        Returns
        -------
        out_prob : torch.Tensor
            The log posterior probabilities of each class ([batch, N_class])
        score : torch.Tensor
            It is the value of the log-posterior for the best class ([batch,])
        index : torch.Tensor
            The indexes of the best class ([batch,])
        text_lab : list of str
            List with the text labels corresponding to the indexes.
            (label encoder should be provided).
        """
        waveform = self.load_audio(path, **kwargs)
        # Fake a batch:
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        emb = self.encode_batch(batch, rel_length)
        out_prob = self.mods.classifier(emb).squeeze(1)
        score, index = torch.max(out_prob, dim=-1)
        text_lab = self.hparams.label_encoder.decode_torch(index)
        return out_prob, score, index, text_lab

    def forward(self, wavs, wav_lens=None):
        """Runs the classification"""
        return self.classify_batch(wavs, wav_lens)


class AudioClassifier(Pretrained):
    """A ready-to-use class for utterance-level classification (e.g, speaker-id,
    language-id, emotion recognition, keyword spotting, etc).

    The class assumes that an encoder called "embedding_model" and a model
    called "classifier" are defined in the yaml file. If you want to
    convert the predicted index into a corresponding text label, please
    provide the path of the label_encoder in a variable called 'lab_encoder_file'
    within the yaml.

    The class can be used either to run only the encoder (encode_batch()) to
    extract embeddings or to run a classification step (classify_batch()).

    Arguments
    ---------
    See ``Pretrained``.

    Example
    -------
    >>> import torchaudio
    >>> from speechbrain.inference.classifiers import AudioClassifier
    >>> tmpdir = getfixture("tmpdir")
    >>> classifier = AudioClassifier.from_hparams(
    ...     source="speechbrain/cnn14-esc50",
    ...     savedir=tmpdir,
    ... )
    >>> signal = torch.randn(1, 16000)
    >>> prediction, _, _, text_lab = classifier.classify_batch(signal)
    >>> print(prediction.shape)
    torch.Size([1, 1, 50])
    """

    def classify_batch(self, wavs, wav_lens=None):
        """Performs classification on the top of the encoded features.

        It returns the posterior probabilities, the index and, if the label
        encoder is specified it also the text label.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model. Make sure the sample rate is fs=16000 Hz.
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        out_prob : torch.Tensor
            The log posterior probabilities of each class ([batch, N_class])
        score : torch.Tensor
            It is the value of the log-posterior for the best class ([batch,])
        index : torch.Tensor
            The indexes of the best class ([batch,])
        text_lab : list of str
            List with the text labels corresponding to the indexes.
            (label encoder should be provided).
        """
        wavs = wavs.to(self.device)
        X_stft = self.mods.compute_stft(wavs)
        X_stft_power = speechbrain.processing.features.spectral_magnitude(
            X_stft, power=self.hparams.spec_mag_power
        )

        if self.hparams.use_melspectra:
            net_input = self.mods.compute_fbank(X_stft_power)
        else:
            net_input = torch.log1p(X_stft_power)

        # Embeddings + sound classifier
        embeddings = self.mods.embedding_model(net_input)
        if embeddings.ndim == 4:
            embeddings = embeddings.mean((-1, -2))

        out_probs = self.mods.classifier(embeddings)
        score, index = torch.max(out_probs, dim=-1)
        text_lab = self.hparams.label_encoder.decode_torch(index)
        return out_probs, score, index, text_lab

    def classify_file(self, path, savedir=None):
        """Classifies the given audiofile into the given set of labels.

        Arguments
        ---------
        path : str
            Path to audio file to classify.
        savedir : str
            Path to folder for caching downloads.

        Returns
        -------
        out_prob
            The log posterior probabilities of each class ([batch, N_class])
        score:
            It is the value of the log-posterior for the best class ([batch,])
        index
            The indexes of the best class ([batch,])
        text_lab:
            List with the text labels corresponding to the indexes.
            (label encoder should be provided).
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

        out_probs, score, index, text_lab = self.classify_batch(batch)
        return out_probs, score, index, text_lab

    def forward(self, wavs, wav_lens=None):
        """Runs the classification"""
        return self.classify_batch(wavs, wav_lens)
