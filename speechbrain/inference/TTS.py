"""Specifies the inference interfaces for Text-To-Speech (TTS) modules.

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

import random
import re

import torch
import torchaudio

import speechbrain
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.inference.encoders import MelSpectrogramEncoder
from speechbrain.inference.interfaces import Pretrained
from speechbrain.inference.text import GraphemeToPhoneme
from speechbrain.utils.fetching import fetch
from speechbrain.utils.logger import get_logger
from speechbrain.utils.text_to_sequence import text_to_sequence

logger = get_logger(__name__)


class Tacotron2(Pretrained):
    """
    A ready-to-use wrapper for Tacotron2 (text -> mel_spec).

    Arguments
    ---------
    *args : tuple
    **kwargs : dict
        Arguments are forwarded to ``Pretrained`` parent class.

    Example
    -------
    >>> tmpdir_tts = getfixture("tmpdir") / "tts"
    >>> tacotron2 = Tacotron2.from_hparams(
    ...     source="speechbrain/tts-tacotron2-ljspeech", savedir=tmpdir_tts
    ... )
    >>> mel_output, mel_length, alignment = tacotron2.encode_text(
    ...     "Mary had a little lamb"
    ... )
    >>> items = [
    ...     "A quick brown fox jumped over the lazy dog",
    ...     "How much wood would a woodchuck chuck?",
    ...     "Never odd or even",
    ... ]
    >>> mel_outputs, mel_lengths, alignments = tacotron2.encode_batch(items)

    >>> # One can combine the TTS model with a vocoder (that generates the final waveform)
    >>> # Initialize the Vocoder (HiFIGAN)
    >>> tmpdir_vocoder = getfixture("tmpdir") / "vocoder"
    >>> from speechbrain.inference.vocoders import HIFIGAN
    >>> hifi_gan = HIFIGAN.from_hparams(
    ...     source="speechbrain/tts-hifigan-ljspeech", savedir=tmpdir_vocoder
    ... )
    >>> # Running the TTS
    >>> mel_output, mel_length, alignment = tacotron2.encode_text(
    ...     "Mary had a little lamb"
    ... )
    >>> # Running Vocoder (spectrogram-to-waveform)
    >>> waveforms = hifi_gan.decode_batch(mel_output)
    """

    HPARAMS_NEEDED = ["model", "text_to_sequence"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_cleaners = getattr(
            self.hparams, "text_cleaners", ["english_cleaners"]
        )
        self.infer = self.hparams.model.infer

    def text_to_seq(self, txt):
        """Encodes raw text into a tensor with a customer text-to-sequence function"""
        sequence = self.hparams.text_to_sequence(txt, self.text_cleaners)
        return sequence, len(sequence)

    def encode_batch(self, texts):
        """Computes mel-spectrogram for a list of texts

        Texts must be sorted in decreasing order on their lengths

        Arguments
        ---------
        texts: List[str]
            texts to be encoded into spectrogram

        Returns
        -------
        tensors of output spectrograms, output lengths and alignments
        """
        with torch.no_grad():
            inputs = [
                {
                    "text_sequences": torch.tensor(
                        self.text_to_seq(item)[0], device=self.device
                    )
                }
                for item in texts
            ]
            inputs = speechbrain.dataio.batch.PaddedBatch(inputs)

            lens = [self.text_to_seq(item)[1] for item in texts]
            assert lens == sorted(lens, reverse=True), (
                "input lengths must be sorted in decreasing order"
            )
            input_lengths = torch.tensor(lens, device=self.device)

            mel_outputs_postnet, mel_lengths, alignments = self.infer(
                inputs.text_sequences.data, input_lengths
            )
        return mel_outputs_postnet, mel_lengths, alignments

    def encode_text(self, text):
        """Runs inference for a single text str"""
        return self.encode_batch([text])

    def forward(self, texts):
        "Encodes the input texts."
        return self.encode_batch(texts)


class MSTacotron2(Pretrained):
    """
    A ready-to-use wrapper for Zero-Shot Multi-Speaker Tacotron2.
    For voice cloning: (text, reference_audio) -> (mel_spec).
    For generating a random speaker voice: (text) -> (mel_spec).

    Arguments
    ---------
    *args : tuple
    **kwargs : dict
        Arguments are forwarded to ``Pretrained`` parent class.

    Example
    -------
    >>> tmpdir_tts = getfixture("tmpdir") / "tts"
    >>> mstacotron2 = MSTacotron2.from_hparams(
    ...     source="speechbrain/tts-mstacotron2-libritts", savedir=tmpdir_tts
    ... )  # doctest: +SKIP
    >>> # Sample rate of the reference audio must be greater or equal to the sample rate of the speaker embedding model
    >>> reference_audio_path = "tests/samples/single-mic/example1.wav"
    >>> input_text = "Mary had a little lamb."
    >>> mel_output, mel_length, alignment = mstacotron2.clone_voice(
    ...     input_text, reference_audio_path
    ... )  # doctest: +SKIP
    >>> # One can combine the TTS model with a vocoder (that generates the final waveform)
    >>> # Initialize the Vocoder (HiFIGAN)
    >>> tmpdir_vocoder = getfixture("tmpdir") / "vocoder"
    >>> from speechbrain.inference.vocoders import HIFIGAN
    >>> hifi_gan = HIFIGAN.from_hparams(
    ...     source="speechbrain/tts-hifigan-libritts-22050Hz",
    ...     savedir=tmpdir_vocoder,
    ... )  # doctest: +SKIP
    >>> # Running the TTS
    >>> mel_output, mel_length, alignment = mstacotron2.clone_voice(
    ...     input_text, reference_audio_path
    ... )  # doctest: +SKIP
    >>> # Running Vocoder (spectrogram-to-waveform)
    >>> waveforms = hifi_gan.decode_batch(mel_output)  # doctest: +SKIP
    >>> # For generating a random speaker voice, use the following
    >>> mel_output, mel_length, alignment = mstacotron2.generate_random_voice(
    ...     input_text
    ... )  # doctest: +SKIP
    """

    HPARAMS_NEEDED = ["model"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_cleaners = ["english_cleaners"]
        self.infer = self.hparams.model.infer
        self.custom_mel_spec_encoder = self.hparams.custom_mel_spec_encoder

        self.g2p = GraphemeToPhoneme.from_hparams(
            self.hparams.g2p, run_opts={"device": self.device}
        )

        self.spk_emb_encoder = None
        if self.custom_mel_spec_encoder:
            self.spk_emb_encoder = MelSpectrogramEncoder.from_hparams(
                source=self.hparams.spk_emb_encoder,
                run_opts={"device": self.device},
            )
        else:
            self.spk_emb_encoder = EncoderClassifier.from_hparams(
                source=self.hparams.spk_emb_encoder,
                run_opts={"device": self.device},
            )

    def __text_to_seq(self, txt):
        """Encodes raw text into a tensor with a customer text-to-sequence function"""
        sequence = text_to_sequence(txt, self.text_cleaners)
        return sequence, len(sequence)

    def clone_voice(self, texts, audio_path):
        """
        Generates mel-spectrogram using input text and reference audio

        Arguments
        ---------
        texts : str or list
            Input text
        audio_path : str
            Reference audio

        Returns
        -------
        tensors of output spectrograms, output lengths and alignments
        """

        # Loads audio
        ref_signal, signal_sr = torchaudio.load(audio_path)

        # Resamples the audio if required
        if signal_sr != self.hparams.spk_emb_sample_rate:
            ref_signal = torchaudio.functional.resample(
                ref_signal, signal_sr, self.hparams.spk_emb_sample_rate
            )
        ref_signal = ref_signal.to(self.device)

        # Computes speaker embedding
        if self.custom_mel_spec_encoder:
            spk_emb = self.spk_emb_encoder.encode_waveform(ref_signal)
        else:
            spk_emb = self.spk_emb_encoder.encode_batch(ref_signal)

        spk_emb = spk_emb.squeeze(0)

        # Converts input texts into the corresponding phoneme sequences
        if isinstance(texts, str):
            texts = [texts]
        phoneme_seqs = self.g2p(texts)
        for i in range(len(phoneme_seqs)):
            phoneme_seqs[i] = " ".join(phoneme_seqs[i])
            phoneme_seqs[i] = "{" + phoneme_seqs[i] + "}"

        # Repeats the speaker embedding to match the number of input texts
        spk_embs = spk_emb.repeat(len(texts), 1)

        # Calls __encode_batch to generate the mel-spectrograms
        return self.__encode_batch(phoneme_seqs, spk_embs)

    def generate_random_voice(self, texts):
        """
        Generates mel-spectrogram using input text and a random speaker voice

        Arguments
        ---------
        texts : str or list
            Input text

        Returns
        -------
        tensors of output spectrograms, output lengths and alignments
        """

        spk_emb = self.__sample_random_speaker().float()
        spk_emb = spk_emb.to(self.device)

        # Converts input texts into the corresponding phoneme sequences
        if isinstance(texts, str):
            texts = [texts]
        phoneme_seqs = self.g2p(texts)
        for i in range(len(phoneme_seqs)):
            phoneme_seqs[i] = " ".join(phoneme_seqs[i])
            phoneme_seqs[i] = "{" + phoneme_seqs[i] + "}"

        # Repeats the speaker embedding to match the number of input texts
        spk_embs = spk_emb.repeat(len(texts), 1)

        # Calls __encode_batch to generate the mel-spectrograms
        return self.__encode_batch(phoneme_seqs, spk_embs)

    def __encode_batch(self, texts, spk_embs):
        """Computes mel-spectrograms for a list of texts
        Texts are sorted in decreasing order on their lengths

        Arguments
        ---------
        texts: List[str]
            texts to be encoded into spectrogram
        spk_embs: torch.Tensor
            speaker embeddings

        Returns
        -------
        tensors of output spectrograms, output lengths and alignments
        """

        with torch.no_grad():
            inputs = [
                {
                    "text_sequences": torch.tensor(
                        self.__text_to_seq(item)[0], device=self.device
                    )
                }
                for item in texts
            ]

            inputs = sorted(
                inputs,
                key=lambda x: x["text_sequences"].size()[0],
                reverse=True,
            )

            lens = [entry["text_sequences"].size()[0] for entry in inputs]

            inputs = speechbrain.dataio.batch.PaddedBatch(inputs)

            assert lens == sorted(lens, reverse=True), (
                "input lengths must be sorted in decreasing order"
            )
            input_lengths = torch.tensor(lens, device=self.device)

            mel_outputs_postnet, mel_lengths, alignments = self.infer(
                inputs.text_sequences.data, spk_embs, input_lengths
            )
        return mel_outputs_postnet, mel_lengths, alignments

    def __sample_random_speaker(self):
        """Samples a random speaker embedding from a pretrained GMM

        Returns
        -------
        x: torch.Tensor
            A randomly sampled speaker embedding
        """

        # Fetches and Loads GMM trained on speaker embeddings
        speaker_gmm_local_path = fetch(
            filename=self.hparams.random_speaker_sampler,
            source=self.hparams.random_speaker_sampler_source,
            savedir=self.hparams.pretrainer.collect_in,
        )
        random_speaker_gmm = torch.load(speaker_gmm_local_path)
        gmm_n_components = random_speaker_gmm["gmm_n_components"]
        gmm_means = random_speaker_gmm["gmm_means"]
        gmm_covariances = random_speaker_gmm["gmm_covariances"]

        # Randomly selects a speaker
        counts = torch.zeros(gmm_n_components)
        counts[random.randint(0, gmm_n_components - 1)] = 1
        x = torch.empty(0, device=counts.device)

        # Samples an embedding for the speaker
        for k in torch.arange(gmm_n_components)[counts > 0]:
            # Considers full covariance type
            d_k = torch.distributions.multivariate_normal.MultivariateNormal(
                gmm_means[k], gmm_covariances[k]
            )
            x_k = torch.stack([d_k.sample() for _ in range(int(counts[k]))])

            x = torch.cat((x, x_k), dim=0)

        return x


class FastSpeech2(Pretrained):
    """
    A ready-to-use wrapper for Fastspeech2 (text -> mel_spec).

    Arguments
    ---------
    *args : tuple
    **kwargs : dict
        Arguments are forwarded to ``Pretrained`` parent class.

    Example
    -------
    >>> tmpdir_tts = getfixture("tmpdir") / "tts"
    >>> fastspeech2 = FastSpeech2.from_hparams(
    ...     source="speechbrain/tts-fastspeech2-ljspeech", savedir=tmpdir_tts
    ... )  # doctest: +SKIP
    >>> mel_outputs, durations, pitch, energy = fastspeech2.encode_text(
    ...     ["Mary had a little lamb."]
    ... )  # doctest: +SKIP
    >>> items = [
    ...     "A quick brown fox jumped over the lazy dog",
    ...     "How much wood would a woodchuck chuck?",
    ...     "Never odd or even",
    ... ]
    >>> mel_outputs, durations, pitch, energy = fastspeech2.encode_text(
    ...     items
    ... )  # doctest: +SKIP
    >>>
    >>> # One can combine the TTS model with a vocoder (that generates the final waveform)
    >>> # Initialize the Vocoder (HiFIGAN)
    >>> tmpdir_vocoder = getfixture("tmpdir") / "vocoder"
    >>> from speechbrain.inference.vocoders import HIFIGAN
    >>> hifi_gan = HIFIGAN.from_hparams(
    ...     source="speechbrain/tts-hifigan-ljspeech", savedir=tmpdir_vocoder
    ... )  # doctest: +SKIP
    >>> # Running the TTS
    >>> mel_outputs, durations, pitch, energy = fastspeech2.encode_text(
    ...     ["Mary had a little lamb."]
    ... )  # doctest: +SKIP
    >>> # Running Vocoder (spectrogram-to-waveform)
    >>> waveforms = hifi_gan.decode_batch(mel_outputs)  # doctest: +SKIP
    """

    HPARAMS_NEEDED = ["spn_predictor", "model", "input_encoder"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lexicon = self.hparams.lexicon
        lexicon = ["@@"] + lexicon
        self.input_encoder = self.hparams.input_encoder
        self.input_encoder.update_from_iterable(lexicon, sequence_input=False)
        self.input_encoder.add_unk()

        self.g2p = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p")

        self.spn_token_encoded = (
            self.input_encoder.encode_sequence_torch(["spn"]).int().item()
        )

    def encode_text(self, texts, pace=1.0, pitch_rate=1.0, energy_rate=1.0):
        """Computes mel-spectrogram for a list of texts

        Arguments
        ---------
        texts: List[str]
            texts to be converted to spectrogram
        pace: float
            pace for the speech synthesis
        pitch_rate : float
            scaling factor for phoneme pitches
        energy_rate : float
            scaling factor for phoneme energies

        Returns
        -------
        tensors of output spectrograms, output lengths and alignments
        """

        # Preprocessing required at the inference time for the input text
        # "label" below contains input text
        # "phoneme_labels" contain the phoneme sequences corresponding to input text labels
        # "last_phonemes_combined" is used to indicate whether the index position is for a last phoneme of a word
        # "punc_positions" is used to add back the silence for punctuations
        phoneme_labels = list()
        last_phonemes_combined = list()
        punc_positions = list()

        for label in texts:
            phoneme_label = list()
            last_phonemes = list()
            punc_position = list()

            words = label.split()
            words = [word.strip() for word in words]
            words_phonemes = self.g2p(words)

            for i in range(len(words_phonemes)):
                words_phonemes_seq = words_phonemes[i]
                for phoneme in words_phonemes_seq:
                    if not phoneme.isspace():
                        phoneme_label.append(phoneme)
                        last_phonemes.append(0)
                        punc_position.append(0)
                last_phonemes[-1] = 1
                if words[i][-1] in ":;-,.!?":
                    punc_position[-1] = 1

            phoneme_labels.append(phoneme_label)
            last_phonemes_combined.append(last_phonemes)
            punc_positions.append(punc_position)

        # Inserts silent phonemes in the input phoneme sequence
        all_tokens_with_spn = list()
        max_seq_len = -1
        for i in range(len(phoneme_labels)):
            phoneme_label = phoneme_labels[i]
            token_seq = (
                self.input_encoder.encode_sequence_torch(phoneme_label)
                .int()
                .to(self.device)
            )
            last_phonemes = torch.LongTensor(last_phonemes_combined[i]).to(
                self.device
            )

            # Runs the silent phoneme predictor
            spn_preds = (
                self.hparams.modules["spn_predictor"]
                .infer(token_seq.unsqueeze(0), last_phonemes.unsqueeze(0))
                .int()
            )

            spn_to_add = torch.nonzero(spn_preds).reshape(-1).tolist()

            for j in range(len(punc_positions[i])):
                if punc_positions[i][j] == 1:
                    spn_to_add.append(j)

            tokens_with_spn = list()

            for token_idx in range(token_seq.shape[0]):
                tokens_with_spn.append(token_seq[token_idx].item())
                if token_idx in spn_to_add:
                    tokens_with_spn.append(self.spn_token_encoded)

            tokens_with_spn = torch.LongTensor(tokens_with_spn).to(self.device)
            all_tokens_with_spn.append(tokens_with_spn)
            if max_seq_len < tokens_with_spn.shape[-1]:
                max_seq_len = tokens_with_spn.shape[-1]

        # "tokens_with_spn_tensor" holds the input phoneme sequence with silent phonemes
        tokens_with_spn_tensor_padded = torch.LongTensor(
            len(texts), max_seq_len
        ).to(self.device)
        tokens_with_spn_tensor_padded.zero_()

        for seq_idx, seq in enumerate(all_tokens_with_spn):
            tokens_with_spn_tensor_padded[seq_idx, : len(seq)] = seq

        return self.encode_batch(
            tokens_with_spn_tensor_padded,
            pace=pace,
            pitch_rate=pitch_rate,
            energy_rate=energy_rate,
        )

    def encode_phoneme(
        self, phonemes, pace=1.0, pitch_rate=1.0, energy_rate=1.0
    ):
        """Computes mel-spectrogram for a list of phoneme sequences

        Arguments
        ---------
        phonemes: List[List[str]]
            phonemes to be converted to spectrogram
        pace: float
            pace for the speech synthesis
        pitch_rate : float
            scaling factor for phoneme pitches
        energy_rate : float
            scaling factor for phoneme energies

        Returns
        -------
        tensors of output spectrograms, output lengths and alignments
        """

        all_tokens = []
        max_seq_len = -1
        for phoneme in phonemes:
            token_seq = (
                self.input_encoder.encode_sequence_torch(phoneme)
                .int()
                .to(self.device)
            )
            if max_seq_len < token_seq.shape[-1]:
                max_seq_len = token_seq.shape[-1]
            all_tokens.append(token_seq)

        tokens_padded = torch.LongTensor(len(phonemes), max_seq_len).to(
            self.device
        )
        tokens_padded.zero_()

        for seq_idx, seq in enumerate(all_tokens):
            tokens_padded[seq_idx, : len(seq)] = seq

        return self.encode_batch(
            tokens_padded,
            pace=pace,
            pitch_rate=pitch_rate,
            energy_rate=energy_rate,
        )

    def encode_batch(
        self, tokens_padded, pace=1.0, pitch_rate=1.0, energy_rate=1.0
    ):
        """Batch inference for a tensor of phoneme sequences

        Arguments
        ---------
        tokens_padded : torch.Tensor
            A sequence of encoded phonemes to be converted to spectrogram
        pace : float
            pace for the speech synthesis
        pitch_rate : float
            scaling factor for phoneme pitches
        energy_rate : float
            scaling factor for phoneme energies

        Returns
        -------
        post_mel_outputs : torch.Tensor
        durations : torch.Tensor
        pitch : torch.Tensor
        energy : torch.Tensor
        """
        with torch.no_grad():
            (
                _,
                post_mel_outputs,
                durations,
                pitch,
                _,
                energy,
                _,
                _,
            ) = self.hparams.model(
                tokens_padded,
                pace=pace,
                pitch_rate=pitch_rate,
                energy_rate=energy_rate,
            )

            # Transposes to make in compliant with HiFI GAN expected format
            post_mel_outputs = post_mel_outputs.transpose(-1, 1)

        return post_mel_outputs, durations, pitch, energy

    def forward(self, text, pace=1.0, pitch_rate=1.0, energy_rate=1.0):
        """Batch inference for a tensor of phoneme sequences

        Arguments
        ---------
        text : str
            A text to be converted to spectrogram
        pace : float
            pace for the speech synthesis
        pitch_rate : float
            scaling factor for phoneme pitches
        energy_rate : float
            scaling factor for phoneme energies

        Returns
        -------
        Encoded text
        """
        return self.encode_text(
            [text], pace=pace, pitch_rate=pitch_rate, energy_rate=energy_rate
        )


class FastSpeech2InternalAlignment(Pretrained):
    """
    A ready-to-use wrapper for Fastspeech2 with internal alignment(text -> mel_spec).

    Arguments
    ---------
    *args : tuple
    **kwargs : dict
        Arguments are forwarded to ``Pretrained`` parent class.

    Example
    -------
    >>> tmpdir_tts = getfixture("tmpdir") / "tts"
    >>> fastspeech2 = FastSpeech2InternalAlignment.from_hparams(
    ...     source="speechbrain/tts-fastspeech2-internal-alignment-ljspeech",
    ...     savedir=tmpdir_tts,
    ... )  # doctest: +SKIP
    >>> mel_outputs, durations, pitch, energy = fastspeech2.encode_text(
    ...     ["Mary had a little lamb."]
    ... )  # doctest: +SKIP
    >>> items = [
    ...     "A quick brown fox jumped over the lazy dog",
    ...     "How much wood would a woodchuck chuck?",
    ...     "Never odd or even",
    ... ]
    >>> mel_outputs, durations, pitch, energy = fastspeech2.encode_text(
    ...     items
    ... )  # doctest: +SKIP
    >>> # One can combine the TTS model with a vocoder (that generates the final waveform)
    >>> # Initialize the Vocoder (HiFIGAN)
    >>> tmpdir_vocoder = getfixture("tmpdir") / "vocoder"
    >>> from speechbrain.inference.vocoders import HIFIGAN
    >>> hifi_gan = HIFIGAN.from_hparams(
    ...     source="speechbrain/tts-hifigan-ljspeech", savedir=tmpdir_vocoder
    ... )  # doctest: +SKIP
    >>> # Running the TTS
    >>> mel_outputs, durations, pitch, energy = fastspeech2.encode_text(
    ...     ["Mary had a little lamb."]
    ... )  # doctest: +SKIP
    >>> # Running Vocoder (spectrogram-to-waveform)
    >>> waveforms = hifi_gan.decode_batch(mel_outputs)  # doctest: +SKIP
    """

    HPARAMS_NEEDED = ["model", "input_encoder"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lexicon = self.hparams.lexicon
        lexicon = ["@@"] + lexicon
        self.input_encoder = self.hparams.input_encoder
        self.input_encoder.update_from_iterable(lexicon, sequence_input=False)
        self.input_encoder.add_unk()

        self.g2p = GraphemeToPhoneme.from_hparams("speechbrain/soundchoice-g2p")

    def encode_text(self, texts, pace=1.0, pitch_rate=1.0, energy_rate=1.0):
        """Computes mel-spectrogram for a list of texts

        Arguments
        ---------
        texts: List[str]
            texts to be converted to spectrogram
        pace: float
            pace for the speech synthesis
        pitch_rate : float
            scaling factor for phoneme pitches
        energy_rate : float
            scaling factor for phoneme energies

        Returns
        -------
        tensors of output spectrograms, output lengths and alignments
        """

        # Preprocessing required at the inference time for the input text
        # "label" below contains input text
        # "phoneme_labels" contain the phoneme sequences corresponding to input text labels

        phoneme_labels = list()
        max_seq_len = -1

        for label in texts:
            phonemes_with_punc = self._g2p_keep_punctuations(self.g2p, label)
            if max_seq_len < len(phonemes_with_punc):
                max_seq_len = len(phonemes_with_punc)
            token_seq = (
                self.input_encoder.encode_sequence_torch(phonemes_with_punc)
                .int()
                .to(self.device)
            )
            phoneme_labels.append(token_seq)

        tokens_padded = torch.LongTensor(len(texts), max_seq_len).to(
            self.device
        )
        tokens_padded.zero_()

        for seq_idx, seq in enumerate(phoneme_labels):
            tokens_padded[seq_idx, : len(seq)] = seq

        return self.encode_batch(
            tokens_padded,
            pace=pace,
            pitch_rate=pitch_rate,
            energy_rate=energy_rate,
        )

    def _g2p_keep_punctuations(self, g2p_model, text):
        """do grapheme to phoneme and keep the punctuations between the words"""
        # find the words where a "-" or "'" or "." or ":" appears in the middle
        special_words = re.findall(r"\w+[-':\.][-':\.\w]*\w+", text)

        # remove intra-word punctuations ("-':."), this does not change the output of speechbrain g2p
        for special_word in special_words:
            rmp = special_word.replace("-", "")
            rmp = rmp.replace("'", "")
            rmp = rmp.replace(":", "")
            rmp = rmp.replace(".", "")
            text = text.replace(special_word, rmp)

        # keep inter-word punctuations
        all_ = re.findall(r"[\w]+|[-!'(),.:;? ]", text)
        try:
            phonemes = g2p_model(text)
        except RuntimeError:
            logger.info(f"error with text: {text}")
            quit()
        word_phonemes = "-".join(phonemes).split(" ")

        phonemes_with_punc = []
        count = 0
        try:
            # if the g2p model splits the words correctly
            for i in all_:
                if i not in "-!'(),.:;? ":
                    phonemes_with_punc.extend(word_phonemes[count].split("-"))
                    count += 1
                else:
                    phonemes_with_punc.append(i)
        except IndexError:
            # sometimes the g2p model cannot split the words correctly
            logger.warning(
                f"Do g2p word by word because of unexpected outputs from g2p for text: {text}"
            )

            for i in all_:
                if i not in "-!'(),.:;? ":
                    p = g2p_model.g2p(i)
                    p_without_space = [i for i in p if i != " "]
                    phonemes_with_punc.extend(p_without_space)
                else:
                    phonemes_with_punc.append(i)

        while "" in phonemes_with_punc:
            phonemes_with_punc.remove("")
        return phonemes_with_punc

    def encode_phoneme(
        self, phonemes, pace=1.0, pitch_rate=1.0, energy_rate=1.0
    ):
        """Computes mel-spectrogram for a list of phoneme sequences

        Arguments
        ---------
        phonemes: List[List[str]]
            phonemes to be converted to spectrogram
        pace: float
            pace for the speech synthesis
        pitch_rate : float
            scaling factor for phoneme pitches
        energy_rate : float
            scaling factor for phoneme energies

        Returns
        -------
        tensors of output spectrograms, output lengths and alignments
        """

        all_tokens = []
        max_seq_len = -1
        for phoneme in phonemes:
            token_seq = (
                self.input_encoder.encode_sequence_torch(phoneme)
                .int()
                .to(self.device)
            )
            if max_seq_len < token_seq.shape[-1]:
                max_seq_len = token_seq.shape[-1]
            all_tokens.append(token_seq)

        tokens_padded = torch.LongTensor(len(phonemes), max_seq_len).to(
            self.device
        )
        tokens_padded.zero_()

        for seq_idx, seq in enumerate(all_tokens):
            tokens_padded[seq_idx, : len(seq)] = seq

        return self.encode_batch(
            tokens_padded,
            pace=pace,
            pitch_rate=pitch_rate,
            energy_rate=energy_rate,
        )

    def encode_batch(
        self, tokens_padded, pace=1.0, pitch_rate=1.0, energy_rate=1.0
    ):
        """Batch inference for a tensor of phoneme sequences

        Arguments
        ---------
        tokens_padded : torch.Tensor
            A sequence of encoded phonemes to be converted to spectrogram
        pace : float
            pace for the speech synthesis
        pitch_rate : float
            scaling factor for phoneme pitches
        energy_rate : float
            scaling factor for phoneme energies

        Returns
        -------
        post_mel_outputs : torch.Tensor
        durations : torch.Tensor
        pitch : torch.Tensor
        energy : torch.Tensor
        """
        with torch.no_grad():
            (
                _,
                post_mel_outputs,
                durations,
                pitch,
                _,
                energy,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = self.hparams.model(
                tokens_padded,
                pace=pace,
                pitch_rate=pitch_rate,
                energy_rate=energy_rate,
            )

            # Transposes to make in compliant with HiFI GAN expected format
            post_mel_outputs = post_mel_outputs.transpose(-1, 1)

        return post_mel_outputs, durations, pitch, energy

    def forward(self, text, pace=1.0, pitch_rate=1.0, energy_rate=1.0):
        """Batch inference for a tensor of phoneme sequences

        Arguments
        ---------
        text : str
            A text to be converted to spectrogram
        pace : float
            pace for the speech synthesis
        pitch_rate : float
            scaling factor for phoneme pitches
        energy_rate : float
            scaling factor for phoneme energies

        Returns
        -------
        Encoded text
        """
        return self.encode_text(
            [text], pace=pace, pitch_rate=pitch_rate, energy_rate=energy_rate
        )
