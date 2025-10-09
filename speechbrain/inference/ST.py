"""Specifies the inference interfaces for Speech Translation (ST) modules.

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


class EncoderDecoderS2UT(Pretrained):
    """A ready-to-use Encoder Decoder for speech-to-unit translation model

    The class can be used  to  run the entire encoder-decoder S2UT model
    (translate_file()) to translate speech. The given YAML must contains the fields
    specified in the *_NEEDED[] lists.

    Arguments
    ---------
    *args : tuple
    **kwargs : dict
        Arguments are forwarded to ``Pretrained`` parent class.

    Example
    -------
    >>> from speechbrain.inference.ST import EncoderDecoderS2UT
    >>> tmpdir = getfixture("tmpdir")
    >>> s2ut_model = EncoderDecoderS2UT.from_hparams(
    ...     source="speechbrain/s2st-transformer-fr-en-hubert-l6-k100-cvss",
    ...     savedir=tmpdir,
    ... )  # doctest: +SKIP
    >>> s2ut_model.translate_file(
    ...     "speechbrain/s2st-transformer-fr-en-hubert-l6-k100-cvss/example-fr.wav"
    ... )  # doctest: +SKIP
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
            predicted_tokens, _, _, _ = self.mods.decoder(encoder_out, wav_lens)
        return predicted_tokens

    def forward(self, wavs, wav_lens):
        """Runs full translation"""
        return self.encode_batch(wavs, wav_lens)
