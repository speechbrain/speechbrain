"""Specifies the inference interfaces for Spoken Language Understanding (SLU) modules.

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

from speechbrain.inference.ASR import EncoderDecoderASR
from speechbrain.inference.interfaces import Pretrained


class EndToEndSLU(Pretrained):
    """An end-to-end SLU model.

    The class can be used either to run only the encoder (encode()) to extract
    features or to run the entire model (decode()) to map the speech to its semantics.

    Arguments
    ---------
    *args : tuple
    **kwargs : dict
        Arguments are forwarded to ``Pretrained`` parent class.

    Example
    -------
    >>> from speechbrain.inference.SLU import EndToEndSLU
    >>> tmpdir = getfixture("tmpdir")
    >>> slu_model = EndToEndSLU.from_hparams(
    ...     source="speechbrain/slu-timers-and-such-direct-librispeech-asr",
    ...     savedir=tmpdir,
    ... )  # doctest: +SKIP
    >>> slu_model.decode_file(
    ...     "tests/samples/single-mic/example6.wav"
    ... )  # doctest: +SKIP
    "{'intent': 'SimpleMath', 'slots': {'number1': 37.67, 'number2': 75.7, 'op': ' minus '}}"
    """

    HPARAMS_NEEDED = ["tokenizer", "asr_model_source"]
    MODULES_NEEDED = ["slu_enc", "beam_searcher"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = self.hparams.tokenizer
        self.asr_model = EncoderDecoderASR.from_hparams(
            source=self.hparams.asr_model_source,
            run_opts={"device": self.device},
        )

    def decode_file(self, path, **kwargs):
        """Maps the given audio file to a string representing the
        semantic dictionary for the utterance.

        Arguments
        ---------
        path : str
            Path to audio file to decode.
        **kwargs : dict
            Arguments forwarded to ``load_audio``.

        Returns
        -------
        str
            The predicted semantics.
        """
        waveform = self.load_audio(path, **kwargs)
        waveform = waveform.to(self.device)
        # Fake a batch:
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        predicted_words, predicted_tokens = self.decode_batch(batch, rel_length)
        return predicted_words[0]

    def encode_batch(self, wavs, wav_lens):
        """Encodes the input audio into a sequence of hidden states

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
        ASR_encoder_out = self.asr_model.encode_batch(wavs.detach(), wav_lens)
        encoder_out = self.mods.slu_enc(ASR_encoder_out)
        return encoder_out

    def decode_batch(self, wavs, wav_lens):
        """Maps the input audio to its semantics

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
        list
            Each waveform in the batch decoded.
        tensor
            Each predicted token id.
        """
        with torch.no_grad():
            wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
            encoder_out = self.encode_batch(wavs, wav_lens)
            predicted_tokens, scores, _, _ = self.mods.beam_searcher(
                encoder_out, wav_lens
            )
            predicted_words = [
                self.tokenizer.decode_ids(token_seq)
                for token_seq in predicted_tokens
            ]
        return predicted_words, predicted_tokens

    def forward(self, wavs, wav_lens):
        """Runs full decoding - note: no gradients through decoding"""
        return self.decode_batch(wavs, wav_lens)
