"""Custom Interface for AISHELL-1 CTC inference
An external tokenizer is used so some special tokens
need to be specified during decoding

Authors
 * Yingzhi Wang 2022
"""

import torch
from speechbrain.pretrained import Pretrained


class CustomEncoderDecoderASR(Pretrained):
    """A ready-to-use Encoder-Decoder ASR model
    The class can be used either to run only the encoder (encode()) to extract
    features or to run the entire encoder-decoder model
    (transcribe()) to transcribe speech. The given YAML must contains the fields
    specified in the *_NEEDED[] lists.
    Example
    -------
    >>> from speechbrain.pretrained import EncoderDecoderASR
    >>> tmpdir = getfixture("tmpdir")
    >>> asr_model = EncoderDecoderASR.from_hparams(
    ...     source="speechbrain/asr-crdnn-rnnlm-librispeech",
    ...     savedir=tmpdir,
    ... )
    >>> asr_model.transcribe_file("tests/samples/single-mic/example2.flac")
    "MY FATHER HAS REVEALED THE CULPRIT'S NAME"
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = self.hparams.tokenizer

    def transcribe_file(self, path):
        """Transcribes the given audiofile into a sequence of words.
        Arguments
        ---------
        path : str
            Path to audio file which to transcribe.
        Returns
        -------
        str
            The audiofile transcription produced by this ASR system.
        """
        waveform = self.load_audio(path)
        # Fake a batch:
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        predicted_words = self.transcribe_batch(batch, rel_length)
        return predicted_words[0]

    def encode_batch(self, wavs, wav_lens):
        """Encodes the input audio into a sequence of hidden states
        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = EncoderDecoderASR.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.
        Arguments
        ---------
        wavs : torch.tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model.
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
        wavs = wavs.to(self.device)
        outputs = self.mods.wav2vec2(wavs, wav_lens)
        outputs = self.mods.enc(outputs)
        outputs = self.mods.ctc_lin(outputs)
        return outputs

    def transcribe_batch(self, wavs, wav_lens):
        """Transcribes the input audio into a sequence of words
        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = EncoderDecoderASR.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.
        Arguments
        ---------
        wavs : torch.tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model.
        wav_lens : torch.tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.
        Returns
        -------
        list
            Each waveform in the batch transcribed.
        tensor
            Each predicted token id.
        """
        with torch.no_grad():
            wav_lens = wav_lens.to(self.device)
            encoder_out = self.encode_batch(wavs, wav_lens)
            p_ctc = self.hparams.log_softmax(encoder_out)
            sequences = self.hparams.decoder(p_ctc, wav_lens)
            predicted_words_list = []
            for sequence in sequences:
                predicted_tokens = self.tokenizer.convert_ids_to_tokens(
                    sequence
                )
                predicted_words = []
                for c in predicted_tokens:
                    if c == "[CLS]":
                        continue
                    elif c == "[SEP]" or c == "[PAD]":
                        break
                    else:
                        predicted_words.append(c)
                predicted_words_list.append(predicted_words)

        return predicted_words_list

    def forward(self, wavs, wav_lens):
        """Runs full transcription - note: no gradients through decoding"""
        return self.transcribe_batch(wavs, wav_lens)
