"""Example extension of interfaces; it's a show-casing Easter Egg.
(Please feel free to extend broader coverage.)

Authors:
 * Andreas Nautsch 2022
"""
import torch
from enum import Enum
from speechbrain.utils.data_utils import undo_padding
from speechbrain.pretrained.interfaces import Pretrained


class InferenceDetailLevel(Enum):
    """
    Verbosity level for return values of pretrained interfaces.

    Examples for EncoderDecoderASR (to give an idea).

    TOP1_HYP
    --------
    > 'SUNDAY IS THE BEST PART OF THE WEEK'

    TOP1_HYP_SCORES
    ---------------
    > (-1.86, 'SUNDAY IS THE BEST PART OF THE WEEK')

    TOP1_HYP_DETAILS
    ----------------
    > (-1.86, 'SUNDAY IS THE BEST PART OF THE WEEK')
    > [(-4.63, '▁SUN'), (-0.99, 'DAY'), (-2.26, '▁IS'), (-1.32, '▁THE'), (-2.48, '▁BE'), (-0.12, 'ST'),
    >  (-2.67, '▁PART'), (-0.52, '▁OF'), (-1.33, '▁THE'), (-2.91, '▁WEEK')]

    TOP_K_HYP
    ---------
    > ['SUNDAY IS THE BEST PART OF THE WEEK',
    >  'SUNDAY IS THE BEST PART OF THE DAY',
    >  'SUNDAY IS THE BEST PART OF A WEEK',
    >  'SUNDAY IS THE BEST PART OF THE WORK',
    >  'SUNDAY WAS THE BEST PART OF THE WEEK']

    TOP_K_HYP_SCORES
    ----------------
    > [(-1.86, 'SUNDAY IS THE BEST PART OF THE WEEK'),
    >  (-2.32, 'SUNDAY IS THE BEST PART OF THE DAY'),
    >  (-2.33, 'SUNDAY IS THE BEST PART OF A WEEK'),
    >  (-2.34, 'SUNDAY IS THE BEST PART OF THE WORK'),
    >  (-2.36, 'SUNDAY WAS THE BEST PART OF THE WEEK')]

    TOP_K_HYP_DETAILS
    -----------------
    > [(-1.86, 'SUNDAY IS THE BEST PART OF THE WEEK'),
    >  (-2.32, 'SUNDAY IS THE BEST PART OF THE DAY'),
    >  (-2.33, 'SUNDAY IS THE BEST PART OF A WEEK'),
    >  (-2.34, 'SUNDAY IS THE BEST PART OF THE WORK'),
    >  (-2.36, 'SUNDAY WAS THE BEST PART OF THE WEEK')]
    > [[(-4.63, '▁SUN'), (-0.99, 'DAY'), (-2.26, '▁IS'), (-1.32, '▁THE'), (-2.48, '▁BE'), (-0.12, 'ST'),
    >   (-2.67, '▁PART'), (-0.52, '▁OF'), (-1.33, '▁THE'), (-2.91, '▁WEEK')],
    >  [(-4.63, '▁SUN'), (-0.99, 'DAY'), (-2.26, '▁IS'), (-1.32, '▁THE'), (-2.48, '▁BE'), (-0.12, 'ST'),
    >   (-2.67, '▁PART'), (-0.52, '▁OF'), (-1.33, '▁THE'), (-8.04, '▁DAY')],
    >  [(-4.63, '▁SUN'), (-0.99, 'DAY'), (-2.26, '▁IS'), (-1.32, '▁THE'), (-2.48, '▁BE'), (-0.12, 'ST'),
    >   (-2.67, '▁PART'), (-0.52, '▁OF'), (-6.57, '▁A'), (-2.78, '▁WEEK')],
    >  [(-4.63, '▁SUN'), (-0.99, 'DAY'), (-2.26, '▁IS'), (-1.32, '▁THE'), (-2.48, '▁BE'), (-0.12, 'ST'),
    >   (-2.67, '▁PART'), (-0.52, '▁OF'), (-1.33, '▁THE'), (-7.90, '▁WORK')],
    >  [(-4.63, '▁SUN'), (-0.99, 'DAY'), (-7.73, '▁WAS'), (-1.30, '▁THE'), (-2.70, '▁BE'), (-0.19, 'ST'),
    >   (-2.65, '▁PART'), (-0.50, '▁OF'), (-1.16, '▁THE'), (-2.78, '▁WEEK')]]
    """

    TOP1_HYP = 1
    TOP1_HYP_SCORES = 2
    TOP1_HYP_DETAILS = 3
    TOP_K_HYP = 4
    TOP_K_HYP_SCORES = 5
    TOP_K_HYP_DETAILS = 6


class EndToEndSLUTopK(Pretrained):
    """A end-to-end SLU model which returns top-k results.

    The class can be used either to run only the encoder (encode()) to extract
    features or to run the entire model (decode()) to map the speech to its semantics.
    """

    HPARAMS_NEEDED = ["tokenizer", "asr_model_source"]
    MODULES_NEEDED = ["slu_enc", "beam_searcher"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = self.hparams.tokenizer
        self.asr_model = EncoderDecoderASRTopK.from_hparams(
            source=self.hparams.asr_model_source,
            run_opts={"device": self.device},
        )

    def decode_file(self, path, detail_level=None):
        """Maps the given audio file to a string representing the
        semantic dictionary for the utterance.

        Arguments
        ---------
        path : str
            Path to audio file to decode.

        Returns
        -------
        str
            The predicted semantics.
        """
        waveform = self.load_audio(path)
        waveform = waveform.to(self.device)
        # Fake a batch:
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        if detail_level is None:
            return self.decode_batch(batch, rel_length, None)[0][0]
        else:
            return self.decode_batch(batch, rel_length, detail_level)

    def encode_batch(self, wavs, wav_lens):
        """Encodes the input audio into a sequence of hidden states

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
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        ASR_encoder_out = self.asr_model.encode_batch(wavs.detach(), wav_lens)
        encoder_out = self.mods.slu_enc(ASR_encoder_out)
        return encoder_out

    def decode_batch(self, wavs, wav_lens, detail_level=None):
        """Maps the input audio to its semantics

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
            Each waveform in the batch decoded.
        tensor
            Each predicted token id.
        """
        with torch.no_grad():
            wav_lens = wav_lens.to(self.device)
            encoder_out = self.encode_batch(wavs, wav_lens)
            if detail_level == InferenceDetailLevel.TOP1_HYP:
                topk_tokens, topk_lens, _, _ = self.mods.beam_searcher(
                    encoder_out, wav_lens
                )
                best_hyps, best_lens = topk_tokens[:, 0, :], topk_lens[:, 0]
                predicted_tokens = undo_padding(best_hyps, best_lens)
                predicted_words = [
                    self.tokenizer.decode_ids(token_seq)
                    for token_seq in predicted_tokens
                ]
                return predicted_words
            elif detail_level == InferenceDetailLevel.TOP1_HYP_SCORES:
                (
                    topk_tokens,
                    topk_lens,
                    topk_scores,
                    _,
                ) = self.mods.beam_searcher(encoder_out, wav_lens)
                best_hyps, best_lens = topk_tokens[:, 0, :], topk_lens[:, 0]
                predicted_tokens = undo_padding(best_hyps, best_lens)
                predicted_words = [
                    self.tokenizer.decode_ids(token_seq)
                    for token_seq in predicted_tokens
                ]
                return predicted_words, list(topk_scores.T[0, :])
            elif detail_level == InferenceDetailLevel.TOP1_HYP_DETAILS:
                (
                    topk_tokens,
                    topk_lens,
                    topk_scores,
                    topk_log_probs,
                ) = self.mods.beam_searcher(encoder_out, wav_lens)
                best_hyps, best_lens, best_prob = (
                    topk_tokens[:, 0, :],
                    topk_lens[:, 0],
                    topk_log_probs[:, 0, :],
                )
                predicted_tokens = undo_padding(best_hyps, best_lens)
                predicted_log_prob = undo_padding(best_prob, best_lens)
                predicted_words = [
                    self.tokenizer.decode_ids(token_seq)
                    for token_seq in predicted_tokens
                ]
                tkns = []
                for pred_tkns in predicted_tokens:
                    tkns.append(self.tokenizer.id_to_piece(pred_tkns))
                return (
                    predicted_words,
                    list(topk_scores.T[0, :]),
                    predicted_tokens,
                    predicted_log_prob,
                    tkns,
                )
            elif detail_level == InferenceDetailLevel.TOP_K_HYP:
                predictions = []
                topk_tokens, topk_lens, _, _ = self.mods.beam_searcher(
                    encoder_out, wav_lens
                )
                for i in range(topk_lens.shape[1]):
                    best_hyps, best_lens = topk_tokens[:, i, :], topk_lens[:, i]
                    predicted_tokens = undo_padding(best_hyps, best_lens)
                    predicted_words = [
                        self.tokenizer.decode_ids(token_seq)
                        for token_seq in predicted_tokens
                    ]
                    predictions.append(predicted_words)
                return predictions
            elif detail_level == InferenceDetailLevel.TOP_K_HYP_SCORES:
                predictions = []
                (
                    topk_tokens,
                    topk_lens,
                    topk_scores,
                    _,
                ) = self.mods.beam_searcher(encoder_out, wav_lens)
                for i in range(topk_lens.shape[1]):
                    best_hyps, best_lens = topk_tokens[:, i, :], topk_lens[:, i]
                    predicted_tokens = undo_padding(best_hyps, best_lens)
                    predicted_words = [
                        self.tokenizer.decode_ids(token_seq)
                        for token_seq in predicted_tokens
                    ]
                    predictions.append(predicted_words)
                return predictions, list(map(list, topk_scores.T))
            elif detail_level == InferenceDetailLevel.TOP_K_HYP_DETAILS:
                predictions = []
                token_ids = []
                log_probs = []
                tokens = []
                (
                    topk_tokens,
                    topk_lens,
                    topk_scores,
                    topk_log_probs,
                ) = self.mods.beam_searcher(encoder_out, wav_lens)
                for i in range(topk_lens.shape[1]):
                    best_hyps, best_lens, best_prob = (
                        topk_tokens[:, i, :],
                        topk_lens[:, i],
                        topk_log_probs[:, i, :],
                    )
                    predicted_tokens = undo_padding(best_hyps, best_lens)
                    predicted_log_prob = undo_padding(best_prob, best_lens)
                    predicted_words = [
                        self.tokenizer.decode_ids(token_seq)
                        for token_seq in predicted_tokens
                    ]
                    predictions.append(predicted_words)
                    token_ids.append(predicted_tokens)
                    log_probs.append(predicted_log_prob)
                    tkns = []
                    for pred_tkns in predicted_tokens:
                        tkns.append(self.tokenizer.id_to_piece(pred_tkns))
                    tokens.append(tkns)
                return (
                    predictions,
                    list(map(list, topk_scores.T)),
                    token_ids,
                    log_probs,
                    tokens,
                )
            else:
                # the legacy default
                topk_tokens, topk_lens, _, _ = self.mods.beam_searcher(
                    encoder_out, wav_lens
                )
                best_hyps, best_lens = topk_tokens[:, 0, :], topk_lens[:, 0]
                predicted_tokens = undo_padding(best_hyps, best_lens)
                predicted_words = [
                    self.tokenizer.decode_ids(token_seq)
                    for token_seq in predicted_tokens
                ]
                return predicted_words, predicted_tokens

    def forward(self, wavs, wav_lens):
        """Runs full decoding - note: no gradients through decoding"""
        return self.decode_batch(wavs, wav_lens)


class EncoderDecoderASRTopK(
    Pretrained
):  # inherits directly from Pretrained (not from EncoderDecoderASR)
    """A ready-to-use Encoder-Decoder ASR model which returns top-k results.
    """

    HPARAMS_NEEDED = ["tokenizer"]
    MODULES_NEEDED = ["encoder", "decoder"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = self.hparams.tokenizer

    def transcribe_file(self, path, detail_level=None):
        """Transcribes the given audiofile into a sequence of words.

        Arguments
        ---------
        path : str
            Path to audio file which to transcribe.
        detail_level : Option[InferenceDetailLevel]
            Verboseness level of result; default: None (legacy output)

        Returns
        -------
        str
            The audiofile transcription produced by this ASR system.
        """
        waveform = self.load_audio(path)
        # Fake a batch:
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        if detail_level is None:
            return self.transcribe_batch(batch, rel_length, None)[0][0]
        else:
            return self.transcribe_batch(batch, rel_length, detail_level)[0]

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
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        encoder_out = self.mods.encoder(wavs, wav_lens)
        return encoder_out

    def transcribe_batch(self, wavs, wav_lens, detail_level=None):
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
        detail_level : Option[InferenceDetailLevel]
            Verboseness level of result; default: None (legacy output)

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
            if detail_level == InferenceDetailLevel.TOP1_HYP:
                topk_tokens, topk_lens, _, _ = self.mods.decoder(
                    encoder_out, wav_lens
                )
                best_hyps, best_lens = topk_tokens[:, 0, :], topk_lens[:, 0]
                predicted_tokens = undo_padding(best_hyps, best_lens)
                predicted_words = [
                    self.tokenizer.decode_ids(token_seq)
                    for token_seq in predicted_tokens
                ]
                return predicted_words
            elif detail_level == InferenceDetailLevel.TOP1_HYP_SCORES:
                topk_tokens, topk_lens, topk_scores, _ = self.mods.decoder(
                    encoder_out, wav_lens
                )
                best_hyps, best_lens = topk_tokens[:, 0, :], topk_lens[:, 0]
                predicted_tokens = undo_padding(best_hyps, best_lens)
                predicted_words = [
                    self.tokenizer.decode_ids(token_seq)
                    for token_seq in predicted_tokens
                ]
                return predicted_words, list(topk_scores.T[0, :])
            elif detail_level == InferenceDetailLevel.TOP1_HYP_DETAILS:
                (
                    topk_tokens,
                    topk_lens,
                    topk_scores,
                    topk_log_probs,
                ) = self.mods.decoder(encoder_out, wav_lens)
                best_hyps, best_lens, best_prob = (
                    topk_tokens[:, 0, :],
                    topk_lens[:, 0],
                    topk_log_probs[:, 0, :],
                )
                predicted_tokens = undo_padding(best_hyps, best_lens)
                predicted_log_prob = undo_padding(best_prob, best_lens)
                predicted_words = [
                    self.tokenizer.decode_ids(token_seq)
                    for token_seq in predicted_tokens
                ]
                tkns = []
                for pred_tkns in predicted_tokens:
                    tkns.append(self.tokenizer.id_to_piece(pred_tkns))
                return (
                    predicted_words,
                    list(topk_scores.T[0, :]),
                    predicted_tokens,
                    predicted_log_prob,
                    tkns,
                )
            elif detail_level == InferenceDetailLevel.TOP_K_HYP:
                predictions = []
                topk_tokens, topk_lens, _, _ = self.mods.decoder(
                    encoder_out, wav_lens
                )
                for i in range(topk_lens.shape[1]):
                    best_hyps, best_lens = topk_tokens[:, i, :], topk_lens[:, i]
                    predicted_tokens = undo_padding(best_hyps, best_lens)
                    predicted_words = [
                        self.tokenizer.decode_ids(token_seq)
                        for token_seq in predicted_tokens
                    ]
                    predictions.append(predicted_words)
                return predictions
            elif detail_level == InferenceDetailLevel.TOP_K_HYP_SCORES:
                predictions = []
                topk_tokens, topk_lens, topk_scores, _ = self.mods.decoder(
                    encoder_out, wav_lens
                )
                for i in range(topk_lens.shape[1]):
                    best_hyps, best_lens = topk_tokens[:, i, :], topk_lens[:, i]
                    predicted_tokens = undo_padding(best_hyps, best_lens)
                    predicted_words = [
                        self.tokenizer.decode_ids(token_seq)
                        for token_seq in predicted_tokens
                    ]
                    predictions.append(predicted_words)
                return predictions, list(map(list, topk_scores.T))
            elif detail_level == InferenceDetailLevel.TOP_K_HYP_DETAILS:
                predictions = []
                token_ids = []
                log_probs = []
                tokens = []
                (
                    topk_tokens,
                    topk_lens,
                    topk_scores,
                    topk_log_probs,
                ) = self.mods.decoder(encoder_out, wav_lens)
                for i in range(topk_lens.shape[1]):
                    best_hyps, best_lens, best_prob = (
                        topk_tokens[:, i, :],
                        topk_lens[:, i],
                        topk_log_probs[:, i, :],
                    )
                    predicted_tokens = undo_padding(best_hyps, best_lens)
                    predicted_log_prob = undo_padding(best_prob, best_lens)
                    predicted_words = [
                        self.tokenizer.decode_ids(token_seq)
                        for token_seq in predicted_tokens
                    ]
                    predictions.append(predicted_words)
                    token_ids.append(predicted_tokens)
                    log_probs.append(predicted_log_prob)
                    tkns = []
                    for pred_tkns in predicted_tokens:
                        tkns.append(self.tokenizer.id_to_piece(pred_tkns))
                    tokens.append(tkns)
                return (
                    predictions,
                    list(map(list, topk_scores.T)),
                    token_ids,
                    log_probs,
                    tokens,
                )
            else:
                # the legacy default
                topk_tokens, topk_lens, _, _ = self.mods.decoder(
                    encoder_out, wav_lens
                )
                best_hyps, best_lens = topk_tokens[:, 0, :], topk_lens[:, 0]
                predicted_tokens = undo_padding(best_hyps, best_lens)
                predicted_words = [
                    self.tokenizer.decode_ids(token_seq)
                    for token_seq in predicted_tokens
                ]
                return predicted_words, predicted_tokens

    def forward(self, wavs, wav_lens, detail_level=None):
        """Runs full transcription - note: no gradients through decoding"""
        return self.transcribe_batch(wavs, wav_lens, detail_level)
