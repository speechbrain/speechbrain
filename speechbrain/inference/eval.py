""" Specifies the inference interfaces for speech quality
evaluation, used to assess the quality/intelligibility of
text-to-speech systems

Authors:
* Artem Ploujnikov 2024
"""

from speechbrain.inference.interfaces import Pretrained
from speechbrain.inference.ASR import EncoderDecoderASR
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.dataio.dataio import read_audio
from speechbrain.utils.metric_stats import ErrorRateStats
from collections import namedtuple
import torch

SpeechEvaluationResult = namedtuple(
    "SpeechEvaluationResult", ["score", "details"]
)


class SpeechEvaluator:
    """A base class for speech evaluators"""

    def evaluate_file(self, file_name, text=None):
        """Evaluates a single file

        Arguments
        ---------
        file_name : str|pathlib.Path
            The file name to evaluate
        text : str
            The ground truth text, if applicable

        Returns
        -------
        result: SpeechEvaluationResult
            the evaluation result
        """
        wav = read_audio(str(file_name)).to(self.device)
        result = self.evaluate(
            wavs=wav.unsqueeze(0),
            length=torch.ones(1).to(self.device),
            text=[text],
        )
        return SpeechEvaluationResult(
            score=result.score.item(),
            details={
                key: _unbatchify(value) for key, value in result.details.items()
            },
        )

    def evaluate_files(self, file_names, text=None):
        """Evaluates multiple files

        Arguments
        ---------
        file_names : list
            A list of files

        text : list
            File transcripts (not required for all evaluators)

        Returns
        -------
        result : list
            a list of SpeechEvaluationResult instances
        """
        if text is None:
            text = [None] * len(file_names)
        items = [
            {"wav": read_audio(str(file_name)), "text": item_text}
            for file_name, item_text in zip(file_names, text)
        ]
        batch = PaddedBatch(items)
        return self.evaluate(
            wavs=batch.wav.data.to(self.device),
            length=batch.wav.lengths.to(self.device),
            text=batch.text,
        )

    def evaluate(self, wavs, length, text=None):
        """Evaluates samples

        Arguments
        ---------
        wavs : torch.Tensor
            the waveforms to evaluate

        length : torch.Tensor
            relative lengths (a 1-D tensor)

        text : list
            Evaluator-specific metadata

        Returns
        -------
        result : list
            A list of SpeechEvaluationResult objects,
            one for each sample"""
        raise NotImplementedError()


def _unbatchify(value):
    """Removes the batch dimension from the tensor. If a single
    number is returned in any shape, the function converts
    the result to a numeric value. Values that are not tensors
    are returned unmodified

    Arguments
    ---------
    value : object
        the value

    Returns
    -------
    value : object
        the value with the batch dimension removed, if applicable
    """
    if torch.is_tensor(value):
        if value.dim() == 0 or not any(dim > 1 for dim in value.shape):
            value = value.item()
        else:
            value = value.squeeze(0)
    return value


class RegressionModelSpeechEvaluator(Pretrained, SpeechEvaluator):
    """A speech evaluator that uses a regression model
    that produces a quality score (e.g. SSL fine-tuning)
    for a sample of speech

    """

    def evaluate(self, wavs, length, text=None):
        """Evaluates a batch of waveforms

        Arguments
        ---------
        wavs : torch.Tensor
            the waveforms to evaluate

        length : torch.Tensor
            relative lengths (a 1-D tensor)

        text : list
            Evaluator-specific metadata (ignored
            for this evaluator)

        Returns
        -------
        result : SpeechEvaluationResult
            an aggregated speech evaluation result with a score
            for each item
        """
        scores = self.mods.model(wavs, length)
        while scores.dim() > 1 and scores.size(-1) == 1:
            scores = scores.squeeze(-1)
        return SpeechEvaluationResult(score=scores, details={"score": scores})


class EncoderDecoderASRSpeechEvaluator(SpeechEvaluator):
    """A speech evaluator implementation based on ASR. Computes the Word Error Rate (WER),
    Character Error Rate (CER) and a few other metrics
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.asr = EncoderDecoderASR.from_hparams(*args, **kwargs)
        self.device = next(self.asr.mods.parameters()).device

    def init_metrics(self):
        """Initializes the WER and CER metrics

        Returns
        -------
        wer_metric : ErrorRateStats
            the Word Error Rate (WER) metric
        cer_metric : ErrorRateStats
            the Character Error Rate (CER) metric"""
        wer_metric = ErrorRateStats()
        cer_metric = ErrorRateStats(split_tokens=True)
        return wer_metric, cer_metric

    def evaluate(self, wavs, length, text=None):
        """Evaluates samples

        Arguments
        ---------
        wavs: torch.Tensor
            the waveforms to evaluate

        length: torch.Tensor
            relative lengths (a 1-D tensor)

        text : list
            Ground truth text

        Returns
        -------
        result : SpeechEvaluationResult
            an aggregated speech evaluation result with a score
            for each item
        """
        if text is None:
            raise ValueError("This evaluator requires ground-truth text")
        predicted_words, scores, log_probs = self.transcribe_batch_with_details(
            wavs, length
        )
        ids = range(1, len(wavs) + 1)
        wer_metric, cer_metric = self.init_metrics()
        wer_metric.append(ids, predicted_words, text)
        cer_metric.append(ids, predicted_words, text)
        wer = torch.tensor(
            [score["WER"] for score in wer_metric.scores], device=wavs.device
        )
        cer = torch.tensor(
            [score["WER"] for score in cer_metric.scores], device=wavs.device
        )
        prob_mean = log_probs.exp().mean(dim=-1)
        return SpeechEvaluationResult(
            score=wer,
            details={
                "wer": wer,
                "cer": cer,
                "beam_score": scores,
                "prob_mean": prob_mean,
                "pred": predicted_words,
                "target": text,
            },
        )

    def transcribe_batch_with_details(self, wavs, wav_lens):
        """Transcribes the input audio into a sequence of words

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = EncoderDecoderASR.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        predicted_words : list
            The raw ASR predictions, fully decoded
        best_scores : list
            The best scores (from beam search)
        best_log_probs : list
            The best predicted log-probabilities (from beam search)


        Returns
        -------
        predicted_words : list
            The predictions

        best_scores : torch.Tensor
            The best scores (from beam search)

        best_log_probs : torch.Tensor
            The best log-probabilities

        """
        with torch.no_grad():
            wav_lens = wav_lens.to(self.device)
            encoder_out = self.asr.encode_batch(wavs, wav_lens)
            (
                hyps,
                best_lens,
                best_scores,
                best_log_probs,
            ) = self.asr.mods.decoder(encoder_out, wav_lens)
            predicted_words = [
                self.asr.tokenizer.decode_ids(token_seq) for token_seq in hyps
            ]
        return predicted_words, best_scores, best_log_probs


def itemize(result):
    """Converts a single batch result into per-item results

    Arguments
    ---------
    result: SpeechEvaluationResult
        a single batch result

    Returns
    -------
    results: list
        a list of individual SpeechEvaluationResult instances"""

    return [
        SpeechEvaluationResult(
            score=result.score[idx],
            details={key: value[idx] for key, value in result.items()},
        )
        for idx in range(len(result.score))
    ]
