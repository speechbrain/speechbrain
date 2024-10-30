""" Specifies the inference interfaces for speech quality
evaluation, used to assess the quality/intelligibility of
text-to-speech systems

Authors:
* Artem Ploujnikov 2024
"""

import csv
import json
import logging
import re
import string
from abc import abstractmethod
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import torch
import torchaudio
from torch import nn

from speechbrain.dataio.batch import PaddedBatch, undo_batch
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.dataio.dataset import FilteredSortedDynamicItemDataset
from speechbrain.decoders.seq2seq import S2SWhisperGreedySearcher
from speechbrain.inference.ASR import EncoderDecoderASR
from speechbrain.inference.interfaces import Pretrained
from speechbrain.lobes.models.eval.utmos import UTMOSModel
from speechbrain.lobes.models.huggingface_transformers import Whisper
from speechbrain.utils.data_utils import pad_right_to
from speechbrain.utils.fetching import fetch
from speechbrain.utils.metric_stats import ErrorRateStats, MetricStats

logger = logging.getLogger(__name__)

RE_PUNCTUATION = re.compile(
    "|".join(re.escape(char) for char in string.punctuation)
)


SpeechEvaluationResult = namedtuple(
    "SpeechEvaluationResult", ["score", "details"]
)

has_transformers = False
try:
    from transformers import AutoModelForAudioXVector

    has_transformers = True
except ImportError:
    logger.warning(
        "transformers library not found - some evaluators may be disabled"
    )


class SpeechEvaluator:
    """A base class for speech evaluators

    Arguments
    ---------
    sample_rate : int
        The audio sample rate this evaluator expects
    """

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

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
        wav = self.read_audio(str(file_name)).to(self.device)
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
            {"wav": self.read_audio(str(file_name)), "text": item_text}
            for file_name, item_text in zip(file_names, text)
        ]
        batch = PaddedBatch(items)
        return self.evaluate(
            wavs=batch.wav.data.to(self.device),
            length=batch.wav.lengths.to(self.device),
            text=batch.text,
        )

    def read_audio(self, file_name):
        """Reads an audio file, resampling if necessary

        Arguments
        ---------
        file_name : str | path-like
            The file path

        Returns
        -------
        audio : torch.Tensor
            the audio
        """
        audio, audio_sample_rate = torchaudio.load(str(file_name))
        return self.resample(audio, audio_sample_rate)

    @abstractmethod
    def evaluate(
        self,
        wavs,
        length,
        text=None,
        wavs_ref=None,
        wavs_length_ref=None,
        sample_rate=None,
    ):
        """Evaluates samples

        Arguments
        ---------
        wavs : torch.Tensor
            the waveforms to evaluate

        length : torch.Tensor
            relative lengths (a 1-D tensor)

        text : list
            Evaluator-specific metadata

        wavs_ref : torch.Tensor
            the reference waveforms

        wavs_length_ref : torch.Tensor
            the reference waveform lengths

        sample_rate: int, optional
            The sample rate of the audio. If not provided,
            the audio is assumed to be at the same sample
            rate as the model

        Returns
        -------
        result : list
            A list of SpeechEvaluationResult objects,
            one for each sample"""
        return None

    def resample(self, audio, sample_rate=None):
        """Resamples the audio, if necessary

        Arguments
        ---------
        audio : torch.Tensor
            the audio to be resampled
        sample_rate : int
            the sample rate of the audio

        Returns
        -------
        audio : torch.Tensor
            the target audio, resampled if necessary
        """
        if sample_rate is not None and sample_rate != self.sample_rate:
            audio = torchaudio.functional.resample(
                audio, orig_freq=sample_rate, new_freq=self.sample_rate
            )
        return audio


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


class SpeechEvaluationRegressionModel(Pretrained):
    """A pretrained wrapper for regression-based evaluaton
    models"""

    def __call__(self, wavs, length):
        return self.mods.model(wavs, length)


class RegressionModelSpeechEvaluator(SpeechEvaluator):
    """A speech evaluator that uses a regression model
    that produces a quality score (e.g. SSL fine-tuning)
    for a sample of speech

    Arguments
    ---------
    source : str
        The source model path or HuggingFace hub name
    sample_rate : int
        The audio sample rate this evaluator expects
    *args : list
        Additional arguments (passed through)
    **kwargs : dict
        Additional keyword arguments (passed through)
    """

    def __init__(self, source, sample_rate=None, *args, **kwargs):
        super().__init__(sample_rate=sample_rate)
        self.model = SpeechEvaluationRegressionModel.from_hparams(
            source, *args, **kwargs
        )

    def evaluate(
        self,
        wavs,
        length,
        text=None,
        wavs_ref=None,
        length_ref=None,
        sample_rate=None,
        sample_rate_ref=None,
    ):
        """Evaluates a batch of waveforms

        Arguments
        ---------
        Arguments
        ---------
        wavs: torch.Tensor
            the waveforms to evaluate

        length: torch.Tensor
            relative lengths (a 1-D tensor)

        text : list, optional
            Ground truth text

        wavs_ref : torch.Tensor
            the reference waveforms

        length_ref : torch.Tensor
            the reference waveform lengths

        sample_rate : int, optional
            The sample rate of the audio. If not provided,
            the audio is assumed to be at the same sample
            rate as the model

        sample_rate_ref : int, optional
            The sample rate of the reference samples

        Returns
        -------
        result : SpeechEvaluationResult
            an aggregated speech evaluation result with a score
            for each item
        """
        wavs = self.resample(wavs, sample_rate)
        scores = self.model(wavs, length)
        while scores.dim() > 1 and scores.size(-1) == 1:
            scores = scores.squeeze(-1)
        return SpeechEvaluationResult(score=scores, details={"score": scores})


class ASRSpeechEvaluator(SpeechEvaluator):
    """A superclass for ASR speech evaluators"""

    def evaluate(
        self,
        wavs,
        length,
        text=None,
        wavs_ref=None,
        length_ref=None,
        sample_rate=None,
        sample_rate_ref=None,
    ):
        """Evaluates samples

        Arguments
        ---------
        wavs: torch.Tensor
            the waveforms to evaluate

        length: torch.Tensor
            relative lengths (a 1-D tensor)

        text : list, optional
            Ground truth text

        wavs_ref : torch.Tensor
            the reference waveforms

        length_ref : torch.Tensor
            the reference waveform lengths


        sample_rate : int, optional
            The sample rate of the audio. If not provided,
            the audio is assumed to be at the same sample
            rate as the model

        sample_rate_ref : int, optional
            The sample rate of the reference samples

        Returns
        -------
        result : SpeechEvaluationResult
            an aggregated speech evaluation result with a score
            for each item
        """
        details = self.evaluate_samples(
            wavs=wavs, length=length, text=text, sample_rate=sample_rate
        )
        if wavs_ref is not None:
            details_ref = self.evaluate_samples(
                wavs=wavs_ref,
                length=length_ref,
                text=text,
                sample_rate=sample_rate_ref,
            )
            details.update(
                {f"{key}_ref": value for key, value in details_ref.items()}
            )
            # Redundant: it is the same
            del details["target_ref"]
            details.update(self.compute_diff_rate(details, device=wavs.device))

        return SpeechEvaluationResult(
            score=details["wer"],
            details=details,
        )

    def compute_diff_rate(self, details, device):
        """Computes the differential token rate

        Arguments
        ---------
        details : dict
            The evaluation details
            Keys:
                "pred": ASR predictions for the TTS sample
                "pred_ref": ASR predictions for the ground
                truth
        device : str | torch.device
            The device to use

        Returns
        -------
        result: dict
            A dictionary with the following keys

            dwer : torch.Tensor
                The differential Word Error Rate (dWER)
            dcer : torch.Tensor
                The differential Character Error Rate (dCER)

        """
        ids = range(1, len(details["pred"]) + 1)
        wer_metric, cer_metric = init_asr_metrics()
        pred = self._replace_blanks(details["pred"])
        pred_ref = self._replace_blanks(details["pred_ref"])
        wer_metric.append(ids, pred, pred_ref)
        cer_metric.append(ids, pred, pred_ref)
        dwer = torch.tensor(
            [score["WER"] for score in wer_metric.scores], device=device
        )
        dcer = torch.tensor(
            [score["WER"] for score in cer_metric.scores], device=device
        )
        return {"dwer": dwer, "dcer": dcer}

    def _replace_blanks(self, preds):
        """Replaces blanks with single spaces, preventing an exception
        in the case of an unintelligible sample

        Arguments
        ---------
        """
        return [" " if item == "" else item for item in preds]


class EncoderDecoderASRSpeechEvaluator(ASRSpeechEvaluator):
    """A speech evaluator implementation based on ASR.
    Computes the Word Error Rate (WER), Character Error Rate (CER)
    and a few other metrics

    Arguments
    ---------
    source : str
        The name of the HuggingFace repository to use or the path to the model
    sample_rate : int
        The audio sample rate this evaluator expects
    *args : list
        Additional arguments (passed through)
    **kwargs : dict
        Additional keyword arguments (passed through)
    """

    def __init__(self, source, sample_rate=None, *args, **kwargs):
        super().__init__(sample_rate=sample_rate)
        self.asr = EncoderDecoderASR.from_hparams(source, *args, **kwargs)
        self.device = next(self.asr.mods.parameters()).device

    def evaluate_samples(self, wavs, length, text, sample_rate):
        """Evaluates a batch of samples

        Arguments
        ---------
        wavs : torch.Tensor
            A batch of waveforms
        length : torch.Tensor
            Relative lengths
        text : list
            Text labels corresponding to the waveforms
        sample_rate : int
            The sample rate of the waveforms

        Returns
        -------
        results : dict
            The evaluation results
        """
        wavs = self.resample(wavs, sample_rate)
        if text is None:
            raise ValueError("This evaluator requires ground-truth text")
        predicted_words, scores, log_probs = self.transcribe_batch_with_details(
            wavs, length
        )
        ids = range(1, len(wavs) + 1)
        wer_metric, cer_metric = init_asr_metrics()
        wer_metric.append(ids, predicted_words, text)
        cer_metric.append(ids, predicted_words, text)
        wer = torch.tensor(
            [score["WER"] for score in wer_metric.scores], device=wavs.device
        )
        cer = torch.tensor(
            [score["WER"] for score in cer_metric.scores], device=wavs.device
        )
        prob_mean = log_probs.exp().mean(dim=-1)
        return {
            "wer": wer,
            "cer": cer,
            "beam_score": scores,
            "prob_mean": prob_mean,
            "pred": predicted_words,
            "target": text,
        }

    def transcribe_batch_with_details(self, wavs, wav_lens):
        """Transcribes the input audio into a sequence of words

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = EncoderDecoderASR.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.Tensor
            A tensor of waveforms
        wav_lens : torch.Tensor
            Relative lengths


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
            hyps = [
                token_seq.tolist() if torch.is_tensor(token_seq) else token_seq
                for token_seq in hyps
            ]
            predicted_words = [
                self.asr.tokenizer.decode_ids(token_seq) for token_seq in hyps
            ]
        return predicted_words, best_scores, best_log_probs

    def to(self, device):
        """Transfers this module to the spcieifed device

        Arguments
        ---------
        device : str | torch.Device
            the target device

        Returns
        -------
        result : EncoderDecoderASRSpeechEvaluator
            The evaluator, on the correct device
        """
        self.asr = self.asr.to(device)
        return self


class WhisperASRSpeechEvaluator(ASRSpeechEvaluator):
    """A speech evaluator implementation based on Whisper ASR

    Arguments
    ---------
    source : str
        The source directory
    savedir : str, optional
        The path where Whisper will be saved
    sample_rate: int, optional
        The audio sample rate
    min_decode_ratio : float, optional
        The minimum decode ratio
    max_decode_ratio : float, optional
        The maximum decode ratio
    run_opts : dict, optional
        Run options for the Whisper model
    unbatch : bool, optional
        If enabled, which is the default, the implementation
        will evaluate samples one by one with a batch size of
        1 and then "reassemble" the original batch. This is
        sometimes needed because batched inference has been
        found to result in decreased performance, primarily
        due to masks not being applied to convolutional layers
    """

    def __init__(
        self,
        source,
        savedir=None,
        sample_rate=22050,
        min_decode_ratio=0.0,
        max_decode_ratio=1.0,
        run_opts=None,
        unbatch=True,
    ):
        super().__init__(sample_rate=sample_rate)
        if run_opts is None:
            run_opts = {}
        if savedir is None:
            savedir = "."
        self.model = Whisper(
            source,
            savedir,
            sample_rate,
            freeze=True,
            freeze_encoder=True,
        )
        self.model.tokenizer.set_prefix_tokens("english", "transcribe", False)
        self.searcher = S2SWhisperGreedySearcher(
            self.model,
            min_decode_ratio=min_decode_ratio,
            max_decode_ratio=max_decode_ratio,
        )
        device = run_opts.get("device", next(self.model.parameters()).device)
        self.unbatch = unbatch
        self.to(device)

    def evaluate_samples(self, wavs, length, text, sample_rate):
        """Evaluates a batch of samples

        Arguments
        ---------
        wavs : torch.Tensor
            A batch of waveforms
        length : torch.Tensor
            Relative lengths
        text : list
            Text labels corresponding to the waveforms
        sample_rate : int
            The sample rate of the waveforms

        Returns
        -------
        results : dict
            The evaluation results
        """
        if self.unbatch:
            batch_size = len(wavs)
            length_abs = (length * wavs.size(1)).int()
            results = [
                self._evaluate_samples(
                    wavs[idx : idx + 1, : length_abs[idx].item()],
                    torch.ones(1, device=wavs.device),
                    text[idx : idx + 1],
                    sample_rate,
                )
                for idx in range(batch_size)
            ]
            result = {
                "wer": torch.stack(
                    [result["wer"] for result in results]
                ).squeeze(-1),
                "cer": torch.stack(
                    [result["cer"] for result in results]
                ).squeeze(-1),
                "pred": [result["pred"][0] for result in results],
                "target": text,
            }
            return result
        else:
            return self._evaluate_samples(wavs, length, text, sample_rate)

    def _evaluate_samples(self, wavs, length, text, sample_rate):
        """Evaluates a batch of samples. This function is meant
        to be used internally. evaluate_samples will call
        it multiple times if unbatch is enabled.

        Arguments
        ---------
        wavs : torch.Tensor
            A batch of waveforms
        length : torch.Tensor
            Relative lengths
        text : list
            Text labels corresponding to the waveforms
        sample_rate : int
            The sample rate of the waveforms

        Returns
        -------
        results : dict
            The evaluation results
        """
        if text is None:
            raise ValueError("This evaluator requires ground-truth text")
        wavs = self.resample(wavs, sample_rate)
        wavs = self.model.pad_or_trim(wavs)
        mels = self.model.log_mel_spectrogram(wavs)
        enc_out = self.model.forward_encoder(mels)
        predicted_words, _, _, _ = self.searcher(enc_out.detach(), length)
        predicted_words = self.model.tokenizer.batch_decode(
            predicted_words, skip_special_tokens=True
        )
        predicted_words = [self.normalize(text) for text in predicted_words]
        ids = range(1, len(wavs) + 1)
        wer_metric, cer_metric = init_asr_metrics()
        wer_metric.append(ids, predicted_words, text)
        cer_metric.append(ids, predicted_words, text)
        wer = torch.tensor(
            [score["WER"] for score in wer_metric.scores], device=wavs.device
        )
        cer = torch.tensor(
            [score["WER"] for score in cer_metric.scores], device=wavs.device
        )
        return {
            "wer": wer,
            "cer": cer,
            "pred": predicted_words,
            "target": text,
        }

    def normalize(self, text):
        """Performs text normalization (uppercase, remove whitespace,
        remove punctuation)

        Arguments
        ---------
        text : str
            Unnormalized text

        Returns
        -------
        text : str
            Normalized text
        """
        text = text.upper()
        text = text.strip()
        text = RE_PUNCTUATION.sub("", text)
        return text

    def to(self, device):
        """Transfers this module to the spcieifed device

        Arguments
        ---------
        device : str | torch.Device
            the target device

        Returns
        -------
        result : EncoderDecoderASRSpeechEvaluator
            The evaluator, on the correct device
        """
        self.model = self.model.to(device)
        return self


class SpkSimWavLM(SpeechEvaluator):
    """A speaker similarity evaluator based on WavLM / XVector

    Arguments
    ---------
    source : str
        The model hub to use
    savedir : str
        The path where the model will be saved
    model_sample_rate : int, optional
        The sample rate to which all samples will be resampled
        before being processed
    run_opts : dict
        Run options for the similarity model
    *args : list
        Additional arguments (passed through)
    **kwargs : dict
        Additional keyword arguments (passed through)
    """

    def __init__(
        self,
        source,
        savedir,
        model_sample_rate=16000,
        run_opts=None,
        *args,
        **kwargs,
    ):
        if not has_transformers:
            raise ValueError(
                "Unable to use the SpkSimWavLM evaluator because the "
                "transformers library is not enabled"
            )
        if run_opts is None:
            run_opts = {}
        device = run_opts.get("device")
        self.model = AutoModelForAudioXVector.from_pretrained(
            source, cache_dir=savedir, *args, **kwargs
        )
        if device is not None:
            self.model = self.model.to(device)

        self.model.eval()
        self.model_sample_rate = model_sample_rate
        self.device = next(self.model.parameters()).device

    def evaluate(
        self,
        wavs,
        length,
        text=None,
        wavs_ref=None,
        length_ref=None,
        sample_rate=None,
        sample_rate_ref=None,
    ):
        # Resample
        if sample_rate is not None:
            wavs = torchaudio.functional.resample(
                wavs, orig_freq=sample_rate, new_freq=self.model_sample_rate
            )
        if sample_rate_ref is not None:
            wavs_ref = torchaudio.functional.resample(
                wavs_ref,
                orig_freq=sample_rate_ref,
                new_freq=self.model_sample_rate,
            )

        # Concatenate
        batch_size, wavs_max_len = wavs.shape
        _, wavs_ref_max_len = wavs_ref.shape
        length_abs = length * wavs_max_len
        length_ref_abs = length_ref * wavs_ref_max_len
        max_len = max(wavs_max_len, wavs_ref_max_len)
        wavs, _ = pad_right_to(wavs, (batch_size, max_len))
        wavs_ref, _ = pad_right_to(wavs_ref, (batch_size, max_len))
        audio = torch.cat([wavs, wavs_ref])

        length_cat_abs = torch.cat([length_abs, length_ref_abs])
        # Attention mask
        attention_mask = None
        attention_mask = length_to_mask(
            length_cat_abs.int()
        ).long()  # 0 for masked tokens
        # Forward
        embs = self.model(
            input_values=audio,
            attention_mask=attention_mask,
            output_attentions=False,
        ).embeddings
        hyp_embs, ref_embs = embs.split([len(wavs), len(wavs_ref)])
        scores = torch.nn.functional.cosine_similarity(
            hyp_embs, ref_embs, dim=-1
        )

        return SpeechEvaluationResult(scores, {"score": scores})


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


def init_asr_metrics():
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


class UTMOSSpeechEvaluator(SpeechEvaluator):
    """The UTMOS speech evaluator wrapper

    Github: https://github.com/sarulab-speech/UTMOS22
    HuggingFace: https://huggingface.co/spaces/sarulab-speech/UTMOS-demo


    Arguments
    ---------
    source : str, optional
        The WavLM source
    save_path : str | path-like, optional
        The path where the model will be saved
    model_name : str
        The name of the model hub
    model_url : str
        The model URL (if applicable)
    domain_id : int
        The domain ID of the underlying model
    judge_id : int
        The judge ID to use (given UTMOS was trained as an ensemble
        of judges)
    run_opts: dict, optional
        The run options
    sample_rate : int
        The sample rate of the underlying model
    """

    def __init__(
        self,
        source=None,
        save_path=None,
        model_name=None,
        model_url=None,
        domain_id=None,
        judge_id=None,
        run_opts=None,
        sample_rate=16000,
    ):
        super().__init__(sample_rate=sample_rate)
        self.model = UTMOSModel(
            source=source,
            save_path=save_path,
        )
        if run_opts is not None:
            device = run_opts.get("device")
            if device:
                self.model = self.model.to(device)
        fetch(model_name, model_url, save_path)
        model_path = Path(save_path) / model_name
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.domain_id = domain_id
        self.judge_id = judge_id

    def evaluate(
        self,
        wavs,
        length,
        text=None,
        wavs_ref=None,
        length_ref=None,
        sample_rate=None,
        sample_rate_ref=None,
    ):
        """Evaluates a batch of waveforms using UTMOS

        Arguments
        ---------
        wavs: torch.Tensor
            the waveforms to evaluate
        length: torch.Tensor
            relative lengths (a 1-D tensor)
        text : list, optional
            Ground truth text. Ignored for UTMOS.
        wavs_ref : torch.Tensor
            the reference waveforms. Ignored for UTMOS.
        length_ref : torch.Tensor
            the reference waveform lengths. Ignored for UTMOS.
        sample_rate : int, optional
            The sample rate of the audio. If not provided,
            the audio is assumed to be at the same sample
            rate as the model
        sample_rate_ref : int, optional
            The sample rate of the reference samples. Ignored for UTMOS.

        Returns
        -------
        result : SpeechEvaluationResult
            an aggregated speech evaluation result with a score
            for each item
        """
        wavs = self.resample(wavs, sample_rate=sample_rate)
        domain_id, judge_id = None, None
        if self.domain_id is not None:
            domain_id = (
                torch.ones(len(wavs), device=wavs.device) * self.domain_id
            )
        if self.judge_id is not None:
            judge_id = torch.ones(len(wavs), device=wavs.device) * self.judge_id

        scores = self.model(wav=wavs, domain_id=domain_id, judge_id=judge_id)
        return SpeechEvaluationResult(score=scores, details={"utmos": scores})


def vocoder_to_device(vocoder, device):
    """A fix for vocoders that do not properly handle
    the .to() function and require the device to be set manually

    Arguments
    ---------
    vocoder : torch.nn.Module
        a vocoder
    device : str | torch.Device
        the target device
    """
    if hasattr(vocoder, "model") and hasattr(vocoder.model, "device"):
        vocoder.model.device = device
    elif hasattr(vocoder, "device"):
        vocoder.device = device


class Tracker:
    """A tracker that makes it possible to resume evaluation

    Arguments
    ---------
    file_name : str | path-like
        The path to the tracker file"""

    def __init__(self, file_name):
        self.file_name = Path(file_name)

    def mark_processed(self, item_id):
        """Marks the specified file as processed

        Arguments
        ---------
        item_id : str|enumerable
            The item ID or a list of IDS
        """
        if isinstance(item_id, str):
            item_id = [item_id]
        with open(self.file_name, "a+") as tracker_file:
            for item in item_id:
                print(item, file=tracker_file)

    def filter(self, dataset):
        """Filters a dataset using the tracker file

        Arguments
        ---------
        dataset : speechbrain.dataio.dataset.DynamicItemDataset
            A dataset

        Returns
        -------
        dataset : speechbrain.dataio.dataset.DynamicItemDataset
            The dataset, possibly filtered
        """
        if self.file_name.exists():
            with open(self.file_name) as tracker_file:
                processed_ids = set(line.strip() for line in tracker_file)
                remaining_ids = [
                    data_id
                    for data_id in dataset.data_ids
                    if data_id not in processed_ids
                ]
                logger.info(
                    "Tracker %s already exists, %d items already processed, %d items remaining",
                    self.file_name,
                    len(processed_ids),
                    len(remaining_ids),
                )
                dataset = FilteredSortedDynamicItemDataset(
                    dataset, remaining_ids
                )
        else:
            logger.info(
                "Tracker %s does not exist, evaluating from the beginning"
            )
        return dataset

    def get_processed(self):
        """Retrieves the IDs of items that have been processed

        Returns
        -------
        processed_ids : list
            The list of file IDs
        """
        if self.file_name.exists():
            with open(self.file_name, "r") as tracker_file:
                processed_ids = [line.strip() for line in tracker_file]
        else:
            processed_ids = []
        return processed_ids


class CommandError(Exception):
    """Thrown when an external command returns an error

    Arguments
    ---------
    cmd : str
        The command that was run
    output : str
        The captured standard output stream
    err : str
        The captured standard error stream
    return_code : int
        The return code"""

    def __init__(self, cmd, output, err, return_code):
        super().__init__(
            f"Command {cmd} returned code {return_code}\n"
            f"Output: {output}\n"
            f"Errors: {err}"
        )
        self.cmd = cmd
        self.output = output


class SpeechEvaluationMetricStats(MetricStats):
    """An aggregate metric combining multiple speech evaluators

    Arguments
    ---------
    hparams : dict | SimpleNamespace | object
        Raw hyperparameters for evaluation

    device : str
        The device on which evaluation will be performed

    """

    def __init__(self, hparams, device="cpu"):
        if isinstance(hparams, dict):
            hparams = SimpleNamespace(**hparams)
        self.hparams = hparams
        self.device = device
        modules = self.hparams.modules
        self.modules = nn.ModuleDict(modules).to(self.device)
        self.enabled_evaluators = set(self.hparams.evaluations.split(","))
        evaluators = hparams.evaluators
        if evaluators:
            self.evaluators = {
                key: evaluator_f(run_opts={"device": device})
                for key, evaluator_f in evaluators.items()
                if key in self.enabled_evaluators
            }
        else:
            self.evaluators = {}

        if not self.evaluators:
            logger.warn(
                "No evaluators were defined - this run will produce samples only"
            )

        self.attention = []

    def on_evaluation_start(self, output_folder="eval"):
        """Invoked at the beginning of the evaluation cycle.

        Arguments
        ---------
        output_folder : str | path-like
            The folder to which results will be output

        """
        logger.info("Starting evaluation")
        output_folder = Path(output_folder)
        self.output_folder = (
            output_folder
            if output_folder.is_absolute()
            else self.hparams.output_folder / output_folder
        )
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.files = []
        details_keys = list(self.evaluators.keys())
        self.details = {evaluator_key: [] for evaluator_key in details_keys}
        self.read_reports()
        self.create_reports()
        self.item_ids = []

    def on_evaluation_end(self):
        """Invoked at the beginning of the evaluation cycle. The default
        implementation is a no-op
        """
        logger.info("Ending evaluation")
        self.write_summary()

    def create_reports(self):
        """Creates report files and report writers"""
        self.report_files = {}
        self.report_writers = {}
        for evaluator_key in self.enabled_evaluators:
            columns = self.get_report_columns(evaluator_key)
            file_name = self.output_folder / f"{evaluator_key}.csv"
            self.files.append(file_name)
            resume = file_name.exists() and file_name.stat().st_size > 0
            report_file = open(file_name, "a+")
            self.report_files[evaluator_key] = report_file
            writer = csv.DictWriter(report_file, columns)
            if not resume:
                writer.writeheader()
            self.report_writers[evaluator_key] = writer

    def read_reports(self):
        """Invoked when resuming"""
        for evaluator_key in self.enabled_evaluators:
            file_name = self.output_folder / f"{evaluator_key}.csv"
            if file_name.exists():
                logger.info("%s exists, reading")
                with open(file_name) as report_file:
                    reader = csv.DictReader(report_file)
                    for row in reader:
                        del row["uttid"]
                        row = {
                            key: handle_number(value)
                            for key, value in row.items()
                        }
                        self.details[evaluator_key].append(row)

    def get_tracker_file_name(self):
        """Determines the file name of the tracker file"""
        suffix = (
            f"_{self.hparams.eval_suffix}" if self.hparams.eval_suffix else ""
        )
        file_name = f"tracker_{self.hparams.eval_dataset}{suffix}.txt"
        return self.output_folder / file_name

    def get_report_columns(self, evaluator_key):
        """Returns the columns for the specified evaluator

        Arguments
        ---------
        evaluator_key : str
            the identifier of the evaluator

        Returns
        -------
        columns : list[str]
            a list of column headers
        """
        bogus_wavs = torch.randn(2, 10000, device=self.device)
        bogus_length = torch.tensor([1.0, 1.0], device=self.device)
        evaluator = self.evaluators[evaluator_key]
        result = evaluator.evaluate(
            wavs=bogus_wavs,
            length=bogus_length,
            text=["BOGUS"] * len(bogus_wavs),
            wavs_ref=bogus_wavs,
            length_ref=bogus_length,
        )

        return ["uttid"] + list(result.details.keys())

    def append(self, ids, wav, length, text, wav_ref, length_ref):
        """Appends the result of a single item

        Arguments
        ---------
        ids : str
            Utterance IDs
        wav : torch.Tensor
            Synthesized waveforms
        length : torch.Tensor
            Relative lengths of the synthesized waveforms
        text : list
            Ground truth text
        wav_ref : torch.Tensor
            Reference (ground truth) waveforms
        length_ref : torch.Tensor
            Reference lengths
        """
        with torch.no_grad():
            self.item_ids.extend(ids)
            for evaluator_key, evaluator in self.evaluators.items():
                result = evaluator.evaluate(
                    wavs=wav,
                    length=length,
                    text=text,
                    wavs_ref=wav_ref,
                    length_ref=length_ref,
                    sample_rate_ref=self.hparams.sample_rate,
                    sample_rate=self.hparams.model_sample_rate,
                )
                details = undo_batch(result.details)
                self.write_result(evaluator_key, ids, details)
                self.details[evaluator_key].extend(details)

    def write_result(self, evaluator_key, ids, details):
        """Outputs the result details to the report for the specified evaluator

        Arguments
        ---------
        evaluator_key : str
            The evaluator key
        ids : list
            The list of IDs
        details : list
            a list of evaluation details, one dictionary per item
        """
        writer = self.report_writers[evaluator_key]
        for uttid, details_item in zip(ids, details):
            report_details = {
                "uttid": uttid,
                **details_item,
            }
            writer.writerow(ascii_only(flatten(report_details)))
        self.report_files[evaluator_key].flush()

    def write_summary(self, file_name=None):
        """Outputs summarized statistics

        Arguments
        ---------
        file_name : str | path-like
            An alternative path to save the file
        """
        summary = self.summarize()
        if file_name is None:
            file_name = self.output_folder / "summary.json"
        self.files.append(file_name)
        with open(file_name, "w") as output_file:
            json.dump(summary, output_file, indent=4)

    def summarize(self, field=None):
        """Computes the summarized statistics

        Arguments
        ---------
        field : str, optional
            If specified, it will return a specific field

        Returns
        -------
        result : dict | float
            The summary - or the specified field from the sum
        """
        result = {
            f"{evaluator_key}_{stat_key}": value
            for evaluator_key in self.enabled_evaluators
            if evaluator_key in self.details
            for metric_key in self.hparams.eval_summary[evaluator_key][
                "descriptive"
            ]
            for stat_key, value in descriptive_statistics(
                items=self.details[evaluator_key],
                key=metric_key,
            ).items()
        }
        if field is not None:
            result = result[field]
        return result

    def clear(self):
        """Deletes all the files that have been created"""
        for file_name in self.files:
            file_name.unlink()


RE_INTEGER = re.compile(r"^-?\d+$")
RE_FLOAT = re.compile(r"^-?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$")


def handle_number(value):
    """Converts a value to a number, if applicable. Strings
    that look like integers or floats will be converted to integers
    or floats.

    Arguments
    ---------
    value : str
        a string value

    Returns
    -------
    result : object
        The processed result"""
    if RE_INTEGER.match(value):
        value = int(value)
    elif RE_FLOAT.match(value):
        value = float(value)
    return value


def descriptive_statistics(items, key):
    """Computes descriptive statistics for the summary

    Arguments
    ---------
    items : list
        a list of dictionaries with metric values for each item
    key : str
        The key of the metric for which the statistics will be computed

    Returns
    -------
    statistics : dict
        The desccriptive statistics computed
            <key>_mean : the arithmetic mean
            <key>_std : the standard deviation
            <key>_min : the minimum value
            <key>_max : the maximum value
            <key>_median : the median value
            <key>_q1 : the first quartile
            <key>_q3 : the third quartile
            <key>_iqr : the interquartile ratio
    """
    values = torch.tensor([item[key] for item in items])
    quantiles = torch.tensor([0.25, 0.5, 0.75])
    q1, median, q3 = values.quantile(quantiles)
    stats = {
        "mean": values.mean(),
        "std": values.std(),
        "min": values.min(),
        "max": values.max(),
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }
    return {
        f"{key}_{stat_key}": value.item() for stat_key, value in stats.items()
    }


RE_NON_ASCII = re.compile(r"[^\x00-\x7F]+")


def ascii_only(values):
    """Removes any non-ASCII characters from a dictionary

    Arguments
    ---------
    values : dict
        A dictionary of values

    Returns
    -------
    result : dict
        The same dictionary - but with non-ASCII strings removed"""
    return {
        key: RE_NON_ASCII.sub("", value) if isinstance(value, str) else value
        for key, value in values.items()
    }


def flatten(value):
    """Converts tensors to scalars and lists of strings to strings

    Arguments
    ---------
    value : dict
        the dictionary to flatten

    Returns
    -------
    result : dict
        a flattened dictionary
    """
    return {
        key: item_value.item() if torch.is_tensor(item_value) else item_value
        for key, item_value in value.items()
    }
