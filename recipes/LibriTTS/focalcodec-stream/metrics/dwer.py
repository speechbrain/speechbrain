"""Differential WER (dWER) (see https://arxiv.org/abs/1911.07953).

Authors
 * Luca Della Libera 2025
"""

import torch
import torchaudio
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

from speechbrain.decoders.seq2seq import S2SWhisperGreedySearcher
from speechbrain.integrations.huggingface import Whisper
from speechbrain.utils.metric_stats import ErrorRateStats, MetricStats

__all__ = ["DWER"]


SAMPLE_RATE = 16000


class DWER(MetricStats):
    """Differentiable Word Error Rate (dWER) metric.

    Arguments
    ---------
    model_hub : str
        Name of the HuggingFace Whisper checkpoint to load.
    sample_rate : int
        Sampling rate.
    save_path : str, optional
        Model cache directory.
    model : Any, optional
        Pre-initialized model.

    Example
    -------
    > import torch
    > device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    > sample_rate = 24000
    > ids = ["A", "B"]
    > hyp_sig = torch.randn(2, 2 * sample_rate, device=device)
    > ref_sig = torch.randn(2, 2 * sample_rate, device=device)
    > dwer = DWER("openai/whisper-small", sample_rate)
    > dwer.append(ids, hyp_sig, ref_sig)
    > print(dwer.summarize("error_rate"))
    > print(dwer.summarize("WER"))
    > print(dwer.summarize("error_rate_char"))
    > print(dwer.summarize("CER"))

    """

    def __init__(
        self,
        model_hub,
        sample_rate,
        save_path=HUGGINGFACE_HUB_CACHE,
        model=None,
    ):
        self.sample_rate = sample_rate
        self.model = model
        if model is None:
            self.model = Whisper(
                model_hub,
                save_path,
                SAMPLE_RATE,
                freeze=True,
                freeze_encoder=True,
            ).cpu()
        self.searcher = S2SWhisperGreedySearcher(
            self.model,
            min_decode_ratio=0.0,
            max_decode_ratio=1.0,
        )
        self.model.tokenizer.set_prefix_tokens("english", "transcribe", False)
        self.wer_computer = ErrorRateStats()
        self.cer_computer = ErrorRateStats(split_tokens=True)

    def clear(self):
        self.wer_computer.clear()
        self.cer_computer.clear()

    @torch.no_grad()
    def append(self, ids, hyp_audio, ref_audio, lens=None):
        assert hyp_audio.shape == ref_audio.shape
        assert hyp_audio.ndim == 2

        # Concatenate
        audio = torch.cat([hyp_audio, ref_audio])
        if lens is not None:
            lens = torch.cat([lens, lens])
        else:
            lens = torch.ones(audio.shape[0], device=audio.device)

        # Resample
        audio = torchaudio.functional.resample(
            audio, self.sample_rate, SAMPLE_RATE
        )

        self.model.to(hyp_audio.device)
        self.model.eval()

        # Forward
        enc_out = self.model.forward_encoder(self.model._get_mel(audio))
        text, _, _, _ = self.searcher(enc_out, lens)
        text = self.model.tokenizer.batch_decode(text, skip_special_tokens=True)
        text = [self.model.tokenizer._normalize(x).split(" ") for x in text]
        hyp_text = text[: hyp_audio.shape[0]]
        ref_text = text[hyp_audio.shape[0] :]

        # Compute WER
        self.wer_computer.append(ids, hyp_text, ref_text)
        self.cer_computer.append(ids, hyp_text, ref_text)

    def summarize(self, field=None):
        wer_summary = self.wer_computer.summarize()
        cer_summary = self.cer_computer.summarize()
        wer_summary["CER"] = wer_summary["error_rate_char"] = cer_summary[
            "error_rate"
        ]
        if field is None:
            return wer_summary
        return wer_summary[field]

    def write_stats(self, filestream, verbose=False):
        self.wer_computer.write_stats(filestream)
