"""Cosine similarity between speaker embeddings.

Authors
 * Luca Della Libera 2025
"""

import torch
import torchaudio
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from transformers import AutoModelForAudioXVector

from speechbrain.dataio.dataio import length_to_mask
from speechbrain.utils.metric_stats import MetricStats

__all__ = ["SpkSimWavLM"]


SAMPLE_RATE = 16000


class SpkSimWavLM(MetricStats):
    """WavLM speaker similarity metric.

    Arguments
    ---------
    model_hub : str
        Name of the HuggingFace WavLM checkpoint to load.
    sample_rate : int
        Sampling rate.
    save_path : str, optional
        Model cache directory.
    model : Any, optional
        Pre-initialized model.

    Example
    -------
    > import torch
    > sample_rate = 24000
    > ids = ["A", "B"]
    > hyp_sig = torch.randn(2, 2 * sample_rate)
    > ref_sig = torch.randn(2, 2 * sample_rate)
    > spk_sim = SpkSimWavLM("microsoft/wavlm-base-sv", sample_rate)
    > spk_sim.append(ids, hyp_sig, ref_sig)
    > print(spk_sim.summarize("average"))

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
            self.model = AutoModelForAudioXVector.from_pretrained(
                model_hub, cache_dir=save_path
            )
        self.clear()

    @torch.no_grad()
    def append(self, ids, hyp_sig, ref_sig, lens=None):
        assert hyp_sig.shape == ref_sig.shape
        assert hyp_sig.ndim == 2

        # Concatenate
        sig = torch.cat([hyp_sig, ref_sig])
        if lens is not None:
            lens = torch.cat([lens, lens])

        # Resample
        sig = torchaudio.functional.resample(sig, self.sample_rate, SAMPLE_RATE)
        if sig.shape[-1] < 4880:
            sig = torch.nn.functional.pad(
                sig, [0, 4880 - sig.shape[-1]], mode="replicate"
            )

        self.model.to(hyp_sig.device)
        self.model.eval()

        # Attention mask
        attention_mask = None
        if lens is not None:
            abs_length = lens * sig.shape[-1]
            attention_mask = length_to_mask(
                abs_length.int()
            ).long()  # 0 for masked tokens

        # Forward
        embs = self.model(
            input_values=sig,
            attention_mask=attention_mask,
            output_attentions=False,
        ).embeddings

        hyp_embs, ref_embs = embs.split([len(hyp_sig), len(ref_sig)])
        scores = torch.nn.functional.cosine_similarity(
            hyp_embs, ref_embs, dim=-1
        )

        self.ids += ids
        self.scores += scores.cpu().tolist()
