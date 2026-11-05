"""UTokyo-SaruLab System for VoiceMOS Challenge 2022 (UTMOS) (see https://arxiv.org/abs/2204.02152).

Authors
 * Luca Della Libera 2025
"""

import torch
import torchaudio

from speechbrain.utils.metric_stats import MetricStats

__all__ = ["UTMOS"]


SAMPLE_RATE = 16000


class UTMOS(MetricStats):
    """UTMOS metric.

    Arguments
    ---------
    sample_rate : int
        Sampling rate.
    model : Any, optional
        Pre-initialized model.

    Example
    -------
    > import torch
    > sample_rate = 24000
    > ids = ["A", "B"]
    > hyp_sig = torch.randn(2, 2 * sample_rate)
    > utmos = UTMOS(sample_rate)
    > utmos.append(ids, hyp_sig)
    > print(utmos.summarize("average"))

    """

    def __init__(self, sample_rate, model=None):
        self.sample_rate = sample_rate
        self.model = model
        if model is None:
            self.model = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
            )
        self.clear()

    @torch.no_grad()
    def append(self, ids, sig, lens=None):
        assert sig.ndim == 2

        # Resample
        hyp_sig = torchaudio.functional.resample(
            sig, self.sample_rate, SAMPLE_RATE
        )

        self.model.to(hyp_sig.device)
        self.model.eval()

        # Forward
        scores = self.model(hyp_sig, SAMPLE_RATE)

        self.ids += ids
        self.scores += scores.cpu().tolist()
