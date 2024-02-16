"""Cosine similarity between speaker embeddings.

Authors
* Luca Della Libera 2024
"""

import os

import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition


__all__ = ["SpkSim"]


SAMPLING_RATE = 16000


class ComputeScore:
    def __init__(self, model_hub, save_path, sampling_rate):
        self.model = SpeakerRecognition.from_hparams(
            model_hub, savedir=save_path
        )
        self.sampling_rate = sampling_rate

    def __call__(self, hyp_audio, ref_audio, sampling_rate):
        # Resample
        hyp_audio = torchaudio.functional.resample(
            hyp_audio, sampling_rate, self.sampling_rate
        )
        ref_audio = torchaudio.functional.resample(
            ref_audio, sampling_rate, self.sampling_rate
        )

        # Forward
        self.model.device = hyp_audio.device
        self.model.to(hyp_audio.device)
        self.model.eval()
        score, _ = self.model.verify_batch(hyp_audio[None], ref_audio[None])
        return score.item()


root_folder = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.join(root_folder, "huggingface")

SpkSim = ComputeScore(
    "speechbrain/spkrec-ecapa-voxceleb", save_path, SAMPLING_RATE
)
