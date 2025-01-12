import torch
import torch.nn.functional as F
import torchaudio
import os
import tqdm
import pickle
import numpy as np
import joblib

MIN_WAV_LEN = 720
MODEL_SR = 16000

class FairseqHuBERT(torch.nn.Module):
    def __init__(
            self, 
            feat_extractor_path, 
            layer, 
            km_path, 
            max_chunk=1600000,
            vocoder=None,
        ):
        super().__init__()
        import fairseq
        # Feature extractor
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [feat_extractor_path]
        )
        self.model = model[0]
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        # Quantizer
        km_model = joblib.load(km_path)
        self.C_np = km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)
        self.register_buffer("C", torch.from_numpy(self.C_np))
        self.register_buffer("Cnorm", torch.from_numpy(self.Cnorm_np))
        self.sample_rate = MODEL_SR
        self.vocoder = vocoder

    def encode(self, x, wav_lens=None):
        if self.task.cfg.normalize:
            x = F.layer_norm(x, x.shape)
        x = x.view(1, -1)

        feat = []
        for start in range(0, x.size(1), self.max_chunk):
            x_chunk = x[:, start : start + self.max_chunk]
            if x_chunk.size(1) < MIN_WAV_LEN:
                continue
            feat_chunk, _ = self.model.extract_features(
                source=x_chunk,
                padding_mask=None,
                mask=False,
                output_layer=self.layer,
            )
            feat.append(feat_chunk)
        feat = torch.cat(feat, 1).squeeze(0)
        dist = (
            feat.pow(2).sum(1, keepdim=True)
            - 2 * torch.matmul(feat, self.C)
            + self.Cnorm
        )
        return dist.argmin(dim=1).unsqueeze(0)

    def decode(self, tokens):
        if self.vocoder is None:
            raise ValueError("Vocoder is not set")
        return self.vocoder(tokens, dur_prediction = True)