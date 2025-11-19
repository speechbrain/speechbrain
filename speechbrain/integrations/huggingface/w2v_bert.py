#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration of HuggingFace w2v-bert-2.0 https://huggingface.co/facebook/w2v-bert-2.0

    - Input  : wav [B, T] (float32, 16)
    - Output  : features [B, T_feat, C] (C = dim of the model, ex. 1024)

Use of w2v-bert-2.0 :
    wav2vec2: !new:speechbrain.integrations.huggingface.w2v_bert.W2VBert
        source: !ref <wav2vec2_hub>
        save_path: !ref <save_folder>/wav2vec2_checkpoint
        freeze: !ref <wav2vec2_frozen>
        freeze_feature_extractor: False
        apply_spec_augment: False
        output_norm: False
        output_all_hiddens: False

Authors :
 * Bouziane Maryem, 2025
 * Salima Mdhaffar, 2025
 * Yannick Estève, 2025
"""

from typing import Optional

import torch
import torch.nn.functional as F

from speechbrain.integrations.huggingface.huggingface import (
    HFTransformersInterface,
)
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class W2VBert(HFTransformersInterface):
    """Parameters
    source : str
        Name on Hugging Face or local path (e.g. "facebook/w2v-bert-2.0").
    save_path : str
        Directory used to cache / save the model.
    output_norm : bool, optional
        If True, applies a layer_norm to the output features.
    freeze : bool, optional
        If True, freezes all model parameters.
    freeze_feature_extractor : bool, optional
        If True, freezes only the convolutional frontend (if freeze=False).
    apply_spec_augment : bool, optional
        Enables or disables the internal SpecAugment of the HF model (if available).
    output_all_hiddens : bool, optional
        If True, returns all Transformer layers [L+1, B, T_feat, C] instead of only the last one [B, T_feat, C].
    sample_rate : int, optional
        Sampling rate of the audio waveforms.
        If None, attempts to read feature_extractor.sampling_rate, otherwise defaults to 16000.
    **kwargs :
        Passed to from_pretrained(source, **kwargs).
        
    Output
    -------
    - If output_all_hiddens = False: [B, T_feat, C]
    - If output_all_hiddens = True: [L+1, B, T_feat, C]

    """
    def __init__(
        self,
        source: str,
        save_path: str,
        output_norm: bool = False,
        freeze: bool = True,
        freeze_feature_extractor: bool = False,
        apply_spec_augment: bool = False,
        output_all_hiddens: bool = False,
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        # Initialise HFTransformersInterface (charge le modèle HF)
        super().__init__(
            source=source,
            save_path=save_path,
            freeze=freeze,
            **kwargs,
        )

        # Charger le feature_extractor HF pour gérer normalisation / padding
        self.load_feature_extractor(source, cache_dir=save_path)

        # Déterminer le sample_rate à utiliser
        if sample_rate is not None:
            self.sample_rate = sample_rate
        else:
            self.sample_rate = getattr(self.feature_extractor, "sampling_rate", 16000)
        logger.info(
            f"[W2VBert] sample_rate utilisé pour le feature_extractor = {self.sample_rate}"
        )

        # SpecAugment interne (si supporté par le modèle)
        if hasattr(self.model.config, "apply_spec_augment"):
            self.model.config.apply_spec_augment = apply_spec_augment

        self.output_norm = output_norm
        self.output_all_hiddens = output_all_hiddens

        self.freeze_feature_extractor = freeze_feature_extractor
        if not self.freeze and self.freeze_feature_extractor:
            if hasattr(self.model, "feature_extractor"):
                logger.warning(
                    "W2VBert - Frozen feature extractor (freeze_feature_extractor=True)."
                )
                self.model.feature_extractor.eval()
                for p in self.model.feature_extractor.parameters():
                    p.requires_grad = False
            else:
                logger.warning(
                    "W2VBert - freeze_feature_extractor=True but the model HF "
                    "does not one `feature_extractor` attributes."
                )

    def forward(self, wav: torch.Tensor, wav_lens: Optional[torch.Tensor] = None):
        """Forward: wav [B, T] -> features HF.

        If freeze=True, gradient is deactivated (no_grad).
        """
        if self.freeze:
            with torch.no_grad():
                return self._forward_hf(wav, wav_lens)
        return self._forward_hf(wav, wav_lens)

    def _forward_hf(self, wav: torch.Tensor, wav_lens: Optional[torch.Tensor]):
        """Internal call to HF model + feature_extractor.

        wav: [B, T].
        wav_lens: ratios [0, 1] (optional).
        """
        device = wav.device
        B, T = wav.shape

        wav_list = []
        if wav_lens is not None:
            for b in range(B):
                t_b = int(wav_lens[b].item() * T)
                t_b = max(t_b, 1)
                wav_list.append(wav[b, :t_b].detach().cpu())
        else:
            for b in range(B):
                wav_list.append(wav[b].detach().cpu())

        # feature_extractor HF: normalisation + padding + attention_mask
        inputs = self.feature_extractor(
            wav_list,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Call of HF model
        out = self.model(
            **inputs,
            output_hidden_states=self.output_all_hiddens,
        )

        if self.output_all_hiddens:
            # Tuple -> [L+1, B, T_feat, C]
            hidden_states = out.hidden_states
            out_tensor = torch.stack(list(hidden_states), dim=0)
            norm_shape = out_tensor.shape[-1:]
        else:
            # Last hidden state : [B, T_feat, C]
            out_tensor = out.last_hidden_state
            norm_shape = out_tensor.shape[-1:]

        # Normalisation of dimension of the output
        if self.output_norm:
            out_tensor = F.layer_norm(out_tensor, norm_shape)

        return out_tensor
