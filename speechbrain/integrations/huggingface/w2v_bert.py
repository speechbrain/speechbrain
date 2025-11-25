"""This lobe enables the integration of HuggingFace pretrained w2v-bert-2.0 models.

Reference: https://arxiv.org/abs/2312.05187
Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Maryem Bouziane 2025
 * Salima Mdhaffar 2025
 * Yannick Estève 2025
"""

from typing import Optional

import torch
import torch.nn.functional as F

from speechbrain.integrations.huggingface.huggingface import (
    HFTransformersInterface,
)
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class W2VBert(HFTransformersInterface):
    """This lobe enables the integration of HuggingFace and SpeechBrain
    pretrained w2v-bert-2.0 models.

    Source paper w2v-BERT: https://arxiv.org/abs/2312.05187
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model can be used as a fixed feature extractor or can be finetuned. It
    will download automatically the model from HuggingFace or use a local path.

    Arguments
    ---------
    source : str
        HuggingFace hub name or local path, e.g. "facebook/w2v-bert-2.0".
    save_path : str
        Path (dir) used to cache / save the model.
    output_norm : bool (default: False)
        If True, a layer_norm is applied to the output features.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model is trained
        alongside the rest of the pipeline.
    freeze_feature_extractor : bool (default: False)
        When ``freeze`` is False and this flag is True, only the convolutional
        feature extractor is frozen.
    apply_spec_augment : bool (default: False)
        If True, the internal SpecAugment of the HF model is enabled.
    output_all_hiddens : bool (default: False)
        If True, the forward method outputs the hidden states from all
        transformer layers.
    sample_rate : int or None (default: None)
        Expected sampling rate of the input waveforms. If None, the sampling
        rate is read from the HF feature extractor when available, otherwise
        it defaults to 16000.
    **kwargs
        Extra keyword arguments passed to the `from_pretrained` function.

    Example
    -------
    >>> inputs = torch.rand([2, 16000])
    >>> model_hub = "facebook/w2v-bert-2.0"
    >>> save_path = "savedir"
    >>> model = W2VBert(model_hub, save_path)
    >>> outputs = model(inputs)
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
        super().__init__(
            source=source,
            save_path=save_path,
            freeze=freeze,
            **kwargs,
        )

        # We load the HF feature extractor
        self.load_feature_extractor(source, cache_dir=save_path)

        # We determine the sampling rate to be used
        if sample_rate is not None:
            self.sample_rate = sample_rate
        else:
            self.sample_rate = getattr(
                self.feature_extractor, "sampling_rate", 16000
            )

        logger.info(
            f"[W2VBert] feature_extractor sample_rate = {self.sample_rate}"
        )

        self.model.config.apply_spec_augment = apply_spec_augment

        self.output_norm = output_norm
        self.output_all_hiddens = output_all_hiddens

        self.freeze_feature_extractor = freeze_feature_extractor
        if not self.freeze and self.freeze_feature_extractor:
            logger.warning(
                "speechbrain.integrations.huggingface.w2v_bert - "
                "w2v-bert feature extractor is frozen."
            )
            self.model.feature_extractor.eval()
            for param in self.model.feature_extractor.parameters():
                param.requires_grad = False

    def forward(
        self,
        wav: torch.Tensor,
        wav_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Takes an input waveform and returns its corresponding w2v-BERT encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        wav_lens : torch.Tensor or None
            The relative length of the wav given in SpeechBrain format.

        Returns
        -------
        torch.Tensor
            w2v-BERT encoded features.
        """
        if self.freeze:
            with torch.no_grad():
                return self._forward_hf(wav, wav_lens)

        return self._forward_hf(wav, wav_lens)

    def _forward_hf(
        self,
        wav: torch.Tensor,
        wav_lens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Takes an input waveform and returns its corresponding w2v-BERT encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of padded audio signals to transform to features.
        wav_lens : torch.Tensor or None
            The relative length of the wav given in SpeechBrain format.

        Returns
        -------
        torch.Tensor
            w2v-BERT encoded features.
        """
        device = wav.device
        B, _ = wav.shape

        if wav_lens is not None:
            wav_list = undo_padding(
                wav.detach().cpu(),
                wav_lens.detach().cpu(),
            )
        else:
            wav_list = [
                wav[b].detach().cpu()
                for b in range(B)
            ]

        inputs = self.feature_extractor(
            wav_list,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = self.model(
            **inputs,
            output_hidden_states=self.output_all_hiddens,
        )

        if self.output_all_hiddens:
            out_tensor = torch.stack(list(out.hidden_states), dim=0)
            norm_shape = out_tensor.shape[-1:]
        else:
            out_tensor = out.last_hidden_state
            norm_shape = out_tensor.shape[-1:]

        if self.output_norm:
            out_tensor = F.layer_norm(out_tensor, norm_shape)

        return out_tensor
