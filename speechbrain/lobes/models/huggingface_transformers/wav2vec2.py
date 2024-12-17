"""This lobe enables the integration of huggingface pretrained wav2vec2 models.

Reference: https://arxiv.org/abs/2006.11477
Reference: https://arxiv.org/abs/1904.05862
Reference: https://arxiv.org/abs/2110.13900
Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Titouan Parcollet 2021
 * Boumadane Abdelmoumene 2021
 * Ha Nguyen 2023
"""

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices

from speechbrain.lobes.models.huggingface_transformers.huggingface import (
    HFTransformersInterface,
    make_padding_masks,
)
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class Wav2Vec2(HFTransformersInterface):
    """This lobe enables the integration of HuggingFace and SpeechBrain
    pretrained wav2vec2.0/Hubert models.

    Source paper wav2vec2.0: https://arxiv.org/abs/2006.11477
    Source paper Hubert: https://arxiv.org/abs/2106.07447
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model can be used as a fixed feature extractor or can be finetuned. It
    will download automatically the model from HuggingFace or use a local path.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
    save_path : str
        Path (dir) of the downloaded model.
    output_norm : bool (default: True)
        If True, a layer_norm (affine) will be applied to the output obtained
        from the wav2vec model.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    freeze_feature_extractor :  bool (default: False)
        When freeze = False and freeze_feature_extractor True, the feature_extractor module of the model is Frozen. If False
        all the wav2vec model will be trained including feature_extractor module.
    apply_spec_augment : bool (default: False)
        If True, the model will apply spec augment on the output of feature extractor
        (inside huggingface Wav2VecModel() class).
        If False, the model will not apply spec augment. We set this to false to prevent from doing it twice.
    output_all_hiddens : bool (default: False)
        If True, the forward function outputs the hidden states from all transformer layers.
        For example wav2vec2-base has 12 transformer layers and the output is of shape (13, B, T, C),
        where a projection of the CNN output is added to the beginning.
        If False, the forward function outputs the hidden states only from the last transformer layer.
    **kwargs
        Extra keyword arguments passed to the `from_pretrained` function.

    Example
    -------
    >>> inputs = torch.rand([10, 600])
    >>> model_hub = "facebook/wav2vec2-base-960h"
    >>> save_path = "savedir"
    >>> model = Wav2Vec2(model_hub, save_path)
    >>> outputs = model(inputs)
    """

    def __init__(
        self,
        source,
        save_path,
        output_norm=False,
        freeze=False,
        freeze_feature_extractor=False,
        apply_spec_augment=False,
        output_all_hiddens=False,
        **kwargs,
    ):
        super().__init__(
            source=source, save_path=save_path, freeze=freeze, **kwargs
        )

        self.model.config.apply_spec_augment = apply_spec_augment

        # We check if inputs need to be normalized w.r.t pretrained wav2vec2
        self.load_feature_extractor(source, cache_dir=save_path)
        self.normalize_wav = self.feature_extractor.do_normalize

        self.freeze_feature_extractor = freeze_feature_extractor
        if not self.freeze and self.freeze_feature_extractor:
            logger.warning(
                "speechbrain.lobes.models.huggingface_transformers.wav2vec2 - wav2vec 2.0 feature extractor is frozen."
            )
            self.model.feature_extractor.eval()
            for param in self.model.feature_extractor.parameters():
                param.requires_grad = False

        self.output_norm = output_norm
        self.output_all_hiddens = output_all_hiddens

    def _modify_state_dict(self, path, replaceables=["wav2vec2"]):
        """A custom loading ensures SpeechBrain compatibility for Pretrain and model
        de/serialization. Here, the scope is to remove '.wav2vec2' before loading.

        Arguments
        ---------
        path : str
            Checkpoint path, file name relative to the repo root.
        replaceables : List[str]
            State dict sub-keys that if found, shall be dropped (incl. the 'model.' parent key), elevating key structures.

        Returns
        -------
        modified_state_dict : see torch.load
            SpeechBrain-valid deserialized pretrained model.
        """
        modified_state_dict = {}
        orig_state_dict = torch.load(path, map_location="cpu")

        # We remove the .wav2vec2 in the state dict.
        for key, params in orig_state_dict.items():
            for tag in replaceables:
                if f"{tag}." in key:
                    save_key = key.replace(f"model.{tag}.", "")
                    modified_state_dict[save_key] = params
        return modified_state_dict

    def forward(self, wav, wav_lens=None):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        wav_lens : torch.Tensor
            The relative length of the wav given in SpeechBrain format.

        Returns
        -------
        Wav2vec encoded features.
        """

        # If we freeze, we simply remove all grads from the graph.
        if self.freeze:
            with torch.no_grad():
                return self.extract_features(wav, wav_lens)

        return self.extract_features(wav, wav_lens)

    def extract_features(self, wav, wav_lens=None):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        wav_lens : torch.Tensor
            The relative length of the wav given in SpeechBrain format.

        Returns
        -------
        out : torch.Tensor
            Wav2vec encoded features.
        """

        padding_mask = make_padding_masks(wav, wav_len=wav_lens)

        if self.normalize_wav:
            wav = F.layer_norm(wav, wav.shape[1:])

        # Extract wav2vec output
        out = self.model(
            wav,
            attention_mask=padding_mask,
            output_hidden_states=self.output_all_hiddens,
        )

        if self.output_all_hiddens:
            out = torch.stack(list(out.hidden_states), dim=0)
            norm_shape = out.shape[-3:]
        else:
            out = out.last_hidden_state
            norm_shape = out.shape

        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out, norm_shape[1:])

        return out


class Wav2Vec2Pretrain(HFTransformersInterface):
    """This lobe enables the integration of HuggingFace
    wav2vec2.0 models to be pretrained.

    Source paper: https://arxiv.org/abs/2006.11477
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The return is an HuggingFace format and the mask indices that contains:
    https://huggingface.co/transformers/model_doc/wav2vec2.html#wav2vec2forpretraining

    For instance, it returns the loss that can be accessed with .loss

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
    save_path : str
        Path (dir) of the downloaded model.
    mask_prob : float (default: 0.65)
        Probability of masking a given frame. Default is taken from the paper.
    mask_length : float (default: 10)
        Length (i.e. number of consecutive masked frames). Default is taken from
        the paper.
    normalize_wav : bool
        Whether to normalize input before processing.

    Example
    -------
    >>> inputs = torch.rand([10, 32000])
    >>> model_hub = "facebook/wav2vec2-base-960h"
    >>> save_path = "savedir"
    >>> model = Wav2Vec2Pretrain(model_hub, save_path)
    >>> outputs, _ = model(inputs, wav_lens=None)
    """

    def __init__(
        self,
        source,
        save_path,
        mask_prob=0.65,
        mask_length=10,
        normalize_wav=True,
    ):
        super().__init__(
            source=source, save_path=save_path, for_pretraining=True
        )

        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.normalize_wav = normalize_wav

        # We check if inputs need to be normalized w.r.t pretrained wav2vec2

    def forward(self, wav, wav_lens=None):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        wav_lens : torch.Tensor
            The relative length of the wav given in SpeechBrain format.

        Returns
        -------
        Wav2vec encoded outputs.
        """
        batch_size, raw_sequence_length = wav.shape

        if self.normalize_wav:
            wav = F.layer_norm(wav, wav.shape)

        sequence_length = self.model._get_feat_extract_output_lengths(
            raw_sequence_length
        ).item()

        # 1. Compute the indices that will be masked
        mask_time_indices = _compute_mask_indices(
            (batch_size, sequence_length),
            mask_prob=self.mask_prob,
            mask_length=self.mask_length,
        )
        torch_mask_time_indices = torch.tensor(
            mask_time_indices,
            device=wav.device,
            dtype=torch.long,
        )
        padding_mask = make_padding_masks(wav, wav_len=wav_lens)

        # 2. Sample the negative samples from the entire sequence.
        # Fairseq does it only on the masked indices, but this only work if you
        # have long sentences. For more versatility, we sample on the entire sequence.
        # value.
        full_sentence_indices = np.ones((batch_size, sequence_length))

        # print(np.sum(mask_time_indices, axis=1))
        negative_sample_indices = torch.tensor(
            transformers.models.wav2vec2.modeling_wav2vec2._sample_negative_indices(
                (batch_size, sequence_length),
                num_negatives=self.config.num_negatives,
                mask_time_indices=full_sentence_indices,
            ),
            device=wav.device,
            dtype=torch.long,
        )

        return (
            self.model(
                wav,
                mask_time_indices=torch_mask_time_indices,
                sampled_negative_indices=negative_sample_indices,
                attention_mask=padding_mask,
            ),
            torch_mask_time_indices,
        )

    def override_config(self, config):
        """If the config needs to be overridden, here is the place

        Arguments
        ---------
        config : Wav2Vec2Config
            The original config needs to be overridden.

        Returns
        -------
        Overridden config
        """
        config.output_hidden_states = True
        return config
