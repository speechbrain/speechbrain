"""This lobe enables the integration of huggingface pretrained wav2vec2 models.

Reference: https://arxiv.org/abs/2006.11477
Reference: https://arxiv.org/abs/1904.05862
Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Titouan Parcollet 2021
 * Boumadane Abdelmoumene 2021
"""

import os
import torch
import logging
import pathlib
import torch.nn.functional as F
from torch import nn
import speechbrain as sb

# We check if transformers is installed.
try:
    from transformers import Wav2Vec2Model, HubertModel
    from transformers import Wav2Vec2Config, HubertConfig
    from transformers import Wav2Vec2FeatureExtractor
    from transformers import Wav2Vec2ForPreTraining
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        _compute_mask_indices,
    )

except ImportError:
    print(
        "Please install transformer from HuggingFace to use wav2vec2/Hubert !"
    )

logger = logging.getLogger(__name__)

HF_models = {"wav2vec2": Wav2Vec2Model, "hubert": HubertModel}

HF_config = {"wav2vec2": Wav2Vec2Config, "hubert": HubertConfig}


class HuggingFaceWav2Vec2(nn.Module):
    """This lobe enables the integration of HuggingFace
    pretrained wav2vec2.0/Hubert models.

    Source paper wav2vec2.0: https://arxiv.org/abs/2006.11477
    Source paper Hubert: https://arxiv.org/abs/2106.07447
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model can be used as a fixed feature extractor or can be finetuned. It
    will download automatically the model from HuggingFace.

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
        When freeze = False and freeze_feature_extractor True, the featue_extractor module of the model is Frozen. If False
        all the wav2vec model will be trained including featue_extractor module.
    pretrain : bool (default: True)
        If True, the model is pretrained with the specified source.
        If False, the randomly-initialized model is instantiated.
    apply_spec_augment : bool (default: False)
        If True, the model will apply spec augment on the output of feature extractor
        (inside huggingface Wav2VecModel() class).
        If False, the model will not apply spec augment. We set this to false to prevent from doing it twice.
    Example
    -------
    >>> inputs = torch.rand([10, 600])
    >>> model_hub = "facebook/wav2vec2-base-960h"
    >>> save_path = "savedir"
    >>> model = HuggingFaceWav2Vec2(model_hub, save_path)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 1,  768])
    """

    def __init__(
        self,
        source,
        save_path,
        output_norm=True,
        freeze=True,
        freeze_feature_extractor=False,
        pretrain=True,
        apply_spec_augment=False,
    ):
        super().__init__()

        # Download the extractor from HuggingFace.
        # The extractor is only used to retrieve the normalisation
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            source, cache_dir=save_path
        )

        # Select specific self-supervised loader (eg. Wav2Vec2, Hubert)
        if "hubert" in source:
            config = HF_config.get("hubert")
            model = HF_models.get("hubert")
        else:
            config = HF_config.get("wav2vec2")
            model = HF_models.get("wav2vec2")

        # Download the model from HuggingFace.
        # if pretrain is False, we do not download the pretrained weights
        # it it is True, we download and load them.
        if not (pretrain):
            config = config.from_pretrained(source, cache_dir=save_path)
            config.gradient_checkpointing = (
                False  # This cause errors with DDP if True.
            )
            self.model = model(config)
        else:
            # We check if the pretrained model comes from HF or SpeechBrain.
            # and we load it accordingly.
            is_sb, ckpt_file = self.check_model_source(source)
            if is_sb:
                config = config.from_pretrained(source, cache_dir=save_path)
                self.model = model(config)
                self.model.gradient_checkpointing_disable()  # Required by DDP
                # We transfer the parameters from the checkpoint.
                sb.utils.checkpoints.torch_parameter_transfer(
                    self.model, ckpt_file, device="cpu"
                )
            else:
                self.model = model.from_pretrained(source, cache_dir=save_path)

        # set apply_spec_augment
        self.model.config.apply_spec_augment = apply_spec_augment

        # We check if inputs need to be normalized w.r.t pretrained wav2vec2
        self.normalize_wav = self.feature_extractor.do_normalize

        self.freeze = freeze
        self.freeze_feature_extractor = freeze_feature_extractor
        self.output_norm = output_norm
        if self.freeze:
            self.model.eval()
        else:
            self.model.train()
            if self.freeze_feature_extractor:
                self.model.feature_extractor._freeze_parameters()

    def forward(self, wav):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        # If we freeze, we simply remove all grads and features from the graph.
        if self.freeze:
            with torch.no_grad():
                return self.extract_features(wav).detach()

        return self.extract_features(wav)

    def extract_features(self, wav):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        if self.normalize_wav:
            wav = F.layer_norm(wav, wav.shape)

        # Extract wav2vec output
        out = self.model(wav)[0]

        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out, out.shape)

        return out

    def check_model_source(self, path):
        """Checks if the pretrained model has been trained with SpeechBrain.
        """
        checkpoint_filename = ""
        source = pathlib.Path(path)

        # If path isn't a path but a HuggingFace repository, return.
        if not source.exists():
            return False, checkpoint_filename

        # Test for HuggingFace model
        if any(File.endswith(".bin") for File in os.listdir(path)):
            return False, checkpoint_filename

        # Test for SpeechBrain model and get the filename.
        for File in os.listdir(path):
            if File.endswith(".ckpt"):
                checkpoint_filename = os.path.join(path, File)
                return True, checkpoint_filename

        return False, checkpoint_filename


class HuggingFaceWav2Vec2Pretrain(nn.Module):
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
    Example
    -------
    >>> inputs = torch.rand([10, 600])
    >>> model_hub = "facebook/wav2vec2-base-960h"
    >>> save_path = "savedir"
    >>> model = HuggingFaceWav2Vec2Pretrain(model_hub, save_path)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 1,  768])
    """

    def __init__(
        self,
        source,
        save_path,
        mask_prob=0.65,
        mask_length=10,
        normalize_wav=True,
    ):
        super().__init__()

        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.normalize_wav = normalize_wav

        # Download the config of the model from HuggingFace.
        config = Wav2Vec2Config.from_pretrained(source, cache_dir=save_path)
        config.output_hidden_states = True  # We want the hidden states as well!

        self.model = Wav2Vec2ForPreTraining(config)
        self.model.gradient_checkpointing_disable()  # Required by DDP
        self.model.train()

        # We check if inputs need to be normalized w.r.t pretrained wav2vec2

    def forward(self, wav):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """
        batch_size, raw_sequence_length = wav.shape

        if self.normalize_wav:
            wav = F.layer_norm(wav, wav.shape)

        # We must compute the masking before forward. This is used by the loss.
        sequence_length = self.model._get_feat_extract_output_lengths(
            raw_sequence_length
        )
        mask_time_indices = _compute_mask_indices(
            (batch_size, sequence_length),
            mask_prob=self.mask_prob,
            mask_length=self.mask_length,
        )

        return (
            self.model(wav, mask_time_indices=mask_time_indices),
            mask_time_indices,
        )
