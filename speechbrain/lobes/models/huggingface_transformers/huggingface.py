"""This lobe is the interface for huggingface transformers models
It enables loading config and model via AutoConfig & AutoModel.

Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Titouan Parcollet 2021, 2022, 2023
 * Mirco Ravanelli 2021
 * Boumadane Abdelmoumene 2021
 * Ju-Chieh Chou 2021
 * Artem Ploujnikov 2021, 2022
 * Abdel Heba 2021
 * Aku Rouhe 2022
 * Arseniy Gorin 2022
 * Ali Safaya 2022
 * Benoit Wang 2022
 * Adel Moumen 2022, 2023
 * Andreas Nautsch 2022, 2023
 * Luca Della Libera 2022
 * Heitor Guimarães 2022
 * Ha Nguyen 2023
"""

import os
import pathlib

import torch
from huggingface_hub import model_info
from torch import nn
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForPreTraining,
    AutoModelForSeq2SeqLM,
    AutoModelWithLMHead,
    AutoTokenizer,
)

from speechbrain.dataio.dataio import length_to_mask
from speechbrain.utils.fetching import fetch
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class HFTransformersInterface(nn.Module):
    """This lobe provides an interface for integrating any HuggingFace transformer model within SpeechBrain.

    We use AutoClasses for loading any model from the hub and its necessary components.
    For example, we build Wav2Vec2 class which inherits HFTransformersInterface for working with HuggingFace's wav2vec models.
    While Wav2Vec2 can enjoy some already built features like modeling loading, pretrained weights loading, all weights freezing,
    feature_extractor loading, etc.
    Users are expected to override the essential forward() function to fit their specific needs.
    Depending on the HuggingFace transformer model in question, one can also modify the state_dict by overwriting the _modify_state_dict() method,
    or adapting their config by modifying override_config() method, etc.
    See:
    https://huggingface.co/docs/transformers/model_doc/auto
    https://huggingface.co/docs/transformers/autoclass_tutorial

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
    save_path : str
        save directory of the downloaded model.
    for_pretraining: bool (default: False)
        If True, build the model for pretraining
    with_lm_head : bool (default: False)
        If True, build the model with lm_head
    with_casual_lm : bool (default: False)
        If True, build casual lm  model
    seq2seqlm : bool (default: False)
        If True, build a sequence-to-sequence model with lm_head
    quantization_config : dict (default: None)
        Quantization config, extremely useful for deadling with LLM
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    cache_dir : str or Path (default: None)
        Location of HuggingFace cache for storing pre-trained models, to which symlinks are created.
    device : any, optional
        Device to migrate the model to.
    **kwargs
        Extra keyword arguments passed to the `from_pretrained` function.

    Example
    -------
    >>> model_hub = "facebook/wav2vec2-base-960h"
    >>> save_path = "tmp"
    >>> model = HFTransformersInterface(model_hub, save_path=save_path)
    """

    def __init__(
        self,
        source,
        save_path="",
        for_pretraining=False,
        with_lm_head=False,
        with_casual_lm=False,
        seq2seqlm=False,
        quantization_config=None,
        freeze=False,
        cache_dir="pretrained_models",
        device=None,
        **kwargs,
    ):
        super().__init__()

        # Fetch config
        self.config, _unused_kwargs = AutoConfig.from_pretrained(
            source,
            # cache_dir=save_path,
            return_unused_kwargs=True,
        )

        self.config = self.override_config(self.config)
        self.quantization_config = quantization_config

        self.for_pretraining = for_pretraining

        if self.for_pretraining:
            self.auto_class = AutoModelForPreTraining
        elif with_lm_head:
            self.auto_class = AutoModelWithLMHead
        elif with_casual_lm:
            self.auto_class = AutoModelForCausalLM
        elif seq2seqlm:
            self.auto_class = AutoModelForSeq2SeqLM
        else:
            self.auto_class = AutoModel

        # Download model
        self._from_pretrained(
            source,
            save_path=save_path,
            cache_dir=cache_dir,
            device=device,
            **kwargs,
        )

        # Prepare for training, fine-tuning, or inference
        self.freeze = freeze
        if self.freeze:
            logger.warning(
                f"speechbrain.lobes.models.huggingface_transformers.huggingface - {type(self.model).__name__} is frozen."
            )
            self.freeze_model(self.model)
        else:
            self.model.gradient_checkpointing_disable()  # Required by DDP
            self.model.train()

    def _from_pretrained(
        self,
        source,
        save_path,
        cache_dir,
        device=None,
        **kwargs,
    ):
        """This function manages the source checking and loading of the params.

        # 1. Is the model from HF or a local path
        # 2. Is the model pretrained with HF or SpeechBrain
        # 3. Download (if appropriate) and load with respect to 1. and 2.

        Arguments
        ---------
        source : str
            HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
        save_path : str
            Path (dir) of the downloaded model.
        cache_dir : str
            Path (dir) in which a downloaded pretrained model configuration should be cached.
        device : any, optional
            Device to migrate the model to.
        **kwargs
            Extra keyword arguments passed to `from_pretrained` function.
        """
        is_sb, ckpt_file, is_local = self._check_model_source(source, save_path)

        if is_sb or self.for_pretraining:
            self.model = self.auto_class.from_config(self.config)

        if is_sb:
            self.model.gradient_checkpointing_disable()  # Required by DDP
            # fetch the checkpoint file
            ckpt_full_path = fetch(
                filename=ckpt_file,
                source=source,
                savedir=save_path,
            )
            # We transfer the parameters from the checkpoint.
            self._load_sb_pretrained_parameters(ckpt_full_path)
        elif not self.for_pretraining:
            self.model = self.auto_class.from_pretrained(
                source,
                config=self.config,
                # cache_dir=save_path,
                quantization_config=self.quantization_config,
                **kwargs,
            )

        if device is not None:
            self.model.to(device)

    def _check_model_source(self, path, save_path):
        """Checks if the pretrained model has been trained with SpeechBrain and
        is hosted locally or on a HuggingFace hub.
        Called as static function in HFTransformersInterface._from_pretrained.

        Arguments
        ---------
        path : str
            Used as "source"; local path or HuggingFace hub name, e.g., "facebook/wav2vec2-large-lv60".
        save_path : str
            Directory where the downloaded model is saved.

        Returns
        -------
        is_sb : bool
            Whether the model is deserializable with SpeechBrain (True) or requires conversion (False).
        checkpoint_filename : str
            Filename of the checkpoint relative to the repository root.
        is_local : bool
            Whether the model is hosted locally or on the HuggingFace hub.

        Raises
        ------
        FileNotFoundError
            If the checkpoint file is not found.
        """
        import os
        from huggingface_hub import snapshot_download, model_info
        import pathlib

        checkpoint_filename = ""
        source = pathlib.Path(path)
        is_local = True
        local_path = path

        # Check if the path exists locally
        if not source.exists():
            is_local = False
            # Attempt to find the model in the local cache
            try:
                local_path = snapshot_download(
                    repo_id=path,
                    local_files_only=True
                )
                is_local = True
            except FileNotFoundError:
                # Model is not cached locally; will need to download
                pass

        if is_local:
            # Check for HuggingFace model files
            for file_name in os.listdir(local_path):
                if file_name.endswith((".bin", ".safetensors")):
                    checkpoint_filename = os.path.join(local_path, file_name)
                    is_sb = False
                    return is_sb, checkpoint_filename, is_local

            # Check for SpeechBrain model files
            for file_name in os.listdir(local_path):
                if file_name.endswith(".ckpt"):
                    checkpoint_filename = os.path.join(local_path, file_name)
                    is_sb = True
                    return is_sb, checkpoint_filename, is_local
        else:
            # Fetch file information from the HuggingFace hub
            files = model_info(path).siblings

            # Check for SpeechBrain model files on the hub
            for file in files:
                if file.rfilename.endswith(".ckpt"):
                    checkpoint_filename = file.rfilename
                    is_sb = True
                    return is_sb, checkpoint_filename, is_local

            # Check for HuggingFace model files on the hub
            for file in files:
                if file.rfilename.endswith((".bin", ".safetensors")):
                    checkpoint_filename = file.rfilename
                    is_sb = False
                    return is_sb, checkpoint_filename, is_local

        err_msg = f"{path} does not contain a .bin, .safetensors, or .ckpt checkpoint!"
        raise FileNotFoundError(err_msg)

    def _modify_state_dict(self, path, **kwargs):
        """A custom loading ensures SpeechBrain compatibility for pretrain and model.

        For example, wav2vec2 model pretrained with SB (Wav2Vec2Pretrain) has slightly different keys from Wav2Vec2.
        This method handle the compatibility between the two.

        Users should modify this function according to their own tasks.

        Arguments
        ---------
        path : str
            Checkpoint path, file name relative to the repo root.
        **kwargs : dict
            Args to forward
        """
        pass

    def _load_sb_pretrained_parameters(self, path):
        """Loads the parameter of a HuggingFace model pretrained with SpeechBrain
        and the HuggingFace Pretrain Object. It is necessary to perform a custom
        loading because HuggingFace adds a level to the checkpoint when storing
        the model breaking the compatibility Pretrain and model de/serialization.

        For example, a typical Wav2Vec2 checkpoint for a given parameter
        would be: model.conv.weight.data while for Wav2Vec2Pretrain it
        is: model.wav2vec2.weight.data (wav2vec2 must be removed before loading).

        Arguments
        ---------
        path : pathlib.Path
            The full path to the checkpoint.
        """
        modified_state_dict = self._modify_state_dict(path)

        if modified_state_dict is None:
            modified_state_dict = torch.load(path, map_location="cpu")

        incompatible_keys = self.model.load_state_dict(
            modified_state_dict, strict=False
        )
        for missing_key in incompatible_keys.missing_keys:
            logger.warning(
                f"During parameter transfer to {self.model} loading from "
                + f"{path}, the transferred parameters did not have "
                + f"parameters for the key: {missing_key}"
            )
        for unexpected_key in incompatible_keys.unexpected_keys:
            logger.warning(
                f"The param with the key: {unexpected_key} is discarded as it "
                + f"is useless for finetuning this {type(self.model).__name__} model."
            )

    def forward(self, **kwargs):
        """Users should modify this function according to their own tasks."""
        raise NotImplementedError

    def forward_encoder(self, **kwargs):
        """Users should modify this function according to their own tasks."""
        raise NotImplementedError

    def forward_decoder(self, **kwargs):
        """Users should modify this function according to their own tasks."""
        raise NotImplementedError

    def decode(self, **kwargs):
        """Might be useful for models like mbart, which can exploit SB's beamsearch for inference
        Users should modify this function according to their own tasks."""
        raise NotImplementedError

    def encode(self, **kwargs):
        """Custom encoding for inference
        Users should modify this function according to their own tasks."""
        raise NotImplementedError

    def freeze_model(self, model):
        """
        Freezes parameters of a model.
        This should be overridden too, depending on users' needs, for example, adapters use.

        Arguments
        ---------
        model : from AutoModel.from_config
            Valid HuggingFace transformers model object.
        """
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def override_config(self, config):
        """Users should modify this function according to their own tasks.

        Arguments
        ---------
        config : HuggingFace config object
            The original config.

        Returns
        -------
        config : HuggingFace config object
            Overridden config.
        """
        return config

    def load_feature_extractor(self, source, cache_dir=None, **kwarg):
        """Load model's feature_extractor from the hub.

        Arguments
        ---------
        source : str
            HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
        cache_dir : str
            Path (dir) in which a downloaded pretrained model configuration should be cached.
        **kwarg
            Keyword arguments to pass to the AutoFeatureExtractor.from_pretrained() method.
        """
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            source, **kwarg
        )

    def load_tokenizer(self, source, **kwarg):
        """Load model's tokenizer from the hub.

        Arguments
        ---------
        source : str
            HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
        **kwarg
            Keyword arguments to pass to the AutoFeatureExtractor.from_pretrained() method.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(source, **kwarg)


def make_padding_masks(src, wav_len=None, pad_idx=0):
    """This method generates the padding masks.

    Arguments
    ---------
    src : tensor
        The sequence to the encoder (required).
    wav_len : tensor
        The relative length of the wav given in SpeechBrain format.
    pad_idx : int
        The index for <pad> token (default=0).

    Returns
    -------
    src_key_padding_mask : tensor
        The padding mask.
    """
    src_key_padding_mask = None
    if wav_len is not None:
        abs_len = torch.round(wav_len * src.shape[1])
        src_key_padding_mask = length_to_mask(abs_len).bool()

    return src_key_padding_mask
