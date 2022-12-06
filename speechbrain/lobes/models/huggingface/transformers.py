"""This lobe enables the integration of huggingface transformers via AutoConfig & AutoModel.

Development started for pretrained wav2vec2/hubert/wavlm models.
Reference: https://arxiv.org/abs/2006.11477
Reference: https://arxiv.org/abs/1904.05862
Reference: https://arxiv.org/abs/2110.13900

Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Titouan Parcollet 2021
 * Boumadane Abdelmoumene 2021
 * Andreas Nautsch 2022
"""
import torch
import logging
import pathlib
from torch import nn
from functools import partial
from typing import Union, List, Callable
from speechbrain.pretrained.fetching import fetch
from speechbrain.lobes.models.huggingface.forward import default
from speechbrain.lobes.models.huggingface.overrides import _check_model_source

# We check if transformers is installed.
try:
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        AutoFeatureExtractor,
        AutoProcessor,
        AutoModel,
    )

except ImportError:
    MSG = "Please install transformers from HuggingFace to use wav2vec2 / Hubert\n"
    MSG += "E.G. run: pip install transformers"
    raise ImportError(MSG)

logger = logging.getLogger(__name__)

# used to check against
HUGGINGFACE_AUTO_CLASSES = [
    AutoTokenizer,
    AutoFeatureExtractor,
    AutoProcessor,
    AutoModel,
]


class HuggingFaceTransformer(nn.Module):
    """This lobe provides AutoClass architecture loading into SpeechBrain modules.

    See:
    https://huggingface.co/docs/transformers/model_doc/auto
    https://huggingface.co/docs/transformers/autoclass_tutorial

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
    save_path : str
        norm_output (dir) of the downloaded model.
    auto_class : Class (default: transformers.AutoModel)
        AutoClass as provided by transformers, see HUGGINGFACE_AUTO_CLASSES
        e.g., AutoTokenizer, AutoFeatureExtractor, AutoProcessor, AutoModel
    for_pretraining_cls : Class
        Specifies a HuggingFace transformers class that is created directly from a Config object
        (e.g. Wav2Vec2ForPreTraining).
    forward_partial_fn : Callable (default: None)
        Partial function that takes `model` and `data` which is assigned to `self.forward_partial_fn` and specified by:
            `self.forward_partial_fn.keywords['model'] = self.model`
             to be invoked later on by: `out, norm_shape = self.forward_partial_fn(data=data)`.
        Default (None) refers to the above `default_forward(model, data)` function by invokig:
            `self.forward_partial_fn = partial(default_forward, model=self.model)`.
    modify_state_dict_partial_fn : Callable (default: None)
        Partial function that adjusts de/serialization to ensure HuggingFace <> SpeechBrain model compatibility
        by invoking: `modified_state_dict = modify_state_dict_partial_fn(path)`.
        Default (None) invokes: `modified_state_dict = torch.load(path, map_location="cpu")`
    override_hf_config_partial_fn : Callable (default: None)
        Partial function that accustoms an AutoConfig by invoking:
        `config = override_hf_config_partial_fn(config)`
        Default (None) skips that step.
    override_hf_model_partial_fn : Callable (default: None)
        Partial function that accustoms an AutoModel by invoking:
        `self.model = override_hf_model_partial_fn(self.model)`
        Default (None) skips that step.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    freeze_nested_models_their_calls :  Lst[str] or str (default: None)
        When freeze = False and freeze_nested_models_their_calls has strings `nested_freeze_call`,
        nested modules of the model are Frozen by invoking: `eval(f"self.model.{nested_freeze_call}()")`
        Default (None) skips that step.
    cache_dir: str or Path (default: None)
        Location of HuggingFace cache for storing pre-trained models, to which symlinks are created.

    Example
    -------
    >>> inputs = torch.rand([10, 600])
    >>> model_hub = "facebook/wav2vec2-base-960h"
    >>> save_path = "tmp"
    >>> model = HuggingFaceTransformer(model_hub, save_path=save_path)
    >>> outputs = model(inputs)
    """

    def __init__(
        self,
        source,
        save_path,
        auto_class=AutoModel,
        for_pretraining_cls=None,
        forward_partial_fn: Union[Callable, None] = None,
        modify_state_dict_partial_fn: Union[Callable, None] = None,
        override_hf_config_partial_fn: Union[Callable, None] = None,
        override_hf_model_partial_fn: Union[Callable, None] = None,
        freeze=True,
        freeze_nested_models_their_calls: Union[List[str], str, None] = None,
        cache_dir: Union[str, pathlib.Path, None] = "pretrained_models",
    ):
        super().__init__()

        # Is the auto_class valid?
        assert (
            auto_class in HUGGINGFACE_AUTO_CLASSES
        ), "Error: please provide a HuggingFace Auto[Class]"

        # Fetch config
        config, _unused_kwargs = AutoConfig.from_pretrained(
            source,
            cache_dir=cache_dir if for_pretraining_cls is None else save_path,
            return_unused_kwargs=True,
        )

        # Adjust config as desired
        if override_hf_config_partial_fn is not None:
            config = override_hf_config_partial_fn(config)

        # Instantiate model
        if for_pretraining_cls is not None:
            # Construct for pretraining
            self.model = for_pretraining_cls(config)
        else:
            # Fetch model architecture
            if (
                hasattr(config, "auto_map")
                and auto_class.__name__ in config.auto_map
            ):
                model = auto_class.from_config(config, cache_dir=cache_dir)
            else:  # AutoModel.from_config case: type(config) in AutoModel._model_mapping.keys() /or: raise ValueError
                model = auto_class.from_config(config)

            # Download model
            self._from_pretrained(
                source,
                config=config,
                model=model,
                modify_state_dict_partial_fn=modify_state_dict_partial_fn,
                save_path=save_path,
                cache_dir=cache_dir,
            )

        # Adjust model as desired
        if override_hf_model_partial_fn is not None:
            self.model = override_hf_model_partial_fn(self.model)

        # Set inner forward function
        if forward_partial_fn is not None:
            # if forward_partial_fn is a partial, add model to the partial's keyword attributes
            if isinstance(forward_partial_fn, partial):
                self.forward_partial_fn = forward_partial_fn
                self.forward_partial_fn.keywords["model"] = self.model
            else:  # if forward_partial_fn is a function, create a partial from it with the model parameter set
                self.forward_partial_fn = partial(
                    forward_partial_fn, model=self.model
                )
        else:
            self.forward_partial_fn = partial(default, model=self.model)

        # Prepare for training, fine-tuning, or inference
        self.freeze = freeze
        if self.freeze:
            logger.warning(
                "speechbrain.lobes.models.HuggingFaceTransformer is frozen."
            )
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model.gradient_checkpointing_disable()  # Required by DDP
            self.model.train()
            if freeze_nested_models_their_calls is not None:
                if type(freeze_nested_models_their_calls) is not list:
                    freeze_nested_models_their_calls = [
                        freeze_nested_models_their_calls
                    ]
                for nested_freeze_call in freeze_nested_models_their_calls:
                    eval(f"self.model.{nested_freeze_call}()")

    def _from_pretrained(
        self,
        source,
        config,
        model,
        modify_state_dict_partial_fn,
        save_path,
        cache_dir,
    ):
        """This function manages the source checking and loading of the params.
        # 1. Is the model from HF or a local path
        # 2. Is the model pretrained with HF or SpeechBrain
        # 3. Download (if appropriate) and load with respect to 1. and 2.
        """
        is_sb, ckpt_file = _check_model_source(source, save_path)
        if is_sb:
            config = config.from_pretrained(source, cache_dir=save_path)
            self.model = model(config)
            self.model.gradient_checkpointing_disable()  # Required by DDP
            # fetch the checkpoint file
            ckpt_full_path = fetch(
                filename=ckpt_file,
                source=source,
                savedir=save_path,
                cache_dir=cache_dir,
            )
            # We transfer the parameters from the checkpoint.
            self._load_sb_pretrained_parameters(
                ckpt_full_path,
                modify_state_dict_partial_fn=modify_state_dict_partial_fn,
            )
        else:
            self.model = model.from_pretrained(source, cache_dir=save_path)

    def _load_sb_pretrained_parameters(
        self, path, modify_state_dict_partial_fn
    ):
        """Loads the parameter of a HuggingFace model pretrained with SpeechBrain
        and the HuggingFace Pretrain Object. It is necessary to perform a custom
        loading because HuggingFace adds a level to the checkpoint when storing
        the model breaking the compatibility Pretrain and model de/serialization.

        For example, a typical HuggingFaceWav2Vec2 checkpoint for a given parameter
        would be: model.conv.weight.data while for HuggingFaceWav2Vec2Pretrain it
        is: model.wav2vec2.weight.data (wav2vec2 must be removed before loading).
        """
        if modify_state_dict_partial_fn is not None:
            modified_state_dict = modify_state_dict_partial_fn(path)
        else:
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
                + "is useless for finetuning this HuggingFaceTransformer."
            )

    def forward(self, data):
        """Process data (token streams, wavs, ...). This function wraps weight-freezing.
        """
        # If we freeze, we simply remove all grads and features from the graph.
        if self.freeze:
            with torch.no_grad():
                return self.forward_partial_fn(data=data)

        return self.forward_partial_fn(data=data)
