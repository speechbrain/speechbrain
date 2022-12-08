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
import os
import torch
import logging
import pathlib
import numpy as np
from torch import nn
from functools import partial

# import torch.nn.functional as F
from typing import Union, List, Callable
from huggingface_hub import model_info
from speechbrain.pretrained.fetching import fetch


# We check if transformers is installed.
try:
    import transformers
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        AutoFeatureExtractor,
        AutoProcessor,
        AutoModel,
    )
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        _compute_mask_indices,
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
            self.forward_partial_fn = partial(forward_default, model=self.model)

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
        is_sb, ckpt_file = check_model_source(source, save_path)
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


def model_set_spectral_augmentation(model, apply_spec_augment):
    """Sets `model.config.apply_spec_augment` the flag to a specific value.

    To be used as HuggingFaceTransformer init argument:
        override_hf_model_partial_fn=partial(
            model_set_spectral_augmentation,
            apply_spec_augment=apply_spec_augment,
        )

    Arguments
    ---------
    model : from AutoModel.from_config
        Valid HuggingFace transformers model object.
    apply_spec_augment : bool
        If True, the model will apply spec augment on the output of feature extractor
        (e.g., inside huggingface Wav2VecModel() class, see: https://arxiv.org/abs/1904.08779).
        If False, the model will not apply spec augment. We set this to `false` to prevent from doing it twice.

    Returns
    -------
    model : from AutoModel.from_config
        Valid HuggingFace transformers model object; with flag set as desired.
    """
    model.config.apply_spec_augment = apply_spec_augment
    return model


def config_return_hidden_states(config):
    """Sets `output_hidden_states = True` for a transformer config.

    To be used as HuggingFaceTransformer init argument `override_hf_config_partial_fn=partial(config_return_hidden_states)`.

    Arguments
    ---------
    config : from AutoConfig.from_pretrained
        Valid HuggingFace transformers config object.

    Returns
    -------
    config : from AutoConfig.from_pretrained
        Valid HuggingFace transformers config object; with `output_hidden_states = True`
    """
    config.output_hidden_states = True  # We want the hidden states as well!
    return config


def check_model_source(path, save_path):
    """Checks if the pretrained model has been trained with SpeechBrain and
    is hosted locally or on a HuggingFace hub.

    Called as static function in HuggingFaceTransformer._from_pretrained.

    Arguments
    ---------
    path : str
        Used as "source"; local path or HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
    save_path : str
        norm_output (dir) of the downloaded model.

    Returns
    -------
    is_sb : bool
        Whether/not the model is deserializable w/ SpeechBrain or not (then, model conversion is needed).
    checkpoint_filename : str
        as of HuggingFace documentation: file name relative to the repo root (guaranteed to be here).
    """
    checkpoint_filename = ""
    source = pathlib.Path(path)
    is_local = True

    # If path is a huggingface hub.
    if not source.exists():
        is_local = False

    # Check if source is downloaded already
    sink = pathlib.Path(
        save_path + "/models--" + path.replace("/", "--") + "/snapshots"
    )
    if sink.exists():
        sink = sink / os.listdir(str(sink))[0]  # there's a hash-id subfolder
        if any(
            File.endswith(".bin") or File.endswith(".ckpt")
            for File in os.listdir(str(sink))
        ):
            is_local = True
            local_path = str(sink)
        else:
            local_path = path
    else:
        local_path = path

    if is_local:
        # Test for HuggingFace model
        if any(File.endswith(".bin") for File in os.listdir(local_path)):
            is_sb = False
            return is_sb, checkpoint_filename

        # Test for SpeechBrain model and get the filename.
        for File in os.listdir(local_path):
            if File.endswith(".ckpt"):
                checkpoint_filename = os.path.join(path, File)
                is_sb = True
                return is_sb, checkpoint_filename
    else:
        files = model_info(path).siblings  # get the list of files of the Hub

        # Test if it's an HuggingFace model or a SB one
        for File in files:
            if File.rfilename.endswith(".ckpt"):
                checkpoint_filename = File.rfilename
                is_sb = True
                return is_sb, checkpoint_filename

        for File in files:
            if File.rfilename.endswith(".bin"):
                checkpoint_filename = File.rfilename
                is_sb = False
                return is_sb, checkpoint_filename

    err_msg = f"{path} does not contain a .bin or .ckpt checkpoint !"
    raise FileNotFoundError(err_msg)


def modify_state_dict_wav2vec2(path, replacables=["wav2vec2"]):
    """A custom loading ensures SpeechBrain compatibility for Pretrain and model
    de/serialization. Here, the scope is to remove '.wav2vec2' before loading.

    To be used as HuggingFaceTransformer init argument:
        `modify_state_dict_partial_fn=partial(modify_state_dict_wav2vec2)`.

    Called in: HuggingFaceTransformer._load_sb_pretrained_parameters; `path` argument is handled in that scope.

    If you have another state dict to be modified:
     * create your function in your recipe (copy this one and modify as you see fit)
     * pass it to HuggingFaceTransformer init as a partial callable
    This function serves as a reference example to your implementation, only.

    Arguments
    ---------
    path : str
        Checkpoint path; file name relative to the repo root.
    replacables : List[str]
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
        for tag in replacables:
            if f"{tag}." in key:
                save_key = key.replace(f"model.{tag}.", "")
                modified_state_dict[save_key] = params

    return modified_state_dict


def forward_default(model, data):
    """Takes input data and returns its forward pass of a given model.

    Default for HuggingFaceTransformer init argument:
        `forward_partial_fn: Union[Callable, None] = None`
    as it is invoked by:
        `self.forward_partial_fn = partial(default_forward, model=self.model)`.

        Note: `model` is a required parameter, and handled in the init function scope.

    Called in: HuggingFaceTransformer.forward - invoked by:
        `return self.forward_partial_fn(data=data)`.

    If you have another forward function:
     * create your function in your recipe (copy this one and modify as you see fit)
     * pass it to HuggingFaceTransformer init as a partial callable
    This function serves as a reference example to your implementation, only.
    Check out `wav2vec2_forward` and `wav2vec2_pretraining_forward` below for more reference examples.

    Arguments
    ---------
    model : transformers.AutoModel
        A valid HuggingFace transformers model.
    data : torch.Tensor (signal)
        A batch of audio signals to transform to features.

    Returns
    -------
    out : torch.Tensor
        Batch of depending model outputs
    """
    out = model(data)
    return out


def forward_wav2vec2(
    model, data, output_all_hiddens=False,
):
    """Takes an input waveform and return its corresponding wav2vec encoding.

    Used in `HuggingFaceWav2Vec2(HuggingFaceTransformer)` init when calling `super().__init__` as:
        forward_partial_fn=partial(
                wav2vec2_forward,
                output_all_hiddens=output_all_hiddens  # here (default) values are assigned to this partial
            )
    Then, `forward_partial_fn` is handled in the HuggingFaceTransformer init by:
        self.forward_partial_fn = forward_partial_fn
        self.forward_partial_fn.keywords["model"] = self.model

    If you have another forward function:
     * create your function in your recipe (copy this one and modify as you see fit)
     * pass it to HuggingFaceTransformer init as a partial callable
    This function serves as a reference example to your implementation, only.

    See above `default_forward` documentation.

    Arguments
    ---------
    model : transformers.AutoModel
        A valid HuggingFace transformers model.
    data : torch.Tensor (signal)
        A batch of audio signals to transform to features.
    output_all_hiddens : bool (default: False)
        If True, the forward function outputs the hidden states from all transformer layers.
        For example wav2vec2-base has 12 transformer layers and the output is of shape (13, B, T, C),
        where a projection of the CNN output is added to the beginning.
        If False, the forward function outputs the hidden states only from the last transformer layer.

    Returns
    -------
    out : torch.Tensor
        Batch of depending model outputs
    norm_shape : List[int]
        Shape to be used in layer norm.
    """
    # Extract wav2vec output
    out = model(data, output_hidden_states=True)

    if output_all_hiddens:
        out = torch.stack(list(out.hidden_states), dim=0)
        # norm_shape = out.shape[-3:]
    else:
        out = out.last_hidden_state
        # norm_shape = out.shape

    return out


def forward_wav2vec2_pretraining(
    model, data, mask_prob, mask_length,
):
    """Takes an input waveform and return its corresponding wav2vec encoding.

    Used in `HuggingFaceWav2Vec2Pretrain(HuggingFaceTransformer)` init when calling `super().__init__` as:
        forward_partial_fn=partial(
                wav2vec2_pretraining_forward,
                mask_prob=mask_prob,  # here (default) values are assigned to this partial
                mask_length=mask_length,  # here (default) values are assigned to this partial
            )
    Then, `forward_partial_fn` is handled in the HuggingFaceTransformer init by:
        self.forward_partial_fn = forward_partial_fn
        self.forward_partial_fn.keywords["model"] = self.model

    If you have another forward function:
     * create your function in your recipe (copy this one and modify as you see fit)
     * pass it to HuggingFaceTransformer init as a partial callable
    This function serves as a reference example to your implementation, only.

    See above `default_forward` documentation.

    Parameters
    ----------
    model : transformers.AutoModel
        A valid HuggingFace transformers model.
    data : torch.Tensor (signal)
        A batch of audio signals to transform to features.
    mask_prob : float
        Probability of masking a given frame. Default is taken from the paper.
    mask_length : int
        Length (i.e. number of consecutive masked frames). Default is taken from
        the paper.

    Returns
    -------
    out : torch.Tensor
        Batch of depending model outputs
    torch_mask_time_indices : List[int]
        Shape to be used in layer norm.
    """
    batch_size, raw_sequence_length = data.shape

    sequence_length = model._get_feat_extract_output_lengths(
        raw_sequence_length
    )

    # 1. Compute the indices that will be masked
    mask_time_indices = _compute_mask_indices(
        (batch_size, sequence_length),
        mask_prob=mask_prob,
        mask_length=mask_length,
    )
    torch_mask_time_indices = torch.tensor(
        mask_time_indices, device=data.device, dtype=torch.long,
    )

    # 2. Sample the negative samples from the entire sequence.
    # Fairseq does it only on the masked indices, but this only work if you
    # have long sentences. For more versatily, we sample on the entire sequence.
    # value.
    full_sentence_indices = np.ones((batch_size, sequence_length))
    sampled_negative_indices = transformers.models.wav2vec2.modeling_wav2vec2._sample_negative_indices(
        (batch_size, sequence_length.numpy()),
        num_negatives=model.config.num_negatives,
        mask_time_indices=full_sentence_indices,
    )

    negative_sample_indices = torch.tensor(
        sampled_negative_indices, device=data.device, dtype=torch.long,
    )

    # 3. prepare the output
    out = model(
        data,
        mask_time_indices=torch_mask_time_indices,
        sampled_negative_indices=negative_sample_indices,
    )

    return out, torch_mask_time_indices
