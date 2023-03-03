"""This lobe enables the integration of huggingface transformers via AutoConfig & AutoModel.

Development started for pretrained wav2vec2/hubert/wavlm models.
Reference: https://arxiv.org/abs/2006.11477
Reference: https://arxiv.org/abs/1904.05862
Reference: https://arxiv.org/abs/2110.13900

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
"""
import os
import torch
import logging
import pathlib
import numpy as np
from torch import nn
from functools import partial
from typing import Union, Callable
from huggingface_hub import model_info
from speechbrain.pretrained.fetching import fetch
from speechbrain.dataio.dataio import length_to_mask


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
        Partial function that takes `model` and `wav` which is assigned to `self.forward_partial_fn` and specified by:
            `self.forward_partial_fn.keywords['model'] = self.model`
             to be invoked later on by: `out, norm_shape = self.forward_partial_fn(wav=wav)`.
        Default (None) refers to the above `default_forward(model, wav)` function by invokig:
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
        freeze=False,
        freeze_model_fn: Union[Callable, None] = None,
        freeze_params_fn: Union[Callable, None] = None,
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
            self.forward_partial_fn = forward_partial_fn
            # self.forward_partial_fn.keywords["model"] = self.model
            # if forward_partial_fn is a partial, add model to the partial's keyword attributes
            """
            if isinstance(forward_partial_fn, partial):
                self.forward_partial_fn = forward_partial_fn
                # self.forward_partial_fn.keywords["model"] = self.model
            else:  # if forward_partial_fn is a function, create a partial from it with the model parameter set
                self.forward_partial_fn = partial(
                    forward_partial_fn,   # model=self.model
                )
            """
        else:
            self.forward_partial_fn = partial(
                forward_default
            )  # , model=self.model)

        # Prepare for training, fine-tuning, or inference
        self.freeze = freeze
        if self.freeze:
            logger.warning(
                "speechbrain.lobes.models.HuggingFaceTransformer is frozen."
            )
            if freeze_model_fn is not None:
                freeze_model_fn(self.model)
            else:
                freeze_model(self.model)
        else:
            self.model.gradient_checkpointing_disable()  # Required by DDP
            self.model.train()
            if freeze_params_fn is not None:
                freeze_params_fn(self.model)

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

    def forward(self, wav, lengths=None, **kwargs):
        """Process data (token streams, wavs, ...). This function wraps weight-freezing.
        """
        # If we freeze, we simply remove all grads and features from the graph.
        kwargs["self"] = self
        kwargs["wav"] = wav
        # there's a toch function calling this, which needs "lengths" so it uses more than 'x' as argument to call this one
        # => in case of despair, package all your extra variables into a 'lengths' dictionary, so it arrives here; then unpack later
        if lengths is not None:
            kwargs["wav_lens"] = lengths
        if self.freeze:
            with torch.no_grad():
                return self.forward_partial_fn(**kwargs)
        return self.forward_partial_fn(**kwargs)


def freeze_model(model):
    """
    Freezes parameters of a model.

    Arguments
    ---------
    model : from AutoModel.from_config
        Valid HuggingFace transformers model object.
    """
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def freeze_model_but_train(model):
    """
    Freezes model but keeps it in training mode.
    Note: we keep it to train to have dropout and LN computed adequaly.

    Arguments
    ---------
    model : from AutoModel.from_config
        Valid HuggingFace transformers model object.
    """
    model.train()
    for param in model.parameters():
        param.requires_grad = False


def freeze_params_feature_extractor(model):
    """
    Freezes parameters of nested feature extractor.

    Arguments
    ---------
    model : from AutoModel.from_config
        Valid HuggingFace transformers model object.
    """
    logger.warning(
        "speechbrain.lobes.models.transformer.HuggingFace - feature extractor is frozen."
    )
    model.feature_extractor.eval()
    for param in model.feature_extractor.parameters():
        param.requires_grad = False


def freeze_params_encoder(model):
    """
    Freezes parameters of nested encoder.

    Arguments
    ---------
    model : from AutoModel.from_config
        Valid HuggingFace transformers model object.
    """
    logger.warning(
        "speechbrain.lobes.models.transformer.HuggingFace - encoder is frozen."
    )
    for param in model.encoder.parameters():
        param.requires_grad = False


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


def forward_default(self, wav):
    """Takes input data and returns its forward pass of a given model.

    Default for HuggingFaceTransformer init argument:
        `forward_partial_fn: Union[Callable, None] = None`
    as it is invoked by:
        `self.forward_partial_fn = partial(default_forward, model=self.model)`.

        Note: `model` is a required parameter, and handled in the init function scope.

    Called in: HuggingFaceTransformer.forward - invoked by:
        `return self.forward_partial_fn(wav=wav)`.

    If you have another forward function:
     * create your function in your recipe (copy this one and modify as you see fit)
     * pass it to HuggingFaceTransformer init as a partial callable
    This function serves as a reference example to your implementation, only.
    Check out `wav2vec2_forward` and `wav2vec2_pretraining_forward` below for more reference examples.

    Arguments
    ---------
    model : transformers.AutoModel
        A valid HuggingFace transformers model.
    wav : torch.Tensor (signal)
        A batch of audio signals to transform to features.

    Returns
    -------
    out : torch.Tensor
        Batch of depending model outputs
    """
    out = self.model(wav)
    return out


def forward_wav2vec2(
    self, wav, wav_lens, output_all_hiddens=False,
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
    wav : torch.Tensor (signal)
        A batch of audio signals to transform to features.
    wav_lens : tensor
            The relative length of the wav given in SpeechBrain format.
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
    padding_mask = make_masks(src=wav, wav_len=wav_lens)

    # Extract wav2vec output
    out = self.model(
        wav,
        attention_mask=padding_mask,
        output_hidden_states=output_all_hiddens,
    )

    if output_all_hiddens:
        out = torch.stack(list(out.hidden_states), dim=0)
    else:
        out = out.last_hidden_state

    return out


def forward_wav2vec2_pretraining(
    self, wav, wav_lens, mask_prob, mask_length,
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
    wav : torch.Tensor (signal)
        A batch of audio signals to transform to features.
    wav_lens : tensor
            The relative length of the wav given in SpeechBrain format.
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
    batch_size, raw_sequence_length = wav.shape
    padding_mask = make_masks(wav, wav_len=wav_lens)

    sequence_length = self.model._get_feat_extract_output_lengths(
        raw_sequence_length
    )

    # 1. Compute the indices that will be masked
    mask_time_indices = _compute_mask_indices(
        (batch_size, sequence_length),
        mask_prob=mask_prob,
        mask_length=mask_length,
    )
    torch_mask_time_indices = torch.tensor(
        mask_time_indices, device=wav.device, dtype=torch.long,
    )

    # 2. Sample the negative samples from the entire sequence.
    # Fairseq does it only on the masked indices, but this only work if you
    # have long sentences. For more versatily, we sample on the entire sequence.
    # value.
    full_sentence_indices = np.ones((batch_size, sequence_length))
    sampled_negative_indices = transformers.models.wav2vec2.modeling_wav2vec2._sample_negative_indices(
        (batch_size, sequence_length.numpy()),
        num_negatives=self.model.config.num_negatives,
        mask_time_indices=full_sentence_indices,
    )

    negative_sample_indices = torch.tensor(
        sampled_negative_indices, device=wav.device, dtype=torch.long,
    )

    # 3. prepare the output
    out = self.model(
        wav,
        attention_mask=padding_mask,
        mask_time_indices=torch_mask_time_indices,
        sampled_negative_indices=negative_sample_indices,
    )

    return out, torch_mask_time_indices


def forward_whisper(
    self,
    wav,
    decoder_input_ids,
    n_samples=480000,
    n_fft=400,
    hop_length=160,
    mel_filters=80,
    output_attentions=True,
    output_all_hiddens=False,
):
    """Perform mel transformation and one step of the whisper (encoder-decoder).

    Arguments
    ---------
    model : transformers.AutoModel
        A valid HuggingFace transformers model.
    wav : torch.Tensor (signal)
        A batch of audio signals to transform to features.
    decoder_input_ids : torch.Tensor
        This is necessary if we want to use the decoder.
    n_samples : int
        For width in padding/trimming audio signal before mel sepctrum.
    n_fft : int
        Size of Fourier transform.
    hop_length : int
        Size of sliding window shift.
    mel_filters : torch.Tensor
        Mel filterbanks as operand in matrix multiplication.
    output_attentions : bool
        Whether/not to return attention values as well.
    output_all_hiddens : bool
        Whether/not to return attention values as well.

        A batch of decoder inputs tokens.
        The first tokens need to dictacte the behavior of the decoder.
        It needs to start with the bos_token, the language token,
        the task token, and finally the timestamp token.

        Please refer to the whisper paper for more details or go to the
        seq2seq2.py file in SpeechBrain to see how to generate the tokens
        with Greedy Search and/or Beam Search.
    """
    out_encoder = forward_mel_encoder(
        model=self.model,
        wav=wav,
        n_samples=n_samples,
        n_fft=n_fft,
        hop_length=hop_length,
        mel_filters=mel_filters,
    )
    if decoder_input_ids is None:
        return out_encoder
    if output_all_hiddens:
        logits, attn = forward_decoder(out_encoder[-1], decoder_input_ids)
    else:
        logits, attn = forward_decoder(
            model=self.model,
            wav=out_encoder,
            decoder_input_ids=decoder_input_ids,
            output_attentions=output_attentions,
        )
    return out_encoder, logits, attn


def forward_mel_encoder(
    model,
    wav,
    n_samples,
    n_fft,
    hop_length,
    mel_filters,
    output_all_hiddens=False,
):
    """Perform one step of the whisper encoder with Mel FBANKs as Input.

    Arguments
    ---------
    model : transformers.AutoModel
        A valid HuggingFace transformers model.
    wav : torch.Tensor (signal)
        A batch of Mel FBANK from HF to transform to features.
    n_samples : int
        For width in padding/trimming audio signal before mel sepctrum.
    n_fft : int
        Size of Fourier transform.
    hop_length : int
        Size of sliding window shift.
    mel_filters : torch.Tensor
        Mel filterbanks as operand in matrix multiplication.
    output_all_hiddens: bool (default: False)
        If True, the forward function outputs the hidden states from all transformer layers of the encoder.
        For example whisper-base has 6 transformer layers and the output is of shape (7, B, T, C),
        where the output of the CNN output is added to the beginning.
        If False, the forward function outputs the hidden states only from the last transformer layer of the encoder.

    Returns
    -------
    torch.Tensor
        A tensor containing Mel spectograms.
    """

    def _log_mel_spectrogram(audio):
        """Compute the Mel spectrogram of a batch of input waveforms.

        Reference: adapted from
        https://github.com/openai/whisper/blob/eff383b27b783e280c089475852ba83f20f64998/whisper/audio.py#L92

        Arguments
        ---------
        audio : torch.Tensor
            A batch of audio waveforms in 16 kHz.

        Returns
        -------
        torch.Tensor
            A tensor that contains the batch of Mel spectrograms.
        """
        window = torch.hann_window(n_fft, device=audio.device)
        stft = torch.stft(
            audio, n_fft, hop_length, window=window, return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2

        filters = mel_filters
        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(
            log_spec,
            (log_spec.flatten(start_dim=1).max(dim=-1)[0] - 8.0)[:, None, None],
        )
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def _pad_or_trim(array, axis=-1):
        """Pad or trim the Mel spectrograms as expected by the encoder.

        Reference: adapted from
        https://github.com/openai/whisper/blob/eff383b27b783e280c089475852ba83f20f64998/whisper/audio.py#L52

        Arguments
        ---------
        array : torch.Tensor
            A tensor that contains the batch of Mel spectrograms.
        axis : int
            The axis along which to pad.

        Returns
        -------
        torch.Tensor
            The padded tensor.
        """
        if array.shape[axis] > n_samples:
            array = array.index_select(
                dim=axis, index=torch.arange(n_samples, device=array.device),
            )

        if array.shape[axis] < n_samples:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (
                0,
                n_samples - array.shape[axis],
            )
            array = nn.functional.pad(
                array, [pad for sizes in pad_widths[::-1] for pad in sizes]
            )

        return array

    def _get_mel(wavs):
        """Takes an input waveform and return its corresponding mel spectrogram
        according to HuggingFace implementation. WARNING: it's slow! Better push this
        in the DataLoader.

        Arguments
        ---------
        wavs : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """
        mels = _pad_or_trim(wavs)
        mels = _log_mel_spectrogram(mels)
        return mels

    def _get_encoder_states(wavs, output_all_hiddens):
        """Takes an input waveform and return its corresponding encoder states.
        Returns the last hidden state of the encoder or all hidden states if
        output_all_hiddens is True.
        Arguments
        ---------
        wavs : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """
        mel = _get_mel(wavs)
        if output_all_hiddens:
            states = model.encoder(mel, output_hidden_states=True)
            return torch.stack(states.hidden_states)
        else:
            return model.encoder(mel).last_hidden_state

    mel = _get_mel(wav)
    return model.encoder(mel).last_hidden_state


def forward_decoder(model, wav, decoder_input_ids, output_attentions=True):
    """Perform one step of the whisper decoder.
    Arguments
    ---------
    model : transformers.AutoModel
        A valid HuggingFace transformers model.
    wav : torch.Tensor
        A batch of audio features (mel + whisper encoding).
    decoder_input_ids : torch.Tensor
        A batch of decoder inputs tokens.
        The first tokens need to dictacte the behavior of the decoder.
        It needs to start with the bos_token, the language token,
        the task token, and finally the timestamp token.

        Please refer to the whisper paper for more details or go to the
        seq2seq2.py file in SpeechBrain to see how to generate the tokens
        with Greedy Search and/or Beam Search.
    output_attentions : bool
        Whether/not to return attention values as well.

    """
    output_states = model.decoder(
        encoder_hidden_states=wav,
        input_ids=decoder_input_ids,
        output_attentions=output_attentions,
    )

    attn = output_states.attentions[-1]
    attn = attn.view(attn.shape[0] * attn.shape[1], *attn.shape[2:])
    output_states = output_states.last_hidden_state

    logits = (
        output_states
        @ torch.transpose(
            model.decoder.embed_tokens.weight.to(output_states.dtype), 0, 1,
        )
    ).to(wav.dtype)

    return logits, attn


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
    """
    src_key_padding_mask = None
    if wav_len is not None:
        abs_len = torch.round(wav_len * src.shape[1])
        src_key_padding_mask = length_to_mask(abs_len).bool()

    return src_key_padding_mask


def make_masks(src, wav_len=None, pad_idx=0):
    """This method generates the padding masks.
    Arguments
    ---------
    src : tensor
        The sequence to the encoder (required).
    wav_len : tensor
        The relative length of the wav given in SpeechBrain format.
    pad_idx : int
        The index for <pad> token (default=0).
    """
    src_key_padding_mask = None
    if wav_len is not None:
        abs_len = torch.round(wav_len * src.shape[1])
        src_key_padding_mask = length_to_mask(abs_len).bool()

    return src_key_padding_mask
