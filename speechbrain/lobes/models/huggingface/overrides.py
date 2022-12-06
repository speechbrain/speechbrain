"""Helpers library to modify huggingface AutoModel instances.

Authors
 * Titouan Parcollet 2021
 * Boumadane Abdelmoumene 2021
 * Andreas Nautsch 2022
"""
import os
import torch
import pathlib
from huggingface_hub import model_info


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


def _check_model_source(path, save_path):
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
