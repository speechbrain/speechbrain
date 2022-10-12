"""Helpers library to modify huggingface AutoModel instances.

Authors
 * Titouan Parcollet 2021
 * Boumadane Abdelmoumene 2021
 * Andreas Nautsch 2022
"""


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
