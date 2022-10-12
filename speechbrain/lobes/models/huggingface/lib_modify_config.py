"""Helpers library to modify huggingface AutoConfig instances.

Authors
 * Titouan Parcollet 2021
 * Boumadane Abdelmoumene 2021
 * Andreas Nautsch 2022
"""


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
