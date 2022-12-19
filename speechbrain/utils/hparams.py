"""Utilities for hparams files

Authors
 * Artem Ploujnikov 2021
"""


def choice(value, choices, default=None):
    """
    The equivalent of a "switch statement" for hparams files. The typical use case
    is where different options/modules are available, and a top-level flag decides
    which one to use

    Arguments
    ---------
    value: any
        the value to be used as a flag
    choices: dict
        a dictionary maps the possible values of the value parameter
        to the corresponding return values
    default: any
        the default value

    Example
    -------
    model: !new:speechbrain.lobes.models.g2p.model.TransformerG2P
        encoder_emb: !apply:speechbrain.utils.hparams.choice
            value: !ref <embedding_type>
            choices:
                regular: !ref <encoder_emb>
                normalized: !ref <encoder_emb_norm>
    """
    return choices.get(value, default)
