"""Utilities for hparams files

Authors
 * Artem Ploujnikov 2021
"""


def choice(value, choices, default=None, apply=False):
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
    apply : bool
        if set to true, the selected choice will be invoked as a callable

    Returns
    -------
    The selected option out of the choices

    Example
    -------
    model: !new:speechbrain.lobes.models.g2p.model.TransformerG2P
        encoder_emb: !apply:speechbrain.utils.hparams.choice
            value: !ref <embedding_type>
            choices:
                regular: !ref <encoder_emb>
                normalized: !ref <encoder_emb_norm>
    """
    result = choices.get(value, default)
    if apply and callable(result):
        result = result()
    return result


def conditional(
    condition, value, condition_value=True, apply=False, fallback=None
):
    """A shorthand for choice for Boolean condittions

    Arguments
    ---------
    condition : any
        the condition to check (truthy if condition_value is not specified)
    value : any
        the value to return if the condition is True
    condition_value : any
        the value of the condition that evaluates to True
    apply : bool
        if set to true, the value is expected to
        be a callable, and the result of the call
        will be returned
    fallback : any
        the value to return if the condition is False

    Returns
    -------
    result : any
        value if the condition evaluates to true, fallback otherwise
    """
    return choice(
        value=condition,
        choices={
            condition_value: value,
        },
        default=fallback,
        apply=apply,
    )
