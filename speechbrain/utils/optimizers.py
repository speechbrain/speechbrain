"""Implements functions to avoid optimizing certain parameters

Authors
 * Titouan Parcollet 2023
"""


def rm_vector_weight_decay(
    modules,
):
    decay = []
    no_decay = []
    for name, param in modules.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay},
    ]
