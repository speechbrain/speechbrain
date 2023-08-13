"""Implements functions to avoid optimizing certain parameters

Authors
 * Titouan Parcollet 2023
"""


def rm_vector_weight_decay(modules):
    """Put vectors in a parameter group without weight decay

    Takes in a list of modules and separates their parameters into two parameter groups,
    which can be passed to a PyTorch Optimizer class. Vector parameters get weight_decay overridden to zero.
    This is particularly useful for biases and norms, which we expect to deviate from zero. Other vectors as parameters are also likely not meant to be pushed toward zero.

    Arguments
    ---------
    modules : torch.ModuleList, torch.Module
        Torch modules to operate on

    Returns
    -------
    list
        The parameter groups in the Pytorch Optimizer specification format.
    """
    decay = []
    no_decay = []
    for _, param in modules.named_parameters():
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
