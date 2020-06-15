"""Convenience functions for the simplest parameter transfer cases.

Use `speechbrain.utils.checkpoints.Checkpointer` to find a checkpoint
and the path to the parameter file.

Example
-------
>>> from speechbrain.utils.checkpoints import Checkpointer
>>> import torch
>>> # SETUP THE EXAMPLE:
>>> class Recoverable(torch.nn.Module):
...     def __init__(self, param):
...         super().__init__()
...         self.param = torch.nn.Parameter(torch.tensor([param]))
...     def forward(self, x):
...         return x * self.param
>>> tempdir = getfixture('tmpdir')
>>> recoverable = Recoverable(1.)
>>> recoverables = {'recoverable': recoverable}
>>> checkpointer = Checkpointer(tempdir, recoverables)
>>> saved_ckpt = checkpointer.save_checkpoint()
>>> # SETUP DONE
>>> # NOW FIND THE CHECKPOINT AND TRANSFER:
>>> new_recoverable = Recoverable(0.)
>>> checkpoint_finder = Checkpointer(tempdir)
>>> checkpoint_to_load = checkpoint_finder.find_checkpoint()
>>> paramfile = checkpoint_to_load.paramfiles["recoverable"]
>>> torch_parameter_transfer(new_recoverable, paramfile)
>>> assert new_recoverable(5.) == 5.

Authors
 * Aku Rouhe 2020
"""
import torch
import logging

logger = logging.getLogger(__name__)


def torch_parameter_transfer(obj, path):
    """Non-strict Torch Module state_dict load

    Loads a parameters from path to obj. If obj has layers for which
    parameters can't be found, only a warning is logged. Same thing
    if the path has parameters for layers which don't find a counterpart
    in obj.

    Arguments
    ---------
    obj : torch.nn.Module
        Instance for which to load the parameters
    path : str
        Path where to load from

    Returns
    -------
    None
        The object is modified in place
    """
    incompatible_keys = obj.load_state_dict(torch.load(path), strict=False)
    for missing_key in incompatible_keys.missing_keys:
        logger.warning(
            f"During parameter transfer to {obj} loading from "
            + f"{path}, the transferred parameters did not have "
            + f"parameters for the key: {missing_key}"
        )
    for unexpected_key in incompatible_keys.unexpected_keys:
        logger.warning(
            f"During parameter transfer to {obj} loading from "
            + f"{path}, the object could not use the parameters loaded"
            + f"with the key: {unexpected_key}"
        )
