"""Convenience functions for the simplest parameter transfer cases.

Use `speechbrain.utils.checkpoints.Checkpointer` to find a checkpoint
and the path to the parameter file.

Example:
    >>> from speechbrain.utils.checkpoints import Checkpointer
    >>> from speechbrain.utils.parameter_transfer \
            import torch_lazy_parameter_transfer
    >>> import torch
    >>> import tempfile
    >>> # SETUP THE EXAMPLE:
    >>> class Recoverable(torch.nn.Module):
    ...     def __init__(self, param):
    ...         super().__init__()
    ...         self.param = torch.nn.Parameter(torch.tensor([param]))
    ...     def forward(self, x):
    ...         return x * self.param
    >>> with tempfile.TemporaryDirectory() as tempdir:
    ...     recoverable = Recoverable(1.)
    ...     recoverables = {'recoverable': recoverable}
    ...     checkpointer = Checkpointer(tempdir, recoverables)
    ...     saved_ckpt = checkpointer.save_checkpoint()
    ...     # SETUP DONE
    ...     # NOW FIND THE CHECKPOINT AND TRANSFER:
    ...     new_recoverable = Recoverable(0.)
    ...     checkpoint_finder = Checkpointer(tempdir)
    ...     checkpoint_to_load = checkpoint_finder.find_checkpoint()
    ...     paramfile = checkpoint_to_load.paramfiles["recoverable"]
    ...     torch_lazy_parameter_transfer(new_recoverable, paramfile)
    ...     assert new_recoverable(5.) == 5.
"""
import torch
import functools
import logging

logger = logging.getLogger(__name__)


def torch_parameter_transfer(obj, path):
    """Non-strict Torch Module state_dict load

    Loads a parameters from path to obj. If obj has layers for which
    parameters can't be found, only a warning is logged. Same thing
    if the path has parameters for layers which don't find a counterpart
    in obj.

    Args:
        obj (torch.nn.Module): Instance for which to
            load the parameters
        path (str, pathlib.Path): Path where to load from
    Returns:
        None - the object is modified in place.
    Author:
        Aku Rouhe 2020
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


def torch_lazy_parameter_transfer(
    obj, path, load_method=torch_parameter_transfer
):
    """Init Torch object from path at first forward() call

    Loads a torch.nn.Module state_dict from the given path.
    The load is added as a lazy hook: the file is loaded and the parameters
    transferred the next time the Module is called.
    This is especially useful for the model initialization style widely
    used in SpeechBrain, where a model is initialized based on the input,
    as that initialization also happens at the first call.

    See the module docstring for example.

    Args:
        obj (instance of torch.nn.Module or derivative): Instance for which to
            load the parameters
        path (str, pathlib.Path): Path where to load from
        load_method (callable, optional): Callable with signature 
            (instance, path) [e.g. def load(self, path)], which actually 
            performs the parameter transfer from the given path. Defaults to
            `torch_parameter_transfer` which works for most torch
    Returns:
        None - Given object is modified in place
    Note:
        The hook is added as _speechbrain_lazy_recovery_hook[_X] where X
        is an integer specifying a unique hook number. You can add multiple
        parameter transfer hooks. (Not thread-safe)
    Author:
        Aku Rouhe 2020
    """
    if hasattr(obj, "_speechbrain_lazy_recovery_hook"):
        MSG = "Recovery hook present before parameter transfer. The recovered \
                parameters would be overwritten. Call parameter transfer before\
                recovery"
        raise RuntimeError(MSG)
    # Use this hook with functools.partial to save objpath and hook name properly
    # Otherwise, objpath is searched for dynamically (and has probably changed)
    def _lazy_transfer_hook(path, hookname, self, *input):
        load_method(self, path)
        getattr(self, hookname).remove()

    # Make a unique hook attribute name
    hookbase = "_speechbrain_lazy_transfer_hook"
    hook_x = 0
    while hasattr(obj, f"{hookbase}_{hook_x}"):
        hook_x += 1
    hook = functools.partial(_lazy_transfer_hook, path, f"{hookbase}_{hook_x}")
    setattr(obj, f"{hookbase}_{hook_x}", obj.register_forward_pre_hook(hook))
