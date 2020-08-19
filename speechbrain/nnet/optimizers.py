"""
Optimizers for neural network training.

Authors
 * Mirco Ravanelli 2020
 * Aku Rouhe 2020
 * Abdel Heba 2020
 * Peter Plantinga 2020
"""

import torch
import logging
from speechbrain.utils import checkpoints

logger = logging.getLogger(__name__)


@checkpoints.register_checkpoint_hooks
class Optimizer:
    """Generic pytorch optimizer wrapper.

    Arguments
    ---------
    optimizer : class
        A ``torch.optim`` class definition.
    *args, **kwargs
        Arguments to pass to the optimizer upon initialization

    Example
    -------
    >>> from torch.optim import SGD
    >>> from torch.nn import Linear
    >>> opt = Optimizer(SGD, lr=0.1)
    >>> model = Linear(10, 10)
    >>> opt.init_params([model])
    """

    def __init__(self, OptClass, *args, **kwargs):
        self.OptClass = OptClass
        self.args = args
        self.kwargs = kwargs

    def init_params(self, module_list):
        # Making sure the input is class with parameters to optimize
        param_lst = []

        # Storing all the parameters to updated in the param_lst
        for module in module_list:
            try:
                param_lst = param_lst + list(module.parameters())
            except AttributeError:
                err_msg = (
                    "The class optimize expected in input a list of"
                    "neural classes (nn.Module), but %s has no parameters"
                    % (module)
                )
                raise ValueError(err_msg)

        self.optim = self.OptClass(param_lst, *self.args, **self.kwargs,)

    def step(self):
        # Parameter update
        self.optim.step()

    def zero_grad(self):
        # Zeroing gradient buffers
        self.optim.zero_grad()

    @checkpoints.mark_as_loader
    def _recovery(self, path, end_of_epoch):
        """Lazy recovery of self.optim

        Need special recovery because here the forward() should not and
        need not be run before recovery of optimize; so we use forward_pre_hook

        (In many cases the forward does need to be run so that submodules
        get initialized first, using forward_hook and rerunning forward())
        """
        del end_of_epoch  # Unused here.
        self.optim.load_state_dict(torch.load(path))

    @checkpoints.mark_as_saver
    def _save(self, path):
        torch.save(self.optim.state_dict(), path)
