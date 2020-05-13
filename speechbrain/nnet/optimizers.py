"""
Optimizers for neural network training.

Author
------
Mirco Ravanelli 2020
Aku Rouhe 2020
"""

import torch
import logging
from speechbrain.utils import checkpoints

logger = logging.getLogger(__name__)


@checkpoints.register_checkpoint_hooks
class Optimize(torch.nn.Module):
    """This function implements different optimizers.

    Supports standard optimizers such as adam, sgd, rmseprop, and some of
    their variations such as as adamw, adamax, adadelta. The function takes
    in input some neural networks and updates their parameters according to
    the optimization algorithm adopted.

    Arguments
    ---------
    optimizer_type: str
        the type of optimizer to be used, one of (rmsprop,
        adam, adamw, adamax, adadelta, sgd, rprop). Refer to torch.nn
        documentation for a more detailed description of each optimizer.
    learning_rate: float
        the learning rate used to update the parameters.
    alpha: float
        smoothing constant used in rmseprop.
    betas: float
        coefficients used for computing running averages of gradient
        and its square in adam optimizer and its variations.
    etas: tuple (eta_minus : float, eta_plus : float)
        that are multiplicative increase and decrease factors, used in Rprop.
    eps: float
        the numerical stability factor.
    step_sizes: tuple (min_step : float, max_step : float)
        used in rprop optimizer to control allowed step sizes.
    weight_decay: float
        weight decay (L2 penalty) factor used as as additionally loss.
    momentum: float
        the momentum factor for the optimizers.
    dampening: float
        dampening factor for SGD momentum.
    rho: float
        it is used in adadelta and it is the coefficient used for
        computing a running average of squared gradients.
    centered: bool
        if True, compute the centered RMSProp, the gradient is
        normalized by an estimation of its variance.
    amsgrad: bool
        if True it uses the AMSGrad variant of the adam optimizer.
    nesterov: bool
        enables Nesterov momentum for SGD.

    Example
    -------
    >>> from speechbrain.nnet.linear import Linear
    >>> from speechbrain.nnet.losses import ComputeCost
    >>> inp_tensor = torch.rand([1,660,3])
    >>> model = Linear(n_neurons=4)
    >>> cost = ComputeCost(cost_type='nll')
    >>> optim = Optimize(optimizer_type='sgd', learning_rate=0.01)
    >>> output = model(inp_tensor, init_params=True)
    >>> prediction = torch.nn.functional.softmax(output, dim=2)
    >>> label = torch.FloatTensor([0,1,3]).unsqueeze(0)
    >>> lengths = torch.Tensor([1.0])
    >>> out_cost = cost(prediction, label, lengths)
    >>> out_cost.backward()
    >>> optim([model], init_params=True)
    """

    def __init__(
        self,
        optimizer_type,
        learning_rate,
        alpha=0.95,
        betas=[0.9, 0.999],
        etas=[0.5, 1.2],
        step_sizes=[1e-06, 50],
        eps=1e-8,
        weight_decay=0.0,
        momentum=0.0,
        dampening=0.0,
        rho=0.0,
        initial_accumulator_value=0.0,
        centered=False,
        amsgrad=False,
        nesterov=False,
    ):
        super().__init__()

        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.betas = betas
        self.etas = etas
        self.step_sizes = step_sizes
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dampening = dampening
        self.rho = rho
        self.initial_accumulator_value = initial_accumulator_value
        self.centered = centered
        self.amsgrad = amsgrad
        self.nesterov = nesterov

        # Dummy input for jitability
        self.optim = torch.Tensor([])

    def init_params(self, modules):
        # Making sure the input is class with parameters to optimize
        param_lst = []

        # Storing all the parameters to updated in the param_lst
        for module in modules:
            try:
                param_lst = param_lst + list(module.parameters())
            except AttributeError:
                err_msg = (
                    "The class optimize expected in input a list of"
                    "neural classes (nn.Module), but %s has no parameters"
                    % (module)
                )
                raise ValueError(err_msg)

        if self.optimizer_type == "rmsprop":
            self.optim = torch.optim.RMSprop(
                param_lst,
                lr=self.learning_rate,
                alpha=self.alpha,
                eps=self.eps,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
                centered=self.centered,
            )

        if self.optimizer_type == "adam":
            self.optim = torch.optim.Adam(
                param_lst,
                lr=self.learning_rate,
                betas=tuple(self.betas),
                eps=self.eps,
                weight_decay=self.weight_decay,
                amsgrad=self.amsgrad,
            )

        if self.optimizer_type == "adamw":
            self.optim = torch.optim.AdamW(
                param_lst,
                lr=self.learning_rate,
                betas=tuple(self.betas),
                eps=self.eps,
                weight_decay=self.weight_decay,
                amsgrad=self.amsgrad,
            )

        if self.optimizer_type == "adamax":
            self.optim = torch.optim.Adamax(
                param_lst,
                lr=self.learning_rate,
                betas=tuple(self.betas),
                eps=self.eps,
            )

        if self.optimizer_type == "adadelta":
            self.optim = torch.optim.Adadelta(
                param_lst,
                lr=self.learning_rate,
                rho=self.rho,
                eps=self.eps,
                weight_decay=self.weight_decay,
            )

        if self.optimizer_type == "sgd":
            self.optim = torch.optim.SGD(
                param_lst,
                lr=self.learning_rate,
                momentum=self.momentum,
                dampening=self.dampening,
                weight_decay=self.weight_decay,
                nesterov=self.nesterov,
            )

        if self.optimizer_type == "rprop":
            self.optim = torch.optim.Rprop(
                param_lst,
                lr=self.learning_rate,
                etas=tuple(self.etas),
                step_sizes=tuple(self.step_sizes),
            )

    def forward(self, input_lst, init_params=False):

        if init_params:
            self.init_params(input_lst)

        # Parameter update
        self.optim.step()

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
