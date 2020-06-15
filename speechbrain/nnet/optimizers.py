"""
Optimizers for neural network training.

Authors
 * Mirco Ravanelli 2020
 * Aku Rouhe 2020
 * Abdel Heba 2020
"""

import torch
import logging
from speechbrain.utils import checkpoints

logger = logging.getLogger(__name__)


@checkpoints.register_checkpoint_hooks
class RMSprop_Optimizer:
    """This class supports RMSprop optimizer.

    The function takes in input some neural networks and updates
    their parameters according to the RMSprop optimization algorithm.

    Arguments
    ---------
    learning_rate: float
        the learning rate used to update the parameters.
    alpha: float
        smoothing constant used in rmseprop.
    eps: float
        the numerical stability factor.
    weight_decay: float
        weight decay (L2 penalty) factor used as as additionally loss.
    momentum: float
        the momentum factor for the optimizers.
    centered: bool
        if True, compute the centered RMSProp, the gradient is
        normalized by an estimation of its variance.

    Example
    -------
    >>> from speechbrain.nnet.linear import Linear
    >>> from speechbrain.nnet.losses import nll_loss
    >>> inp_tensor = torch.rand([1,660,3])
    >>> model = Linear(n_neurons=4)
    >>> optim = RMSprop_Optimizer(learning_rate=0.01)
    >>> output = model(inp_tensor, init_params=True)
    >>> optim.init_params([model])
    >>> prediction = torch.nn.functional.log_softmax(output, dim=2)
    >>> label = torch.randint(3, (1, 660))
    >>> lengths = torch.Tensor([1.0])
    >>> out_cost = nll_loss(prediction, label, lengths)
    >>> out_cost.backward()
    >>> optim.step()
    >>> optim.zero_grad()
    """

    def __init__(
        self,
        learning_rate,
        alpha=0.95,
        eps=1e-8,
        weight_decay=0.0,
        momentum=0.0,
        centered=False,
    ):

        self.learning_rate = learning_rate
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered

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

        self.optim = torch.optim.RMSprop(
            param_lst,
            lr=self.learning_rate,
            alpha=self.alpha,
            eps=self.eps,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            centered=self.centered,
        )

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


@checkpoints.register_checkpoint_hooks
class Adam_Optimizer:
    """This class supports Adam optimizer.

    The function takes in input some neural networks and updates
    their parameters according to the Adam optimization algorithm.

    Arguments
    ---------
    learning_rate: float
        the learning rate used to update the parameters.
    betas: float
        coefficients used for computing running averages of gradient
        and its square in adam optimizer and its variations.
    eps: float
        the numerical stability factor.
    weight_decay: float
        weight decay (L2 penalty) factor used as as additionally loss.
    amsgrad: bool
        if True it uses the AMSGrad variant of the adam optimizer.

    Example
    -------
    >>> from speechbrain.nnet.linear import Linear
    >>> from speechbrain.nnet.losses import nll_loss
    >>> inp_tensor = torch.rand([1,660,3])
    >>> model = Linear(n_neurons=4)
    >>> optim = Adam_Optimizer(learning_rate=0.01)
    >>> output = model(inp_tensor, init_params=True)
    >>> optim.init_params([model])
    >>> prediction = torch.nn.functional.log_softmax(output, dim=2)
    >>> label = torch.randint(3, (1, 660))
    >>> lengths = torch.Tensor([1.0])
    >>> out_cost = nll_loss(prediction, label, lengths)
    >>> out_cost.backward()
    >>> optim.step()
    >>> optim.zero_grad()
    """

    def __init__(
        self,
        learning_rate,
        betas=[0.9, 0.999],
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
    ):

        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

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

        self.optim = torch.optim.Adam(
            param_lst,
            lr=self.learning_rate,
            betas=tuple(self.betas),
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )

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


@checkpoints.register_checkpoint_hooks
class Adadelta_Optimizer:
    """This class supports Adadelta optimizer.

    The function takes in input some neural networks and updates
    their parameters according to the Adadelta optimization algorithm.

    Arguments
    ---------
    learning_rate: float
        the learning rate used to update the parameters.
    rho: float
        it is used in adadelta and it is the coefficient used for
        computing a running average of squared gradients.
    eps: float
        the numerical stability factor.
    weight_decay: float
        weight decay (L2 penalty) factor used as as additionally loss.

    Example
    -------
    >>> from speechbrain.nnet.linear import Linear
    >>> from speechbrain.nnet.losses import nll_loss
    >>> inp_tensor = torch.rand([1,660,3])
    >>> model = Linear(n_neurons=4)
    >>> optim = Adadelta_Optimizer(learning_rate=0.01)
    >>> output = model(inp_tensor, init_params=True)
    >>> optim.init_params([model])
    >>> prediction = torch.nn.functional.log_softmax(output, dim=2)
    >>> label = torch.randint(3, (1, 660))
    >>> lengths = torch.Tensor([1.0])
    >>> out_cost = nll_loss(prediction, label, lengths)
    >>> out_cost.backward()
    >>> optim.step()
    >>> optim.zero_grad()
    """

    def __init__(
        self, learning_rate, rho=0.0, eps=1e-8, weight_decay=0.0,
    ):

        self.learning_rate = learning_rate
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay

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

        self.optim = torch.optim.Adadelta(
            param_lst,
            lr=self.learning_rate,
            rho=self.rho,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

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


@checkpoints.register_checkpoint_hooks
class SGD_Optimizer:
    """This class supports standard SGD optimizer.

    The function takes in input some neural networks and updates
    their parameters according to the SGD optimization algorithm.

    Arguments
    ---------
    learning_rate: float
        the learning rate used to update the parameters.
    momentum: float
        the momentum factor for the optimizers.
    dampening: float
        dampening factor for SGD momentum.
    weight_decay: float
        weight decay (L2 penalty) factor used as as additionally loss.
    nesterov: bool
        enables Nesterov momentum for SGD.

    Example
    -------
    >>> from speechbrain.nnet.linear import Linear
    >>> from speechbrain.nnet.losses import nll_loss
    >>> inp_tensor = torch.rand([1,660,3])
    >>> model = Linear(n_neurons=4)
    >>> optim = SGD_Optimizer(learning_rate=0.01)
    >>> output = model(inp_tensor, init_params=True)
    >>> optim.init_params([model])
    >>> prediction = torch.nn.functional.log_softmax(output, dim=2)
    >>> label = torch.randint(3, (1, 660))
    >>> lengths = torch.Tensor([1.0])
    >>> out_cost = nll_loss(prediction, label, lengths)
    >>> out_cost.backward()
    >>> optim.step()
    >>> optim.zero_grad()
    """

    def __init__(
        self,
        learning_rate,
        momentum=0.0,
        dampening=0.0,
        weight_decay=0.0,
        nesterov=False,
    ):

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
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

        self.optim = torch.optim.SGD(
            param_lst,
            lr=self.learning_rate,
            momentum=self.momentum,
            dampening=self.dampening,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,
        )

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


@checkpoints.register_checkpoint_hooks
class Rprop_Optimizer:
    """This class supports standard Rprop optimizer.

    The function takes in input some neural networks and updates
    their parameters according to the Rprop optimization algorithm.

    Arguments
    ---------
    learning_rate: float
        the learning rate used to update the parameters.
    etas: tuple (eta_minus : float, eta_plus : float)
        that are multiplicative increase and decrease factors, used in Rprop.
    step_sizes: tuple (min_step : float, max_step : float)
        used in rprop optimizer to control allowed step sizes.

    Example
    -------
    >>> from speechbrain.nnet.linear import Linear
    >>> from speechbrain.nnet.losses import nll_loss
    >>> inp_tensor = torch.rand([1,660,3])
    >>> model = Linear(n_neurons=4)
    >>> optim = Rprop_Optimizer(learning_rate=0.01)
    >>> output = model(inp_tensor, init_params=True)
    >>> optim.init_params([model])
    >>> prediction = torch.nn.functional.log_softmax(output, dim=2)
    >>> label = torch.randint(3, (1, 660))
    >>> lengths = torch.Tensor([1.0])
    >>> out_cost = nll_loss(prediction, label, lengths)
    >>> out_cost.backward()
    >>> optim.step()
    >>> optim.zero_grad()
    """

    def __init__(
        self, learning_rate, etas=[0.5, 1.2], step_sizes=[1e-06, 50],
    ):
        self.learning_rate = learning_rate
        self.etas = etas
        self.step_sizes = step_sizes

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

        self.optim = torch.optim.Rprop(
            param_lst,
            lr=self.learning_rate,
            etas=tuple(self.etas),
            step_sizes=tuple(self.step_sizes),
        )

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
