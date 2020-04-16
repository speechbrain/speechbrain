"""
Optimizers.
"""

import torch
import logging
import functools
from speechbrain.utils import checkpoints 
logger = logging.getLogger(__name__)


@checkpoints.register_checkpoint_hooks
class optimize(torch.nn.Module):
    """This function implements different optimizers.

    Supports standard optimizers such as adam, sgd, rmseprop, and some of
    their variations such as as adamw, adamax, adadelta. The function takes
    in input some neural networks and updates their parameters according to
    the optimization algorithm adopted.

    Args:
        optimizer_type: the type of optimizer to be used, one of (rmsprop,
            adam, adamw, adamax, adadelta, sgd, rprop). Refer to torch.nn
            documentation for a more detailed description of each optimizer.
        learning_rate: the learning rate used to update the parameters.
        alpha: smoothing constant used in rmseprop.
        betas: coefficients used for computing running averages of gradient
            and its square in adam optimizer and its variations.
        etas: (etaminus, etaplis), that are multiplicative increase and
            decrease factors, used in Rprop.
        eps: it is the numerical stability factor.
        step_sizes: used in rprop optimizer and contains a pair of minimal
            and maximal allowed step sizes.
        weight_decay: it is the weight decay (L2 penalty) factor
            used as as additionally loss.
        momentum: it is the momentum factor for the optimizers.
        dampening: dampening factor for SGD momentum.
        rho: it is used in adadelta and it is the coefficient used for
            computing a running average of squared gradients.
        centered: if True, compute the centered RMSProp, the gradient is
            normalized by an estimation of its variance.
        amsgrad: if True it uses the AMSGrad variant of the adam optimizer.
        nesterov: enables Nesterov momentum for SGD.

    Example:
       >>> import torch
       >>> from speechbrain.nnet.architectures import linear
       >>> from speechbrain.nnet.architectures import activation
       >>> from speechbrain.nnet.losses import compute_cost
       >>> inp_tensor = torch.rand([1,660,3])
       >>> model = linear(n_neurons=4))
       >>> cost = compute_cost(cost_type='nll')
       >>> optim = optimize(optimizer_type='sgd', learning_rate=0.01)
       >>> prediction = torch.nn.functional.softmax(model(inp_tensor))
       >>> label = torch.FloatTensor([0,1,3]).unsqueeze(0)
       >>> lengths = torch.Tensor([1.0])
       >>> out_cost = cost(pred, label, lengths)
       >>> out_cost.backward()
       >>> optim([model])

    Author:
        Mirco Ravanelli 2020
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
        weight_decay=0.,
        momentum=0.,
        dampening=0.,
        rho=0.,
        initial_accumulator_value=0.,
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
                    % (inp)
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

    def forward(self, input_lst):

        # Gradient combination for the multi-gpu case
        self.sum_grad_multi_gpu(input_lst)

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
        Author:
            Aku Rouhe 2020
        """
        del end_of_epoch  # Unused here.
        self.optim.load_state_dict(torch.load(path))

    @checkpoints.mark_as_saver
    def _save(self, path):
        torch.save(self.optim.state_dict(), path)

    def sum_grad_multi_gpu(self, input_lst):
        """Sum all gradients from different gpus

        Args:
            input_list: list of all neural models to optimize

        Author:
            Mirco Ravanelli 2020
        """

        # Loops over all the input models
        for inp in input_lst:

            # Check if the computations are multi-gpu
            if hasattr(inp, "multi_gpu_models"):

                # list of all the parameters
                for index, param in enumerate(inp.parameters()):

                    first = True

                    # look for the models replicated over the various gpus
                    for model in inp.multi_gpu_models:

                        # model parameter in the current gpu
                        par_gpu = list(model.parameters())[index].grad

                        if first:
                            par_sum = par_gpu.to("cuda:0")
                            first = False
                        else:
                            par_sum = par_sum + par_gpu.to("cuda:0")

                    # Summing up all the gradients
                    param.grad = param.grad + par_sum

