"""
-----------------------------------------------------------------------------
optimizers.py

Description: This library implements different optimizers.
-----------------------------------------------------------------------------
"""

import torch
import logging
from speechbrain.data_io.data_io import recovery
logger = logging.getLogger(__name__)


class optimize(torch.nn.Module):
    """
    -------------------------------------------------------------------------
    nnet.optimizers.optimize (author: Mirco Ravanelli)

    Description:
        This function implements different optimizers.
        It supports standard optimizers such as adam, sgd, rmseprop,
        and some of their variations such as as adamw, adamax,
        adadelta. The function takes in input some neural networks
        and updates their parameters according to the optimization
        algorithm adopted.

    Input:
       - optimizer_type (one_of(rmsprop,adam,adamw,adamax,
                        adadelta,sgd,rprop), mandatory):
          it is the type of optimizer to be used.
          Refer to torch.nn documentation of a more
          detailed description of each optimizer.


      - learning_rate (float, mandatory):
          it is the learning rate used to update the
          parameters.

      - alpha (float, optional, Default:0.95):
          it is used the smoothing constant used in
          rmseprop.

      - betas (float_list, optional, Default:0.95):
           are coefficients used for computing running
           averages of gradient and its square in adam
           optimizer and its variations.

      - etas (float_list, optional, Default:0.5,1.2):
          yt is used in Rprop optimizer. It is a
          pair of (etaminus, etaplis), that are
          multiplicative increase and decrease factors.

      - eps (float, optional, Default:1e-8):
          it is the numerical stability factor.

      - step_sizes (float_list, optional,
                    Default: 1e-06, 50):
         It is used in rprop optimizer and contains a
         pair of minimal and maximal allowed step sizes.

      - weight_decay (int, optional, Default: 0):
          it is the weight decay (L2 penalty) factor
          used as as additionally loss.

      - momentum (float, optional, Default: 0.0):
         it is the momentum factor for the optimizers.

      - dampening (float, optional, Default: 0.0):
          it is  dampening facror for SGD momentum.

      - rho (float, optional, Default: 0.0):
          it is used in adadelta and it is the coefficient
          used for computing a running average of
          squared gradients.

      - centered (bool, optional, Default: False):
          if True, compute the centered RMSProp, the
          gradient is normalized by an estimation of
          its variance.

      - amsgrad (bool, optional, Default: False):
           if True it uses the AMSGrad variant of the
           adam optimizer.

      - nesterov (bool, optional, Default: False):
           it enables Nesterov momentum for SGD.

      - do_recovery (type: bool, optional, Default:True):
          if True, the system restarts from the last
          epoch correctly executed.


    Example:
       import torch
       from speechbrain.nnet.architectures import linear
       from speechbrain.nnet.architectures import activation
       from speechbrain.nnet.losses import compute_cost
       from speechbrain.nnet.optimizers import optimize

       # Definition the input tensor
       inp_tensor = torch.rand([1,660,3])

       # Initialization of the linear class
       config={'class_name':'speechbrain.nnet.architectures.linear',
               'n_neurons':'4'}

       model=linear(config,first_input=[inp_tensor])


       # Initialization of the log_softmax class
       config={'class_name':'speechbrain.nnet.architectures.activation',
               'act_type':'log_softmax',
               }

       softmax=activation(config, first_input=[inp_tensor])


       # Initialization of the loss function
       config={'class_name':'speechbrain.nnet.losses.compute_cost',
               'cost_type':'nll'}

       cost=compute_cost(config)

       # Initialization of the optimizer
       config={'class_name':'speechbrain.nnet.optimizers.optimizer',
               'optimizer_type': 'sgd',
               'learning_rate': '0.01'
               }

       optim=optimize(config, first_input=[model])


       # Computatitions of the prediction for the current input
       pre_act=model([inp_tensor])
       pred = softmax([pre_act])

       # fake label
       label=torch.FloatTensor([0,1,3]).unsqueeze(0)
       lengths=torch.Tensor([1.0])

       out_cost= cost([pred,label,lengths])

       print(out_cost)

       # back propagation
       out_cost.backward()

       print(list(model.parameters()))

       # applying optimization
       optim([model])

       print(list(model.parameters()))

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
        do_recovery=True,
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
        self.recovery = do_recovery

        def hook(self, input):

            # Making sure the input is class with parameters to optimize
            param_lst = []

            # Storing all the parameters to updated in the param_lst
            for inp in input[0]:

                try:
                    param_lst = param_lst + list(inp.parameters())
                except Exception:

                    err_msg = (
                        "The class optimize expected in input a list of"
                        "neural classes (nn.Module), but %s has no parameters"
                        % (inp)
                    )

                    logger.error(err_msg, exc_info=True)

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

            # Automatic recovery
            # if global_config is not None:
            #    recovery(self)

            self.hook.remove()

        self.hook = self.register_forward_pre_hook(hook)

    def forward(self, input_lst):
        """
        Input (call): - inp_lst(type, list, mandatory):
                       it is a list containing the neural networks to optimize.
        """

        # Gradient combination for the multi-gpu case
        self.sum_grad_multi_gpu(input_lst)

        # Parameter update
        self.optim.step()

        # Zeroing gradient buffers
        self.optim.zero_grad()

    def sum_grad_multi_gpu(self, input_lst):
        """
         ----------------------------------------------------------------------
         nnet.optimizers.optimize.sum_grad_multi_gpu (author: Mirco Ravanelli)

         Description: This support function is used in the multi-gpu scenario
                      and sums all the gradients from the different gpus.

         Input (call):    - input_lst (type: list, mandatory):
                               list of all the neural models to optimize.


         Output (call):  None:
                          the gradient is directly updated in the reference
                          device (which is by default cuda:0).

         ----------------------------------------------------------------------
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
