"""
 -----------------------------------------------------------------------------
 lr_scheduling.py

 Description: This library implements different learning rate schedulers.
 -----------------------------------------------------------------------------
"""

import math
import torch.nn as nn
from speechbrain.utils.input_validation import check_opts, check_inputs
from speechbrain.utils.logger import logger_write


class lr_annealing(nn.Module):
    """
     -------------------------------------------------------------------------
     nnet.lr_scheduling.lr_annealing (author: Mirco Ravanelli)

     Description:  This function implements different strategies for lerarning
                   rate annealing. It supports time decay, step decay, exp
                   decay, new bob, custom, and constant annealing. Unless,
                   differently specified in the optim_list parameter, this
                   function searches for all the optimizers defined by the
                   user and anneals their learning rate with the selected
                   strategy.

     Input (init):  - config (type, dict, mandatory):
                       it is a dictionary containing the keys described below.

                           - annealing_type (one_of(constant,time_decay,
                                             step_decay,exp_decay,newbob,
                                             custom), mandatory):
                               it is the type of learning rate annealing used.
                               "constant": no learning rate annealing
                               "time_decay": linear decay over the epochs
                                             lr=lr_ini/(epoch_decay*(epoch))

                               "step_decay":  decay over the epochs with the
                                              selected epoch_decay factor
                                              lr=self.lr_int*epoch_decay^\
                                                  ((1+epoch)/self.epoch_drop)


                               "exp_decay":   exponential over the epochs
                                              selected epoch_decay factor
                                              lr=lr_ini*exp^(-self.\
                                                  exp_decay*epoch)

                               "newbob":     the learning rate is annealed
                                              based on the validation
                                              peformance. In particular:

                                              if (past_loss-current_loss)/
                                              past_loss < impr_threshold:

                                              lr=lr * annealing_factor


                              "custom":       the learning rate is set by the
                                              user with an external array (with
                                              length equal to the number of
                                              epochs)


                           - annealing_factor (float, optional, Default: 0.5):
                               it is annealing factor used in new_bob strategy.

                           - improvement_threshold (float, optional, Default:\
                               0.0025):
                               it is improvement rate between losses used to
                               perform learning annealing in new_bob strategy.

                           - lr_at_epoch (float_list, optional, Default: None):
                               it is a float containing the learning rates to
                               use for each epoch in the "custom" setting.
                               The length of the list must be equal to the
                               number of epochs.

                           - N_epochs (int(1,inf), optional, Default: None):
                               it is the total number of epoch.

                           - decay (float, optional, Default: None):
                               it is improvement rate between losses used to
                               perform learning annealing in new_bob strategy.

                          - lr_initial (float, optional, Default: None):
                               it is the initial learning rate (i.e. the lr
                               used at epoch 0).

                          - epoch_decay (float, optional, Default: 0.5):
                               it is the decay factor used in time and step
                               decay strategies.

                          - epoch_decay (float, optional, Default: 0.1):
                               it is the decay factor used in the exponential
                               decay strategy.

                          - epoch_drop (float, optional, Default: 2):
                               it is the drop factor used in step decay.

                           - patient (int(0,inf), optional, Default: 0):
                               it is used in new_bob setting. When the
                               annealing condition is violeted patient times,
                               the learning rate is finally reduced.

                           - optim_list (str_lst, optional, Default: None):
                               If None, the code search for all the optimizers
                               defined by the users and perform annealing of
                               all of them. If this is not what the user wants,
                               one can specify here the list on optimizers
                               whose lr must be annaled.


                   - funct_name (type, str, optional, default: None):
                       it is a string containing the name of the parent
                       function that has called this method.

                   - global_config (type, dict, optional, default: None):
                       it a dictionary containing the global variables of the
                       parent config file.

                   - logger (type, logger, optional, default: None):
                       it the logger used to write debug and error messages.
                       If logger=None and root_cfg=True, the file is created
                       from scratch.

                   - first_input (type, list, optional, default: None)
                      this variable allows users to analyze the first input
                      given when calling the class for the first time.


     Input (call): - inp_lst(type, list, mandatory):
                       it is a list containing [current_epoch, current_loss]



     Output (call): - None
                       This function returns "None" when called. It directly
                       changes the learning rates withing all the optimizers.



     Example:   import torch
                from speechbrain.nnet.optimizers import optimize
                from speechbrain.nnet.lr_scheduling import lr_annealing


                # Initialization of the optimizer
                config={'class_name':'speechbrain.nnet.optimizers.optimizer',
                        'optimizer_type': 'sgd',
                        'learning_rate': '0.01'
                        }

                optim=optimize(config, first_input=[model])


                # Initialization of the lr scheduler
                config={'class_name':'speechbrain.nnet.lr_scheduler.lr_annealing',
                        'annealing_type':'exp_decay',
                        'lr_initial' : '0.01'
                        }

                # Creating the list of function (that must contain the
                # optimizer)

                funct_dict={}
                funct_dict['optimizer']=optim
                update_lr=lr_annealing(config, functions=funct_dict)

                print(funct_dict['optimizer'].optim.param_groups[0]['lr'])
                update_lr([0, 1.2])
                print(funct_dict['optimizer'].optim.param_groups[0]['lr'])
                update_lr([2, 1.2])
                print(funct_dict['optimizer'].optim.param_groups[0]['lr'])
     """

    def __init__(
        self,
        config,
        funct_name=None,
        global_config=None,
        functions=None,
        logger=None,
        first_input=None,
    ):
        super(lr_annealing, self).__init__()

        # Logger setup
        self.logger = logger

        # Function list
        self.functions = functions

        # Here are summarized the expected options for this class
        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "annealing_type": (
                "one_of(constant,time_decay,step_decay,exp_decay,\
                    newbob,custom)",
                "mandatory",
            ),
            "annealing_factor": ("float", "optional", "0.50"),
            "improvement_threshold": ("float", "optional", "0.0025"),
            "lr_at_epoch": ("float_list", "optional", "None"),
            "N_epochs": ("int", "optional", "None"),
            "lr_initial": ("float", "optional", "None"),
            "epoch_decay": ("float", "optional", "0.5"),
            "exp_decay": ("float", "optional", "0.1"),
            "epoch_drop": ("float", "optional", "2"),
            "patient": ("int", "optional", "0"),
            "optim_list": ("str_list", "optional", "None"),
        }

        # Check, cast , and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, self.logger
        )

        # Definition of the expected input
        self.expected_inputs = ["int", "torch.Tensor"]

        # Check the first input
        check_inputs(
            self.conf, self.expected_inputs, first_input, logger=self.logger
        )

        # Getting the list of optimizers
        self.optimizers = self.get_optimizer_list()

        # Additional checks on the input when annealing_type=='custom
        self.check_annealing_type()

        # Initalizing the list that stored the losses/errors
        self.losses = []

        # Setting current patient
        self.current_patient = self.patient

    def forward(self, input_lst):

        # Current epoch
        current_epoch = input_lst[0] + 1

        # Current loss
        current_loss = input_lst[1]

        for opt in self.optimizers:

            # Current learning rate
            current_lr = self.optimizers[opt].optim.param_groups[0]["lr"]

            # Managing newbob annealing
            if self.annealing_type == "newbob":

                next_lr = current_lr

                if len(self.losses) > 0:
                    if (self.losses[-1] - current_loss) / self.losses[
                        -1
                    ] < self.improvement_threshold:
                        if self.current_patient == 0:
                            next_lr = current_lr * self.annealing_factor
                            self.current_patient = self.patient
                        else:
                            self.current_patient = self.current_patient - 1

            # Managing newbob annealing
            if self.annealing_type == "custom":
                next_lr = self.lr_at_epoch[current_epoch]

            # Managing time_decay
            if self.annealing_type == "time_decay":
                next_lr = self.lr_initial / (
                    1 + self.epoch_decay * (current_epoch)
                )

            # Managing step_decay
            if self.annealing_type == "step_decay":
                next_lr = self.lr_initial * math.pow(
                    self.epoch_decay,
                    math.floor((1 + current_epoch) / self.epoch_drop),
                )

            # Managing exp_decay
            if self.annealing_type == "exp_decay":
                next_lr = self.lr_initial * math.exp(
                    -self.exp_decay * current_epoch
                )

            # Changing the learning rate
            self.optimizers[opt].optim.param_groups[0]["lr"] = next_lr
            self.optimizers[opt].optim.param_groups[0]["prev_lr"] = current_lr

        # Appending current loss
        self.losses.append(current_loss)

        return

    def get_optimizer_list(self):
        """
         ----------------------------------------------------------------------
         nnet.lr_scheduling.lr_annealing.get_optimizer_list (M. Ravanelli)

         Description:  This support function searches which of the functions
                       collected in the self.function dict are actually
                       optimizers. The function returns a dictionary containing
                       all the optimizers found.

         Input (call):  None


         Output (call): - optimizers(type, list):
                           dictionary contaning the optimizer objects.

         Example:   import torch
                    from speechbrain.nnet.architectures import linear
                    from speechbrain.nnet.optimizers import optimize
                    from speechbrain.nnet.lr_scheduling import lr_annealing

                    # Definition the input tensor
                    inp_tensor = torch.rand([1,660,3])

                    # Initialization of the linear class
                    config={'class_name':'speechbrain.nnet.architectures.linear',
                            'n_neurons':'4'}

                    model=linear(config,first_input=[inp_tensor])

                    # Initialization of the optimizer
                    config={'class_name':'speechbrain.nnet.optimizers.optimizer',
                            'optimizer_type': 'sgd',
                            'learning_rate': '0.01'
                            }

                    optim=optimize(config, first_input=[model])


                    # Initialization of the lr scheduler
                    config={'class_name':'speechbrain.nnet.lr_scheduler.lr_annealing',
                            'annealing_type':'exp_decay',
                            'lr_initial' : '0.01'
                            }

                    # Creating the list of function (that must contain the
                    # optimizer)

                    funct_dict={}
                    funct_dict['optimizer']=optim
                    funct_dict['nnet']=model


                    update_lr=lr_annealing(config, functions=funct_dict)

                    print(funct_dict.keys())

                    print(update_lr.get_optimizer_list())

         """
        # Dictionary of possible optimizers
        optimizers = {}

        # Looking for all the possible optimizers
        for funct in self.functions:

            # the the function has the parameter 'optim' it is an optimizer
            if "optim" in self.functions[funct].__dict__:

                optimizers[funct] = self.functions[funct]

        # Additional checks on optim_list
        if self.optim_list is not None:

            selected_optim = {}

            # Check if the optimizers in the list actually exist
            for opt_name in self.optim_list:

                if opt_name not in optimizers:
                    err_msg = (
                        "The optimizer %s specified in the list optim_list "
                        "does not exisit (it is not specified in any config "
                        "file). Got %s, possible optimizers are %s"
                        % (opt_name, optimizers.keys())
                    )

                    logger_write(err_msg, logfile=self.logger)
                else:
                    # Adding optimizer
                    selected_optim[opt_name] = optimizers[opt_name]

            optimizers = selected_optim

        return optimizers

    def check_annealing_type(self):
        """
         ----------------------------------------------------------------------
         nnet.lr_scheduling.lr_annealing.check_annealing_type (M. Ravanelli)

         Description:  This support function that all the fields are
                       correctly specified when using the custom learning
                       rate scheduler.

         Input (call):  None

         Output (call): None

         Example:   import torch

                    from speechbrain.nnet.lr_scheduling import lr_annealing

                    # Initialization of the lr scheduler
                    config={'class_name':'speechbrain.nnet.lr_scheduler.lr_annealing',
                            'annealing_type':'custom',
                            }

                    # The check_annealing_type is called during init
                    # and raises an error becuase some of the needed parameters
                    # are not specified.

                    update_lr=lr_annealing(config, functions={})

         """

        if self.annealing_type == "custom":

            # Making sure that lr_at_epoch is a list that contains a number
            # of elements equal to the number of epochs.
            if self.lr_at_epoch is None:

                err_msg = (
                    "The field lr_at_epoch must be a list composed of"
                    "N_epochs elements when annealing_type=custom"
                )

                logger_write(err_msg, logfile=self.logger)

            # Checking the N_epochs field
            if self.N_epochs is None:

                err_msg = (
                    "The field N_epochs must be specified when"
                    "annealing_type=custom=custom"
                )

                logger_write(err_msg, logfile=self.logger)

            # Making sure that the list of learning rate specified by
            # the user has length = N_epochs:
            if len(self.lr_at_epoch) != self.N_epochs:

                err_msg = (
                    "The field lr_at_epoch must be a list composed of"
                    "N_epochs elements when annealing_type=custom."
                    "Got a list of %i elements (%i expected)"
                ) % (len(self.lr_at_epoch), len(self.N_epochs))

                logger_write(err_msg, logfile=self.logger)
