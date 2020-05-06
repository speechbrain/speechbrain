"""
Learning rate schedulers.

Author
------
Mirco Ravanelli 2020
"""

import math
import torch
import logging
from speechbrain.utils import checkpoints

logger = logging.getLogger(__name__)


@checkpoints.register_checkpoint_hooks
class LRScheduler(torch.nn.Module):
    """Learning rate schedulers

    This function implements different strategies for lerarning rate
    annealing. It supports time decay, step decay, exp decay, new bob,
    custom, and constant annealing. Unless, differently specified in the
    optim_list parameter, this function searches for all the optimizers
    defined by the user and changes their learning rate according to the
    selected strategy.

    Arguments
    ---------
    annealing_type : str
        It is the type of learning rate annealing used.
        - "constant": no learning rate annealing

        - "time_decay": linear decay over the epochs
        lr=lr_ini/(epoch_decay*(epoch))

        - "step_decay": decay over the epochs with the selected epoch_decay
        factor r=self.lr_int*epoch_decay^((1+epoch)/self.epoch_drop)

        -"exp_decay": exponential over the epochs selected epoch_decay factor
        r=lr_ini*exp^(-self.exp_decay*epoch)

        -"newbob": the learning rate is annealed based on the validation
        peformance. In particular: if (past_loss-current_loss)/past_loss
        < impr_threshold: r=lr * annealing_factor

        -"custom": the learning rate is set by the user with an external
        array (with length equal to the number of epochs)

    annealing_factor : float
        It is annealing factor used in new_bob strategy.
    improvement_threshold : float
        It is improvement rate between losses used to perform learning
        annealing in new_bob strategy.
    lr_at_epoch : float
        It is a float containing the learning rates to use for each epoch in
        the "custom" setting. The length of the list must be equal to the
        number of epochs.
    N_epochs: int
        It is the total number of epoch.
    decay : float
        It is improvement rate between losses used to perform learning
        annealing in new_bob strategy.
    lr_initial : float
        It is the initial learning rate (i.e. the lr used at epoch 0).
    epoch_decay : float
        It is the decay factor used in time and step decay strategies.
    epoch_decay : float
        It is the decay factor used in the exponential decay strategy.
    epoch_drop: float
        It is the drop factor used in step decay.
    patient : int
        It is used in new_bob setting. When the annealing condition is
        violeted patient times, the learning rate is finally reduced.
    optim_list : list
        If None, the code search for all the optimizers defined by the users
        and performs annealing of all of them. If this is not what the user
        wants, one can specify here the list on optimizers.
    """

    def __init__(
        self,
        annealing_type,
        annealing_factor=0.5,
        improvement_threshold=0.0025,
        lr_at_epoch=None,
        N_epochs=None,
        lr_initial=None,
        epoch_decay=0.5,
        exp_decay=0.1,
        epoch_drop=2,
        patient=0,
    ):
        super().__init__()

        self.annealing_type = annealing_type
        self.annealing_factor = annealing_factor
        self.improvement_threshold = improvement_threshold
        self.lr_at_epoch = lr_at_epoch
        self.N_epochs = N_epochs
        self.lr_initial = lr_initial
        self.epoch_decay = epoch_decay
        self.exp_decay = exp_decay
        self.epoch_drop = epoch_drop
        self.patient = patient

        # Additional checks on the input when annealing_type=='custom
        self._check_annealing_type()

        # Initalizing the list that stored the losses/errors
        self.losses = []

        # Setting current patient
        self.current_patient = self.patient

    def forward(self, optim_list, current_epoch, current_loss):
        """
        Arguments
        ---------
        optim_list : list of optimizers
            The optimizers to update using this scheduler.
        current_epoch : int
            Number of times the dataset has been iterated.
        current_loss : int
            A number for determining whether to change the learning rate.

        Returns
        -------
        float
            The updated (possibly to the same value) learning rate.
        """

        for opt in optim_list:

            # Current learning rate
            current_lr = opt.optim.param_groups[0]["lr"]
            next_lr = current_lr

            # Managing newbob annealing
            if self.annealing_type == "newbob":

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
            opt.optim.param_groups[0]["lr"] = next_lr
            opt.optim.param_groups[0]["prev_lr"] = current_lr

            if next_lr != current_lr:
                logger.info(
                    "Changing lr from %.2g to %.2g" % (current_lr, next_lr)
                )

        # Appending current loss
        self.losses.append(current_loss)

        return current_lr, next_lr

    def _check_annealing_type(self):
        """This support function checks that all the fields are specified
        correctly when using the custom learning rate scheduler.
        """

        if self.annealing_type == "custom":
            # Making sure that lr_at_epoch is a list that contains a number
            # of elements equal to the number of epochs.
            if self.lr_at_epoch is None:

                err_msg = (
                    "The field lr_at_epoch must be a list composed of"
                    "N_epochs elements when annealing_type=custom"
                )

                logger.error(err_msg, exc_info=True)

            # Checking the N_epochs field
            if self.N_epochs is None:

                err_msg = (
                    "The field N_epochs must be specified when"
                    "annealing_type=custom=custom"
                )

                logger.error(err_msg, exc_info=True)

            # Making sure that the list of learning rate specified by
            # the user has length = N_epochs:
            if len(self.lr_at_epoch) != self.N_epochs:

                err_msg = (
                    "The field lr_at_epoch must be a list composed of"
                    "N_epochs elements when annealing_type=custom."
                    "Got a list of %i elements (%i expected)"
                ) % (len(self.lr_at_epoch), len(self.N_epochs))

                logger.error(err_msg, exc_info=True)

    @checkpoints.mark_as_saver
    def save(self, path):
        data = {"losses": self.losses, "current_patient": self.current_patient}
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch):
        del end_of_epoch  # Unused in this class
        data = torch.load(path)
        self.losses = data["losses"]
        self.current_patient = data["current_patient"]
