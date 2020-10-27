"""
Schedulers for updating hyperparameters (such as learning rate).

Authors
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
"""

import math
import torch
import logging
from speechbrain.utils import checkpoints

logger = logging.getLogger(__name__)


def update_learning_rate(optimizer, new_lr, param_group=None):
    """Change the learning rate value within an optimizer.

    Arguments
    ---------
    optimizer : torch.optim object
        Updates the learning rate for this optimizer
    new_lr : float
        The new value to use for the learning rate.
    param_group : list of int
        The param group indices to update. If not provided, all groups updated.

    Example
    -------
    >>> from torch.optim import SGD
    >>> from speechbrain.nnet import Linear
    >>> model = Linear(n_neurons=10, input_size=10)
    >>> optimizer = SGD(model.parameters(), lr=0.1)
    >>> update_learning_rate(optimizer, 0.2)
    >>> optimizer.param_groups[0]["lr"]
    0.2
    """
    # Iterate all groups if none is provided
    if param_group is None:
        groups = range(len(optimizer.param_groups))

    for i in groups:
        old_lr = optimizer.param_groups[i]["lr"]

        # Change learning rate if new value is different from old.
        if new_lr != old_lr:
            optimizer.param_groups[i]["lr"] = new_lr
            optimizer.param_groups[i]["prev_lr"] = old_lr
            logger.info("Changing lr from %.2g to %.2g" % (old_lr, new_lr))


@checkpoints.register_checkpoint_hooks
class NewBobScheduler:
    """Scheduler with new-bob technique, used for LR annealing.

    The learning rate is annealed based on the validation peformance.
    In particular: if (past_loss-current_loss)/past_loss< impr_threshold:
    lr=lr * annealing_factor

    Arguments
    ---------
    initial_value : float
        The initial hyperparameter value.
    annealing_factor : float
        It is annealing factor used in new_bob strategy.
    improvement_threshold : float
        It is improvement rate between losses used to perform learning
        annealing in new_bob strategy.
    patient : int
        When the annealing condition is violeted patient times,
        the learning rate is finally reduced.

    Example
    -------
    >>> scheduler = NewBobScheduler(initial_value=1.0)
    >>> scheduler(metric_value=10.0)
    (1.0, 1.0)
    >>> scheduler(metric_value=2.0)
    (1.0, 1.0)
    >>> scheduler(metric_value=2.5)
    (1.0, 0.5)
    """

    def __init__(
        self,
        initial_value,
        annealing_factor=0.5,
        improvement_threshold=0.0025,
        patient=0,
    ):
        self.hyperparam_value = initial_value
        self.annealing_factor = annealing_factor
        self.improvement_threshold = improvement_threshold
        self.patient = patient
        self.metric_values = []
        self.current_patient = self.patient

    def __call__(self, metric_value):
        """Returns the current and new value for the hyperparameter.

        Arguments
        ---------
        metric_value : int
            A number for determining whether to change the hyperparameter value.
        """
        old_value = new_value = self.hyperparam_value
        if len(self.metric_values) > 0:
            prev_metric = self.metric_values[-1]

            # Update value if improvement too small and patience is 0
            improvement = (prev_metric - metric_value) / prev_metric
            if improvement < self.improvement_threshold:
                if self.current_patient == 0:
                    new_value *= self.annealing_factor
                    self.current_patient = self.patient
                else:
                    self.current_patient -= 1

        # Store relevant info
        self.metric_values.append(metric_value)
        self.hyperparam_value = new_value

        return old_value, new_value

    @checkpoints.mark_as_saver
    def save(self, path):
        data = {
            "hyperparam_value": self.hyperparam_value,
            "metric_values": self.metric_values,
            "current_patient": self.current_patient,
        }
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch, device=None):
        del end_of_epoch  # Unused in this class
        data = torch.load(path)
        self.hyperparam_value = data["hyperparam_value"]
        self.metric_values = data["metric_values"]
        self.current_patient = data["current_patient"]


class LinearScheduler:
    """Scheduler with linear annealing technique.

    The learning rate linearly decays over the specified number of epochs.

    Arguments
    ---------
    initial_value : float
        The value upon initialization.
    final_value : float
        The value used when the epoch count reaches ``epoch_count - 1``.
    epoch_count : int
        Number of epochs.

    Example
    -------
    >>> scheduler = LinearScheduler(1.0, 0.0, 4)
    >>> scheduler(current_epoch=1)
    (1.0, 0.666...)
    >>> scheduler(current_epoch=2)
    (0.666..., 0.333...)
    >>> scheduler(current_epoch=3)
    (0.333..., 0.0)
    >>> scheduler(current_epoch=4)
    (0.0, 0.0)
    """

    def __init__(self, initial_value, final_value, epoch_count):
        self.value_at_epoch = torch.linspace(
            initial_value, final_value, steps=epoch_count
        ).tolist()

    def __call__(self, current_epoch):
        """Returns the current and new value for the hyperparameter.

        Arguments
        ---------
        current_epoch : int
            Number of times the dataset has been iterated.
        """
        old_index = max(0, current_epoch - 1)
        index = min(current_epoch, len(self.value_at_epoch) - 1)
        return self.value_at_epoch[old_index], self.value_at_epoch[index]


class StepScheduler:
    """Learning rate scheduler with step annealing technique.

    The hyperparameter's value decays over the epochs with the
    selected ``epoch_decay`` factor

    ``value = init_value * decay_factor ^ floor((1 + epoch) / decay_drop)``

    Arguments
    ---------
    initial_value : float
        Initial value for the hyperparameter being updated.
    decay_factor : float
        Factor multiplied with the initial_value
    decay_drop : float
        Annealing factor (the decay of the hyperparameter value is faster
        with higher ``decay_drop`` values).

    Example
    -------
    >>> scheduler = StepScheduler(initial_value=1.0)
    >>> scheduler(current_epoch=1)
    (1.0, 0.5)
    >>> scheduler(current_epoch=2)
    (0.5, 0.5)
    >>> scheduler(current_epoch=3)
    (0.5, 0.25)
    """

    def __init__(
        self, initial_value, decay_factor=0.5, decay_drop=2,
    ):
        self.initial_value = initial_value
        self.decay_factor = decay_factor
        self.decay_drop = decay_drop

    def __call__(self, current_epoch):
        """Returns current and new hyperparameter value.

        Arguments
        ---------
        current_epoch : int
            Number of times the dataset has been iterated.
        """
        current_value = self._compute_value(current_epoch - 1)
        next_value = self._compute_value(current_epoch)

        return current_value, next_value

    def _compute_value(self, current_epoch):
        return self.initial_value * math.pow(
            self.decay_factor,
            math.floor((1 + current_epoch) / self.decay_drop),
        )


@checkpoints.register_checkpoint_hooks
class NoamScheduler:
    """The is an implementation of the transformer's learning rate scheduler with warmup.
    Reference: https://arxiv.org/abs/1706.03762

    Note: this schdualer aneals lr at each update of the model's weight, and n_steps must be saved for restarting

    Arguments
    ---------
    lr_initial : float
        Initial learning rate (i.e. the lr used at epoch 0).
    n_warmup_steps : int
        numer of warm up steps

    Example
    -------
    >>> from speechbrain.nnet.linear import Linear
    >>> inp_tensor = torch.rand([1,660,3])
    >>> model = Linear(input_size=3, n_neurons=4)
    >>> optim = torch.optim.Adam(model.parameters(), lr=1)
    >>> output = model(inp_tensor)
    >>> scheduler =NoamScheduler(optim.param_groups[0]["lr"], 3)
    >>> curr_lr,next_lr=scheduler(optim)
    >>> optim.param_groups[0]["lr"]
    0.33333333333333337
    >>> curr_lr,next_lr=scheduler(optim)
    >>> optim.param_groups[0]["lr"]
    0.6666666666666667
    >>> curr_lr,next_lr=scheduler(optim)
    >>> optim.param_groups[0]["lr"]
    1.0
    """

    def __init__(self, lr_initial, n_warmup_steps, model_size=None):
        self.lr_initial = lr_initial
        self.n_warmup_steps = n_warmup_steps
        self.current_lr = lr_initial
        self.losses = []

        self.n_steps = 0
        self.normalize = 1 / (n_warmup_steps * n_warmup_steps ** -1.5)
        if model_size is not None:
            self.normalize = model_size ** (-0.5)

    def __call__(self, opt):
        """
        Arguments
        ---------
        opt : optimizer
            The optimizer to update using this scheduler.
        Returns
        -------
        float
            The learning rate before the update.
        float
            The learning rate after the update.
        """
        self.n_steps += 1

        current_lr = opt.param_groups[0]["lr"]

        lr = self.lr_initial * self._get_lr_scale()

        # Changing the learning rate within the optimizer
        for param_group in opt.param_groups:
            param_group["lr"] = lr

        self.current_lr = current_lr
        return current_lr, lr

    def _get_lr_scale(self):
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return self.normalize * min(
            n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5)
        )

    @checkpoints.mark_as_saver
    def save(self, path):
        data = {"losses": self.losses, "n_steps": self.n_steps}
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch, device=None):
        del end_of_epoch  # Unused in this class
        data = torch.load(path)
        self.losses = data["losses"]
        self.n_steps = data["n_steps"]


@checkpoints.register_checkpoint_hooks
class CyclicCosineScheduler:
    """The is an implementation of the Cyclic-Cosine learning rate scheduler with warmup.
    Reference:  https://openreview.net/pdf?id=BJYwwY9ll

    Note: this schdualer aneals lr at each update of the model's weight, and n_steps must be saved for restarting

    Arguments
    ---------
    lr_initial : float
        Initial learning rate (i.e. the lr used at epoch 0).
    n_warmup_steps : int
        numer of warm up steps
    total_steps: int
        total number of updating steps

    Example
    -------
    >>> from speechbrain.nnet.linear import Linear
    >>> inp_tensor = torch.rand([1,660,3])
    >>> model = Linear(input_size=3, n_neurons=4)
    >>> optim = torch.optim.Adam(model.parameters(), lr=1)
    >>> output = model(inp_tensor)
    >>> scheduler =CyclicCosineScheduler(3, optim.param_groups[0]["lr"])
    >>> curr_lr,next_lr=scheduler(optim)
    >>> optim.param_groups[0]["lr"]
    0.9999999990130395
    >>> curr_lr,next_lr=scheduler(optim)
    >>> optim.param_groups[0]["lr"]
    0.9999999997532598
    >>> curr_lr,next_lr=scheduler(optim)
    >>> optim.param_groups[0]["lr"]
    1.0
    """

    def __init__(self, n_warmup_steps, lr_initial=None, total_steps=100000):
        self.n_warmup_steps = n_warmup_steps
        self.losses = []
        self.initial_lr = lr_initial
        self.current_lr = lr_initial
        self.total = total_steps

        self.n_steps = 0
        self.normalize = 1 / (n_warmup_steps * n_warmup_steps ** -1.5)

    def __call__(self, opt):
        """
        Arguments
        ---------
        opt : list of optimizers
            The optimizers to update using this scheduler.
        current_epoch : int
            Number of times the dataset has been iterated.
        current_loss : int
            A number for determining whether to change the learning rate.
        Returns
        -------
        float
            The learning rate before the update.
        float
            The learning rate after the update.
        """
        self.n_steps += 1

        if self.initial_lr is None:
            current_lr = opt.param_groups[0]["lr"]
        else:
            current_lr = self.current_lr

        lr = current_lr * self._get_lr_scale()

        # Changing the learning rate within the optimizer
        for param_group in opt.param_groups:
            param_group["lr"] = lr

        self.current_lr = current_lr
        return current_lr, lr

    def _get_lr_scale(self):
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return 0.5 * (
            math.cos(math.pi * (n_steps - n_warmup_steps) / self.total) + 1
        )

    @checkpoints.mark_as_saver
    def save(self, path):
        data = {"losses": self.losses, "n_steps": self.n_steps}
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch, device=None):
        del end_of_epoch  # Unused in this class
        data = torch.load(path)
        self.losses = data["losses"]
        self.n_steps = data["n_steps"]
