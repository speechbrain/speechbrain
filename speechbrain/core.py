"""Core SpeechBrain code for running experiments.

Authors
 * Peter Plantinga 2020
"""

import os
import sys
import torch
import shutil
import logging
import inspect
import argparse
import subprocess
import ruamel.yaml
import speechbrain as sb
import torch.distributed as dist
import torch.multiprocessing as mp
from io import StringIO
from datetime import date
from enum import Enum, auto
from tqdm.contrib import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)
DEFAULT_LOG_CONFIG = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_CONFIG = os.path.join(DEFAULT_LOG_CONFIG, "log-config.yaml")


def create_experiment_directory(
    experiment_directory,
    hyperparams_to_save=None,
    overrides={},
    log_config=DEFAULT_LOG_CONFIG,
    save_env_desc=True,
):
    """Create the output folder and relevant experimental files.

    Arguments
    ---------
    experiment_directory : str
        The place where the experiment directory should be created.
    hyperparams_to_save : str
        A filename of a yaml file representing the parameters for this
        experiment. If passed, references are resolved and the result is
        written to a file in the experiment directory called "hyperparams.yaml"
    overrides : dict
        A mapping of replacements made in the yaml file, to save in yaml.
    log_config : str
        A yaml filename containing configuration options for the logger.
    save_env_desc : bool
        If True, a basic environment state description is saved to the experiment
        directory, in a file called env.log in the experiment directory
    """
    if not os.path.isdir(experiment_directory):
        os.makedirs(experiment_directory)

    # Write the parameters file
    if hyperparams_to_save is not None:
        hyperparams_filename = os.path.join(
            experiment_directory, "hyperparams.yaml"
        )
        with open(hyperparams_to_save) as f:
            resolved_yaml = sb.resolve_references(f, overrides)
        with open(hyperparams_filename, "w") as w:
            print("# Generated %s from:" % date.today(), file=w)
            print("# %s" % os.path.abspath(hyperparams_to_save), file=w)
            print("# yamllint disable", file=w)
            shutil.copyfileobj(resolved_yaml, w)

    # Copy executing file to output directory
    module = inspect.getmodule(inspect.currentframe().f_back)
    if module is not None:
        callingfile = os.path.realpath(module.__file__)
        shutil.copy(callingfile, experiment_directory)

    # Log exceptions to output automatically
    log_file = os.path.join(experiment_directory, "log.txt")
    logger_overrides = {"handlers": {"file_handler": {"filename": log_file}}}
    sb.setup_logging(log_config, logger_overrides)
    sys.excepthook = _logging_excepthook

    # Log beginning of experiment!
    logger.info("Beginning experiment!")
    logger.info(f"Experiment folder: {experiment_directory}")
    commit_hash = subprocess.check_output(["git", "describe", "--always"])
    logger.debug("Commit hash: '%s'" % commit_hash.decode("utf-8").strip())

    # Save system description:
    if save_env_desc:
        description_str = sb.get_environment_description()
        with open(os.path.join(experiment_directory, "env.log"), "w") as fo:
            fo.write(description_str)


def _logging_excepthook(exc_type, exc_value, exc_traceback):
    """Interrupt exception raising to log the error."""
    logger.error("Exception:", exc_info=(exc_type, exc_value, exc_traceback))


def parse_arguments(arg_list):
    r"""Parse command-line arguments to the experiment.

    Arguments
    ---------
    arg_list: list
        a list of arguments to parse, most often from `sys.argv[1:]`

    Returns
    -------
    param_file : str
        The location of the parameters file.
    overrides : str
        The yaml-formatted overrides, to pass to ``load_extended_yaml``.

    Example
    -------
    >>> filename, overrides = parse_arguments(['hyperparams.yaml', '--seed', '10'])
    >>> filename
    'hyperparams.yaml'
    >>> overrides
    'seed: 10\n'
    """
    parser = argparse.ArgumentParser(
        description="Run a SpeechBrain experiment",
    )
    parser.add_argument(
        "param_file",
        help="a yaml-formatted file using the extended YAML syntax "
        "defined by SpeechBrain.",
    )
    parser.add_argument(
        "--yaml_overrides",
        help="A yaml-formatted string representing a dictionary of "
        "overrides to the parameters in the param file. The keys of "
        "the dictionary can use dots to represent levels in the yaml "
        'hierarchy. For example: "{model.param1: value1}" would '
        "override the param1 parameter of the model node.",
    )
    parser.add_argument(
        "--output_folder",
        help="A folder for storing all experiment-related outputs.",
    )
    parser.add_argument(
        "--data_folder", help="A folder containing the data used for training",
    )
    parser.add_argument(
        "--save_folder",
        help="A folder for storing checkpoints that allow restoring "
        "progress for testing or re-starting training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="A random seed to reproduce experiments on the same machine",
    )
    parser.add_argument(
        "--log_config",
        help="A file storing the configuration options for logging",
    )

    # Ignore items that are "None", they were not passed
    parsed_args = vars(parser.parse_args(arg_list))

    param_file = parsed_args["param_file"]
    del parsed_args["param_file"]

    # Convert yaml_overrides to dictionary
    yaml_overrides = ""
    if parsed_args["yaml_overrides"] is not None:
        yaml_overrides = parsed_args["yaml_overrides"]
        del parsed_args["yaml_overrides"]

    # Only return non-empty items
    items = {k: v for k, v in parsed_args.items() if v is not None}

    # Convert to string and append to overrides
    ruamel_yaml = ruamel.yaml.YAML()
    overrides = ruamel_yaml.load(yaml_overrides) or {}
    sb.recursive_update(overrides, items)
    yaml_stream = StringIO()
    ruamel_yaml.dump(overrides, yaml_stream)

    return param_file, yaml_stream.getvalue()


class Stage(Enum):
    """Simple enum to track stage of experiments."""

    TRAIN = auto()
    VALID = auto()
    TEST = auto()


class Brain:
    r"""Brain class abstracts away the details of data loops.

    The primary purpose of the `Brain` class is the implementation of
    the ``fit()`` method, which iterates epochs and datasets for the
    purpose of "fitting" a set of modules to a set of data.

    In order to use the ``fit()`` method, one should sub-class the ``Brain``
    class and override any methods for which the default behavior does not
    match the use case. For a simple use case (e.g. training a single model
    with a single dataset) the only methods that need to be overridden are:

    * ``compute_forward()``
    * ``compute_objectives()``

    The example below illustrates how overriding these two methods is done.

    For more complicated use cases, such as multiple modules that need to
    be updated, the following methods can be overridden:

    * ``fit_batch()``
    * ``evaluate_batch()``

    Arguments
    ---------
    modules : dict
        Each element should consist of a string key for indicating which
        modules need to be passed to an optimizer, as well as a module
        to be used by the Brain class (is not required to be a torch module).
    optimizers : dict
        Pairs where the key is a string referring to a specific module, or a
        tuple referring to several modules, and the value is an optimizer.
        The optimizer will be used to update the listed modules.
    first_inputs : list of torch.Tensor
        An example of the input to the Brain class, for parameter init.
        Arguments are passed individually to the ``compute_forward`` method,
        for cases where a different signature is desired.
    device : str
        The location for performing computations.
    auto_mix_prec: bool
        If True, automatic mixed-precision is used. Activate it only with cuda.

    Example
    -------
    >>> from speechbrain.nnet import Optimizer
    >>> from torch.optim import SGD
    >>> class SimpleBrain(Brain):
    ...     def compute_forward(self, x, stage):
    ...         return self.model(x)
    ...     def compute_objectives(self, predictions, targets, stage):
    ...         return torch.nn.functional.l1_loss(predictions, targets)
    >>> model = torch.nn.Linear(in_features=10, out_features=10)
    >>> brain = SimpleBrain(
    ...     modules={'model': model},
    ...     optimizers={'model': Optimizer(SGD, 0.01)},
    ...     device='cpu',
    ... )
    >>> brain.fit(
    ...     epoch_counter=range(1),
    ...     train_set=([torch.rand(10, 10), torch.rand(10, 10)],)
    ... )
    """

    def __init__(
        self,
        modules,
        optimizers,
        jit_modules=None,
        device="cuda:0",
        torch_ddp_procs=0,
        auto_mix_prec=False,
    ):
        self.device = device
        self.optimizers = optimizers
        self.jit_modules = jit_modules
        self.torch_ddp_procs = torch_ddp_procs

        # Set module attributes, so compute_forward can access modules
        modulelist = []
        for name, module in modules.items():
            if isinstance(module, torch.nn.Module):
                module = module.to(device)
                modulelist.append(module)
            setattr(self, name, module)

        # Initialize optimizers
        for module_list, optimizer in optimizers.items():
            if isinstance(module_list, str):
                optimizer.init_params([modules[module_list]])
            else:
                optimizer.init_params([modules[k] for k in module_list])

        # Store modules as ModuleList, primarily for calling train()/eval()
        self.modules = torch.nn.ModuleList(modulelist)
        self.auto_mix_prec = auto_mix_prec

        # Automatic mixed precision init
        self.scaler = torch.cuda.amp.GradScaler()

        total_params = sum(
            p.numel() for p in self.modules.parameters() if p.requires_grad
        )
        clsname = self.__class__.__name__
        fmt_num = sb.format_order_of_magnitude(total_params)
        logger.info(f"Initialized {fmt_num} trainable parameters in {clsname}")

    def compute_forward(self, x, stage):
        """Forward pass, to be overridden by sub-classes.

        Arguments
        ---------
        x : torch.Tensor or list of tensors
            The input tensor or tensors for processing.
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST

        Returns
        -------
        torch.Tensor
            A tensor representing the outputs after all processing is complete.
        """
        raise NotImplementedError

    def compute_objectives(self, predictions, targets, stage):
        """Compute loss, to be overridden by sub-classes.

        Arguments
        ---------
        predictions : torch.Tensor or list of tensors
            The output tensor or tensors to evaluate.
        targets : torch.Tensor or list of tensors
            The gold standard to use for evaluation.
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST

        Returns
        -------
        loss : torch.Tensor
            A tensor with the computed loss
        """
        raise NotImplementedError

    def on_stage_start(self, stage, epoch=None):
        """Gets called when a stage starts.

        Useful for defining class variables used during the stage.

        Arguments
        ---------
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST
        epoch : int
            The current epoch count.
        """
        pass

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of a stage.

        Arguments
        ---------
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST
        stage_loss : float
            The average loss over the completed stage.
        epoch : int
            The current epoch count.
        """
        pass

    def fit_batch(self, batch):
        """Fit one batch, override to do multiple updates.

        The default impementation depends on a few methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Also depends on having optimizers passed at initialization.

        Arguments
        ---------
        batch : list of torch.Tensors
            batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.

        Returns
        -------
        detached loss
        """
        inputs, labels = batch

        for optimizer in self.optimizers.values():

            # Managing automatic mixed precision
            if self.auto_mix_prec:
                with torch.cuda.amp.autocast():
                    outputs = self.compute_forward(inputs, Stage.TRAIN)
                    loss = self.compute_objectives(outputs, labels, Stage.TRAIN)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer.optim)
                    optimizer.zero_grad()
                    self.scaler.update()
            else:
                outputs = self.compute_forward(inputs, Stage.TRAIN)
                loss = self.compute_objectives(outputs, labels, Stage.TRAIN)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Evaluate one batch, override for different procedure than train.

        The default impementation depends on two methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Arguments
        ---------
        batch : list of torch.Tensors
            batch of data to use for evaluation. Default implementation assumes
            this batch has two elements: inputs and targets.
        stage : Stage
            The stage of the experiment: Stage.VALID, Stage.TEST

        Returns
        -------
        detached loss
        """
        inputs, targets = batch
        out = self.compute_forward(inputs, stage=stage)
        loss = self.compute_objectives(out, targets, stage=stage)
        return loss.detach().cpu()

    def ddp_fit(self, *args, **kwargs):
        """Use torch DistributedDataParallel to fit over multiple devices.

        For kwargs, see ``fit()``.
        """
        mp.spawn(
            ddp_init,
            args=(self, args, kwargs),
            nprocs=self.torch_ddp_procs,
            join=True,
        )

    def fit(
        self, epoch_counter, train_set, valid_set=None, progressbar=True,
    ):
        """Iterate epochs and datasets to improve objective.

        Relies on the existence of mulitple functions that can (or should) be
        overridden. The following methods are used and expected to have a
        certain behavior:

        * ``fit_batch()``
        * ``evaluate_batch()``
        * ``update_average()``

        Arguments
        ---------
        epoch_counter : iterable
            each call should return an integer indicating the epoch count.
        train_set : DataLoader
            A set of data to use for training.
        valid_set : DataLoader
            A set of data to use for validation.
        progressbar : bool
            Whether to display the progress of each epoch in a progressbar.
        """
        # Compile jit modules if requested
        self._compile_jit()

        for epoch in epoch_counter:

            # Training stage
            self.on_stage_start(Stage.TRAIN, epoch)
            self.modules.train()
            avg_train_loss = 0.0
            disable = not progressbar
            with tqdm(train_set, dynamic_ncols=True, disable=disable) as t:
                for i, batch in enumerate(t):
                    if (
                        isinstance(self.device, int)
                        and i % self.torch_ddp_procs != self.device
                    ):
                        continue
                    loss = self.fit_batch(batch)
                    avg_train_loss = self.update_average(
                        loss, avg_train_loss, iteration=i + 1
                    )
                    t.set_postfix(train_loss=avg_train_loss)
            self.on_stage_end(Stage.TRAIN, avg_train_loss, epoch)

            # Validation stage
            avg_valid_loss = None
            if valid_set is not None:
                self.on_stage_start(Stage.VALID, epoch)
                self.modules.eval()
                avg_valid_loss = 0.0
                with torch.no_grad():
                    for i, batch in enumerate(
                        tqdm(valid_set, dynamic_ncols=True, disable=disable)
                    ):
                        if (
                            isinstance(self.device, int)
                            and i % self.torch_ddp_procs != self.device
                        ):
                            continue
                        loss = self.evaluate_batch(batch, stage=Stage.VALID)
                        avg_valid_loss = self.update_average(
                            loss, avg_valid_loss, iteration=i + 1
                        )
                self.on_stage_end(Stage.VALID, avg_valid_loss, epoch)

    def _compile_jit(self):
        if self.jit_modules is None:
            return

        for jit_module in self.jit_modules:
            module = getattr(self, jit_module)
            module = torch.jit.script(module)
            module = module.to(self.device)
            setattr(self, jit_module, module)

    def evaluate(self, test_set, progressbar=True):
        """Iterate test_set and evaluate brain performance.

        Arguments
        ---------
        test_set : list of DataLoaders
            This list will be zipped before iterating.
        progressbar : bool
            Whether to display the progress in a progressbar.

        Returns
        -------
        average test loss
        """
        self.on_stage_start(Stage.TEST)
        self.modules.eval()
        avg_test_loss = 0.0
        disable = not progressbar
        with torch.no_grad():
            for i, batch in enumerate(
                tqdm(test_set, dynamic_ncols=True, disable=disable)
            ):
                loss = self.evaluate_batch(batch, stage=Stage.TEST)
                avg_test_loss = self.update_average(
                    loss, avg_test_loss, iteration=i + 1
                )
        self.on_stage_end(Stage.TEST, avg_test_loss)

        return avg_test_loss

    def update_average(self, loss, avg_loss, iteration):
        """Update running average of the loss.

        Arguments
        ---------
        loss : torch.tensor
            detached loss, a single float value.
        avg_loss : float
            current running average.
        iteration : int
            The iteration count.

        Returns
        -------
        float
            The average loss
        """
        if not torch.isfinite(loss):
            raise ValueError(
                "Loss is not finite. To debug, wrap `fit()` with autograd's "
                "`detect_anomaly()`, e.g.\n\nwith "
                "torch.autograd.detect_anomaly():\n\tbrain.fit(...)"
            )

        # Compute moving average
        avg_loss -= avg_loss / iteration
        avg_loss += float(loss) / iteration
        return avg_loss


def ddp_init(rank, brain, args, kwargs):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12321"
    dist.init_process_group(
        backend="nccl", world_size=brain.torch_ddp_procs, rank=rank
    )

    # Move to correct device
    brain.device = rank
    for i, module in enumerate(brain.modules):
        if any(p.requires_grad for p in module.parameters()):
            brain.modules[i] = DDP(module.to(rank), device_ids=[rank])

    brain.fit(*args, **kwargs)
