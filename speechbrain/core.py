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
from io import StringIO
from datetime import date
from tqdm.contrib import tqdm
from speechbrain.utils.logger import setup_logging
from speechbrain.utils.logger import format_order_of_magnitude
from speechbrain.utils.data_utils import recursive_update

logger = logging.getLogger(__name__)
DEFAULT_LOG_CONFIG = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_CONFIG = os.path.join(DEFAULT_LOG_CONFIG, "log-config.yaml")


def create_experiment_directory(
    experiment_directory,
    params_to_save=None,
    overrides={},
    log_config=DEFAULT_LOG_CONFIG,
):
    """Create the output folder and relevant experimental files.

    Arguments
    ---------
    experiment_directory : str
        The place where the experiment directory should be created.
    params_to_save : str
        A filename of a yaml file representing the parameters for this
        experiment. If passed, references are resolved and the result
        is written to a file in the experiment directory called "params.yaml"
    overrides : dict
        A mapping of replacements made in the yaml file, to save in yaml.
    log_config : str
        A yaml filename containing configuration options for the logger.
    """
    if not os.path.isdir(experiment_directory):
        os.makedirs(experiment_directory)

    # Write the parameters file
    if params_to_save is not None:
        params_filename = os.path.join(experiment_directory, "params.yaml")
        with open(params_to_save) as f:
            resolved_yaml = sb.yaml.resolve_references(f, overrides)
        with open(params_filename, "w") as w:
            print("# Generated %s from:" % date.today(), file=w)
            print("# %s" % os.path.abspath(params_to_save), file=w)
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
    setup_logging(log_config, logger_overrides)
    sys.excepthook = _logging_excepthook

    # Log beginning of experiment!
    logger.info("Beginning experiment!")
    logger.info(f"Experiment folder: {experiment_directory}")
    commit_hash = subprocess.check_output(["git", "describe", "--always"])
    logger.debug("Commit hash: '%s'" % commit_hash.decode("utf-8").strip())


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
    >>> filename, overrides = parse_arguments(['params.yaml', '--seed', '10'])
    >>> filename
    'params.yaml'
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
    recursive_update(overrides, items)
    yaml_stream = StringIO()
    ruamel_yaml.dump(overrides, yaml_stream)

    return param_file, yaml_stream.getvalue()


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
    modules : list of torch.Tensors
        The modules that will be updated using the optimizer.
    optimizer : optimizer
        The class to use for updating the modules' parameters.
    first_inputs : list of torch.Tensor
        An example of the input to the Brain class, for parameter init.
        Arguments are passed individually to the ``compute_forward`` method,
        for cases where a different signature is desired.

    Example
    -------
    >>> from speechbrain.nnet.optimizers import SGD_Optimizer
    >>> class SimpleBrain(Brain):
    ...     def compute_forward(self, x, init_params=False):
    ...         return self.modules[0](x)
    ...     def compute_objectives(self, predictions, targets, train=True):
    ...         return torch.nn.functional.l1_loss(predictions, targets), {}
    >>> model = torch.nn.Linear(in_features=10, out_features=10)
    >>> brain = SimpleBrain(
    ...     modules=[model],
    ...     optimizer=SGD_Optimizer(0.01),
    ...     first_inputs=[torch.rand(10, 10)],
    ... )
    >>> brain.fit(
    ...     epoch_counter=range(1),
    ...     train_set=([torch.rand(10, 10),torch.rand(10, 10)],)
    ... )
    """

    def __init__(self, modules=None, optimizer=None, first_inputs=None):
        self.modules = torch.nn.ModuleList(modules)
        self.optimizer = optimizer
        self.avg_train_loss = 0.0

        # Initialize parameters
        if first_inputs is not None:
            self.compute_forward(*first_inputs, init_params=True)

            if self.optimizer is not None:
                self.optimizer.init_params(self.modules)

        total_params = sum(
            p.numel() for p in self.modules.parameters() if p.requires_grad
        )
        clsname = self.__class__.__name__
        fmt_num = format_order_of_magnitude(total_params)
        logger.info(f"Initialized {fmt_num} trainable parameters in {clsname}")

    def compute_forward(self, x, stage="train", init_params=False):
        """Forward pass, to be overridden by sub-classes.

        Arguments
        ---------
        x : torch.Tensor or list of tensors
            The input tensor or tensors for processing.
        stage : str
            The stage of the training process, one of "train", "valid", "test"
        init_params : bool
            Whether this pass should initialize parameters rather
            than return the results of the forward pass.

        Returns
        -------
        torch.Tensor
            A tensor representing the outputs after all processing is complete.
        """
        raise NotImplementedError

    def compute_objectives(self, predictions, targets, stage="train"):
        """Compute loss, to be overridden by sub-classes.

        Arguments
        ---------
        predictions : torch.Tensor or list of tensors
            The output tensor or tensors to evaluate.
        targets : torch.Tensor or list of tensors
            The gold standard to use for evaluation.
        stage : str
            The stage of the training process, one of "train", "valid", "test"

        Returns
        -------
        loss : torch.Tensor
            A tensor with the computed loss
        stats : dict
            A mapping with additional statistics about the batch
            (e.g. ``{"accuracy": .9}``)
        """
        raise NotImplementedError

    def on_epoch_end(self, epoch, train_stats, valid_stats=None):
        """Gets called at the end of each epoch.

        Arguments
        ---------
        epoch : int
            The current epoch count.
        train_stats : dict of str:list pairs
            Each key refers to a statstic, and the list contains the values
            for this statistic, generated in a training pass.
        valid_stats : dict of str:list pairs
            Each key refers to a statstic, and the list contains the values
            for this statistic, generated in a training pass.
        """
        pass

    def fit_batch(self, batch):
        """Fit one batch, override to do multiple updates.

        The default impementation depends on three methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``
        * ``optimizer()``

        Arguments
        ---------
        batch : list of torch.Tensors
            batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.

        Returns
        -------
        dict
            A dictionary of the same format as `evaluate_batch()` where each
            item includes a statistic about the batch, including the loss.
            (e.g. ``{"loss": 0.1, "accuracy": 0.9}``)
        """
        inputs, targets = batch
        predictions = self.compute_forward(inputs)
        loss, stats = self.compute_objectives(predictions, targets)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        stats["loss"] = loss.detach()
        return stats

    def evaluate_batch(self, batch, stage="test"):
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
        stage : str
            The stage of the training process, one of "valid", "test"

        Returns
        -------
        dict
            A dictionary of the same format as ``fit_batch()`` where each item
            includes a statistic about the batch, including the loss.
            (e.g. ``{"loss": 0.1, "accuracy": 0.9}``)
        """
        inputs, targets = batch
        out = self.compute_forward(inputs, stage=stage)
        loss, stats = self.compute_objectives(out, targets, stage=stage)
        stats["loss"] = loss.detach()
        return stats

    def add_stats(self, dataset_stats, batch_stats):
        """Add the stats for a batch to the set of stats for a dataset.

        Arguments
        ---------
        dataset_stats : dict
            A mapping of stat name to a list of the stats in the dataset.
        batch_stats : dict
            A mapping of stat name to the value for that stat in a batch.
        """
        for key in batch_stats:
            if key not in dataset_stats:
                dataset_stats[key] = []
            if isinstance(batch_stats[key], list):
                dataset_stats[key].extend(batch_stats[key])
            else:
                dataset_stats[key].append(batch_stats[key])

    def fit(self, epoch_counter, train_set, valid_set=None):
        """Iterate epochs and datasets to improve objective.

        Relies on the existence of mulitple functions that can (or should) be
        overridden. The following functions are used and expected to have a
        certain behavior:

        * ``fit_batch()``
        * ``evaluate_batch()``
        * ``add_stats()``

        Arguments
        ---------
        epoch_counter : iterable
            each call should return an integer indicating the epoch count.
        train_set : list of DataLoaders
            a list of datasets to use for training, zipped before iterating.
        valid_set : list of Data Loaders
            a list of datasets to use for validation, zipped before iterating.
        """
        for epoch in epoch_counter:
            self.modules.train()
            train_stats = {}
            with tqdm(train_set, dynamic_ncols=True) as t:
                for i, batch in enumerate(t):
                    stats = self.fit_batch(batch)
                    self.add_stats(train_stats, stats)
                    average = self.update_average(stats, iteration=i + 1)
                    t.set_postfix(train_loss=average)

            valid_stats = {}
            if valid_set is not None:
                self.modules.eval()
                with torch.no_grad():
                    for batch in tqdm(valid_set, dynamic_ncols=True):
                        stats = self.evaluate_batch(batch, stage="valid")
                        self.add_stats(valid_stats, stats)

            self.on_epoch_end(epoch, train_stats, valid_stats)

    def evaluate(self, test_set):
        """Iterate test_set and evaluate brain performance.

        Arguments
        ---------
        test_set : list of DataLoaders
            This list will be zipped before iterating.

        Returns
        -------
        dict
            The test stats, where each item
            has a list of all the statistics from the test pass.
            (e.g. ``{"loss": [0.1, 0.2, 0.05], "accuracy": [0.8, 0.8, 0.9]}``)
        """
        test_stats = {}
        self.modules.eval()
        with torch.no_grad():
            for batch in tqdm(test_set, dynamic_ncols=True):
                stats = self.evaluate_batch(batch, stage="test")
                self.add_stats(test_stats, stats)

        return test_stats

    def update_average(self, stats, iteration):
        """Update running average of the loss.

        Arguments
        ---------
        stats : dict
            Result of `compute_objectives()`
        iteration : int
            The iteration count.

        Returns
        -------
        float
            The average loss
        """
        if not torch.isfinite(stats["loss"]):
            raise ValueError(
                "Loss is not finite. To debug, wrap `fit()` with `debug_anomaly`"
                ", e.g.\nwith torch.autograd.detect_anomaly():\n\tbrain.fit(...)"
            )

        # Compute moving average
        self.avg_train_loss -= self.avg_train_loss / iteration
        self.avg_train_loss += float(stats["loss"]) / iteration
        return self.avg_train_loss
