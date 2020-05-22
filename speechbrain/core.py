"""Core SpeechBrain code for running experiments.

Author(s): Peter Plantinga 2020
"""

import os
import sys
import torch
import shutil
import logging
import inspect
import argparse
import subprocess
import speechbrain as sb
from datetime import date
from tqdm.contrib import tzip
from speechbrain.utils.logger import setup_logging

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
    log_filepath = os.path.join(experiment_directory, "log.txt")
    logger_overrides = parse_overrides(
        "{handlers.file_handler.filename: %s}" % log_filepath
    )
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
    """Parse command-line arguments to the experiment.

    Arguments
    ---------
    arg_list: list
        a list of arguments to parse, most often from `sys.argv[1:]`

    Example
    -------
    >>> filename, overrides = parse_arguments(['params.yaml', '--seed', '10'])
    >>> filename
    'params.yaml'
    >>> overrides
    {'seed': 10}
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
        "--hyperparams",
        help="A yaml-formatted string representing a dictionary of "
        "overrides to the parameters in the param_file. The keys of "
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
        "--experiment_id",
        help="An id for the experiment, default is the parameter file name, "
        "without the '.yaml' extension.",
    )
    parser.add_argument(
        "--log_config",
        help="A file storing the configuration options for logging",
    )

    # Ignore items that are "None", they were not passed
    parsed_args = vars(parser.parse_args(arg_list))

    param_file = parsed_args["param_file"]
    del parsed_args["param_file"]

    # Set the experiment_id to a reasonable default
    if parsed_args["experiment_id"] is None:
        parsed_args["experiment_id"] = os.path.splitext(
            os.path.basename(param_file)
        )[0]

    # Convert hyperparams to dictionary
    hyperparams = {}
    if parsed_args["hyperparams"] is not None:
        hyperparams = parse_overrides(parsed_args["hyperparams"])
        del parsed_args["hyperparams"]

    # Only return non-empty items
    overrides = {k: v for k, v in parsed_args.items() if v is not None}
    return param_file, overrides, hyperparams


def parse_overrides(override_string):
    """Parse overrides from a yaml string representing paired args and values.

    Arguments
    ---------
    override_string: str
        A yaml-formatted string, where each (key: value) pair
        overrides the same pair in a loaded file.

    Example
    -------
    >>> parse_overrides("{model.arg1: val1, model.arg2.arg3: 3.}")
    {'model': {'arg1': 'val1', 'arg2': {'arg3': 3.0}}}
    """
    preview = {}
    if override_string:
        preview = sb.yaml.load_extended_yaml(override_string)

    overrides = {}
    for arg, val in preview.__dict__.items():
        if "." in arg:
            nest(overrides, arg.split("."), val)
        else:
            overrides[arg] = val

    return overrides


def nest(dictionary, args, val):
    """Create a nested sequence of dictionaries, based on an arg list.

    Arguments
    ---------
    dictionary : dict
        this object will be updated with the nested arguments.
    args : list
        a list of parameters specifying a nested location.
    val : obj
        The value to store at the specified nested location.

    Example
    -------
    >>> params = {}
    >>> nest(params, ['arg1', 'arg2', 'arg3'], 'value')
    >>> params
    {'arg1': {'arg2': {'arg3': 'value'}}}
    """
    if len(args) == 1:
        dictionary[args[0]] = val
        return

    if args[0] not in dictionary:
        dictionary[args[0]] = {}

    nest(dictionary[args[0]], args[1:], val)


class Brain:
    r"""Brain class abstracts away the details of data loops.

    The primary purpose of the `Brain` class is the implementation of
    the `fit()` method, which iterates epochs and datasets for the
    purpose of "fitting" a set of modules to a set of data.

    In order to use the `fit()` method, one should sub-class the `Brain` class
    and override any methods for which the default behavior does not match
    the use case. For a simple use case (e.g. training a single model with
    a single dataset) the only methods that need to be overridden are:

    * `compute_forward()`
    * `compute_objectives()`

    The example below illustrates how overriding these two methods is done.

    For more complicated use cases, such as multiple modules that need to
    be updated, the following methods can be overridden:

    * `fit_batch()`
    * `evaluate_batch()`

    Arguments
    ---------
    modules : list of torch.Tensors
        The modules that will be updated using the optimizer.
    optimizer : optimizer
        The class to use for updating the modules' parameters.
    first_inputs : list of torch.Tensor
        An example of the input to the Brain class, for parameter init.
        Arguments are passed individually to the `compute_forward` method,
        for cases where a different signature is desired.

    Example
    -------
    >>> from speechbrain.nnet.optimizers import Optimize
    >>> class SimpleBrain(Brain):
    ...     def compute_forward(self, x, init_params=False):
    ...         return self.modules[0](x)
    ...     def compute_objectives(self, predictions, targets, train=True):
    ...         return torch.nn.functional.l1_loss(predictions, targets)
    >>> model = torch.nn.Linear(in_features=10, out_features=10)
    >>> brain = SimpleBrain(
    ...     modules=[model],
    ...     optimizer=Optimize('sgd', 0.01),
    ...     first_inputs=[torch.rand(10, 10)],
    ... )
    >>> brain.fit(
    ...     epoch_counter=range(1),
    ...     train_set=([torch.rand(10, 10)], [torch.rand(10, 10)]),
    ... )
    """

    def __init__(self, modules=None, optimizer=None, first_inputs=None):
        self.modules = torch.nn.ModuleList(modules)
        self.optimizer = optimizer

        # Initialize parameters
        if first_inputs is not None:
            self.compute_forward(*first_inputs, init_params=True)

            if self.optimizer is not None:
                self.optimizer.init_params(self.modules)

    def compute_forward(self, x, train_mode=True, init_params=False):
        """Forward pass, to be overridden by sub-classes.

        Arguments
        ---------
        x : torch.Tensor or list of tensors
            The input tensor or tensors for processing.
        train_mode : bool
            Whether this pass is done in train mode or not. Models such
            as seq2seq may have different behavior in train and eval.
        init_params : bool
            Whether this pass should initialize parameters rather
            than return the results of the forward pass.
        """
        raise NotImplementedError

    def compute_objectives(self, predictions, targets, train_mode=True):
        """Compute loss, to be overridden by sub-classes.

        Arguments
        ---------
        predictions : torch.Tensor or list of tensors
            The output tensor or tensors to evaluate.
        targets : torch.Tensor or list of tensors
            The gold standard to use for evaluation.
        train_mode : bool
            Whether this is computed for training or not. During training,
            sometimes fewer stats will be computed for the sake of efficiency
            (e.g. WER might only be computed for valid and test, not train).
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

        * `compute_forward()`
        * `compute_objectives()`
        * `optimizer()`

        Arguments
        ---------
        batch : list of torch.Tensors
            batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.
        """
        inputs, targets = batch
        predictions = self.compute_forward(inputs)
        loss = self.compute_objectives(predictions, targets)
        loss.backward()
        self.optimizer(self.modules)
        return {"loss": loss.detach()}

    def evaluate_batch(self, batch):
        """Evaluate one batch, override for different procedure than train.

        The default impementation depends on two methods being defined
        with a particular behavior:

        * `compute_forward()`
        * `compute_objectives()`

        Arguments
        ---------
        batch : list of torch.Tensors
            batch of data to use for evaluation. Default implementation assumes
            this batch has two elements: inputs and targets.
        """
        inputs, targets = batch
        out = self.compute_forward(inputs, train_mode=False)
        loss, stats = self.compute_objectives(out, targets, train_mode=False)
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

        * `fit_batch()`
        * `evaluate_batch()`
        * `add_stats()`

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
            for batch in tzip(*train_set):
                stats = self.fit_batch(batch)
                self.add_stats(train_stats, stats)

            valid_stats = {}
            if valid_set is not None:
                self.modules.eval()
                with torch.no_grad():
                    for batch in tzip(*valid_set):
                        stats = self.evaluate_batch(batch)
                        self.add_stats(valid_stats, stats)

            self.on_epoch_end(epoch, train_stats, valid_stats)

    def evaluate(self, test_set):
        """Iterate test_set and evaluate brain performance.

        Arguments
        ---------
        test_set : list of DataLoaders
            This list will be zipped before iterating.
        """
        test_stats = {}
        self.modules.eval()
        with torch.no_grad():
            for batch in tzip(*test_set):
                stats = self.evaluate_batch(batch)
                self.add_stats(test_stats, stats)

        return test_stats
