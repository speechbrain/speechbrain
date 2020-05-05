"""Core SpeechBrain code for running experiments.

Author(s): Peter Plantinga 2020
"""

import os
import sys
import yaml
import torch
import shutil
import logging
import inspect
import argparse
import subprocess
from tqdm.contrib import tzip
from speechbrain.yaml import resolve_references
from speechbrain.utils.logger import setup_logging
from speechbrain.utils.epoch_loop import EpochCounter
from speechbrain.utils.checkpoints import ckpt_recency
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
            resolved_yaml = resolve_references(f, overrides)
        with open(params_filename, "w") as w:
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
    if parsed_args["yaml_overrides"] is not None:
        overrides = parse_overrides(parsed_args["yaml_overrides"])
        del parsed_args["yaml_overrides"]
        recursive_update(parsed_args, overrides)

    # Only return non-empty items
    return param_file, {k: v for k, v in parsed_args.items() if v is not None}


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
        preview = yaml.safe_load(override_string)

    overrides = {}
    for arg, val in preview.items():
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

    * `forward()`
    * `compute_objectives()`

    The example below illustrates how overriding these two methods is done.

    For more complicated use cases, such as multiple modules that need to
    be updated, the following methods can be overridden:

    * `fit_batch()`
    * `evaluate_batch()`

    If there is more than one objective (either for training or evaluation),
    the method for summarizing the losses (e.g. averaging) can be specified
    by overriding the `summarize()` method.

    Arguments
    ---------
    modules : list of torch.Tensors
        The modules that will be updated using the optimizer.
    optimizer : optimizer
        The class to use for updating the modules' parameters.
    scheduler : scheduler
        An object that changes the learning rate based on performance.
    saver : Checkpointer
        This is called by default at the end of each epoch to save progress.

    Example
    -------
    >>> from speechbrain.nnet.optimizers import Optimize
    >>> class SimpleBrain(Brain):
    ...     def forward(self, x, init_params=False):
    ...         return self.modules[0](x)
    ...     def compute_objectives(self, predictions, targets, train=True):
    ...         return torch.nn.functional.l1_loss(predictions, targets)
    >>> tmpdir = getfixture('tmpdir')
    >>> model = torch.nn.Linear(in_features=10, out_features=10)
    >>> brain = SimpleBrain([model], Optimize('sgd', 0.01))
    >>> brain.fit(
    ...     train_set=([torch.rand(10, 10)], [torch.rand(10, 10)]),
    ... )
    """

    def __init__(self, modules, optimizer=None, scheduler=None, saver=None):
        self.modules = torch.nn.ModuleList(modules)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.saver = saver

    def forward(self, x, init_params=False):
        """Forward pass, to be overridden by sub-classes.

        Arguments
        ---------
        x : torch.Tensor or list of tensors
            The input tensor or tensors for processing.
        init_params : bool
            Whether this pass should initialize parameters rather
            than return the results of the forward pass.
        """
        raise NotImplementedError

    def compute_objectives(self, predictions, targets, train=True):
        """Compute loss, to be overridden by sub-classes.

        Arguments
        ---------
        predictions : torch.Tensor or list of tensors
            The output tensor or tensors to evaluate.
        targets : torch.Tensor or list of tensors
            The gold standard to use for evaluation.
        train : bool
            Whether this is computed for training or not. During training,
            sometimes fewer stats will be computed for the sake of efficiency
            (e.g. WER might only be computed for valid and test, not train).
        """
        raise NotImplementedError

    def fit_batch(self, batch):
        """Fit one batch, override to do multiple updates.

        The default impementation depends on three methods being defined
        with a particular behavior:

        * `forward()`
        * `compute_objectives()`
        * `optimizer()`

        Arguments
        ---------
        batch : list of torch.Tensors
            batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.
        """
        inputs, targets = batch
        predictions = self.forward(inputs)
        loss = self.compute_objectives(predictions, targets)
        loss.backward()
        self.optimizer(self.modules)
        return {"loss": loss.detach()}

    def evaluate_batch(self, batch):
        """Evaluate one batch, override for different procedure than train.

        The default impementation depends on two methods being defined
        with a particular behavior:

        * `forward()`
        * `compute_objectives()`

        Arguments
        ---------
        batch : list of torch.Tensors
            batch of data to use for evaluation. Default implementation assumes
            this batch has two elements: inputs and targets.
        """
        inputs, targets = batch
        output = self.forward(inputs)
        loss, stats = self.compute_objectives(output, targets, train=False)
        stats["loss"] = loss.detach()
        return stats

    def summarize(self, stats, write=False):
        """Take a list of stats from a pass through data and summarize it.

        By default, averages the loss and returns the average.

        Arguments
        ---------
        stats : list of dicts
            A list of stats to summarize.
        """
        return {"loss": float(sum(s["loss"] for s in stats) / len(stats))}

    def fit(
        self,
        train_set,
        valid_set=None,
        number_of_epochs=1,
        max_keys=[],
        min_keys=[],
    ):
        """Iterate epochs and datasets to improve objective.

        Relies on the existence of mulitple functions that can (or should) be
        overridden. The following functions are used and expected to have a
        certain behavior:

        * `fit_batch()`
        * `evaluate_batch()`
        * `summarize()`
        * `save()`

        Arguments
        ---------
        train_set : list of DataLoaders
            a list of datasets to use for training, zipped before iterating.
        valid_set : list of Data Loaders
            a list of datasets to use for validation, zipped before iterating.
        number_of_epochs : int
            number of epochs to iterate
        max_keys : list of str
            a list of keys to use for checkpointing, highest value is kept.
        min_keys : list of str
            a list of keys to use for checkpointing, lowest value is kept.
        """
        self.forward(next(iter(train_set[0])), init_params=True)
        self.optimizer.init_params(self.modules)
        epoch_counter = EpochCounter(number_of_epochs)
        if self.saver is not None:
            self.saver.add_recoverable("counter", epoch_counter)
            self.saver.recover_if_possible()

        for epoch in epoch_counter:
            self.modules.train()
            train_stats = []
            for batch in tzip(*train_set):
                train_stats.append(self.fit_batch(batch))
            summary = self.summarize(train_stats)

            logger.info(f"Epoch {epoch} complete")
            for key in summary:
                logger.info(f"Train {key}: {summary[key]:.2f}")

            if valid_set is not None:
                self.modules.eval()
                valid_stats = []
                with torch.no_grad():
                    for batch in tzip(*valid_set):
                        valid_stats.append(self.evaluate_batch(batch))
                summary = self.summarize(valid_stats)
                for key in summary:
                    logger.info(f"Valid {key}: {summary[key]:.2f}")

            self.on_epoch_end(epoch, summary, max_keys, min_keys)

    def on_epoch_end(self, epoch, summary, max_keys, min_keys):
        """Gets called at the end of each epoch.

        Arguments
        ---------
        summary : mapping
            This dict defines summary info about the validation pass, if
            the validation data was passed, otherwise training pass. The
            output of the `summarize` method is directly passed.
        max_keys : list of str
            A sequence of strings that match keys in the summary. Highest
            value is the relevant value.
        min_keys : list of str
            A sequence of strings that match keys in the summary. Lowest
            value is the relevant value.
        """
        if self.scheduler is not None:
            min_val = summary[min_keys[0]]
            self.scheduler([self.optimizer], epoch, min_val)
        if self.saver is not None:
            self.save(summary, max_keys, min_keys)

    def evaluate(self, test_set, max_key=None, min_key=None):
        """Iterate test_set and evaluate brain performance.

        Arguments
        ---------
        test_set : list of DataLoaders
            This list will be zipped before iterating.
        max_key : str
            This string references a key in the checkpoint, the
            checkpoint with the highest value for this key will be loaded.
        min_key : str
            This string references a key in the checkpoint, the
            checkpoint with the lowest value for this key will be loaded.
        """
        if self.saver is None and (max_key or min_key):
            raise ValueError("max_key and min_key require a saver.")
        elif max_key is not None and min_key is not None:
            raise ValueError("Can't use both min_key and max_key.")

        if self.saver is not None:
            if max_key is not None:
                self.saver.recover_if_possible(
                    lambda c, key=max_key: c.meta[key]
                )
            elif min_key is not None:
                self.saver.recover_if_possible(
                    lambda c, key=min_key: -c.meta[key]
                )
            else:
                self.saver.recover_if_possible()

        test_stats = []
        self.modules.eval()
        with torch.no_grad():
            for batch in tzip(*test_set):
                test_stats.append(self.evaluate_batch(batch))

        test_summary = self.summarize(test_stats, write=True)
        for key in test_summary:
            logger.info(f"Test {key}: {test_summary[key]:.2f}")

    def save(self, stats, max_keys=[], min_keys=[]):
        """Record relevant data into a checkpoint file.

        Arguments
        ---------
        stats : mapping
            A set of meta keys and their values to store.
        max_keys : list of str
            A set of keys from the meta, keep the maximum of each.
        min_keys : list of str
            A set of keys from the meta, keep the minimum of each.
        """
        importance_keys = [ckpt_recency]
        for key in max_keys:
            importance_keys.append(lambda c, key=key: c.meta[key])
        for key in min_keys:
            importance_keys.append(lambda c, key=key: -c.meta[key])
        self.saver.save_and_keep_only(
            meta=stats, importance_keys=importance_keys,
        )
