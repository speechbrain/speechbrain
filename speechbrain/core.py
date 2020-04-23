"""Core SpeechBrain code for running experiments.

Author(s): Peter Plantinga 2020
"""

import re
import os
import sys
import yaml
import torch
import pydoc
import shutil
import logging
import inspect
import argparse
import subprocess
from tqdm.contrib import tzip
from speechbrain.utils.logger import setup_logging
from speechbrain.utils.checkpoints import Checkpointer
from speechbrain.utils.checkpoints import ckpt_recency
from speechbrain.utils.data_utils import load_extended_yaml, resolve_references
logger = logging.getLogger(__name__)


class Experiment:
    r'''A class for reading configuration files and creating folders

    The primary method for specifying parameters for an experiment is
    to provide a YAML file with the syntax extensions described by
    `speechbrain.utils.data_utils.load_extended_yaml`. In addition,
    the Experiment class expects the loaded YAML to have three
    top-level sections: `constants`, `saveables`, and `functions`.
    These three sections indicate the following:

    1. `constants:` These items are expected to be scalar nodes that
        can be referenced throughout the file by using the extended YAML
        reference tags `!ref`. In addition, a few constants are treated as
        experimental arguments:

        * output_folder (str): The experimental directory for logs, results,
            labels, and other experimental files.
        * local_folder (str): The directory to use for local storage.
        * save_folder (str): The directory to use for storing parameters
            (see next section).
        * ckpts_to_keep (int): The number of "best" checkpoints to keep.
        * seed (int): The random seed for reproducing the experiment on the
            same device on the same machine.
        * log_config (str): A yaml file specifying how logging should be done.

        These arguments can be specified in the command line, for easier
        spawning of multiple experiments.

    2. `saveables:` These items are expected to be instances of sub-classes
        of `torch.nn.Module` or to implement a custom saver method, so that
        the relevant parameters can be automatically discovered and saved.
        The parameters are saved in the `save_folder` which should be
        specified in the `Constants` section above. In addition, these
        items will be made available as attributes of
        the `Experiment` instance, so they can be easily accessed.

    3. `functions:` Additional functions that do not need to have their
        state saved by the saver. Like the items defined in `saveables`,
        all of these items will be added as attributes to the `Experiment`
        instance, so that they can be easily accessed.

    Overrides
    ---------
    Any of the keys listed in the yaml file may be overriden via the
    command-line argument: `yaml_overrides`. The value of this argument should
    be a yaml-formatted string, though a shortcut has been provided for
    nested items, e.g.

        "{model.arg1: value, model.arg2.arg3: 3., model.arg2.arg4: True}"

    will be interpreted as:

        {'model': {'arg1': 'value', 'arg2': {'arg3': 3., 'arg4': True}}}

    Arguments
    ---------
    yaml_stream : stream
        A file-like object or string containing
        experimental parameters. The format of the file is described in
        the method `speechbrain.utils.data_utils.load_extended_yaml()`.
    commandline_args : list
        The arguments from the command-line for
        overriding the experimental parameters.

    Example
    -------
    >>> yaml_string = """
    ... constants:
    ...     output_folder: exp
    ...     save_folder: !ref <constants.output_folder>/save
    ... """
    >>> sb = Experiment(yaml_string)
    core - Beginning experiment!
    core - Output folder: exp
    >>> sb.save_folder
    'exp/save'
    '''
    def __init__(
        self,
        yaml_stream,
        commandline_args=[],
    ):
        """"""
        # Parse yaml overrides, with command-line args taking precedence
        # precedence over the parameters listed in the file.
        overrides = {'constants': {}}
        cmd_args = parse_arguments(commandline_args)
        if 'yaml_overrides' in cmd_args:
            overrides.update(parse_overrides(cmd_args['yaml_overrides']))
            del cmd_args['yaml_overrides']
        overrides['constants'].update(cmd_args)

        # Load parameters file and store
        parameters = load_extended_yaml(yaml_stream, overrides)
        for toplevel_field in ['constants', 'saveables', 'functions']:
            if toplevel_field in parameters:
                self._update_attributes(parameters[toplevel_field])
        self._update_attributes(cmd_args, override=True)

        # Use experimental parameters to initialize experiment
        if hasattr(self, 'seed'):
            torch.manual_seed(self.seed)

        # Stuff depending on having an output_folder
        logger_overrides = {}
        if hasattr(self, 'output_folder'):
            if not os.path.isdir(self.output_folder):
                os.makedirs(self.output_folder)

            # Write the parameters file
            if hasattr(yaml_stream, 'seek'):
                yaml_stream.seek(0)
            resolved_yaml = resolve_references(yaml_stream, overrides)
            params_filename = os.path.join(self.output_folder, 'params.yaml')
            with open(params_filename, 'w') as w:
                shutil.copyfileobj(resolved_yaml, w)

            # Copy executing folder to output directory
            module = inspect.getmodule(inspect.currentframe().f_back)
            if module is not None:
                callingdir = os.path.dirname(os.path.realpath(module.__file__))
                parentdir, zipname = os.path.split(callingdir)
                archivefile = os.path.join(self.output_folder, zipname)
                shutil.make_archive(archivefile, 'zip', parentdir, zipname)

            # Change logging file to be in output dir
            logger_override_string = (
                '{handlers.file_handler.filename: %s}'
                % os.path.join(self.output_folder, 'log.txt')
            )
            logger_overrides = parse_overrides(logger_override_string)

            # Create checkpointer for loading/saving state
            if hasattr(self, 'save_folder') and 'saveables' in parameters:
                self.saver = Checkpointer(
                    checkpoints_dir=self.save_folder,
                    recoverables=parameters['saveables'],
                )

        # Log exceptions automatically
        if not hasattr(self, 'log_config'):
            self.log_config = 'logging.yaml'
        setup_logging(self.log_config, logger_overrides)
        sys.excepthook = _logging_excepthook

        # Log beginning of experiment!
        logger.info('Beginning experiment!')
        if hasattr(self, 'output_folder'):
            logger.info('Output folder: %s' % self.output_folder)

        # Log commit hash
        commit_hash = subprocess.check_output(["git", "describe", "--always"])
        logger.debug("Commit hash: '%s'" % commit_hash.decode('utf-8').strip())

    def _update_attributes(self, attributes, override=False):
        r'''Update the attributes of this class to reflect a set of parameters

        Arguments
        ---------
        attributes : mapping
            A dict that contains the essential parameters for
            running the experiment. Usually loaded from a yaml file using
            `load_extended_yaml()`.
        '''
        for param, new_value in attributes.items():
            if isinstance(new_value, dict):
                value = getattr(self, param, {})
                value.update(new_value)
            else:
                if hasattr(self, param) and not override:
                    raise KeyError('Parameter %s is defined multiple times')
                value = new_value
            setattr(self, param, value)


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
    >>> parse_arguments(['--seed', '10'])
    {'seed': 10}
    """
    parser = argparse.ArgumentParser(
        description='Run a SpeechBrain experiment',
    )
    parser.add_argument(
        '--yaml_overrides',
        help='A yaml-formatted string representing a dictionary of '
        'overrides to the parameters in the param file. The keys of '
        'the dictionary can use dots to represent levels in the yaml '
        'hierarchy. For example: "{model.param1: value1}" would '
        'override the param1 parameter of the model node.',
    )
    parser.add_argument(
        '--output_folder',
        help='A folder for storing all experiment-related outputs.',
    )
    parser.add_argument(
        '--data_folder',
        help='A folder containing the data used for training',
    )
    parser.add_argument(
        '--save_folder',
        help='A folder for storing checkpoints that allow restoring '
        'progress for testing or re-starting training.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='A random seed to reproduce experiments on the same machine',
    )
    parser.add_argument(
        '--log_config',
        help='A file storing the configuration options for logging',
    )

    # Ignore items that are "None", they were not passed
    parsed_args = vars(parser.parse_args(arg_list))
    return {k: v for k, v in parsed_args.items() if v is not None}


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
        if '.' in arg:
            nest(overrides, arg.split('.'), val)
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


class Brain(torch.nn.Module):
    """A class for abstracting details of training.

    Sub-class this class to use the `learn` function, and override methods
    to specify behavior. In particular, the  `loss_computation` method should
    be overridden to compute your particular loss, and if the models are not
    simply applied in sequence, then `init_params` and `forward` should be
    overridden as well.

    Arguments
    ---------
    models : list
        A sequence of models that are being trained.
    optimizer : optimizer
        Takes a list of models for which the backwards has been
        computed, and applies the updates to all the parameters.
    scheduler : scheduler
        Takes an optimizer and changes the learning rate appropriately.
    saver : checkpointer
        A checkpointer that has been initialized with all the items
        that need to be saved.

    Example
    -------
    >>> from speechbrain.nnet.optimizers import optimize
    >>> model = torch.nn.Linear(in_features=10, out_features=10)
    >>> class SimpleBrain(Brain):
    ...     def forward(self, x, init_params=False):
    ...         return model(x)
    ...     def compute_objectives(self, predictions, targets, train=True):
    ...         return torch.nn.functional.l1_loss(predictions, targets)
    >>> brain = SimpleBrain([model], optimize('sgd', 0.01))
    >>> brain.learn(
    ...     epoch_counter=range(1),
    ...     train_set=([torch.rand(10, 10)], [torch.rand(10, 10)]),
    ... )
    """
    def __init__(
        self,
        models,
        optimizer,
        scheduler=None,
        saver=None,
    ):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
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

    def train_batch(self, batch):
        """Train on one batch, override to do multiple updates.

        Arguments
        ---------
        batch : list of torch.Tensors
            batch to use for training, usually including both inputs
            and targets.
        """
        inputs, targets = batch
        predictions = self(inputs)
        loss = self.compute_objectives(predictions, targets)
        loss.backward()
        self.optimizer(self.models)
        return {"loss": loss.detach()}

    def evaluate_batch(self, batch):
        """Evaluate one batch, override for different procedure than train.

        Arguments
        ---------
        batch : list of torch.Tensors
            batch to evaluate, usually including inputs and targets.
        """
        inputs, targets = batch
        output = self(inputs)
        loss, stats = self.compute_objectives(output, targets, train=False)
        stats["loss"] = loss.detach()
        return stats

    def summarize(self, stats):
        """Take a list of stats from a pass through data and summarize it.

        By default, averages the loss and returns the average.

        Arguments
        ---------
        stats : list of dicts
            A list of stats to summarize.
        """
        return {"loss": float(sum(s["loss"] for s in stats) / len(stats))}

    def learn(
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        max_keys=[],
        min_keys=[],
    ):
        """Iterate epochs and datasets to improve objective.

        Relies on the existence of mulitple functions that can (or should) be
        overridden. The following are functions not defined in `nn.Module`
        but are used and expected to have a certain behavior:

        * `train_batch`
        * `evaluate_batch`
        * `summarize`
        * `save`

        Arguments
        ---------
        epoch_counter : epoch_counter
            Keeps track of what epoch we're on, to save if necessary.
        train_set : list of DataLoaders
            a list of datasets to use for training, zipped before iterating.
        valid_set : list of Data Loaders
            a list of datasets to use for validation, zipped before iterating.
        max_keys : list of str
            a list of keys to use for checkpointing, highest value is kept.
        min_keys : list of str
            a list of keys to use for checkpointing, lowest value is kept.
        """
        self(next(iter(train_set[0])), init_params=True)
        self.optimizer.init_params(self.models)
        if self.saver is not None:
            self.saver.recover_if_possible()

        for epoch in epoch_counter:
            self.train()
            train_stats = []
            for batch in tzip(*train_set):
                train_stats.append(self.train_batch(batch))
            summary = self.summarize(train_stats)

            logger.info(f"Epoch {epoch} complete")
            for key in summary:
                logger.info(f"Train {key}: {summary[key]:.2f}")

            if valid_set is not None:
                self.eval()
                valid_stats = []
                with torch.no_grad():
                    for batch in tzip(*valid_set):
                        valid_stats.append(self.evaluate_batch(batch))
                summary = self.summarize(valid_stats)

                if self.scheduler is not None:
                    min_val = summary[min_keys[0]]
                    self.scheduler([self.optimizer], epoch, min_val)
                if self.saver is not None:
                    self.save(valid_summary, max_keys, min_keys)

                for key in valid_summary:
                    logger.info(f"Valid {key}: {valid_summary[key]:.2f}")

    def evaluate(self, test_set, max_key=None, min_key=None):
        """Iterate test_set and evaluate model performance.

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

        if max_key is not None:
            self.saver.recover_if_possible(lambda c, key=max_key: c.meta[key])
        elif min_key is not None:
            self.saver.recover_if_possible(lambda c, key=min_key: -c.meta[key])
        else:
            self.saver.recover_if_possible()

        test_stats = []
        self.eval()
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
            meta=stats,
            importance_keys=importance_keys,
        )
