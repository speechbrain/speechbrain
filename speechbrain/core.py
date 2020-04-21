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

    def recover_if_possible(self, max_key=None, min_key=None):
        """
        See `speechbrain.utils.checkpoints.Checkpointer.recover_if_possible`

        If neither `max_key` nor `min_key` is passed, the default
        for `recover_if_possible` is used (most recent checkpoint).

        Arguments
        ---------
        max_key : str
            A key that was stored in meta when the checkpoint
            was created. The checkpoint with the `highest` stored value
            for this key will be loaded.
        min_key : str
            A key that was stored in meta when the checkpoint
            was created. The checkpoint with the `lowest` stored value
            for this key will be loaded.
        """
        if not (max_key is None or min_key is None): 
            raise ValueError("Can't use both max and min")
        if hasattr(self, 'saver'):
            if max_key is None and min_key is None:
                self.saver.recover_if_possible()
            # NB: the lambdas need the key=key to actually store the key.
            # Otherwise the value of key is looked up dynamically 
            # (and will have changed)
            elif max_key is not None:
                self.saver.recover_if_possible(lambda c, key=max_key: c.meta[key])
            elif min_key is not None:
                self.saver.recover_if_possible(lambda c, key=min_key: -c.meta[key])
        else:
            raise KeyError(
                'The field <constants.output_folder> and the field '
                '<constants.save_folder> must both be '
                'specified in order to load a checkpoint.'
            )

    def save_and_keep_only(
        self,
        meta={},
        end_of_epoch=True,
        num_to_keep=1,
        max_keys=[],
        min_keys=[],
    ):
        """
        See `speechbrain.utils.checkpoints.Checkpointer.save_and_keep_only`

        Arguments
        ---------
        meta : mapping
            a set of key, value pairs to store alongside the checkpoint.
        end_of_epoch : bool
            Whether the checkpoint happens at the end of an epoch (last thing)
            or not. This may affect recovery. Default: True
        num_to_keep : int
            The number of checkpoints to keep for each metric.
        max_keys : iterable
            a set of keys in the meta to use for determining which checkpoints
            to keep. The highest N of each listed key will be kept.
        min_keys : iterable
            a set of keys in the meta to use for determining which checkpoints
            to keep. The lowest N of each listed key will be kept.
        """
        if hasattr(self, 'saver'):
            for key in max_keys:
                if not key in meta:
                    raise ValueError('Max key {} must be in meta'.format(key))
            for key in min_keys:
                if not key in meta:
                    raise ValueError('Min key {} must be in meta'.format(key))

            # Always save the most recent checkpoint, as well:
            importance_keys = [ckpt_recency]
            # NB: the lambdas need the key=key to actually store the key.
            # Otherwise the value of key is looked up dynamically 
            # (and will have changed)
            for key in max_keys:
                importance_keys.append(lambda c, key=key: c.meta[key])
            for key in min_keys:
                importance_keys.append(lambda c, key=key: -c.meta[key])
            self.saver.save_and_keep_only(
                meta=meta,
                end_of_epoch=end_of_epoch,
                importance_keys=importance_keys,
                num_to_keep=num_to_keep
            )
        else:
            raise KeyError(
                'The field <constants.output_folder> and the field '
                '<constants.save_folder> must both be '
                'specified in order to save a checkpoint.'
            )

    def log_epoch_stats(self, epoch, train_stats, valid_stats):
        """Log key stats about the epoch.

        Arguments
        ---------
        epoch : int
            The epoch to log.
        train_stats : mapping
            The training statistics to log, e.g. `{'wer': 22.1}`
        valid_stats : mapping
            The validation statistics to log, same format as above.

        Example
        -------
        >>> yaml_string = "{Constants: {output_folder: exp}}"
        >>> sb = Experiment(yaml_string)
        >>> sb.log_epoch_stats(3, {'loss': 4}, {'loss': 5})
        core - epoch: 3 - train loss: 4.00 - valid loss: 5.00
        """
        log_string = "epoch: {} - ".format(epoch)
        train_str = ['train %s: %.2f' % i for i in train_stats.items()]
        valid_str = ['valid %s: %.2f' % i for i in valid_stats.items()]
        log_string += ' - '.join(train_str + valid_str)
        logger.info(log_string)

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
        '--ckpts_to_save',
        type=int,
        help='The number of checkpoints to keep before deleting.',
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
    checkpointer : checkpointer
        A checkpointer that has been initialized with all the items
        that need to be saved.

    Example
    -------
    >>> class SimpleBrain(Brain):
    ...     def loss_computation(self, predictions, targets):
    ...         return torch.abs(predictions - targets)
    >>> model = torch.nn.Linear(in_features=10, out_features=10)
    >>> opt = torch.optim.SGD(model.parameters(), lr=0.01)
    >>> brain = SimpleBrain([model], opt)
    >>> brain.init_params(torch.rand(10, 10))
    >>> brain.train_batch(torch.rand(10, 10), torch.rand(10, 10))
    """
    def __init__(
        self,
        models,
        optimizer,
        scheduler=None,
        checkpointer=None,
    ):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpointer = checkpointer

    def forward(self, x):
        """Override if the models aren't simply applied in order."""
        for model in self.models:
            x = model(x)
        return x

    def compute_objectives(self, predictions, targets, train=True):
        """Compute loss, to be overridden by sub-classes."""
        raise NotImplementedError

    def train_batch(self, batch):
        """Train on one batch, override to do multiple updates."""
        inputs, targets = batch
        predictions = self(inputs)
        loss = self.compute_objectives(predictions, targets)
        loss.backward()
        self.optimizer(self.models)
        return {'loss': loss.detach()}

    def evaluate_batch(self, batch):
        """Evaluate one batch, override for different procedure than train."""
        inputs, targets = batch
        output = self(inputs)
        loss, stats = self.compute_objectives(output, targets, train=False)
        stats['loss'] = loss.detach()
        return stats

    def summarize(self, stats):
        """Take a list of stats from a pass through data and summarize it.

        By default, averages the loss and returns the average.

        Arguments
        ---------
        stats : list
            A list of stats to summarize.
        """
        return float(sum(s['loss'] for s in stats) / len(stats))

    def learn(
        self,
        epoch_counter,
        train_set,
        valid_set,
        max_keys=[],
        min_keys=[],
    ):
        """Iterate epochs and datasets to improve objective.

        Should not need to override, but still possible.

        Arguments
        ---------
        epoch_counter : epoch_counter
            Keeps track of what epoch we're on, to save if necessary.
        train_set : list
            a list of datasets to use for training, zipped before iterating.
        valid_set : list
            a list of datasets to use for validation, zipped before iterating.
        max_keys : list
            a list of keys to use for checkpointing, highest value is kept.
        min_keys : list
            a list of keys to use for checkpointing, lowest value is kept.
        """
        self(next(iter(train_set[0])), init_params=True)
        self.optimizer.init_params(self.models)
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible()

        for epoch in epoch_counter:
            self.train()
            train_stats = []
            for batch in tzip(*train_set):
                train_stats.append(self.train_batch(batch))
            train_summary = self.summarize(train_stats)

            self.eval()
            valid_stats = []
            with torch.no_grad():
                for batch in tzip(*valid_set):
                    valid_stats.append(self.evaluate_batch(batch))
            valid_summary = self.summarize(valid_stats)

            if self.scheduler is not None:
                min_val = valid_summary[min_keys[0]]
                self.scheduler([self.optimizer], epoch, min_val)
            if self.checkpointer is not None:
                self.save(valid_summary, max_keys, min_keys)

            logger.info(f'Epoch {epoch} complete')
            for key in train_summary:
                logger.info(f'Train {key}: {train_summary[key]:.2f}')
            for key in valid_summary:
                logger.info(f'Valid {key}: {valid_summary[key]:.2f}')

    def evaluate(self, test_set):
        test_stats = []
        self.eval()
        with torch.no_grad():
            for batch in tzip(*test_set):
                test_stats.append(self.evaluate_batch(batch))

        test_summary = self.summarize(test_stats, write=True)
        for key in test_summary:
            logger.info(f'Test {key}: {test_summary[key]:.2f}')

    def save(self, stats, max_keys=[], min_keys=[]):
        importance_keys = [ckpt_recency]
        for key in max_keys:
            importance_keys.append(lambda c, key=key: c.meta[key])
        for key in min_keys:
            importance_keys.append(lambda c, key=key: -c.meta[key])
        self.checkpointer.save_and_keep_only(
            meta=stats,
            importance_keys=importance_keys,
        )
