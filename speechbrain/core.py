"""Core SpeechBrain code for running experiments.

Author(s): Peter Plantinga 2020
"""

import re
import os
import sys
import yaml
import torch
import shutil
import logging
import inspect
import argparse
from speechbrain.utils.logger import setup_logging
from speechbrain.utils.checkpoints import Checkpointer
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
        overrides = {}
        cmd_args = parse_arguments(commandline_args)
        if 'yaml_overrides' in cmd_args:
            overrides = parse_overrides(cmd_args['yaml_overrides'])

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
        logger = setup_logging(self.log_config, logger_overrides)
        sys.excepthook = _logging_excepthook

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
            elif max_key is not None:
                def max_sort(ckpt):
                    return ckpt.meta[max_key]
                self.saver.recover_if_possible(max_sort)
            elif min_key is not None:
                def min_sort(ckpt):
                    return -ckpt.meta[min_key]
                self.saver.recover_if_possible(min_sort)
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
        end_of_spoech : bool
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

            importance_keys = []
            for key in ['unixtime'] + max_keys:
                importance_keys.append(lambda x: x.meta[key])
            for key in min_keys:
                importance_keys.append(lambda x: -x.meta[key])
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
        '--local_folder',
        help='A folder for storing temporary on-disk information. Useful '
        'for functions like `copy_locally()`.',
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
