import re
import os
import yaml
import logging
import inspect
import argparse
from speechbrain.utils.logger import setup_logger
from speechbrain.utils.data_utils import recursive_update
from speechbrain.utils.data_utils import load_extended_yaml
logger = logging.getLogger(__name__)


class Experiment:
    """
    Description:
        The experiment class implements important functionality related to
        running experiments, such as setting up the experimental directories
        and loading hyperparameters. A few key parameters, listed below,
        can be set in three ways, in increasing priority order.

            1. They may be passed to the `__init__()` method for this class
            2. They may be stored in a yaml file, the name of which is
                passed to the `__init__()` method of this class.
            3. They may be passed as command-line arguments.

        Any of the keys listed in the yaml file may be overriden using
        the `overrides` parameter passed either via `__init__()` or
        via the command-line. The value of this parameter should be a
        yaml-formatted string, though a shortcut has been provided for
        nested items, e.g.

        ```
        {model.arg1: value, model.arg2.arg3: 3., model.arg2.arg4: True}
        ```
        will be interpreted as:
        ```
        {'model': {'arg1': 'value', 'arg2': {'arg3': 3., 'arg4': True}}}
        ```

    Input:
        - param_filename (type: file, default: None)
            A file for reading experimental hyperparameters. This file is
            expected to be in the same folder as the function that calls
            this one. The format of the file is described in the method
            `load_extended_yaml()`. The rest of the parameters to this
            function may also be specified in the command-line parameters
            or in the `constants:` section of the yaml file.

        - overrides (type: string, default: '')
            A yaml-formatted string containing overrides for the parameters
            listed in the file passed to `param_filename`.

        - output_folder (type: file, default: 'exp')
            A folder to store the results of the experiment, as well as
            any checkpoints, logs, or other generated data.

        - verbosity (type: int, default: 0)
            How much information to give the user about the progress
            of the experiment. Levels range from 0 to 2.

        - device (type: one_of(cuda, cpu), default: 'cuda')
            The device to execute the experiment on.

        - seed (type: int, default: None)
            The random seed used to ensure the experiment is reproducible
            if executed on the same device on the same machine.

    Example:
        >>> sb = Experiment()
        >>> sb.constants['verbosity']
        0

    Author:
        Peter Plantinga 2020
    """
    def __init__(
        self,
        param_filename=None,
        overrides='',
        output_folder=None,
        verbosity=0,
        device='cuda',
        seed=None,
        log_config=None,
    ):
        # Initialize stored values
        self.constants = {
            'output_folder': output_folder,
            'verbosity': verbosity,
            'device': device,
            'seed': seed,
            'log_config': log_config,
        }

        # Parse overrides, with command-line args taking precedence
        # over the parameters passed to this method. These overrides
        # will take precedence over the parameters listed in the file.
        commandline_args = parse_arguments()
        overrides = parse_overrides(overrides)
        overrides.update(parse_overrides(commandline_args['overrides']))

        # Find path of the calling file, so we can load the yaml
        # file from the same directory
        calling_filename = inspect.getfile(inspect.currentframe().f_back)
        calling_dirname = os.path.dirname(os.path.abspath(calling_filename))
        relative_filepath = os.path.join(calling_dirname, param_filename)
        if os.path.isfile(relative_filepath):
            param_filename = relative_filepath

        # Load parameters file and store
        parameters = load_extended_yaml(open(param_filename), overrides)
        self.update_attributes(parameters)

        # Set up output folder and logger
        if (self.constants['output_folder']
                and not os.path.isdir(self.constants['output_folder'])):
            os.makedirs(self.constants['output_folder'])
        # logger = setup_logger(log_config, self.constants['verbosity'])

    def update_attributes(self, parameters):
        """
        Description:
            Update the attributes of this class to reflect the parameters
            passed via the config file.

        Input:
            - parameters (type: dict, mandatory)
                A dict that contains the essential parameters for running
                the experiment. Usually loaded from a yaml file using
                `load_extended_yaml()`.

        Author:
            Peter Plantinga 2020
        """
        for param, new_value in parameters.items():
            if isinstance(new_value, dict):
                value = getattr(self, param, {})
                value.update(new_value)
            else:
                value = new_value
            setattr(self, param, value)


def parse_arguments():
    """
    Description:
        Parse command-line arguments to the experiment. The parsed
        arguments are returned as a dictionary.

    Author:
        Peter Plantinga 2020
    """
    parser = argparse.ArgumentParser(
        description='Run a SpeechBrain experiment',
    )
    parser.add_argument(
        '--param_filename',
        help='An extended-yaml formatted file containing experimental '
        'parameters.',
    )
    parser.add_argument(
        '--overrides',
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
        '--verbosity',
        type=int,
        choices=[0, 1, 2],
        help='The amount of output to print to stdout.',
    )
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        help='The device to use for PyTorch computations.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='A random seed to reproduce experiments on the same machine',
    )
    return parser.parse_args().__dict__


def parse_overrides(override_string):
    """
    Description:
        Parse overrides from a yaml string representing paired args and values

    Input:
        - override_string (type: string, mandatory)
            A yaml-formatted string, where each (key: value) pair overrides
            the same pair in a loaded file.

    Example:
        >>> parse_overrides("{model.arg1: val1, model.arg2.arg3: 3.}")
        {'model': {'arg1': 'val1', 'arg2': {'arg3': 3.}}}
    """
    overrides = {}
    if override_string:
        overrides = yaml.safe_load(override_string)
        for arg, val in overrides.items():
            if '.' in arg:
                nest(overrides, arg.split('.'), val)
                del overrides[arg]

    return overrides


def nest(dictionary, args, val):
    """
    Description:
        Create a nested sequence of dictionaries, based on an arg list.

    Input:
        - dictionary: dict
            this object will be updated with the nested arguments.

        - args: list
            a list of parameters specifying a nested location.

        - val: any
            The value to store at the specified nested location.

    Example:
        >>> params = {}
        >>> nest(params, ['arg1', 'arg2', 'arg3'], 'value')
        >>> params
        {'arg1': {'arg2': {'arg3': 'value'}}}

    Author:
        Peter Plantinga 2020
    """
    if len(args) == 1:
        overrides[args[0]] = val
        return

    if arg[0] not in overrides:
        overrides[arg[0]] = {}

    nest(overrides[arg[0]], arg[1:], val)
