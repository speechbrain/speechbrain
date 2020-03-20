"""
 -----------------------------------------------------------------------------
 core.py

 Description: This library gathers important classes that implement crucial
              functionalities of SpeechBrain.
 -----------------------------------------------------------------------------
"""

# Importing libraries
import os
import re
import sys
import ast
import yaml
import torch
import inspect
import argparse
import itertools
import torch.nn as nn
from tqdm import tqdm
from pydoc import locate
from types import SimpleNamespace
from speechbrain.data_io.data_io import create_dataloader, load_pkl, save_pkl
from speechbrain.utils import (
    check_opts,
    import_class,
    logger_write,
    read_config,
    setup_logger,
    process_cmd_string,
    conf_to_text,
    write_config,
)


def load_params(params_filename, global_params={}):

    # Find path of the calling file, so we can load the param
    # file from the same directory
    calling_filename = inspect.getfile(inspect.currentframe().f_back)
    calling_dirname = os.path.dirname(os.path.abspath(calling_filename))
    params_filepath = os.path.join(calling_dirname, params_filename)

    # Load all parameters
    with open(params_filepath) as f:
        params = yaml.load(f, Loader=yaml.Loader)

    # Create global parameters object
    if 'global' in params:
        global_params.update(params['global'])
        del params['global']
    else:
        raise ValueError('global section required in params file')

    # Parse overrides to global variables, e.g. --local_folder=xxx
    if len(sys.argv) > 2:
        parse_overrides(global_params, sys.argv[2:])

    # Function to use for variable replacement
    def var_replace(match_obj):
        try:
            return global_params[match_obj.group(1)]
        except KeyError:
            print("ERROR: %s not in global" % match_obj.group(1))
            return match_obj.group(0)

    # Replace variables with their values
    def simple_string_replace(config, global_config):
        full_pattern = re.compile(r'^\$(\w*)$')
        sub_pattern = re.compile(r'\$(\w*)')
        for key in config:
            try:
                # If the whole variable is matched, preserve type
                match = full_pattern.match(config[key])
                if match is not None:
                    config[key] = var_replace(match)

                # If a sub-string is matched, replace with a string
                else:
                    config[key] = sub_pattern.sub(var_replace, config[key])

            # Ignore errors due to variable not being a string
            except TypeError:
                pass

    # Update global_params
    simple_string_replace(global_params, global_params)

    # Check for required variables in params file
    for required in ['output_folder', 'verbosity']:
        if required not in global_params:
            raise ValueError('%s required in global section of config'
                             % required)

    # Set up output folder
    if not os.path.isdir(global_params['output_folder']):
        os.makedirs(global_params['output_folder'])

    # Setup logger
    log_file = global_params['output_folder'] + "/log.log"
    logger = setup_logger(
        "logger",
        log_file,
        verbosity_stdout=global_params["verbosity"],
    )

    # Setup functions
    functions = {}
    for function in params:

        # Replace variable names
        simple_string_replace(params[function], global_params),

        # Load class
        _class = locate(params[function]['class_name'])
        del params[function]['class_name']
        print(function)
        print(_class)

        # Initialize class
        functions[function] = _class(
            **params[function],
            global_config=global_params,
            logger=logger,
        )

    return SimpleNamespace(**functions), global_params


def parse_overrides(global_params, argv):
    """
    Parse the overrides to the global variables. (Author: Peter Plantinga)

    Input: - global_params (type: dict)
               The set of parameters in the global section of params file.

           - argv (type: list)
               The list of arguments passed on the command-line
    """

    # Initialize parser with global params
    parser = argparse.ArgumentParser()
    for key in global_params:
        parser.add_argument("--" + key)

    # Parse command line
    for key, value in vars(parser.parse_args(argv)).items():
        if value is not None:
            global_params[key] = value
