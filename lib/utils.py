"""
-------------------------------------------------------------------------------
 utils.py

 Description: This library contains support functions that implement useful
              functionalities.
-------------------------------------------------------------------------------
"""

import os
import re
import sys
import numpy
import torch
import random
import logging
import traceback
import importlib
import subprocess
from pydoc import locate


def read_config(
    config_file, cfg_change=None, global_config={}, root_cfg=False, logger=None
):
    """
     -------------------------------------------------------------------------
     utils.read_config (author: Mirco Ravanelli)

     Description: This function reads a speechbrain config file and converts
                  into a config dictionary. Errors are raised when the format
                  is different from the one expected.

     Input (call):  - config_file (type, file, mandatory):
                       it is the configuration file formatted as expected
                       (with [global], [functions],[computations] sections).

                    - cfg_change (type:str,optional,default: None):
                       it can be used to change the param of the cfg_file
                       (e.g, cfg_change=--global,device=cuda)

                   - global_config (type, dict, optional, default: None):
                       it a dictionary containing the global variables of the
                       parent config file.

                   - root_cfg (type: bool,optional, default: False):
                       it is a flag that indicates if the current config file
                       is the root one or not.

                   - logger (type, logger, optional, default: None):
                       it the logger used to write debug and error messages.
                       If logger=None and root_cfg=True, the file is created
                       from scratch.

     Output (call):  - config (type:dict):
                       the output is a dictionary summarizing the content of
                       the config file.


     Example:   from utils import read_config

                cfg_file='cfg/minimal_examples/data_reading/\
                read_data_example.cfg'

                # reading the config file
                config=read_config(cfg_file)
                print(config)
     -------------------------------------------------------------------------
     """

    # Definition of the mandatory section
    mandatory_sections = ["functions", "computations"]

    # Reading the config file
    try:
        config_lst = open(config_file, "r")
    except Exception:
        err_msg = "Cannot read the config file %s" % (config_file)
        logger_write(err_msg, logfile=logger)

    # Initialization of the config dictionary
    config = {}

    # Active tags tracking list
    active_tags = []

    # Search patterns
    open_tag = r"\[.*\]"  # open tags (e.g., [exp])
    close_tag = r"\[/.*\]"  # close tags (e.g, [/exp])
    value_pattern = r"(.*)=(.*)"  # field=value lines

    # Reading the config file line by line
    for line in config_lst:

        line = line.rstrip()
        
        if "[computations]" not in active_tags or  "[/computations]" in line:
            
            # Removing empty characters
            line = line.lstrip()
    
        # Removing comments
        line = remove_comments(line)
            
        # Skipping empty lines
        if len(line) == 0:
            continue
        
        # Skipping empty lines composed of spaces only
        if len(line)==line.count(' '):
            continue

        # Detecting open tags [..]
        if bool(re.search(open_tag, line)) and not (
            bool(re.search(close_tag, line))
        ):

            # Do not look for new tags in the computation section.
            # This allows us to support python computations.
            if (
                bool(re.search(open_tag, line))
                and "[computations]" in active_tags
            ):

                # Replacing tabs with spaces
                line = line.replace('\t','        ')
                
                if len(config['computations'])==0:
                    left_spaces=len(line)-len(line.lstrip())
                
                config['computations']= config['computations']+\
                                    line[left_spaces:]+'\n'
                                    
                continue

            # Setting tag_closed
            tag_closed = False

            # Remove spaces
            active_tags.append(line.replace(" ", ""))

            # Initialize the curr_dict
            curr_dict = config

            for tag in active_tags:
                tag = tag.replace("[", "")
                tag = tag.replace("]", "")
                tag = tag.strip()

                # Finding the current position within the dictionary
                if tag in curr_dict.keys():
                    curr_dict = curr_dict[tag]
                else:
                    # If tag does not exist, create entry
                    curr_dict[tag] = {}

                    # For the special tag "computations", initialize a list
                    # that will contain the list of computations to perform.
                    if tag == "computations":
                        config['computations'] = ''

                    curr_dict = curr_dict[tag]
            continue

        # Detecting close tags [/..]
        if bool(re.search(close_tag, line)):

            # Remove spaces
            closed_tag = line.replace(" ", "")

            # Check if tags are correctly closed
            if len(active_tags) == 0:

                err_msg = (
                    "the tag %s is closed but never opened in cfg file %s!"
                    % (closed_tag, config_file)
                )

                logger_write(err_msg, logfile=logger)

            if closed_tag.replace("[/", "[") != active_tags[-1]:

                err_msg = (
                    'the tag %s of the cfg file %s is not closed '
                    'properly! It should be closed before %s.'
                    % (active_tags[-1], config_file, closed_tag)
                )

                logger_write(err_msg, logfile=logger)

            else:
                # Removing from the active tag list the closed element
                del active_tags[-1]

            tag_closed = True

            continue

        # Check if tag is closed before opening it
        if tag_closed and not bool(re.search(open_tag, line)):

            err_msg = (
                'after closing the tag %s in cfg file %s, you must '
                'open a new tag!'
                % (closed_tag, config_file)
            )

            logger_write(err_msg, logfile=logger)

        # Detecting value lines and adding them into the dictionary
        if bool(re.search(value_pattern, line)) and tag != "computations":
            entries = line.split("=")
            field = entries[0].strip()
            value = "=".join(entries[1:]).strip()
            # adding the field in the dictionary
            curr_dict[field] = value

        if tag == "computations":
            
            # Replacing tabs with spaces
            line = line.replace('\t','        ')
            if len(config['computations'])==0:
                left_spaces=len(line)-len(line.lstrip())
                
            config['computations']= config['computations']+\
                                    line[left_spaces:]+'\n'

    # check if all the tags are closed
    if len(active_tags) > 0:

        err_msg = (
            'the following tags are opened but not closed in '
            'cfg file %s! %s.'
            % (active_tags, config_file)
        )

        logger_write(err_msg, logfile=logger)

    # Closing the config file
    config_lst.close()

    # check if mandatory sections are specified
    if mandatory_sections is not None:
        for sec in mandatory_sections:
            if sec not in config.keys():

                err_msg = (
                    'the section [%s] is mandatory and not present in '
                    'the config file %s (got %s)'
                    % (sec, config_file, config.keys())
                )

                logger_write(err_msg, logfile=logger)

    # If needed, replace fields with the values specified in the command line
    if cfg_change is not None and len(cfg_change) > 0:

        if not isinstance(cfg_change, list):
            cfg_change = cfg_change.split(" ")

        for arg in cfg_change:

            if len(arg.split("=")) != 2:

                err_msg = (
                    'the argument specified in the command line '
                    '(or cfg_change field) must be formatted in the '
                    'following way: --section,sub-section,field=value '
                    '(e.g, functions,prepare_timit,splits=train,dev). '
                    'Got "%s"'
                    % (arg)
                )

                logger_write(err_msg, logfile=logger)

            if len(arg.split("=")[0].split(",")) < 2:

                err_msg = (
                    'the argument specified in the command line '
                    '(or cfg_change field)  must be formatted in the '
                    'following way: --section,sub-section,field=value '
                    '(e.g, functions,prepare_timit,splits=train,dev). '
                    'Got "%s"'
                    % (arg)
                )

                logger_write(err_msg, logfile=logger)

            args = arg.split("=")[0].split(",")

            # Replacing values
            out = config
            for i in range(len(args) - 1):

                try:
                    out = out[args[i].replace("--", "")]
                except Exception:

                    err_msg = (
                        'the section [%s] specified in the command line '
                        '(or cfg_change field) does not exist in the '
                        'config file %s'
                        % (args[i].replace("--", ""), config_file)
                    )

                    logger_write(err_msg, logfile=logger)

            values = arg.split("=")[1]
            out[args[-1]] = values

    # Make sure the root config file contains mandatory fields
    # such as output_folder

    if root_cfg and "global" not in config:

        err_msg = (
            'The root config file %s must  contain '
            'a section [global]'
            % (config_file)
        )

        logger_write(err_msg, logfile=logger)


    # Replacing patterns with global variables where needed
    if "global" in config:

        # Update the global_variable (only with keyword that are different)
        # This way I give the priority to higher-level config files.

        for gvar in global_config:
            config["global"][gvar] = global_config[gvar]

        # Replacing variables
        replace_global_variable(config, config["global"], logger=logger)

        # Updating the global dictionary
        global_config.update(config["global"])

        # Check if the root cfg file has the mandatory fields
        if root_cfg:

            if "output_folder" not in global_config:

                err_msg = (
                    'the section [global] in the cfg file %s must contain '
                    'a field "output_folder=" (i.e, the folder where the logs '
                    'and results are saved)'
                    % (config_file)
                )

                logger_write(err_msg, logfile=logger)

            if "verbosity" not in global_config:
                global_config["verbosity"] = "2"
            else:
                try:
                    int(global_config["verbosity"])
                except Exception:

                    err_msg = (
                        'the field "verbosity" in section [global] of the '
                        'cfg file %s must contain an integer between 0 and 2. '
                        'Got %s'
                        % (config_file, global_config["verbosity"])
                    )

                    logger_write(err_msg, logfile=logger)

        # Check if all the functions have the mandatory field "class_name="
        if "functions" in config:
            for funct in config["functions"].keys():
                if "class_name" not in config["functions"][funct]:

                    err_msg = (
                        'the function %s of the config file %s does not '
                        'contain the mandatory field "class_name=" '
                        '(e.g, class_name=core.loop)'
                        % (funct, config_file)
                    )

                    logger_write(err_msg, logfile=logger)

        # Saving the config file in the output folder:
        conf_text = conf_to_text(config)

        config_fn = config_file.split("/")[-1]

        if "output_folder" in global_config:
            write_config(
                conf_text,
                global_config["output_folder"] + "/" + config_fn,
                modality="w",
            )

    # Setting seeds
    if 'seed' in config['global']:

        try:
            seed = int(config['global']['seed'])
        except Exception:

            err_msg = ('The seed in the [global] section of the config file '
                       '%s must be an integer' % (config_file))
            logger_write(err_msg, logfile=logger)

        # Setting seeds for random, numpy, torch
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)


    # Returning the processed config dictionary
    return config


def write_config(text, filename, modality="w", logger=None):
    """
     -------------------------------------------------------------------------
     utils.write_config (author: Mirco Ravanelli)

     Description: This function writes a text into the specified file.

     Input (call):  - text (type:str or lst, mandatory):
                       it is the text to save.

                    - filename (type:path,mandatory):
                       it is the file where to save the text.

                   - modality (type: str, optional, default: 'w'):
                       is the modality used to write. 'w' for writing, 'a' for
                       appending

                   - logger (type: logger, optional, default: None):
                       it the logger used to write debug and error messages.
                       If logger=None and root_cfg=True, the file is created
                       from scratch.

     Output (call):  None


     Example:   from utils import write_config

                text='hello world!'

                # writing text
                write_config(text,'exp/text.txt')
     --------------------------------------------.----------------------------
     """

    # Creating the directory (if it does not exist)
    if not os.path.exists(os.path.dirname(filename)):

        try:
            os.makedirs(os.path.dirname(filename))
        except Exception:
            err_msg = "cannot create the config file %s." % (filename)
            logger_write(err_msg, logfile=logger)

    # Creating the file
    try:
        file_id = open(filename, modality)

    except Exception:
        err_msg = "cannot create the config file %s." % (filename)
        logger_write(err_msg, logfile=logger)

    # Writing list on the file
    if isinstance(text, list):
        for line in text:
            print(line, file=file_id)

    # Writing string on the file
    if isinstance(text, str):
        print(text, file=file_id)

    # Closing the file
    file_id.close()


def conf_to_text(config, conf_text=None, tabs=None):
    """
     -------------------------------------------------------------------------
     utils.conf_to_text (author: Mirco Ravanelli)

     Description: This recursive function converts a config dictionary into
                  the corresponding text.

     Input (call):  - config (type: dict, mandatory):
                       it is the text to save.

                     - conf_text (type: str, optional, default: None):
                       this is a support variable used to accumulate the text
                       over the recursions.

                     - tabs (type: str, optional, default: None):
                       this is a support variable used to retrieve the right
                       number of tabs to be use in the text.

     Output (call):  - conf_text (type: str)


     Example:   from utils import conf_to_text

                config={'global':{'verbosity':'1','ouput_folder':'exp/exp_1'}}

                # writing text
                text=conf_to_text(config)
                print(text)
     --------------------------------------------.----------------------------
     """

    # Initialization of the text
    if conf_text is None:
        conf_text = ""

    # Initialization of the tabs
    if tabs is None:
        tabs = ""

    # If the current section is a dictionary, process it
    if isinstance(config, dict):

        for key in config.keys():
            
            if isinstance(config[key],dict):
                continue

            # If next element is a dictionary, call the function recursively
            if isinstance(config[key], dict):
                conf_text = conf_text + "\n" + tabs + "[" + key + "]\n"
                conf_text = conf_to_text(
                    config[key], conf_text, tabs=tabs + "\t"
                )
                conf_text = conf_text + tabs + "[/" + key + "]\n"

            # Otherwise, concatenate the text
            else:
                # Managing the special section computations
                if key == "computations":
                    conf_text = conf_text + tabs + "\n[computations]\n\n"
                    if isinstance(config[key],list):
                        config[key]='\n'.join(config[key])
                        
                    lines = [tabs + "\t" + line for line in config[key].split('\n')]
                    conf_text = conf_text + "\n".join(lines)
                    conf_text = conf_text + tabs + "\n[/computations]\n"
                else:
                    # Text concatenation for standard sections
                    conf_text = (
                        conf_text + tabs + key + "=" + str(config[key]) + "\n"
                    )

    return conf_text


def replace_global_variable(
    value, global_var, curr_key=None, parent=None, logger=None
):
    """
    -------------------------------------------------------------------------
     utils.replace_global_variable (author: Mirco Ravanelli)

     Description: This recursive function replaces the variable of the config
                  file with the ones defined in the global_var dictionary.

     Input (call):  - value (type: dict or str, mandatory):
                       it contains the the original config dictionary

                     - global_var (type: dict, mandatory):
                       it is the dictionary containing the variable to replace

                     - curr_key (type: str, optional, default: None):
                       it is a support variable to manage the recursion.

                     - parent (type: str, optional, default: None):
                       it is a support variable to manage the recursion.

                      - logger (type, logger, optional, default: None):
                       it the logger used to write debug and error messages.
                       If logger=None and root_cfg=True, the file is created
                       from scratch.

     Output (call):  None


     Example:   from utils import replace_global_variable

                # Initialization of the config dictionary
                config={'functions':{'param1':'1','param2':'$glob_value'}}

                # Initialization of the global_variables
                global_var={'glob_value':'2'}

                # writing text
                replace_global_variable(config,global_var)
                print(config)
     --------------------------------------------.----------------------------
     """

    # Checking if the current value is a dictionary
    if isinstance(value, dict):
        for key in value.keys():
            replace_global_variable(
                value[key],
                global_var,
                curr_key=key,
                parent=value,
                logger=logger,
            )

    # If it is not a dictionary, read the variables
    else:
        for var in global_var.keys():
            if isinstance(value, str):
                #  It the current value contains the pattern '$' replace it
                #  with the corresponding variable of the global_var dictionary
                if "$" + var in value:
                    parent[curr_key] = value.replace(
                        "$" + var, global_var[var]
                    )
                if "$" in value and var not in global_var.keys():
                    err_msg = (
                        'The variable %s is not defined in the [global] of the'
                        ' config file'
                        % ("$" + value)
                    )
                    logger_write(err_msg, logfile=logger)

            # Managing list entries
            if isinstance(value, list):
                parent[curr_key] = []
                for element in value:
                    parent[curr_key].append(
                        element.replace("$" + var, global_var[var])
                    )


        
def process_cmd_string(cmd_str, all_funct):
    """
     ---------------------------------------------------------------------
     utils.process_cmd_string
     (author: Mirco Ravanelli)

     Description: This function processes the string in the section 
                  computations and replaces all the functions in all_funct
                  with  the method self.run_fuction. The latter initializes
                  the class only the first time it is called and then
                  executes it.

     Input:       - cmd_str (type: str, mandatory):
                      it is the string containing the command to execute.
                  - all_funct (type: list, mandatory):
                      it is a list containing all the functions already 
                      defined.


     Output:      - cmd_str  (type: str):
                     it is a string contaning the commands to run where 
                     the functions are executed through self.run_functions
                     

     Example:    from utils import process_cmd_string
     
                 cmd_str='out=my_function(input1,input2)'
                 all_funct=['my_function','my_function2']
                 print(process_cmd_string(cmd_str,all_funct))
     ---------------------------------------------------------------------
     """
    
    # Looping overl all the functions defined 
    for funct_name in all_funct:
                    
        # Manage the case in which the function object is given in input
        # to another function
        pattern_lst = [' ', ',',')']
        
        for pattern in pattern_lst:                
            inp_pattern = funct_name + pattern
            out_pattern = 'self.functions["' + funct_name + '"]' + pattern
            
            # Replace patterns
            cmd_str = cmd_str.replace(inp_pattern,out_pattern)
        
        # Replace function name with the method run_function
        inp_pattern = funct_name + '('
        out_pattern = 'self.run_function("' + funct_name + '",'
        
        # Replace patterns
        cmd_str = cmd_str.replace(inp_pattern,out_pattern)
        
    # Manage special function get_input_var
    cmd_str = cmd_str.replace('get_input_var','self.get_input_var')
    
    return cmd_str


       
def logger_write(msg, logfile=None, level="error"):
    """
     -------------------------------------------------------------------------
     utils.logger_write (author: Mirco Ravanelli)

     Description: This function write error, debug, or info msg in the
                  logfile.

     Input (call):    - msg (type: str, mandatory):
                           it contains the the original config dictionary

                      - logfile (type: logger, optional, default: None):
                           it the logfile used to write messages.

                      - level ('error','debug','info', optional,
                       default: 'error'):
                           it is level associated with the message. Errors
                           will stop the execution of the scirpt.

     Output (call):  None


     Example:   from utils import setup_logger
                from utils import logger_write

                # Initialization of the logger
                logfile=setup_logger('logger','exp/log.log')

               #  Writing an error message
               logger_write('Running log experiment',level='info',
               logfile=logfile)
               logger_write('Error1',level='error',logfile=logfile)

     -------------------------------------------------------------------------
     """

    # Managing error messages
    if level == "error":
        msg = "\n ------------------------------\nERROR: " + msg
        if logfile is not None:
            logfile.error("".join(traceback.format_stack()[-5:]) + msg)
        assert 0, msg

    # Managing debug messages
    if level == "debug":
        if logfile is not None:
            logfile.debug(msg)

    # Managing info messages
    if level == "info":
        if logfile is not None:
            logfile.info(msg)


def str_to_bool(s):
    """
     -------------------------------------------------------------------------
     utils.str_to_bool (author: Mirco Ravanelli)

     Description: This function converts a string to a boolean.

     Input (call):    - s (type: str, mandatory):
                           string to conver to a boolean.

     Output (call):  None


     Example:   from utils import str_to_bool

                print(str_to_bool('False'))
                print(str_to_bool('true'))
     -------------------------------------------------------------------------
     """

    if s == "True" or s == "true":
        return True
    if s == "False" or s == "false":
        return False
    else:
        raise ValueError


def set_default_values(field, expected, logger=None):
    """
     -------------------------------------------------------------------------
     utils.set_default_values (author: Mirco Ravanelli)

     Description: This function sets the default value when not specified.

     Input (call):    - field (type: str, mandatory):
                           it is string containing the name of the field to
                           set.

                      - expected (type: str, mandatory):
                            it is string with the expected type.

                      - logger (type, logger, optional, default: None):
                       it the logger used to write debug and error messages.

     Output (call):  value(type: str,int,float,bool)


     Example:   from utils import set_default_values

                # Setting expected value
                expected=('float(0,inf)','optional','0.8')

                # Cast and check
                print(set_default_values('drop_out',expected))
     -------------------------------------------------------------------------
     """

    # Reading default value
    default_value = expected[2]

    # Reading expected value string
    expected_type = expected[0]

    if default_value == "None":
        return None

    # Check and cast the default value
    value = check_and_cast_type(
        field, default_value, expected_type, logger=logger
    )

    return value


def check_and_cast_type(option_field, option, option_type, logger=None):
    """
     -------------------------------------------------------------------------
     utils.check_and_cast_type (author: Mirco Ravanelli)

     Description: This function checks an option and return the cast version
                  of it (depending on the option_type)

     Input (call):    - option_field(type: str, mandatory):
                           it is a string containing the name of the option.

                      - option_field (type: str, mandatory):
                            it is string contaning the option.

                      - option_type (type: str, mandatory):
                           it is a string containing the option type.

     Output (call):  option (type: str,int,float,bool):
                       it is the cas (and checked) version of option


     Example:   from utils import check_and_cast_type

                # Cast option drop_out
                print(check_and_cast_type('drop_out','0.1','float(0,1)'))

                # Cast option drop_out (with error)
                print(check_and_cast_type('drop_out','1.1','float(0,1)'))
     -------------------------------------------------------------------------
     """

    # Managing strings or list of strings
    if option_type == "str" or option_type == "str_list":

        elements = option.split(",")
        option_lst = []

        for element in elements:
            element = element.strip()
            try:
                option_lst.append(str(element))
            except Exception:

                err_msg = 'the field "%s" must contain a string (got %s)' % (
                    option_field,
                    elements,
                )

                logger_write(err_msg, logfile=logger)

        if option_type != "str":
            option = option_lst

    # Managing file or list of files
    if option_type == "file" or option_type == "file_list":

        elements = option.split(",")
        option = []

        for element in elements:
            element = element.strip()

            if not os.path.exists(element):

                err_msg = (
                    'the field "%s" contains a path "%s" that does '
                    'not exist!'
                    % (option_field, element)
                )

                logger_write(err_msg, logfile=logger)

            option.append(element)

        if option_type == "file":
            option = option[0]

    # Managing directory or list of directories
    if option_type == "directory" or option_type == "directory_list":

        elements = option.split(",")
        option = []

        for element in elements:

            element = element.strip()

            if not os.path.isdir(element):

                err_msg = (
                    'the field "%s" contains a directory "%s" that does '
                    'not exist!'
                    % (option_field, element)
                )

                logger_write(err_msg, logfile=logger)

            option.append(element)

        if option_type == "directory":
            option = option[0]

    # Managing one_of_ option
    if "one_of(" in option_type or "one_of_list(" in option_type:
        possible_values = (
            option_type.strip()
            .replace("one_of(", "")
            .replace("one_of_list(", "")
            .replace(")", "")
            .split(",")
        )

        elements = option.split(",")

        option = []

        for element in elements:
            element = element.strip()

            if element not in possible_values:

                err_msg = (
                    'the field "%s" must contain one of the following '
                    'values %s (got %s).'
                    % (option_field, possible_values, element)
                )

                logger_write(err_msg, logfile=logger)

            option.append(element)

        if "one_of(" in option_type:
            option = option[0]

    # Managing bool or list of booleans
    if option_type == "bool" or option_type == "bool_list":

        elements = option.split(",")
        option = []

        for element in elements:
            element = element.strip()
            try:
                option.append(str_to_bool(element))
            except Exception:

                err_msg = 'the field "%s" must contain a boolean (got %s).' % (
                    option_field,
                    elements,
                )

                logger_write(err_msg, logfile=logger)

        if option_type == "bool" or len(option) == 1:
            if '_list' not in option_type:
                option = option[0]

    # Managing integer or list of integers
    if "int" in option_type:

        elements = option.split(",")
        option = []

        for element in elements:
            element = element.strip()
            if "(" not in option_type:
                lower_value = -float("inf")
                upper_value = float("inf")
            else:
                if "int_list" in option_type:
                    lower_value = float(
                        option_type.split(",")[0].replace("int_list(", "")
                    )
                else:
                    lower_value = float(
                        option_type.split(",")[0].replace("int(", "")
                    )

                upper_value = float(option_type.split(",")[1].replace(")", ""))

            try:
                element = int(element)
                option.append(element)
            except Exception:

                err_msg = (
                    'the field "%s" must contain an integer (got %s).'
                    % (option_field, elements)
                )

                logger_write(err_msg, logfile=logger)

            if element < lower_value or element > upper_value:

                err_msg = (
                    'the field "%s" must contain an integer ranging '
                    'between %f and %f (got %s).'
                    % (option_field, lower_value, upper_value, elements)
                )

                logger_write(err_msg, logfile=logger)

        if "int(" in option_type or len(option) == 1:
            if '_list' not in option_type:
                option = option[0]

    # Managing float or list of floats
    if "float" in option_type:

        elements = option.split(",")
        option = []

        for element in elements:
            element = element.strip()
            if "(" not in option_type:
                lower_value = -float("inf")
                upper_value = float("inf")
            else:
                if "float_list" in option_type:
                    lower_value = float(
                        option_type.split(",")[0].replace("float_list(", "")
                    )
                else:
                    lower_value = float(
                        option_type.split(",")[0].replace("float(", "")
                    )
                upper_value = float(option_type.split(",")[1].replace(")", ""))

            try:
                element = float(element)
                option.append(element)
            except Exception:
                err_msg = 'the field "%s" must contain a float (got %s).' % (
                    option_field,
                    elements,
                )
                logger_write(err_msg, logfile=logger)

            if element < lower_value or element > upper_value:

                err_msg = (
                    'the field "%s" must contain a float ranging '
                    'between %f and %f (got %s).'
                    % (option_field, lower_value, upper_value, elements)
                )

                logger_write(err_msg, logfile=logger)

        if "float(" in option_type or len(option) == 1:
            if '_list' not in option_type:
                option = option[0]

    return option


def check_expected_options(expected_options, logger=None):
    """
     -------------------------------------------------------------------------
     utils.check_expected_options (author: Mirco Ravanelli)

     Description: This function checks if the variable expected_options
                  contains a valid option.

     Input (call):    - expected_options(type: str, mandatory):
                           it is a string containing the expected options

                      - logger (type, logger, optional, default: None):
                       it the logger used to write debug and error messages.


     Output (call):  None


     Example:   from utils import check_expected_options

                # Expected options
                expected_options={
               'class_name': ('str','mandatory'),
               'cfg_file': ('path','mandatory'),
               'cfg_change': ('str','optional','None'),
               'stop_at': ('str_list','optional','None'),
               'root_cfg': ('bool','optional','False')}

               # check expected options
               check_expected_options(expected_options)

     -------------------------------------------------------------------------
     """

    # List of all the supported types
    field = []
    field.append(
        [
            "str",
            "str_list",
            "file",
            "file_list",
            "directory",
            "directory_list",
            "one_of",
            "one_of_list",
            "bool",
            "bool_list",
            "int",
            "int_list",
            "float",
            "float_list",
        ]
    )

    field.append(["mandatory", "optional"])

    # Check if the option is supported. Otherwise, raise and error
    for option in expected_options.keys():

        if not isinstance(expected_options[option], tuple):

            err_msg = (
                'the option "%s" reported in self.expected_options '
                'must be a tuple composed of two or three elements '
                '(e.g, %s=(int,mandatory) or %s=(int,optional,0)). Got %s'
                % (option, option, option, expected_options[option],)
            )

            logger_write(err_msg, logfile=logger)

        if len(expected_options[option]) <= 1:

            err_msg = (
                'the option "%s" reported in self.expected_options '
                'must be a tuple composed of two or three elements '
                '(e.g, %s=(int,mandatory) or %s=(int,optional,0)).Got %s '
                % (option, option, option, expected_options[option],)
            )

            logger_write(err_msg, logfile=logger)

        if expected_options[option][1] not in field[1]:

            err_msg = (
                'the type reported in self.expected_options for the '
                'option "%s" must be a tuple composed of two or three elements'
                '(e.g, %s=(int,mandatory) or %s=(int,optional,0)). "mandatory"'
                ' or "optional" are the only options supported. Got ("%s")'
                % (option, option, option, expected_options[option],)
            )

            logger_write(err_msg, logfile=logger)

        if (
            expected_options[option][1] == "mandatory"
            and len(expected_options[option]) != 2
        ):

            err_msg = (
                'the type "mandatory" reported in '
                'self.expected_options for the option "%s" must be a tuple '
                'composed of two elements (e.g, %s=(int,mandatory)). Got %s'
                % (option, option, expected_options[option],)
            )

            logger_write(err_msg, logfile=logger)

        if (
            expected_options[option][1] == "optional"
            and len(expected_options[option]) != 3
        ):

            err_msg = (
                'the type "optional" reported in self.expected_options '
                'for the option "%s" must be a tuple composed of three elem '
                '(e.g, %s=(int,mandatory,0)). The last element is the default '
                'value. Got %s'
                % (option, option, expected_options[option],)
            )

            logger_write(err_msg, logfile=logger)


def check_opts(self, expected_options, data_opts, logger=None):
    """
     -------------------------------------------------------------------------
     utils.check_opts (author: Mirco Ravanelli)

     Description: This function compares the data options with the related
                  expected options and raise and error in case of mismatch.
                  The function return the options cast as reported in the
                  expected_options dictionary. Moreover is stored in the given
                  input class the cast variables.

     Input (call):     - self (type: class, mandatory):
                           it is the input class where the cast variables will
                           be defined.

                       - expected_options(type: dict, mandatory):
                           it is a dictionary containing the expected options.

                      - data_options(type: dict, mandatory):
                           it is a dictionary containing the expected options.

                      - logger (type: logger, optional, default: None):
                       it the logger used to write debug and error messages.


     Output (call):  cast_options (type: dict):
                       it is the output dictionary contaning the cast options.


     Example:   from utils import check_opts

                # Expected options
                expected_options={
               'class_name': ('str','mandatory'),
               'cfg_file': ('path','mandatory'),
               'cfg_change': ('str','optional','None'),
               'stop_at': ('str_list','optional','None'),
               'root_cfg': ('bool','optional','False')}

               # creation of a dummy class

               class dummy_class:
                   def __init__(self):
                       pass

               class_example=dummy_class()

               # Data options
               data_opts={'class_name': 'speechbrain.py',
                          'cfg_file': 'cfg/minimal_examples/features/\
                                       compute_mfccs_example.cfg',
                           'root_cfg': 'True'}

               # check expected options
               cast_opts=check_opts(class_example,expected_options,data_opts)
               print(cast_opts)
               print(class_example.root_cfg)

     -------------------------------------------------------------------------
     """

    # Check expected options
    check_expected_options(expected_options, logger=logger)

    # List of mandatory fields
    mandatory_fields = []
    optional_fields = []

    for field in expected_options:
        option_mandatory = expected_options[field][1]
        if option_mandatory == "mandatory":
            mandatory_fields.append(field)
        if option_mandatory == "optional":
            optional_fields.append(field)

    # Check and cast options
    cast_options = {}

    for option in data_opts.keys():

        # Check if the current option is in the list of expected options
        if option not in expected_options.keys():
            continue

        option_value = data_opts[option]
        option_type = expected_options[option][0]
        option_mandatory = expected_options[option][1]

        if len(option_value.strip()) == 0:
            err_msg = 'the field "%s" contains an empty value.' % (option)
            logger_write(err_msg, logfile=logger)

        # Check and cast options
        option_value = check_and_cast_type(
            option, option_value, option_type, logger=logger
        )

        cast_options[option] = option_value

    # Check if all the mandatory fields are specified
    specified_fields = set(cast_options.keys()).intersection(
        set(mandatory_fields)
    )

    if specified_fields != set(mandatory_fields):
        err_msg = 'the following mandatory fields are not specified: "%s".' % (
            set(mandatory_fields) - set(specified_fields)
        )
        logger_write(err_msg, logfile=logger)

    # Set default value for optional fields not specified
    specified_fields = set(cast_options.keys()).intersection(
        set(optional_fields)
    )
    missing_opts_fields = set(optional_fields) - set(specified_fields)

    for option in missing_opts_fields:

        value = set_default_values(
            option, expected_options[option], logger=logger
        )
        cast_options[option] = value

    # Expand the options (after that you have self.option_name)
    for option in cast_options:
        exec_line = "self." + option + "=cast_options[option]"
        exec(exec_line)

    return cast_options


def check_inputs(config, expected_inputs, input_lst, logger=None):
    """
     -------------------------------------------------------------------------
     utils.check_inputs (author: Mirco Ravanelli)

     Description: This function checks if the input matches with the type
                   expected.

     Input (call):     - config (type:dict, mandatory):
                           it is the configuration dictionary.

                       - expected_inputs (type: list, mandatory):
                           it is a list containing the expected types.

                      - input_lst(type: list, mandatory):
                           it is a  list containing the current inputs.

                      - logger (type: logger, optional, default: None):
                       it the logger used to write debug and error messages.


     Output (call):  None

     Example:   import torch
                from utils import check_inputs

                # Dummy config file initialization
                config={'class_name':'loop'}

                # Expected inputs
                expected_inputs=['int','torch.Tensor']

               # Current inputs
               current_inputs=[0,torch.rand(10)]

               # Check input:
               check_inputs(config,expected_inputs,current_inputs)

     -------------------------------------------------------------------------
     """

    # Check input_lst
    if input_lst is None:
        return

    # Making sure that the input and expected input lists have the same length
    if len(expected_inputs) != len(input_lst):

        err_msg = (
            'the number of inputs got to the function %s is different '
            'from the exepcted number (got %s, expected %s of type %s)'
            % (config["class_name"], len(input_lst),
               len(expected_inputs), expected_inputs)
        )

        logger_write(err_msg, logfile=logger)

    # Checking all the elements of the input_lst
    for i, elem in enumerate(input_lst):

        if not isinstance(expected_inputs[i],list):
            expected_inputs[i]=[expected_inputs[i]]
        
        type_ok = False
        
        for exp_inp in expected_inputs[i]:
            
            # Continue if expected_inputs[i] is a class
            if exp_inp == 'class':
                continue
    
            # Check if expected option is a valid type
            exp_type = locate(exp_inp)
    
            if exp_type is None:
    
                err_msg = (
                    "the input type %s set in the function %s does not exist"
                    % (exp_inp, config["class_name"])
                )
    
                logger_write(err_msg, logfile=logger)
    
            if isinstance(input_lst[i], exp_type):
                type_ok = True
                
                
        if not(type_ok):
          
            err_msg = (
                "the input %i of the function %s must be a %s (got %s)"
                % (
                    i,
                    config["class_name"],
                    exp_inp,
                    type(input_lst[i]),
                )
            )

            logger_write(err_msg, logfile=logger)


def get_all_files(
    dirName, match_and=None, match_or=None, exclude_and=None, exclude_or=None
):
    """
     -------------------------------------------------------------------------
     utils.get_all_files (author: Mirco Ravanelli)

     Description: This function get a list of files within found within a
                  folder. Different options can be used to restrict the search
                  to some specific patterns.

     Input (call):     - dirName (type: directory, mandatory):
                           it is the configuration dictionary.

                       - match_and (type: list, optional, default:None):
                           it is a list that contains pattern to match. The
                           file is returned if all the entries in match_and
                           are founded.

                       - match_or (type: list, optional, default:None):
                           it is a list that contains pattern to match. The
                           file is returned if one the entries in match_or are
                           founded.

                       - exclude_and (type: list, optional, default:None):
                           it is a list that contains pattern to match. The
                           file is returned if all the entries in match_or are
                            not founded.

                       - exclude_or (type: list, optional, default:None):
                           it is a list that contains pattern to match. The
                           file is returned if one of the entries in match_or
                           is not founded.

                      - logger (type: logger, optional, default: None):
                       it the logger used to write debug and error messages.


     Output (call):  - allFiles(type:list):
                       it is the output list of files.


     Example:   from utils import get_all_files

                # List of wav files
                print(get_all_files('samples',match_and=['.wav']))

               # List of cfg files
               print(get_all_files('exp',match_and=['.cfg']))

     -------------------------------------------------------------------------
     """

    # Match/exclude variable initialization
    match_and_entry = True
    match_or_entry = True
    exclude_or_entry = False
    exclude_and_entry = False

    # Create a list of file and sub directories
    listOfFile = os.listdir(dirName)
    allFiles = list()

    # Iterate over all the entries
    for entry in listOfFile:

        # Create full path
        fullPath = os.path.join(dirName, entry)

        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_all_files(
                fullPath,
                match_and=match_and,
                match_or=match_or,
                exclude_and=exclude_and,
                exclude_or=exclude_or,
            )
        else:

            # Check match_and case
            if match_and is not None:
                match_and_entry = False
                match_found = 0

                for ele in match_and:
                    if ele in fullPath:
                        match_found = match_found + 1
                if match_found == len(match_and):
                    match_and_entry = True

            # Check match_or case
            if match_or is not None:
                match_or_entry = False
                for ele in match_or:
                    if ele in fullPath:
                        match_or_entry = True
                        break

            # Check exclude_and case
            if exclude_and is not None:
                match_found = 0

                for ele in exclude_and:
                    if ele in fullPath:
                        match_found = match_found + 1
                if match_found == len(exclude_and):
                    exclude_and_entry = True

            # Check exclude_and case
            if exclude_or is not None:
                exclude_or_entry = False
                for ele in exclude_or:
                    if ele in fullPath:
                        exclude_or_entry = True
                        break

            # If needed, append the current file to the output list
            if (
                match_and_entry
                and match_or_entry
                and not (exclude_and_entry)
                and not (exclude_or_entry)
            ):
                allFiles.append(fullPath)

    return allFiles


def setup_logger(
    name, log_file, level_file=logging.DEBUG, verbosity_stdout="2"
):
    """
     -------------------------------------------------------------------------
     utils.setup_logger (author: Mirco Ravanelli)

     Description: This function sets up the logger use to save error, debug,
                  and info messages.

     Input (call):    - name (type: str, mandatory):
                           it is the name given to the logger

                      - logfile (type: logger, optional, default: None):
                           it the logfile used to write messages.

                      - level_file ('logging.DEBUG','logging.INFO',
                        'logging.ERROR', optional, default: 'logging.DEBUG'):
                           it is level associated with the logger.
                           A Message is written into the logfile if its level
                           is the same or higher that the one set here.

                      - verbosity_stdout ('0','1','2','optional',
                        default: '2'):
                           it defines the verbosity at stdout level.

     Output (call):  logger (type class logger)


     Example:   from utils import setup_logger
                from utils import logger_write

                # Initialization of the logger
                logfile=setup_logger('logger','exp/log.log')

               #  Writing an error message
               logger_write('Running log experiment',level='info',
               logfile=logfile)
               logger_write('Error1',level='error',logfile=logfile)

     -------------------------------------------------------------------------
     """

    # Logger initialization
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Getting logger file handler
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(level_file)

    # Getting stream handler
    stream_handler = logging.StreamHandler(sys.stdout)

    # Setting verbosity std_out
    verbosity_stdout = int(verbosity_stdout)

    if verbosity_stdout <= 0:
        stream_handler.setLevel(logging.ERROR)

    if verbosity_stdout == 1:
        stream_handler.setLevel(logging.INFO)

    if verbosity_stdout > 1:
        stream_handler.setLevel(logging.DEBUG)

    # Adding Handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def import_class(library, logfile=None):
    """
     -------------------------------------------------------------------------
     utils.import_class (author: Mirco Ravanelli)

     Description: This function is used to import classes from a string.

     Input (call):    - library (type: str, mandatory):
                           it is the string containing the library to load

                      - logfile (type: logger, optional, default: None):
                           it the logfile used to write messages.

     Output (call):  target_class (type: class)


     Example:  from utils import import_class

               # importing class
               loop=import_class('core.loop')

     -------------------------------------------------------------------------
     """

    # Check the library string
    if len(library.split(".")) == 1:

        err_msg = (
            'the class to import must be in the following format: '
            'library.class (e.g., data_io.load_data). Got %s".'
            % (library)
        )

        logger_write(err_msg, logfile=logfile)

    # Split library and class
    module_name = '.'.join(library.split(".")[:-1])
    class_name =  library.split(".")[-1]   

    # Loading the module
    try:
        module_object = importlib.import_module(module_name)
    except Exception:
        err_msg = 'cannot import module %s".' % (module_name)
        logger_write(err_msg, logfile=logfile)

    # Loading the class
    try:
        target_class = getattr(module_object, class_name)

    except Exception:

        err_msg = 'cannot import the class %s from the module %s".' % (
            class_name,
            module_name,
        )

        logger_write(err_msg, logfile=logfile)

    return target_class


def list2dict(list_keys, list_values):
    """
     -------------------------------------------------------------------------
     utils.list2dict (author: Mirco Ravanelli)

     Description: This function converts two lists into a dictionary

     Input (call):    - list_keys (type: list, mandatory):
                           it is a list containing the keys.

                      - list_values (type: list, mandatory):
                           it is a list containing the values.

     Output (call):  dict_data (type: dict)


     Example:  from utils import list2dict

               print(list2dict(['n_loops','drop_last'],[10,False]))

     -------------------------------------------------------------------------
     """

    # Converting the lists to a dict
    dict_data = dict(zip(list_keys, list_values))

    return dict_data


def split_list(seq, num):
    """
     -------------------------------------------------------------------------
     utils.split_list (author: Mirco Ravanelli)

     Description: This function splits the input list in N parts.

     Input (call):    - seq (type: list, mandatory):
                           it is the input list

                      - nums (type: int(1,inf), mandatory):
                           it is the number of chunks to produce

     Output (call):  out (type: list):
                       it is a list containing all chunks created.


     Example:  from utils import split_list

               print(split_list([1,2,3,4,5,6,7,8,9],4))

     -------------------------------------------------------------------------
     """

    # Average length of the chunk
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    # Creating the chunks
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def remove_comments(line):
    """
     -------------------------------------------------------------------------
     utils.split_list (author: Mirco Ravanelli)

     Description: This function removes comments from a line

     Input (call):    - line (type: str, mandatory):
                           it is the input line

     Output (call):  line (type: str):
                       it is the output line (without comments)


     Example:  from utils import remove_comments

               print(remove_comments('Hello! # World'))

     -------------------------------------------------------------------------
     """

    # Removing comments
    if "#" in line:
        if line[0] != "#":
            line = line[0:line.find("#") - 1]
        else:
            line = ""
    return line

def csv_to_dict(csv_file):
    """
     -------------------------------------------------------------------------
     utils.csv_to_dict (author: Mirco Ravanelli)

     Description: This function reads the csv_file and coverts into into a
                  a dictionary.

     Input (call):    - csv_file (type: file, mandatory):
                           it is the csv file to convert.

     Output (call):   - data_dict (type: dict):
                           it is a dictionary containing the sentences
                           reported in the input csv file.


     Example:   from utils import csv_to_dict

                csv_file='samples/audio_samples/csv_example.csv'

                print(csv_to_dict(csv_file))

     -------------------------------------------------------------------------
     """

    # Setting regex to read the data entries
    value_regex = re.compile(r"([\w]*)=([\w\$\(\)\/'\"\,\-\_\.\:\#]*)")
    del_spaces = re.compile(r"=([\s]*)")

    # Initialization of the data_dict function
    data_dict = {}

    # Reading the csv file line by line
    for data_line in open(csv_file):

        # Removing spaces
        data_line = data_line.strip()

        # Removing comments
        data_line = remove_comments(data_line)

        # Skip empty lines
        if len(data_line) == 0:
            continue

        # Replacing multiple spaces
        data_line = (
            re.sub(" +", " ", data_line)
            .replace(" ,", ",")
            .replace(", ", ",")
            .replace("= ", "=")
            .replace(" =", "=")
        )

        # Extracting key=value patterns in the csv file
        data_line = del_spaces.sub("=", data_line)
        values = value_regex.findall(data_line)

        # Creating a dictionary from values
        data = dict(values)

        # Retrieving the sentence_id
        snt_id = data["ID"]

        # Adding the current data into the data_dict
        data_dict[snt_id] = data

    return data_dict


def recursive_items(dictionary):
    
    for key, value in dictionary.items():
        if type(value) is dict:
            yield from recursive_items(value)
        else:
            yield (key, value)
            
            
def dict_to_str(d):
    """
     -------------------------------------------------------------------------
     utils.csv_to_dict (author: Mirco Ravanelli)

     Description: This function converts a dictionary into a string

     Input (call):    - d (type: dictionary, mandatory):

     Output (call):   - str_out(type: string):
                           it a string containing the information of the input
                           dict.


     Example:   from utils import dict_to_str

                print(dict_to_str({'key1':'value1','key2':'value2'}))

     -------------------------------------------------------------------------
     """

    # String initialization
    str_out = ""
    for key in d:
        str_out = str_out + key + "=" + str(d[key]) + " "

    # Add new line
    str_out = str_out + "\n"
    return str_out


def run_shell(cmd, logger=None):
    """
     -------------------------------------------------------------------------
     utils.run_shell (author: Mirco Ravanelli)

     Description: This function can be used to run command from the bash shell

     Input (call):    - cmd (type: str, mandatory):
                       it a string containing the command.

     Output (call):   - output(type: str):
                       it is a string containign the standard output


     Example:   from utils import run_shell

                run_shell("echo 'hello world'")

     -------------------------------------------------------------------------
     """

    # Executing the command
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )

    # Capturing standard output and error
    (output, err) = p.communicate()
    p.wait()

    # Adding information in the logger
    if logger is not None:
        msg = output.decode("utf-8") + "\n" + err.decode("utf-8")
        logger_write(msg, logfile=logger, level="debug")

    return output,err


def create_exec_config(cfg_file, cmd_arg):
    """
     -------------------------------------------------------------------------
     utils.create_exec_config (author: Mirco Ravanelli)

     Description: This function creates the exec_config dict for the root cfg
                   file.

     Input (call):    - cfg_file, (type: file, mandatory):
                       it the configuration file.

                      - cmd_arg, (type: file, mandatory):
                       it contains the argument passed through the command
                       file.

     Output (call):   - output(type: dict):
                       it is the dictionary to be used to initialize the first
                       computations.


     Example:   from utils import create_exec_config

                cfg_file='cfg/minimal_examples/data_reading\
                /read_data_example.cfg'
                cmd_arg=''
                print(create_exec_config(cfg_file,cmd_arg))
     -------------------------------------------------------------------------
     """

    # Dictionary initialization
    exec_config = {
        "class_name": "speechbrain.py",
        "cfg_file": cfg_file,
        "root_cfg": "True",
    }

    # Managing command-line arguments
    cmd_arg = " ".join(cmd_arg)

    if len(cmd_arg) > 0:
        exec_config.update({"cfg_change": cmd_arg})

    return exec_config


