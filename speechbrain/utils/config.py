"""
 -----------------------------------------------------------------------------
 config.py

 Description: This library gathers utils functions for reading and writing
              configuration files.
 -----------------------------------------------------------------------------
"""

import os
import re
import numpy
import torch
import random
from speechbrain.utils.logger import logger_write

def read_config(
    config_file, cfg_change=None, global_config={}, root_cfg=False, logger=None
):
    """
     -------------------------------------------------------------------------
     speechbrain.utils.config.read_config (author: Mirco Ravanelli)

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


     Example:   from speechbrain.utils.config import read_config

                cfg_file='cfg/minimal_examples/basic_processing/\
                minimal_processing_read_write_example_noise.cfg'

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
     speechbrain.utils.config.write_config (author: Mirco Ravanelli)

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


     Example:   from lib.utils.config import write_config

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
     speechbrain.utils.config.conf_to_text (author: Mirco Ravanelli)

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


     Example:   from lib.utils.config import conf_to_text

                config={'global':'verbosity:1'}

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
    speechbrain.utils.config.replace_global_variable (author: Mirco Ravanelli)

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


     Example:   from lib.utils.config import replace_global_variable

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
     speechbrain.utils.config.process_cmd_string
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
                     

     Example:    from lib.utils.config import process_cmd_string
     
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

def create_exec_config(cfg_file, cmd_arg):
    """
     -------------------------------------------------------------------------
     speechbrain.utils.config.create_exec_config (author: Mirco Ravanelli)

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


     Example:   from lib.utils.config import create_exec_config

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

def remove_comments(line):
    """
     -------------------------------------------------------------------------
     speechbrain.utils.config.remove_comments (author: Mirco Ravanelli)

     Description: This function removes comments from a line

     Input (call):    - line (type: str, mandatory):
                           it is the input line

     Output (call):  line (type: str):
                       it is the output line (without comments)


     Example:  from lib.utils.config import remove_comments

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
