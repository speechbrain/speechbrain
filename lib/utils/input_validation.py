"""
 -----------------------------------------------------------------------------
 input_validation.py

 Description: This library contains functions for validation of input 
              and hyperparameters.
 -----------------------------------------------------------------------------
"""

import os
from pydoc import locate
from lib.utils.logger import logger_write

def str_to_bool(s):
    """
     -------------------------------------------------------------------------
     utils.str_to_bool (author: Mirco Ravanelli)

     Description: This function converts a string to a boolean.

     Input (call):    - s (type: str, mandatory):
                           string to conver to a boolean.

     Output (call):  None


     Example:   from lib.utils.input_validation import str_to_bool

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


     Example:   from lib.utils.input_validation import set_default_values

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


     Example:   from lib.utils.input_validation import check_and_cast_type

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


     Example:   from lib.utils.input_validation import check_expected_options

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


     Example:   from lib.utils.input_validation import check_opts

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
                from lib.utils.input_validation import check_inputs

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
