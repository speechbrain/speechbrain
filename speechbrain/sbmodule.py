"""
-----------------------------------------------------------------------------
 speechbrain/sbmodule.py

 Description: This class provides the base class for speechbrain classes.
 -----------------------------------------------------------------------------
"""

# Importing libraries
import torch.nn as nn

from speechbrain.utils.input_validation import (check_opts,
                                                check_inputs)


class SBModule(nn.Module):
    """
    -------------------------------------------------------------------------
    speechbrain.sbmodule.SBModule (author: Peter Plantinga)

    Description: This class defines input type checking and other functions
                 common to all speechbrain classes. All new speechbrain
                 classes should inherit from this class, and pass expected
                 inputs to init, so inputs can be checked.

    Input (init): - config (type, dict, mandatory):
                      If a config dict is passed, all options are checked
                      against the list of expected types given by
                      the `expected_options` variable.

                  - expected_options(type, dict, mandatory):
                      This dict defines the types expected in the config
                      dict. Available types are defined in `check_opts`.

                  - expected_inputs(type, list, mandatory):
                      This dict defines the types of the inputs when
                      the class is called as a function (i.e. passed
                      to the `forward` or `call` method). This input
                      will be checked on the first call, but not
                      thereafter, for efficiency. If a tensor is
                      passed, the shape that it is expected to
                      have must also be passed.

                  - funct_name (type, str, optional, default: None):
                      it is a string containing the name of the parent
                      function that has called this method.

                  - global_config (type, dict, optional, default: None):
                      it a dictionary containing the global variables of the
                      parent config file.

                  - functions (type, dict, optional, default: None):
                      dictionary for storing user-defined functions. Keys are
                      the function names, values are their corresponding
                      objects.

                  - logger (type, logger, optional, default: None):
                      it the logger used to write debug and error messages.
                      If logger=None and root_cfg=True, the file is created
                      from scratch.
    -------------------------------------------------------------------------
    """

    def __init__(
        self,
        config,
        expected_options,
        expected_inputs,
        funct_name=None,
        global_config=None,
        functions=None,
        logger=None,
        first_input=None,
    ):
        # Initialize the torch module code
        super().__init__()

        # Store vars
        self.funct_name = funct_name
        self.global_config = global_config
        self.functions = functions
        self.logger = logger

        # Check, cast, and expand the options
        self.conf = check_opts(self, expected_options, config, logger)

        # This records if the first input has been checked
        self.first_input = True
        self.expected_inputs = expected_inputs

        # Register a hook to check the first input passed to forward
        def first_pass_check_inputs(self, input, output):
            if self.first_input:
                self.first_input = False
                check_inputs(
                    self.conf,
                    self.expected_inputs,
                    input[0],
                    self.logger,
                )

        self.register_forward_hook(first_pass_check_inputs)

