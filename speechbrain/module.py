"""
-----------------------------------------------------------------------------
speechbrain/module.py

Description: This class provides the base class for speechbrain classes.
-----------------------------------------------------------------------------
"""

# Importing libraries
import torch.nn as nn
from speechbrain.utils.input_validation import check_inputs


class SpeechBrainModule(nn.Module):
    """
    -------------------------------------------------------------------------
    speechbrain.module.SpeechBrainModule (author: Peter Plantinga)

    Description: This class defines input type checking and other functions
                 common to all speechbrain classes. All new speechbrain
                 classes should inherit from this class, and pass expected
                 inputs to init, so inputs can be checked.

    Input:
      - expected_inputs(type, list, mandatory):
          This dict defines the types of the inputs when
          the class is called as a function (i.e. passed
          to the `forward` or `call` method). This input
          will be checked on the first call, but not
          thereafter, for efficiency. If a tensor is
          passed, the shape that it is expected to
          have must also be passed.

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

      - first_input_hook (type, function, optional, default: None)
          A function to be run when `forward` is called for the
          first time. For example: defining the size of a
          neural network layer based on the input size.

    Example:

        >>> class example(SpeechBrainModule):
        ...     def __init__(param1, **kwargs):
        ...         super().__init__(expected_inputs=[], **kwargs)
        ...         self.param1 = param1
        ...     def forward():
        ...         return self.param1
        >>> instance = example(3)
        >>> instance()
        3

    -------------------------------------------------------------------------
    """

    def __init__(
        self,
        expected_inputs,
        first_input_hook=None,
        global_config=None,
        logger=None,
    ):
        # Initialize the torch module code
        super().__init__()

        # Store vars
        self.global_config = global_config
        self.logger = logger

        # Prepare for checking the first input
        self.expected_inputs = expected_inputs
        self.first_input_hook = first_input_hook
        self.hook = self.register_forward_pre_hook(first_pass_check_inputs)


def first_pass_check_inputs(self, input):
    """
    Description: Register a hook to check the first input passed to forward.
    (Author: Peter Plantinga)

    Input: - self: class with registered hook

           - input: the first input to get passed to forward
    """

    # Check the type and shape of the inputs
    check_inputs(self.expected_inputs, input, self.logger)

    # Run user-defined first input hook
    if self.first_input_hook is not None:
        self.first_input_hook(self, input)

    # Remove hook, since we only want to do this once
    self.hook.remove()
