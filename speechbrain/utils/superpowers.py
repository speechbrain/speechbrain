"""
 -----------------------------------------------------------------------------
 superpowers.py

 Description: This library contains functions for importing python classes and
              for running shell commands. Remember, great power comes great
              responsibility.
 -----------------------------------------------------------------------------
"""

import importlib
import subprocess
from speechbrain.utils.logger import logger_write


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


     Example:  from speechbrain.utils.superpowers import import_class

               # importing class
               loop=import_class('speechbrain.core.execute_computations')

     -------------------------------------------------------------------------
     """

    # Check the library string
    if len(library.split(".")) == 1:

        err_msg = (
            "the class to import must be in the following format: "
            'library.class (e.g., data_io.load_data). Got %s".' % (library)
        )

        logger_write(err_msg, logfile=logfile)

    # Split library and class
    module_name = ".".join(library.split(".")[:-1])
    class_name = library.split(".")[-1]

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


def run_shell(cmd, logger=None):
    """
     -------------------------------------------------------------------------
     utils.run_shell (author: Mirco Ravanelli)

     Description: This function can be used to run command from the bash shell

     Input (call):    - cmd (type: str, mandatory):
                       it a string containing the command.

     Output (call):   - output(type: str):
                       it is a string containing the standard output


     Example:   from speechbrain.utils.superpowers import run_shell

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

    return output, err
