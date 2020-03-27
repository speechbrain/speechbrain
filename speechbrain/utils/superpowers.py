"""
 -----------------------------------------------------------------------------
 superpowers.py

 Description: This library contains functions for importing python classes and
              for running shell commands. Remember, great power comes great
              responsibility.
 -----------------------------------------------------------------------------
"""

import logging
import importlib
import subprocess
logger = logging.getLogger(__name__)


def import_class(library):
    """
     -------------------------------------------------------------------------
     utils.import_class (author: Mirco Ravanelli)

     Description: This function is used to import classes from a string.

     Input (call):    - library (type: str, mandatory):
                           it is the string containing the library to load

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

        logger.error(err_msg, exc_info=True)

    # Split library and class
    module_name = ".".join(library.split(".")[:-1])
    class_name = library.split(".")[-1]

    # Loading the module
    try:
        module_object = importlib.import_module(module_name)
    except Exception:
        err_msg = 'cannot import module %s".' % (module_name)
        logger.error(err_msg, exc_info=True)

    # Loading the class
    try:
        target_class = getattr(module_object, class_name)

    except Exception:

        err_msg = 'cannot import the class %s from the module %s".' % (
            class_name,
            module_name,
        )

        logger.error(err_msg, exc_info=True)

    return target_class


def run_shell(cmd):
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

    # Adding information in the logger
    msg = output.decode("utf-8") + "\n" + err.decode("utf-8")
    logger.debug(msg)

    return output, err, p.returncode
