"""
 -----------------------------------------------------------------------------
 logger.py

 Description: This library contains functions for managing the logger.
 -----------------------------------------------------------------------------
"""

import logging
import sys
import traceback

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


     Example:   from speechbrain.utils.logger import setup_logger
                from speechbrain.utils.logger import logger_write

                # Initialization of the logger
                logfile=setup_logger('logger','exp/log.log')

                # Writing an error message
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


     Example:   from from speechbrain.utils.logger import setup_logger
                from from speechbrain.utils.logger import logger_write

                # Initialization of the logger
                logfile=setup_logger('logger','exp/log.log')

                # Writing an error message
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
