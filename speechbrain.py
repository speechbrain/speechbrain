
"""
-------------------------------------------------------------------------------
 speechbrain.py (author: Mirco Ravanelli)

 Description: This is the main script of SpeechBrain.
              It can be used to run experiments.

 Input:       config_file (type: file)

 Output:      None

 Example:     python speechbrain.py config_file
              where config_file is a configuration file (formatted as
              described in the documentation).
-------------------------------------------------------------------------------
"""

# Importing libraries
import sys
from lib.utils import create_exec_config
from lib.core import execute_computations


# Definition of the main function
if __name__ == "__main__":

    # Reading the arguments from the command line
    cfg_file = sys.argv[1]
    cmd_arg = sys.argv[2:]

    # Creating config dict for executing computations
    exec_config = create_exec_config(cfg_file, cmd_arg)
    
    # Initializing the execute computation class
    computations = execute_computations(exec_config)
    
    # Executing the computations specified in the config file
    computations([])
