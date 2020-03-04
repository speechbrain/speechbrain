"""
-------------------------------------------------------------------------------
 speechbrain.utils

 Description: This library contains support functions that implement useful
              functionalities.
-------------------------------------------------------------------------------
"""

#TODO: get rid of the imports here, replace most with explicit import
# (e.g. import speechbrain.utils.config)
# for the core utilities we may still keep them in the speechbrain.utils namespace
from speechbrain.utils.config import (read_config, write_config, conf_to_text, process_cmd_string, 
        replace_global_variable, create_exec_config)
from speechbrain.utils.data_utils import (get_all_files, split_list, csv_to_dict, recursive_items)
from speechbrain.utils.input_validation import (str_to_bool, set_default_values, check_and_cast_type,
        check_expected_options, check_opts, check_inputs)
from speechbrain.utils.logger import setup_logger, logger_write
from speechbrain.utils.superpowers import import_class, run_shell
