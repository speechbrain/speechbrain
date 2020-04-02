"""
-------------------------------------------------------------------------------
 speechbrain.utils

 Description: This library contains support functions that implement useful
              functionalities.
-------------------------------------------------------------------------------
"""

# TODO: get rid of the imports here, replace most with explicit import
# (e.g. import speechbrain.utils.config)
# some core utilities we may still keep them in the speechbrain.utils namespace
from speechbrain.utils.config import (
    read_config,
    write_config,
    conf_to_text,
    process_cmd_string,
    replace_global_variable,
    create_exec_config,
)
from speechbrain.utils.data_utils import (
    get_all_files,
    split_list,
    recursive_items,
)
from speechbrain.utils.superpowers import import_class, run_shell
