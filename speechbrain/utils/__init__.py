"""This package contains support functions.
"""

# TODO: get rid of the imports here, replace most with explicit import
# (e.g. import speechbrain.utils.config)
# some core utilities we may still keep them in the speechbrain.utils namespace
from speechbrain.utils.data_utils import (
    get_all_files,
    split_list,
    recursive_items,
)
from speechbrain.utils.superpowers import import_class, run_shell
