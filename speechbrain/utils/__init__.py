"""This package contains support functions.
"""

# TODO: get rid of the imports here, replace most with explicit import
# (e.g. import speechbrain.utils.config)
# some core utilities we may still keep them in the speechbrain.utils namespace
# TODO: Specify in __all__ what the public API of speechbrain.utils is
from speechbrain.utils.data_utils import (  # noqa: F401
    get_all_files,
    split_list,
    recursive_items,
)
from speechbrain.utils.superpowers import run_shell  # noqa: F401
