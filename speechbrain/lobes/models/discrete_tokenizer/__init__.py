"""High level processing blocks.

This subpackage gathers higher level blocks, or "lobes" for discrete tokenzier.
"""

# Transformers is required for this package.
try:
    import transformers  # noqa
except ImportError:
    MSG = "Please install transformers from HuggingFace.\n"
    MSG += "E.G. run: pip install transformers \n"
    MSG += "For more information, visit: https://huggingface.co/docs/transformers/installation"
    raise ImportError(MSG)

from .dac import *  # noqa
from .discrete_ssl import *  # noqa
from .encodec import *  # noqa

