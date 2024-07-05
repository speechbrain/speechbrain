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

try:
    import speechtokenizer  # noqa
except ImportError:
    MSG = "Please install speechtokenizer.\n"
    MSG += "E.G. run: pip install speechtokenizer \n"
    raise ImportError(MSG)

from .dac import *  # noqa
from .discrete_ssl import *  # noqa
from .encodec import *  # noqa
from .speechtokenizer_interface import *  # noqa
