"""
speechbrain k2 integration
Intended loading manner:

    >>> import speechbrain.k2_integration as sbk2
    >>> # Then use: sbk2.graph_compiler.CharCtcTrainingGraphCompiler for example

"""

try:
    import k2
except ImportError:
    MSG = "Please install k2 to use k2 training \n"
    MSG += "E.G. run: pip install k2\n"
    raise ImportError(MSG)

from . import utils
from . import graph_compiler
from . import lattice_decode
from . import lexicon
from . import losses
from . import prepare_lang
