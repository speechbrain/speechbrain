"""
Package providing `k2-fsa <https://github.com/k2-fsa/k2>`_ integration.

Intended loading manner:

    >>> import speechbrain.k2_integration as sbk2
    >>> # Then use: sbk2.graph_compiler.CtcGraphCompiler for example

"""


__all__ = [
    "k2",
    "utils",
    "graph_compiler",
    "lattice_decoder",
    "lexicon",
    "losses",
    "prepare_lang",
]

try:
    import k2
except ImportError:
    MSG = "Please install k2 to use k2\n"
    MSG += "Checkout: https://k2-fsa.github.io/k2/installation/from_wheels.html"
    raise ImportError(MSG)

from . import utils
from . import graph_compiler
from . import lattice_decoder
from . import lexicon
from . import losses
from . import prepare_lang
