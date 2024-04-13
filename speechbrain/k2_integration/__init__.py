"""
Package providing `k2-fsa <https://github.com/k2-fsa/k2>`_ integration.

Intended loading manner:

    >>> import speechbrain.k2_integration as sbk2
    >>> # Then use: sbk2.graph_compiler.CtcGraphCompiler for example

"""

try:
    import k2  # noqa
except ImportError as e:
    MSG = "Please install k2 to use k2\n"
    MSG += "Checkout: https://k2-fsa.github.io/k2/installation/from_wheels.html"
    raise ImportError(MSG) from e

from speechbrain.utils.importutils import lazy_export_all

lazy_export_all(__file__, __name__, export_subpackages=True)
