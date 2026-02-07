"""
Package providing `Numba <https://numba.pydata.org/>`_ integration.

This package contains modules that depend on the optional ``numba`` dependency,
such as the CUDA-accelerated Transducer loss.
"""

try:
    import numba  # noqa: F401
except ImportError as e:
    MSG = "Please install numba to use this module.\n"
    MSG += "pip install numba\n"
    MSG += "For more information, visit: https://numba.pydata.org/"
    raise ImportError(MSG) from e

from speechbrain.utils.importutils import lazy_export_all

lazy_export_all(__file__, __name__, export_subpackages=True)
