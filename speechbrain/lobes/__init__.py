""" Package defining common blocks (DNN models, processing ...)

This subpackage gathers higher level blocks, or "lobes".
The classes here may leverage the extended YAML syntax.
"""

from speechbrain.utils.importutils import lazy_export_all

lazy_export_all(__file__, __name__, export_subpackages=True)
