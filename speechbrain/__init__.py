"""Comprehensive speech processing toolkit"""

import os

from .core import Brain, Stage, create_experiment_directory
from .utils.importutils import deprecated_redirect, lazy_export_all
from .utils.run_opts import RunOptions

with open(
    os.path.join(os.path.dirname(__file__), "version.txt"), encoding="utf-8"
) as f:
    version = f.read().strip()
parse_arguments = RunOptions.from_command_line_args()
__all__ = [
    "Stage",
    "Brain",
    "create_experiment_directory",
    "parse_arguments",
]

__version__ = version


def make_deprecated_redirections():
    sb1_0_redirect_str = (
        "This is a change from SpeechBrain 1.0. "
        "See: https://github.com/speechbrain/speechbrain/releases/tag/v1.0.0"
    )

    deprecated_redirect(
        "speechbrain.pretrained",
        "speechbrain.inference",
        extra_reason=sb1_0_redirect_str,
        also_lazy_export=True,
    )


make_deprecated_redirections()

lazy_export_all(__file__, __name__, export_subpackages=True)
