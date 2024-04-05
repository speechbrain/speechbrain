""" Comprehensive speech processing toolkit
"""

import os
from .core import Stage, Brain, create_experiment_directory, parse_arguments
from . import alignment  # noqa
from . import dataio  # noqa
from . import decoders  # noqa
from . import lobes  # noqa
from . import lm  # noqa
from . import nnet  # noqa
from . import processing  # noqa
from . import tokenizers  # noqa
from . import utils  # noqa
from .utils.importutils import deprecated_redirect

with open(os.path.join(os.path.dirname(__file__), "version.txt")) as f:
    version = f.read().strip()

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
    )


make_deprecated_redirections()
