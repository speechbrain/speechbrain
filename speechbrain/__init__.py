""" Comprehensive speech processing toolkit
"""

import os

from . import alignment  # noqa
from . import dataio  # noqa
from . import decoders  # noqa
from . import lm  # noqa
from . import lobes  # noqa
from . import nnet  # noqa
from . import processing  # noqa
from . import tokenizers  # noqa
from . import utils  # noqa
from .core import Brain, Stage, create_experiment_directory, parse_arguments

with open(os.path.join(os.path.dirname(__file__), "version.txt")) as f:
    version = f.read().strip()

__all__ = [
    "Stage",
    "Brain",
    "create_experiment_directory",
    "parse_arguments",
]

__version__ = version
