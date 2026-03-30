"""Comprehensive speech processing toolkit"""

import os

# For redirect of HF transformers
import speechbrain.lobes.models  # noqa: F401

from .core import Brain, Stage, create_experiment_directory
from .utils.importutils import deprecated_redirect, lazy_export_all
from .utils.run_opts import RunOptions

with open(
    os.path.join(os.path.dirname(__file__), "version.txt"), encoding="utf-8"
) as f:
    version = f.read().strip()

# Create an alias to the refactored function
parse_arguments = RunOptions.from_command_line_args

__all__ = [
    "Stage",
    "Brain",
    "create_experiment_directory",
    "parse_arguments",
]

__version__ = version


deprecations = {
    "speechbrain.k2_integration": "speechbrain.integrations.k2_fsa",
    "speechbrain.wordemb": "speechbrain.integrations.huggingface.wordemb",
    "speechbrain.lobes.models.huggingface_transformers": "speechbrain.integrations.huggingface",
    "speechbrain.lobes.models.spacy": "speechbrain.integrations.nlp",
    "speechbrain.lobes.models.flair": "speechbrain.integrations.nlp",
}


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

    for old_path, new_path in deprecations.items():
        deprecated_redirect(old_path, new_path, also_lazy_export=True)

    # speechbrain.nnet.loss is not yet loaded at this point, so we cannot use
    # also_lazy_export (it would try to access sys.modules['speechbrain.nnet.loss']).
    # The sys.modules redirect alone is sufficient for import compatibility.
    deprecated_redirect(
        "speechbrain.nnet.loss.transducer_loss",
        "speechbrain.integrations.numba.transducer_loss",
        extra_reason=(
            "This module depends on the optional 'numba' package. "
            "If you encounter an ImportError here, please install numba, "
            "for example with: pip install numba"
        ),
    )


make_deprecated_redirections()

lazy_export_all(__file__, __name__, export_subpackages=True)
