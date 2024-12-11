"""Package providing simple wrappers for flair models."""

try:
    import flair  # noqa
except ImportError as e:
    raise ImportError(
        f"Failed to import flair: {e}\n"
        f"Please install flair e.g. using `pip install flair`.\n"
        f"For more details, see https://github.com/flairNLP/flair"
    ) from e

from .embeddings import *  # noqa
from .sequencetagger import *  # noqa
