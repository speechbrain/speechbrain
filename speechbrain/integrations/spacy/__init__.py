"""Package providing simple wrappers for spaCy models."""

try:
    import spacy  # noqa
except ImportError as e:
    raise ImportError(
        f"Failed to import spaCy: {e}\n"
        f"Please install spaCy e.g. using `pip install spacy`.\n"
        f"For more details, see https://github.com/explosion/spaCy"
    ) from e

from .nlp import *  # noqa
