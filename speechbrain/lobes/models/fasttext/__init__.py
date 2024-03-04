"""Package providing simple wrappers for fastText models."""

try:
    import fasttext
except ImportError as e:
    raise ImportError(
        f"Failed to import fastText: {e}\n"
        f"Please install fastText e.g. using `pip install fasttext`. You may need to install the `wheel` package first.\n"
        f"You may also try `pip install fasttext-wheel` if the above fails.\n"
        f"For more details, see https://pypi.org/project/fasttext/"
    )

from .embeddings import *
