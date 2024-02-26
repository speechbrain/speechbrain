try:
    import flair
except ImportError as e:
    raise ImportError(
        f"Failed to import flair: {e}\n"
        f"Please install flair e.g. using `pip install flair`.\n"
        f"For more details, see https://github.com/flairNLP/flair"
    )

from .sequencetagger import FlairSequenceTagger
