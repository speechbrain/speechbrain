"""Wrapper for fastText embeddings.

Authors
* Sylvain de Langen 2024
"""

import fasttext
from typing import Optional
import torch
from speechbrain.utils.fetching import fetch


class FasttextModel:
    """
    fastText is a library for learning of word representations and sentence
    classification. This wrapper lets you easily load and define a fastText
    model, optionally loading it from the HuggingFace hub.

    Arguments
    ---------
    model
        Fasttext model object, e.g. loaded through
        `fasttext.load_model("model.bin")`.
    """

    def __init__(self, model):
        self.model = model

    @staticmethod
    def load_hf(source, filename, *args, **kwargs) -> "FasttextModel":
        """
        Fetches and loads the fastText model from the HuggingFace hub.

        source : str
            A HuggingFace repository identifier or a path.
        filename : str
            The name of the file in the repo.
        *args
            Extra positional arguments to pass to :func:`speechbrain.utils.fetching.fetch`
        **kwargs
            Extra keyword arguments to pass to :func:`speechbrain.utils.fetching.fetch`
        """

        model_path = fetch(
            filename=filename,
            source=source,
            *args,
            **kwargs,
        )
        return FasttextModel(fasttext.load_model(str(model_path)))

    def get_word_vector(self, word: str) -> Optional[torch.Tensor]:
        """Returns the (single) word embedding associated with the string
        argument.

        Arguments
        ---------
        word : str
            The word to find the embedding of. Must be a single word.
            Potentially case-sensitive!
        """

        return torch.from_numpy(self.model[word])
