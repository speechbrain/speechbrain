"""Models and tooling for sequence tagging using Flair

Authors
* Sylvain de Langen 2024
"""

from flair.data import Sentence
from flair.models import SequenceTagger

from typing import List, Union

from speechbrain.utils.fetching import fetch


class FlairSequenceTagger:
    """
    Sequence tagger using the flair toolkit, e.g. for part-of-speech (POS)
    extraction.

    Arguments
    ---------
    model : SequenceTagger
        The Flair sequence tagger model. If you do not have one initialized, use
        :meth:`~FlairSequenceTagger.from_hf` instead.
    """

    def __init__(self, model: SequenceTagger):
        self.model = model

    def from_hf(self, source, save_path="", filename="pytorch_model.bin") -> "FlairSequenceTagger":
        """Fetches and load a flair PyTorch model according to the
        :func:`speechbrain.utils.fetching.fetch` semantics.
        
        Arguments
        ---------
        source : str
            The location of the model (a directory or HF repo, for instance).
        save_path : str, optional
            The saving location for the model (i.e. the download or symlink
            location).
        filename : str, optional
            The filename of the model. The default is the usual filename for
            this kind of model.
        """

        local_path = fetch(filename, source, savedir=save_path)
        return FlairSequenceTagger(SequenceTagger.load(local_path))

    def __call__(
        self, inputs: Union[List[str], List[List[str]]]
    ) -> List[List[str]]:
        """Tag a batch of sentences.

        Arguments
        ---------
        inputs: list of sentences (str or list of tokens)
            Sentences to tag, in the form of batches of lists of tokens
            (list of str) or a str.
            In the case of token lists, tokens do *not* need to be already
            tokenized for this specific sequence tagger.

        Returns
        -------
        list of list of str
            For each sentence, the sequence of extracted tags as `str`s."""

        if isinstance(inputs, str):
            raise ValueError("Expected a list of sentences, not a single str")

        sentences = [Sentence(sentence) for sentence in inputs]

        self.model.predict(sentences)

        return [
            [label.value for label in sentence.get_labels()]
            for sentence in sentences
        ]
