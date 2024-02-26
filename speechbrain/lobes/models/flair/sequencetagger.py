"""Models and tooling for part-of-speech tagging

Authors
* Sylvain de Langen 2024
"""

from flair.data import Sentence
from flair.models import SequenceTagger

from typing import List, Union


class FlairSequenceTagger:
    """
    Sequence tagger using the flair toolkit, e.g. for part-of-speech (POS)
    extraction.
    """

    def __init__(self, model_path):
        self.model = SequenceTagger.load(model_path)

    def __call__(self, inputs: Union[List[str], List[List[str]]]) -> List[List[str]]:
        """Tag a batch of list of tokens.
        
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

        sentences = [
            Sentence(sentence)
            for sentence in inputs
        ]

        self.model.predict(sentences)

        return [
            [label.value for label in sentence.get_labels()]
            for sentence in sentences
        ]
