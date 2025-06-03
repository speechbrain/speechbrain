"""Models and tooling for natural language processing using spaCy

Authors
* Sylvain de Langen 2024
"""

from typing import Iterable, Iterator, List, Union

import spacy
import spacy.tokens


def _as_sentence(sentence: Union[str, List[str]]):
    """Ensures that a sentence is a `str` rather than a list of `str` tokens to
    be passed to spaCy pipelines correctly.

    Arguments
    ---------
    sentence: str or list of str
        Sentence to return or list of tokens.

    Returns
    -------
    str
        The sentence, returned from the `sentence` argument as-is or joined with
        spaces from a list of tokens."""

    if isinstance(sentence, str):
        return sentence

    return " ".join(sentence)


def _extract_lemmas(docs: Iterable[spacy.tokens.Doc]):
    """Returns a batch of list of lemmas from a list of Doc (as returned by the
    pipeline).

    Arguments
    ---------
    docs: iterable of Doc
        Documents, typically as returned by `nlp.pipe`.

    Returns
    -------
    list of list of str
        For each sentence, the sequence of extracted lemmas as `str`s."""
    return [[tok.lemma_ for tok in doc] for doc in docs]


class SpacyPipeline:
    """Wraps a `spaCy pipeline <https://spacy.io/usage/processing-pipelines>`_
    with methods that makes it easier to deal with SB's typical sentence format,
    and adds some convenience functions if you only care about a specific task.

    Arguments
    ---------
    nlp : spacy.language.Language
        spaCy text processing pipeline to use."""

    def __init__(self, nlp: spacy.language.Language):
        self.nlp = nlp

    @staticmethod
    def from_name(name, *args, **kwargs):
        """Create a pipeline by loading a model using `spacy.load`.
        Unlike other toolkits, you must explicitly download the model if you
        want to use a remote model (e.g. `spacy download fr_core_news_md`)
        rather than just specifying a HF hub name.

        .. note::
            If you only need a subset of modules enabled in the pipeline,
            e.g. for lemmatization, consider
            `excluding <https://spacy.io/usage/processing-pipelines#disabling>_`
            using the `exclude=[...]` argument.

        Arguments
        ---------
        name: str | Path
            Package name or model path.
        *args
            Extra positional arguments passed to `spacy.load`.
        **kwargs
            Extra keyword arguments passed to `spacy.load`.

        Returns
        -------
        New SpacyPipeline
        """

        return SpacyPipeline(spacy.load(name, *args, **kwargs))

    def __call__(
        self, inputs: Union[List[str], List[List[str]]]
    ) -> Iterator[spacy.tokens.Doc]:
        """Processes a batch of sentences into an iterator of spaCy documents.

        Arguments
        ---------
        inputs: list of sentences (str or list of tokens)
            Sentences to process, in the form of batches of lists of tokens
            (list of str) or a str.
            In the case of token lists, tokens do *not* need to be already
            tokenized for this specific sequence tagger, and they will be joined
            with spaces instead.

        Returns
        -------
        iterator of spacy.tokens.Doc
            Iterator of documents for the passed sentences."""

        return self.nlp.pipe(map(_as_sentence, inputs))

    def lemmatize(
        self, inputs: Union[List[str], List[List[str]]]
    ) -> List[List[str]]:
        """Lemmatize a batch of sentences by processing the input sentences,
        discarding other irrelevant outputs.

        Arguments
        ---------
        inputs: list of sentences (str or list of tokens)
            Sentences to lemmatize, in the form of batches of lists of tokens
            (list of str) or a str.
            In the case of token lists, tokens do *not* need to be already
            tokenized for this specific sequence tagger, and they will be joined
            with spaces instead.

        Returns
        -------
        list of list of str
            For each sentence, the sequence of extracted lemmas as `str`s."""

        return _extract_lemmas(self(inputs))
