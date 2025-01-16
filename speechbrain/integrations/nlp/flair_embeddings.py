"""Wrappers for Flair embedding classes

Authors
* Sylvain de Langen 2024
"""

from typing import List, Union

import torch

try:
    import flair
    from flair.data import Sentence
    from flair.embeddings import Embeddings
except ImportError as e:
    raise ImportError(
        f"Failed to import flair: {e}\n"
        f"Please install flair e.g. using `pip install flair`.\n"
        f"For more details, see https://github.com/flairNLP/flair"
    ) from e


class FlairEmbeddings:
    """
    Simple wrapper for generic Flair embeddings.

    Arguments
    ---------
    embeddings : Embeddings
        The Flair embeddings object. If you do not have one initialized, use
        :meth:`~FlairEmbeddings.from_hf` instead.

    Example
    -------
    >>> from speechbrain.utils.metric_stats import EmbeddingErrorRateSimilarity
    >>> from speechbrain.utils.metric_stats import WeightedErrorRateStats
    >>> from speechbrain.utils.metric_stats import ErrorRateStats
    >>> ember = FlairEmbeddings.from_hf(
    ...     embeddings_class=flair.embeddings.TransformerWordEmbeddings,
    ...     source="google-bert/bert-base-uncased",
    ... )
    >>> ember_metric = EmbeddingErrorRateSimilarity(
    ...     embedding_function=lambda x: FlairEmbeddings.embed_word(ember, x),
    ...     low_similarity_weight=1.0,
    ...     high_similarity_weight=0.1,
    ...     threshold=0.4,
    ... )
    >>> weighted_wer = WeightedErrorRateStats(
    ...     base_stats=ErrorRateStats(),
    ...     cost_function=ember_metric,
    ...     weight_name="ember",
    ... )
    >>> weighted_wer.base_stats.append(["id"], ["hi friend"], ["hi buddy"])
    >>> weighted_wer.summarize()
    {'ember_wer': 16.6..., 'ember_insertions': 1.0, 'ember_substitutions': 0.5, 'ember_deletions': 0.0, 'ember_num_edits': 1.5}
    """

    def __init__(self, embeddings: Embeddings) -> None:
        self.embeddings = embeddings

    @staticmethod
    def from_hf(embeddings_class, source, *args, **kwargs) -> "FlairEmbeddings":
        """Fetches and load flair embeddings.

        Arguments
        ---------
        embeddings_class : class
            The class to use to initialize the model, e.g. `FastTextEmbeddings`.
        source : str
            The location of the model (a directory or HF repo, for instance).
        *args
            Extra positional arguments to pass to the flair class constructor
        **kwargs
            Extra keyword arguments to pass to the flair class constructor

        Returns
        -------
        FlairEmbeddings
        """

        return FlairEmbeddings(embeddings_class(source, *args, **kwargs))

    def __call__(
        self,
        inputs: Union[List[str], List[List[str]]],
        pad_tensor: torch.Tensor = torch.zeros((1,)),
    ) -> torch.Tensor:
        """Extract embeddings for a batch of sentences.

        Arguments
        ---------
        inputs : list of sentences (str or list of tokens)
            Sentences to embed, in the form of batches of lists of tokens
            (list of str) or a str.
            In the case of token lists, tokens do *not* need to be already
            tokenized for this specific sequence tagger. However, a token may be
            considered as a single word.
            Similarly, out-of-vocabulary handling depends on the underlying
            embedding class.
        pad_tensor : torch.Tensor, optional
            What embedding tensor (of shape `[]`, living on the same device as
            the embeddings to insert as padding.

        Returns
        -------
        torch.Tensor
            Batch of shape `[len(inputs), max_len, embed_size]`
        """

        if isinstance(inputs, str):
            raise ValueError("Expected a list of sentences, not a single str")

        sentences = [Sentence(sentence) for sentence in inputs]
        self.embeddings.embed(sentences)

        # migrate pad to device & broadcast if it's just a scalar
        pad_tensor = pad_tensor.to(flair.device)
        pad_tensor = pad_tensor.broadcast_to(
            self.embeddings.embedding_length
        ).unsqueeze(0)

        sentence_embs = [
            torch.stack([token.embedding for token in sentence])
            for sentence in sentences
        ]
        longest_emb = max(emb.size(0) for emb in sentence_embs)
        sentence_embs = [
            torch.cat(
                [emb, pad_tensor.repeat(longest_emb - emb.size(0), 1)], dim=0
            )
            for emb in sentence_embs
        ]
        return torch.stack(sentence_embs)

    def embed_word(self, word: str) -> torch.Tensor:
        """Embeds a single word.

        Arguments
        ---------
        word : str
            Word to embed. Out-of-vocabulary handling depends on the underlying
            embedding class.

        Returns
        -------
        torch.Tensor
            Embedding for a single word, of shape `[embed_size]`
        """

        return self([word])[0, 0, :]
