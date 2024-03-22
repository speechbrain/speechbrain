"""Wrappers for Flair embedding classes

Authors
* Sylvain de Langen 2024
"""

from flair.data import Sentence
from flair.embeddings import Embeddings
from typing import List, Union
import torch

from speechbrain.utils.fetching import fetch


class FlairEmbeddings:
    """
    Simple wrapper for generic Flair embeddings.

    Arguments
    ---------
    embeddings : Embeddings
        The Flair embeddings object. If you do not have one initialized, use
        :meth:`~FlairEmbeddings.from_hf` instead.
    """

    def __init__(self, embeddings: Embeddings) -> None:
        self.embeddings = embeddings

    @staticmethod
    def from_hf(
        embeddings_class,
        source,
        save_path="./model_checkpoints",
        filename="model.bin",
    ) -> "FlairEmbeddings":
        target = save_path + "/flair-emb--" + source.replace("/", "--") + "/"
        local_path = fetch(filename, source, savedir=target)
        return FlairEmbeddings(embeddings_class(local_path))

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
            tokenized for this specific sequence tagger.
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
        sample_emb = sentences[0][0].embedding
        pad_tensor = pad_tensor.to(sample_emb.device)
        pad_tensor = pad_tensor.broadcast_to(sample_emb.shape).unsqueeze(0)

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
