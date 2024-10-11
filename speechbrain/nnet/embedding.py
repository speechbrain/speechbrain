"""Library implementing embedding.

Authors
 * Abdelwahab Heba 2020
"""

import math

import torch
import torch.nn as nn

from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class Embedding(nn.Module):
    """Computes an embedding x = wx.

    Arguments
    ---------
    num_embeddings : int
        Size of the dictionary of embeddings.
    embedding_dim : int
        It is the dim of embedding (i.e, the dimensionality of the output).
    consider_as_one_hot : bool
        Create non-trainable one-hot vector.
    blank_id : int
        If consider_as_one_hot == True: consider the embedding as one_hot
        and use blank_index as zero one_hot vector.

    Example
    -------
    >>> from speechbrain.nnet.embedding import Embedding
    >>> import torch
    >>> emb = Embedding(
    ...     num_embeddings=40,
    ...     embedding_dim=39,
    ...     consider_as_one_hot=True,
    ...     blank_id=39
    ... )
    >>> inputs = torch.Tensor([10,5,2,0,39]).long()
    >>> output = emb(inputs)
    >>> output.shape
    torch.Size([5, 39])
    >>> output
    tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0.]])
    >>> emb = Embedding(num_embeddings=5, embedding_dim=3, consider_as_one_hot=False)
    >>> e = emb(torch.LongTensor([[0, 1, 2], [3, 4, 2]]))
    >>> e.shape
    torch.Size([2, 3, 3])
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim=128,
        consider_as_one_hot=False,
        blank_id=0,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.consider_as_one_hot = consider_as_one_hot
        if self.consider_as_one_hot:
            self.embedding_dim = self.num_embeddings - 1
        else:
            self.embedding_dim = embedding_dim
        self.blank_id = blank_id

        if self.consider_as_one_hot:
            # deal with blank_id, the output should be embedding_dim-1 as we consider blank output as zeros one_hot vect
            # padding_idx fix the idx row to zeros
            self.Embedding = nn.Embedding(
                self.num_embeddings,
                self.embedding_dim,
                padding_idx=self.blank_id,
            )
            one_hot = torch.eye(self.embedding_dim)
            if self.blank_id + 1 != self.num_embeddings:
                self.Embedding.weight.data[self.blank_id + 1 :] = one_hot[
                    self.blank_id :
                ]
            if self.blank_id != 0:
                self.Embedding.weight.data[: self.blank_id] = one_hot[
                    : self.blank_id
                ]
            self.Embedding.weight.requires_grad = False
        else:
            self.Embedding = nn.Embedding(
                self.num_embeddings, self.embedding_dim
            )

    def forward(self, x):
        """Returns the embedding of input tensor.

        Arguments
        ---------
        x : torch.Tensor
           Input to embed.

        Returns
        -------
        The embedded outputs.
        """
        # pytorch embedding layer only accept long dtype
        return self.Embedding(x.long())


class MultiEmbedding(nn.Module):
    """A wrapper module with multiple embedding 'heads' - for
    cases with multiple tokens per sequence

    Arguments
    ---------
    num_embeddings : int
        Size of the dictionary of embeddings.
    embedding_dim : int
        It is the dim of embedding (i.e, the dimensionality of the output).
    num_heads : int
        The number of embedding "heads" (i.e. tokens per step)
    normalized : bool, optional
        Whether to normalize the embeddings (for transformers)
    d_model : int, optional
        The model dimension (ignored if not normalized)
    norm_factor : float, optional
        The normalization factor (multiplier)
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        num_heads,
        normalized=False,
        d_model=512,
        norm_factor=None,
    ):
        super().__init__()
        self.emb = torch.nn.ModuleList(
            torch.nn.Embedding(num_embeddings, embedding_dim)
            for _ in range(num_heads)
        )
        self.normalized = normalized
        if norm_factor is None:
            norm_factor = math.sqrt(d_model) if normalized else 1.0
        self.norm_factor = norm_factor

    def forward(self, x):
        """Computes the forward pass

        Arguments
        ---------
        x : torch.Tensor
            A tensor of indexes

        Returns
        -------
        emb : torch.Tensor
            An embedding tensor"""
        emb = (
            torch.cat(
                [
                    emb(x[..., idx].int()).unsqueeze(-2)
                    for idx, emb in enumerate(self.emb)
                ],
                dim=-2,
            )
            * self.norm_factor
        )
        return emb

    def initialize(self, emb):
        """Initializes the embeddings with the specified embedding tensor

        Arguments
        ---------
        emb : torch.Tensor
            A (Layer x Embeddings x Embedding Dim) tensor"""
        with torch.no_grad():
            for head, head_emb in zip(
                self.emb,
                emb,
            ):
                head.weight.copy_(head_emb)

    def all_weights(self):
        """Returns all embedding weights as a single tensor"""
        return torch.stack([emb.weight for emb in self.emb])
