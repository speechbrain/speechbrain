"""Library implementing embedding.

Author
    Abdelwahab HEBA 2020
"""

import torch
import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


class Embedding(nn.Module):
    """Computes an embedding x = wx.

    Arguments
    ---------
    num_embeddings : int
        size of the dictionary of embeddings
    embeddings_dim : int
        it is the dim of embedding (i.e, the dimensionality of the output)
    consider_as_one_hot: bool - create non-trainable one-hot vect
    blank_id: int
        if consider_as_one_hot == True: consider the embedding as one_hot and use blank_index as zero one_hot vector

    Example
    -------
    >>> from speechbrain.nnet.embedding import Embedding
    >>> emb = Embedding(num_embeddings=40, embeddings_dim=39, consider_as_one_hot=True, blank_id=39)
    >>> inputs = torch.Tensor([10,5,2,0,39]).long()
    >>> output = emb(inputs, init_params=True)
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
     """

    def __init__(
        self,
        num_embeddings,
        embeddings_dim=128,
        consider_as_one_hot=False,
        blank_id=0,
    ):
        assert isinstance(num_embeddings, int), "num_embeddings must be integer"
        assert isinstance(embeddings_dim, int), "embeddings_dim must be integer"
        assert isinstance(
            consider_as_one_hot, bool
        ), "consider_as_one_hot must be boolean"
        assert isinstance(blank_id, int), "blank_id must be integer"

        super().__init__()
        self.num_embeddings = num_embeddings
        self.consider_as_one_hot = consider_as_one_hot
        if self.consider_as_one_hot:
            self.embeddings_dim = self.num_embeddings - 1
        else:
            self.embeddings_dim = embeddings_dim
        self.blank_id = blank_id

    def init_params(self, first_input):
        """

        Arguments
        ---------
        first_input : tensor
                      A first input used for initializing the parameters.
        """
        if self.consider_as_one_hot:
            # deal with blank_id, the output should be embeddings_dim-1 as we consider blank output as zeros one_hot vect
            # padding_idx fix the idx row to zeros
            self.Embedding = nn.Embedding(
                self.num_embeddings,
                self.embeddings_dim,
                padding_idx=self.blank_id,
            ).to(first_input.device)
            one_hot = torch.eye(self.embeddings_dim).to(first_input.device)
            if self.blank_id + 1 != self.num_embeddings:
                self.Embedding.weight.data[self.blank_id + 1 :] = one_hot[
                    self.blank_id + 1 :
                ]
            if self.blank_id != 0:
                self.Embedding.weight.data[: self.blank_id] = one_hot[
                    : self.blank_id
                ]
            self.Embedding.weight.requires_grad = False
        else:
            self.Embedding = nn.Embedding(
                self.num_embeddings, self.embeddings_dim
            ).to(first_input.device)

    def forward(self, x, init_params=False):
        """Returns the embedding of input tensor.

        Arguments
        ---------
        x: torch.Tensor
           input to embed.
        """
        if init_params:
            self.init_params(x)

        return self.Embedding(x)
