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
    embeddings_dim : int
        it is the dim of embedding (i.e, the dimensionality of the output)
    consider_as_one_hot: bool - create non-trainable one-hot vect
    blank_id: int
        if consider_as_one_hot == True: consider the embedding as one_hot and use blank_index as zero one_hot vector
    
    Example
    -------
    >>> emb = Embedding(embeddings_dim=39,consider_as_one_hot=True,blank_id=39)
    >>> inputs = torch.Tensor([10,5,2,0,39])
    >>> output = emb(inputs,init_params=True)
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
        embeddings_dim,
        consider_as_one_hot=True,
        blank_id = 0
    ):
        assert isinstance(embeddings_dim,int), "embeddings_dim must be integer"
        assert isinstance(consider_as_one_hot, bool), "consider_as_one_hot must be boolean"
        assert isinstance(blank_id, int), "blank_id must be integer"

        super().__init__()
        self.embeddings_dim = embeddings_dim
        self.consider_as_one_hot = consider_as_one_hot
        self.blank_id = blank_id

    def init_params(self, first_input):
        """Initialization of the parameters
        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the paramaters.
        """
        if self.consider_as_one_hot:
            # deal with blank_id, the output should be embeddings_dim-1 as we consider blank output as zeros one_hot vect
            self.Embedding = nn.Embedding(self.embeddings_dim+1, self.embeddings_dim,padding_idx=self.blank_id).to(first_input.device)
            one_hot=torch.eye(self.embeddings_dim).to(first_input.device)
            if self.blank_id+1!=self.embeddings_dim+1:
                self.Embedding.weight.data[self.blank_id+1:] = one_hot[self.blank_id+1:]
            if self.blank_id!=0:
                self.Embedding.weight.data[:self.blank_id] = one_hot[:self.blank_id]
            self.Embedding.weight.requires_grad = False
        else:
            self.Embedding = nn.Embedding(first_input.shape[-1], self.n_embeddings)

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