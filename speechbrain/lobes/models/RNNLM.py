"""A popular speech model.

Authors
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
 * Ju-Chieh Chou 2020
 * Titouan Parcollet 2020
 * Abdel 2020
"""
import torch
from torch import nn
from speechbrain.nnet.embedding import Embedding
from speechbrain.nnet.RNN import LSTM
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.containers import Sequential
from speechbrain.nnet.normalization import LayerNorm


class RNNLM(nn.Module):
    """This model is a combination of embedding layer, RNN, DNN.
    It can be used for RNNLM.

    Arguments
    ---------
    num_embeddings : int
        Number of entries in embedding table.
    embedding_dim : int
        Default : 128
        Size of embedding vectors.
    activation : torch class
        A class used for constructing the activation layers. For dnn.
    dropout : float
        Neuron dropout rate, applied to embedding, rnn, and dnn.
    rnn_class : torch class
        The type of rnn to use in CRDNN network (LiGRU, LSTM, GRU, RNN)
    rnn_layers : int
        The number of recurrent LiGRU layers to include.
    rnn_neurons : int
        Number of neurons in each layer of the RNN.
    rnn_re_init : bool
        Whether to initialize rnn with orthogonal initialization.
    rnn_return_hidden : bool
        Default : True
        Whether to return hidden states.
    dnn_blocks : int
        The number of linear neural blocks to include.
    dnn_neurons : int
        The number of neurons in the linear layers.

    Example
    -------
    >>> model = RNNLM(num_embeddings=5)
    >>> inputs = torch.Tensor([[1, 2, 3]])
    >>> outputs = model(inputs, init_params=True)
    >>> outputs.shape
    torch.Size([1, 3, 512])
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim=128,
        activation=torch.nn.LeakyReLU,
        dropout=0.15,
        rnn_class=LSTM,
        rnn_layers=2,
        rnn_neurons=1024,
        rnn_re_init=False,
        return_hidden=False,
        dnn_blocks=1,
        dnn_neurons=512,
    ):
        super().__init__()
        self.embedding = Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )
        self.dropout = nn.Dropout(p=dropout)
        self.rnn = rnn_class(
            hidden_size=rnn_neurons,
            num_layers=rnn_layers,
            dropout=dropout,
            re_init=rnn_re_init,
            return_hidden=True,
        )
        self.return_hidden = return_hidden
        self.reshape = False

        dnn_blocks_lst = []
        for block_index in range(dnn_blocks):
            dnn_blocks_lst.extend(
                [
                    Linear(
                        n_neurons=dnn_neurons, bias=True, combine_dims=False,
                    ),
                    LayerNorm(),
                    activation(),
                    torch.nn.Dropout(p=dropout),
                ]
            )

        self.dnn = Sequential(*dnn_blocks_lst)

    def forward(self, x, hx=None, init_params=False):

        x = self.embedding(x, init_params=init_params)
        x = self.dropout(x)

        # If 2d tensor, add a time-axis
        # this is used for inference situation
        if len(x.shape) == 2:
            x = x.unsqueeze(dim=1)
            self.reshape = True

        x, h = self.rnn(x, hx, init_params=init_params)
        x = self.dnn(x, init_params=init_params)

        if self.reshape:
            x = x.squeeze(dim=1)

        if self.return_hidden:
            return x, h
        else:
            return x
