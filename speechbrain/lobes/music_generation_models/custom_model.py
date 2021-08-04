"""
This file contains a simple PyTorch module to use for language modeling.
It defines an LSTM model.
Authors
 * Edward Son 2021
"""
import torch
import torch.nn as nn
import speechbrain as sb


class MusicRNN(torch.nn.Module):
    """Basic LSTM model for language modeling.
    Arguments
    ---------
    embedding_dim : int
        The dimension of the embeddings.The input indexes are transformed into
        a latent space with this dimensionality.
    rnn_size : int
        Number of neurons to use in rnn (for each direction -> and <-).
    layers : int
        Number of RNN layers to use.
    output_dim : int
        Dimensionality of the output.
    return_hidden : bool
        If True, returns the hidden state of the RNN as well.
    """

    def __init__(
        self,
        embedding_dim,
        rnn_latent_size,
        n_rnn_layers=2,
        output_dim=388,
        return_hidden=False,
        dropout=0,
        representation="event",
    ):
        super().__init__()
        self.reshape = False

        if representation == "event":
            self.emb_layer = nn.Embedding(
                num_embeddings=output_dim, embedding_dim=embedding_dim
            )
        else:
            self.emb_layer = lambda x: x

        # LSTM
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=rnn_latent_size,
            num_layers=n_rnn_layers,
            bidirectional=False,
            batch_first=True,
            dropout=dropout,
        )

        # Final output transformation
        self.out = sb.nnet.linear.Linear(
            input_size=rnn_latent_size, n_neurons=output_dim
        )

    def forward(self, x, hx=None):
        """List of computations from input to output predictions"""

        # If 2d tensor, add a time-axis
        # This is used for inference time

        x = self.emb_layer(x)

        if hx is None:
            x, hidden = self.rnn(x)
        else:
            x, hidden = self.rnn(x, hx)

        x = self.out(x)

        return x, hidden
