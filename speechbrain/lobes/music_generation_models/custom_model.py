"""
This file contains a simple PyTorch module to use for language modeling.
It defines an LSTM model.
Authors
 * Edward Son 2021
"""
import torch
import speechbrain as sb


class CustomModel(torch.nn.Module):
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
        embedding_dim=88,
        rnn_size=100,
        layers=2,
        output_dim=88,
        return_hidden=False,
        dropout=0,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.return_hidden = return_hidden
        self.reshape = False

        # LSTM
        self.rnn = torch.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=rnn_size,
            num_layers=layers,
            bidirectional=False,
            batch_first=True,
            dropout=dropout,
        )

        # Final output transformation
        self.out = sb.nnet.linear.Linear(
            input_size=rnn_size, n_neurons=output_dim
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, hx=None):
        """List of computations from input to output predictions"""

        # If 2d tensor, add a time-axis
        # This is used for inference time
        if len(x.shape) == 2:
            x = x.unsqueeze(dim=0)
            self.reshape = True

        if hx is None:
            x, hidden = self.rnn(x)
        else:
            x, hidden = self.rnn(x, hx)

        x = self.out(x)
        x = self.sigmoid(x)

        if self.reshape:
            x = x.squeeze(dim=0)

        return x, hidden
