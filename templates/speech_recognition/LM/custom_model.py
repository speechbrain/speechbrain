"""
This file contains a very simple PyTorch module to use for language modeling.

To replace this model, change the `!new:` tag in the hyperparameter file
to refer to a built-in SpeechBrain model or another file containing
a custom PyTorch module. Instead of this simple model, we suggest using one
of the following built-in neural models:

RNN-LM: speechbrain.lobes.models.RNNLM.RNNLM
transformer: speechbrain.lobes.models.transformers.TransformerLM.TransformerLM

Authors
 * Mirco Ravanelli 2021

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
        embedding_dim=128,
        rnn_size=256,
        layers=2,
        output_dim=1000,
        return_hidden=False,
    ):
        super().__init__()
        self.return_hidden = return_hidden
        self.reshape = False

        # Embedding model
        self.embedding = sb.nnet.embedding.Embedding(
            num_embeddings=output_dim, embedding_dim=embedding_dim
        )

        # LSTM
        self.rnn = torch.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=rnn_size,
            bidirectional=False,
            num_layers=layers,
        )

        # Final output transformation + softmax
        self.out = sb.nnet.linear.Linear(
            input_size=rnn_size, n_neurons=output_dim
        )
        self.log_softmax = sb.nnet.activations.Softmax(apply_log=True)

    def forward(self, x, hx=None):
        """List of computations from input to output predictions"""
        x = self.embedding(x)

        # If 2d tensor, add a time-axis
        # This is used for inference time (during beamforming)
        if len(x.shape) == 2:
            x = x.unsqueeze(dim=1)
            self.reshape = True

        x = x.transpose(0, 1)
        x, hidden = self.rnn(x, hx)
        x = x.transpose(0, 1)
        x = self.out(x)
        x = self.log_softmax(x)

        if self.reshape:
            x = x.squeeze(dim=1)

        if self.return_hidden:
            return x, hidden
        else:
            return x
