"""Library implementing recurrent neural networks.

Author
    Mirco Ravanelli 2020, Ju-Chieh Chou 2020
"""

import math
import torch
import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


class RNN(torch.nn.Module):
    """ This function implements basic RNN, LSTM and GRU models.

    This function implements different RNN models. It accepts in input tensors
    formatted as (batch, time, fea). In the case of 4d inputs
    like (batch, time, fea, channel) the tensor is flattened in this way:
    (batch, time, fea*channel).

    Arguments
    ---------
    rnn_type: str
        Type of recurrent neural network to use (rnn, lstm, gru, ligru).
    n_neurons: int
        Number of output neurons (i.e, the dimensionality of the output).
        values (i.e, time and frequency kernel sizes respectively).
     nonlinearity: str
         Type of nonlinearity (tanh, relu).
    num_layers: int
         Number of layers to employ in the RNN architecture.
    bias: bool
        If True, the additive bias b is adopted.
    dropout: float
        It is the dropout factor (must be between 0 and 1).
    bidirectional: bool
         if True, a bidirectioal model that scans the sequence both
         right-to-left and left-to-right is used.
.
    Example
    -------
    >>> inp_tensor = torch.rand([4, 10, 20])
    >>> net = RNN(rnn_type='lstm', n_neurons=5)
    >>> out_tensor = net(inp_tensor, init_params=True)
    >>> out_tensor.shape
    torch.Size([4, 10, 5])
    >>> net = RNN(rnn_type='ligru', n_neurons=5)
    >>> out_tensor = net(inp_tensor, init_params=True)
    >>> out_tensor.shape
    torch.Size([4, 10, 5])
    """

    def __init__(
        self,
        rnn_type,
        n_neurons,
        nonlinearity="tanh",
        num_layers=1,
        bias=True,
        dropout=0.0,
        bidirectional=False,
        return_hidden=False,
    ):
        super().__init__()
        self.rnn_type = rnn_type
        self.n_neurons = n_neurons
        self.nonlinearity = (nonlinearity,)
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.reshape = False
        self.return_hidden = return_hidden

    def init_params(self, first_input):
        """
        Initializes the parameters of the recurrent layer.

        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        if len(first_input.shape) > 3:
            self.reshape = True

        # Computing the feature dimensionality
        self.fea_dim = torch.prod(torch.tensor(first_input.shape[2:]))

        kwargs = {
            "input_size": self.fea_dim,
            "hidden_size": self.n_neurons,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "bias": self.bias,
            "batch_first": True,
        }

        # Vanilla RNN
        if self.rnn_type == "rnn":
            kwargs.update({"nonlinearity": self.nonlinearity})
            self.rnn = torch.nn.RNN(**kwargs)

        if self.rnn_type == "lstm":
            self.rnn = torch.nn.LSTM(**kwargs)

        if self.rnn_type == "gru":
            self.rnn = torch.nn.GRU(**kwargs)

        if self.rnn_type == "ligru":
            del kwargs["bias"]
            del kwargs["batch_first"]
            kwargs["batch_size"] = first_input.shape[0]
            kwargs["device"] = first_input.device
            self.rnn = LiGRU(**kwargs)

        self.rnn.to(first_input.device)

    def forward(self, x, init_params=False):
        """Returns the output of the RNN.

        Arguments
        ---------
        x : torch.Tensor
        """
        if init_params:
            self.init_params(x)

        # Reshaping input tensors for 4d inputs
        if self.reshape:
            if len(x.shape) == 4:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        # Needed for multi-gpu
        if self.rnn_type != "ligru":
            self.rnn.flatten_parameters()

        output, hn = self.rnn(x)

        if self.return_hidden:
            return output, hn
        else:
            return output


class LiGRU(torch.jit.ScriptModule):
    """ This function implements Light-Gated Recurrent Units (ligru).

    Ligru is a customized GRU model based on batch-norm + relu
    activations + recurrent dropout. For more info see
    "M. Ravanelli, P. Brakel, M. Omologo, Y. Bengio,
    Light Gated Recurrent Units for Speech Recognition,
    in IEEE Transactions on Emerging Topics in Computational Intelligence,
    2018"
    To speed it up, it is compiled with the torch just-in-time compiler (jit)
    right before using it.

    Arguments
    ---------
    input_size: int
        Feature dimensionality of the input tensors.
    batch_size: int
        Batch size of the input tensors.
    hidden_size: int
         Number of output neurons .
    num_layers: int
         Number of layers to employ in the RNN architecture.
    dropout: float
        It is the dropout factor (must be between 0 and 1).
    bidirectional: bool
         if True, a bidirectioal model that scans the sequence both
         right-to-left and left-to-right is used.
    device: str
         Device used for running the computations (e.g, 'cpu', 'cuda').

    Example
    -------
    >>> inp_tensor = torch.rand([4, 10, 20])
    >>> net = LiGRU(20, 5, 1, 4, device='cpu')
    >>> out_tensor, h = net(inp_tensor)
    >>> out_tensor.shape
    torch.Size([4, 10, 10])
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        batch_size,
        dropout=0.0,
        bidirectional=True,
        device="cuda",
    ):

        super().__init__()
        current_dim = int(input_size)
        self.model = torch.nn.ModuleList([])
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        for i in range(num_layers):
            rnn_lay = LiGRU_Layer(
                current_dim,
                self.hidden_size,
                num_layers,
                batch_size,
                dropout=dropout,
                bidirectional=self.bidirectional,
                device=device,
            )

            self.model.append(rnn_lay)

            if self.bidirectional:
                current_dim = self.hidden_size * 2
            else:
                current_dim = self.hidden_size

    @torch.jit.script_method
    def forward(self, x):
        """Returns the output of the liGRU.

        Arguments
        ---------
        x : torch.Tensor
        """
        h = []
        for ligru_lay in self.model:
            x = ligru_lay(x)
            h.append(x[:, -1, :])
        h = torch.stack(h, dim=1)
        if self.bidirectional:
            h = h.reshape(h.shape[0], h.shape[1] * 2, self.hidden_size)

        return x, (h,)


class LiGRU_Layer(torch.jit.ScriptModule):
    """ This function implements Light-Gated Recurrent Units (ligru) layer.

    Arguments
    ---------
    input_size: int
        Feature dimensionality of the input tensors.
    batch_size: int
        Batch size of the input tensors.
    hidden_size: int
         Number of output neurons .
    num_layers: int
         Number of layers to employ in the RNN architecture.
    dropout: float
        It is the dropout factor (must be between 0 and 1).
    bidirectional: bool
         if True, a bidirectioal model that scans the sequence both
         right-to-left and left-to-right is used.
    device: str
         Device used for running the computations (e.g, 'cpu', 'cuda').
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        batch_size,
        dropout=0.0,
        nonlinearity="relu",
        bidirectional=True,
        device="cuda",
    ):

        super(LiGRU_Layer, self).__init__()
        self.hidden_size = int(hidden_size)
        self.input_size = int(input_size)
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.device = device

        self.w = nn.Linear(
            self.input_size, 2 * self.hidden_size, bias=False
        ).to(device)

        self.u = nn.Linear(
            self.hidden_size, 2 * self.hidden_size, bias=False
        ).to(device)

        # Adding orthogonal initialization for recurrent connection
        nn.init.orthogonal_(self.u.weight)

        # Initializing batch norm
        self.bn_w = nn.BatchNorm1d(2 * self.hidden_size, momentum=0.05).to(
            device
        )

        # Initilizing dropout
        self.drop = torch.nn.Dropout(p=self.dropout, inplace=False).to(
            self.device
        )

        self.drop_mask_te = torch.tensor([1.0], device=self.device).float()

        # Setting the activation function
        self.act = torch.nn.ReLU().to(device)

    @torch.jit.script_method
    def _init_h(self, x):
        """Initializes the initial state h_0 with zeros.
        """
        h_init = torch.zeros(x.shape[0], self.hidden_size, device=self.device,)
        return h_init

    @torch.jit.script_method
    def forward(self, x):
        """Returns the output of the liGRU layer.

        Arguments
        ---------
        x : torch.Tensor
        """
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Feed-forward affine transformations (all steps in parallel)
        w = self.w(x)

        # Apply batch normalization
        w_bn = self.bn_w(w.reshape(w.shape[0] * w.shape[1], w.shape[2]))

        w = w_bn.reshape(w.shape[0], w.shape[1], w.shape[2])

        # Processing time steps
        h = self._ligru_cell(w)

        if self.bidirectional:
            h_f, h_b = h.chunk(2, dim=0)
            h_b = h_b.flip(1)
            h = torch.cat([h_f, h_b], dim=2)

        return h

    @torch.jit.script_method
    def _ligru_cell(self, w):
        """Returns the hidden states for each time step.

        Arguments
        ---------
        wx : torch.Tensor
            Linearly transformed input.
        """
        hiddens = []
        ht = self._init_h(w)
        drop_mask = self._sample_drop_mask(w)

        # Loop over time axis
        for k in range(w.shape[1]):
            gates = w[:, k] + self.u(ht)
            at, zt = gates.chunk(2, 1)
            zt = torch.sigmoid(zt)
            hcand = self.act(at) * drop_mask
            ht = zt * ht + (1 - zt) * hcand
            hiddens.append(ht)

        # Stacking hidden states
        h = torch.stack(hiddens, dim=1)
        return h

    @torch.jit.script_method
    def _sample_drop_mask(self, x):
        """Selects one of the pre-defined dropout masks
        """
        if self.training:
            drop_mask = (
                self.drop(torch.ones(x.shape[0], self.hidden_size,))
                .to(self.device)
                .data
            )

        else:
            drop_mask = self.drop_mask_te

        return drop_mask


def init_rnn_module(module):
    """
    This function is used to initialize the RNN weight with orthogonality.
    Arguments
    ---------
    module: torch.nn.Module
        Reccurent neural network module.
    Example
    -------
    >>> inp_tensor = torch.rand([4, 10, 20])
    >>> net = RNN(rnn_type='lstm', n_neurons=5)
    >>> out_tensor = net(inp_tensor, init_params=True)
    >>> init_rnn_module(net)
    """
    for param in module.parameters():
        if len(param.shape) >= 2:
            nn.init.orthogonal_(param)
        else:
            stdv = 1.0 / math.sqrt(param.shape[0])
            nn.init.uniform_(param, -stdv, stdv)
