"""Library implementing recurrent neural networks.

Author
    Mirco Ravanelli 2020
"""

import torch
import logging
import torch.nn as nn
import math

from speechbrain.nnet.attention import (
    LocationAwareAttention,
    ContentBasedAttention,
)

logger = logging.getLogger(__name__)


def init_rnn_module(module):
    """
    This function is used to initialize the RNN weight with orthogonality.
    """
    for param in module.parameters():
        if len(param.shape) >= 2:
            nn.init.orthogonal_(param)
        else:
            stdv = 1.0 / math.sqrt(param.shape[0])
            nn.init.uniform_(param, -stdv, stdv)


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

    def init_params(self, first_input):
        """
        Initializes the parameters of the RNN layer.

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

        output, hn = self.rnn(x)

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

        for i in range(num_layers):
            rnn_lay = LiGRU_Layer(
                current_dim,
                hidden_size,
                num_layers,
                batch_size,
                dropout=dropout,
                bidirectional=bidirectional,
                device=device,
            )

            self.model.append(rnn_lay)

            if bidirectional:
                current_dim = hidden_size * 2
            else:
                current_dim = hidden_size

    @torch.jit.script_method
    def forward(self, x):
        """Returns the output of the liGRU.

        Arguments
        ---------
        x : torch.Tensor
        """
        for ligru_lay in self.model:
            x = ligru_lay(x)
        return x, 0


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
        self._init_drop()

        # Initilizing initial state h
        self._init_h()

        # Setting the activation function
        self.act = torch.nn.ReLU().to(device)

    def _init_drop(self,):
        """Initializes the recurrent dropout operation. To speed it up,
        the dropout masks are sampled in advance.
        """
        self.drop = torch.nn.Dropout(p=self.dropout, inplace=False).to(
            self.device
        )
        self.drop_mask_te = torch.tensor([1.0], device=self.device).float()
        self.N_drop_masks = 1000
        self.drop_mask_cnt = 0

        if self.bidirectional:
            self.drop_masks = self.drop(
                torch.ones(
                    self.N_drop_masks,
                    2 * self.batch_size,
                    self.hidden_size,
                    device=self.device,
                )
            ).data

        else:
            self.drop_masks = self.drop(
                torch.ones(
                    self.N_drop_masks,
                    self.batch_size,
                    self.hidden_size,
                    device=self.device,
                )
            ).data

    def _init_h(self,):
        """Initializes the initial state h_0 with zeros.
        """
        if self.bidirectional:
            self.h_init = torch.zeros(
                2 * self.batch_size,
                self.hidden_size,
                requires_grad=False,
                device=self.device,
            )

        else:
            self.h_init = torch.zeros(
                self.batch_size,
                self.hidden_size,
                requires_grad=False,
                device=self.device,
            )

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
        ht = self.h_init
        drop_mask = self._sample_drop_mask()

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
    def _sample_drop_mask(self,):
        """Selects one of the pre-defined dropout masks
        """
        if self.training:
            drop_mask = self.drop_masks[self.drop_mask_cnt]
            self.drop_mask_cnt = self.drop_mask_cnt + 1

            # Sample new masks when needed
            if self.drop_mask_cnt >= self.N_drop_masks:
                self.drop_mask_cnt = 0
                if self.bidirectional:
                    self.drop_masks = (
                        self.drop(
                            torch.ones(
                                self.N_drop_masks,
                                2 * self.batch_size,
                                self.hidden_size,
                            )
                        )
                        .to(self.device)
                        .data
                    )
                else:
                    self.drop_masks = (
                        self.drop(
                            torch.ones(
                                self.N_drop_masks,
                                self.batch_size,
                                self.hidden_size,
                            )
                        )
                        .to(self.device)
                        .data
                    )

        else:
            drop_mask = self.drop_mask_te

        return drop_mask


class AttentionalRNNDecoder(nn.Module):
    def __init__(
        self,
        rnn_type,
        attn_type,
        n_neurons,
        attn_dim,
        attn_out_dim,
        vocab_dim,
        emb_dim,
        num_layers,
        weight_tying=False,
        scaling=1.0,
        channels=None,
        kernel_size=None,
        bias=True,
        dropout=0.0,
        bos_index=-1,
        eos_index=-1,
    ):
        super(AttentionalRNNDecoder, self).__init__()

        self.rnn_type = rnn_type
        self.attn_type = attn_type
        self.n_neurons = n_neurons
        self.attn_dim = attn_dim
        self.attn_out_dim = attn_out_dim
        self.vocab_dim = vocab_dim
        self.weight_tying = weight_tying

        if self.weight_tying and emb_dim != n_neurons:
            raise ValueError(
                "Weight tying must have the same emb_dim and n_neurons"
            )

        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.weight_tying = weight_tying
        self.scaling = scaling
        self.bias = bias
        self.dropout = dropout

        # only for location-aware attention
        self.channels = channels
        self.kernel_size = kernel_size

        self.bos_index = (
            bos_index if bos_index >= 0 else self.vocab_dim + bos_index
        )
        self.eos_index = (
            eos_index if eos_index >= 0 else self.vocab_dim + eos_index
        )

        self.reshape = False

    def init_params(self, first_input):
        enc_states, wav_len, tokens = first_input

        self.enc_dim = torch.prod(torch.tensor(enc_states.shape[2:]))
        if len(enc_states.shape) == 4:
            self.reshape = True

        self.emb = nn.Embedding(self.vocab_dim, self.emb_dim).to(
            enc_states.device
        )

        # dimension reduction
        self.proj = nn.Linear(
            self.n_neurons + self.attn_out_dim, self.n_neurons
        ).to(enc_states.device)
        self.out = nn.Linear(self.n_neurons, self.vocab_dim).to(
            enc_states.device
        )

        if self.weight_tying:
            self.out.weight = self.emb.weight

        if self.attn_type == "content":
            self.attn = ContentBasedAttention(
                enc_dim=self.enc_dim,
                dec_dim=self.n_neurons * self.num_layers,
                attn_dim=self.attn_dim,
                output_dim=self.attn_out_dim,
                scaling=self.scaling,
            ).to(enc_states.device)

        elif self.attn_type == "location":
            self.attn = LocationAwareAttention(
                enc_dim=self.enc_dim,
                dec_dim=self.n_neurons * self.num_layers,
                attn_dim=self.attn_dim,
                output_dim=self.attn_out_dim,
                conv_channels=self.channels,
                kernel_size=self.kernel_size,
                scaling=self.scaling,
            ).to(enc_states.device)

        else:
            raise ValueError(f"{self.attn_type} is not implemented.")

        self.drop = nn.Dropout(p=self.dropout).to(enc_states.device)

        # input = [context, emb]
        input_dim = self.emb_dim + self.attn_out_dim
        # Recurrent connections
        if self.rnn_type == "rnn":
            self.rnn = torch.nn.RNN(
                input_size=input_dim,
                hidden_size=self.n_neurons,
                num_layers=self.num_layers,
                bias=self.bias,
                dropout=self.dropout,
                batch_first=True,
            ).to(enc_states.device)
            self.state_has_cell = False
        if self.rnn_type == "lstm":
            self.rnn = torch.nn.LSTM(
                input_size=input_dim,
                hidden_size=self.n_neurons,
                num_layers=self.num_layers,
                bias=self.bias,
                dropout=self.dropout,
                batch_first=True,
            ).to(enc_states.device)

            # LSTMs have two hidden variables
            self.state_has_cell = True
        if self.rnn_type == "gru":
            self.rnn = torch.nn.GRU(
                input_size=input_dim,
                hidden_size=self.n_neurons,
                num_layers=self.num_layers,
                bias=self.bias,
                dropout=self.dropout,
                batch_first=True,
            ).to(enc_states.device)

            self.state_has_cell = False

        # orthogonal initialization
        init_rnn_module(self.rnn)

    def init_state(self, batch_size, device):
        """Initialize the state for RNN
        """
        if self.state_has_cell:
            hidden_state = (
                torch.zeros(
                    self.num_layers, batch_size, self.n_neurons, device=device
                ),
                torch.zeros(
                    self.num_layers, batch_size, self.n_neurons, device=device
                ),
            )
        else:
            hidden_state = torch.zeros(
                self.num_layers, batch_size, self.n_neurons, device=device
            )

        return hidden_state

    def forward_step(self, y_t, hs, c, enc_states, enc_len):
        e = self.emb(y_t)
        cell_inp = torch.cat([e, c], dim=-1).unsqueeze(1)
        cell_inp = self.drop(cell_inp)
        cell_out, hs = self.rnn(cell_inp, hs)
        cell_out = cell_out.squeeze(1)

        # flattening the decoder hidden states
        if self.state_has_cell:
            dec_flatten = hs[0].reshape(-1, self.n_neurons * self.num_layers)
        else:
            dec_flatten = hs.reshape(-1, self.n_neurons * self.num_layers)

        c, w = self.attn(enc_states, enc_len, dec_flatten)
        dec_out = torch.cat([c, cell_out], dim=1)
        dec_out = self.proj(dec_out)
        logits = self.out(dec_out)

        return logits, hs, c, w

    def forward(self, enc_states, wav_len, tokens, init_params=False):
        if init_params:
            self.init_params([enc_states, wav_len, tokens])

        if self.reshape:
            enc_states = enc_states.reshape(
                -1, enc_states.size(1), enc_states.size(2) * enc_states.size(3)
            )

        # calculating the actual length of enc_states
        enc_len = torch.round(enc_states.size(1) * wav_len).long()

        tokens = tokens.long()
        batch_size = enc_states.size(0)

        # preparing input for the decoder
        bos = tokens.new_ones(batch_size, 1).long() * self.bos_index
        y_in = torch.cat([bos, tokens], dim=1)

        # initialization
        hs = self.init_state(batch_size, enc_states.device)
        self.attn.reset()
        c = torch.zeros(batch_size, self.attn_out_dim).to(enc_states.device)

        # store predicted tokens
        logits_lst, attn_lst = [], []

        for t in range(y_in.size(1)):
            logits, hs, c, w = self.forward_step(
                y_in[:, t], hs, c, enc_states, enc_len
            )
            logits_lst.append(logits)
            attn_lst.append(w)

        # [B, L_d, vocab_size]
        logits = torch.stack(logits_lst, dim=1)

        # [B, L_d, L_e]
        attn = torch.stack(attn_lst, dim=1)

        return logits, attn
