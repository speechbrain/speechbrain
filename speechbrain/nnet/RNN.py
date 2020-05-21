"""Library implementing recurrent neural networks.

Author
    Mirco Ravanelli 2020, Ju-Chieh Chou 2020
"""

import torch
import logging
import torch.nn as nn
from speechbrain.nnet.attention import (
    ContentBasedAttention,
    LocationAwareAttention,
)

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
         Type of nonlinearity (tanh, relu). This option is active for
         rnn and ligru models only. For lstm and gru tanh is used.
    normalization: str
         Type of normalization for the ligru model (batchnorm, layernorm).
         Every string different from batchnorm and layernorm will result
         in no normalization.
    num_layers: int
         Number of layers to employ in the RNN architecture.
    bias: bool
        If True, the additive bias b is adopted.
    dropout: float
        It is the dropout factor (must be between 0 and 1).
    re_init: bool:
        It True, orthogonal initialization is used for the recurrent weights.
        Xavier initialization is used for the input connection weights.
    return_hidden: bool:
        It True, the function returns the last hidden layer.
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
    >>> inp_tensor = torch.rand([4, 10, 20])
    >>> net = RNN(rnn_type='ligru', n_neurons=5, num_layers=2, return_hidden=True, bidirectional=True)
    >>> out_tensor0, hn = net(inp_tensor, init_params=True)
    >>> out_tensor1, hn = net(inp_tensor, hn, init_params=True)
    >>> out_tensor1.shape
    torch.Size([4, 10, 10])
    >>> hn.shape
    torch.Size([4, 4, 5])
    """

    def __init__(
        self,
        rnn_type,
        n_neurons,
        nonlinearity="relu",
        normalization="batchnorm",
        num_layers=1,
        bias=True,
        dropout=0.0,
        re_init=False,
        bidirectional=False,
        return_hidden=False,
    ):
        super().__init__()
        self.rnn_type = rnn_type
        self.n_neurons = n_neurons
        self.nonlinearity = nonlinearity
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.re_init = re_init
        self.bidirectional = bidirectional
        self.reshape = False
        self.return_hidden = return_hidden
        self.normalization = normalization

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

        if len(first_input.shape) > 4:
            err_msg = (
                "Class RNN doesn't support tensors with more than",
                "4 dimensions. Got %i" % (str(len(first_input.shape))),
            )
            raise ValueError(err_msg)

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
            kwargs["normalization"] = self.normalization
            kwargs.update({"nonlinearity": self.nonlinearity})
            self.rnn = LiGRU(**kwargs)

        if self.re_init:
            if self.rnn_type in ["gru", "lstm"]:
                rnn_init(self.rnn, act="tanh")
            else:
                rnn_init(self.rnn, act=self.nonlinearity)

        self.rnn.to(first_input.device)

    def forward(self, x, hx=None, init_params=False):
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

        # Support custom inital state
        if hx is not None:
            output, hn = self.rnn(x, hx=hx)
        else:
            output, hn = self.rnn(x)

        if self.return_hidden:
            return output, hn
        else:
            return output


class AttentionalRNNDecoder(nn.Module):
    def __init__(
        self,
        rnn_type,
        attn_type,
        n_neurons,
        attn_dim,
        num_layers,
        nonlinearity="relu",
        re_init=False,
        normalization="batchnorm",
        scaling=1.0,
        channels=None,
        kernel_size=None,
        bias=True,
        dropout=0.0,
    ):
        """This funtion implements RNN decoder model with attention.

        This function implements different RNN models. It accepts in enc_states tensors
        formatted as (batch, time, fea). In the case of 4d inputs
        like (batch, time, fea, channel) the tensor is flattened in this way:
        (batch, time, fea*channel).

        Arguments
        ---------
        rnn_type: str
            Type of recurrent neural network to use (rnn, lstm, gru, ligru).
        attn_type: str
            type of attention to use (location, content).
        n_neurons: int
            Number of internal and output neurons.
        attn_dim: int
            Number of attention module internal and output neurons.
        num_layers: int
             Number of layers to employ in the RNN architecture.
        nonlinearity: str
             Type of nonlinearity (tanh, relu). This option is active for
             rnn and ligru models only. For lstm and gru tanh is used.
        re_init: bool:
            It True, orthogonal initialization is used for the recurrent weights.
            Xavier initialization is used for the input connection weights.
        normalization: str
             Type of normalization for the ligru model (batchnorm, layernorm).
             Every string different from batchnorm and layernorm will result
             in no normalization.
        scaling: float
            The scaling factor to sharpen or smoothen the attention distribution.
        channels: int
            Number of channels for location-aware attention.
        kernel_size: int
            Size of the kernel for location-aware attention.
        bias: bool
            If True, the additive bias b is adopted.
        dropout: float
            It is the dropout factor (must be between 0 and 1).

        Example
        -------
        >>> enc_states = torch.rand([4, 10, 20])
        >>> wav_len = torch.rand([4])
        >>> inp_tensor = torch.rand([4, 5, 6])
        >>> net = AttentionalRNNDecoder(
        ...     rnn_type='lstm',
        ...     attn_type='content',
        ...     n_neurons=7,
        ...     attn_dim=5,
        ...     num_layers=1)
        >>> out_tensor, attn = net(inp_tensor, enc_states, wav_len, init_params=True)
        >>> out_tensor.shape
        torch.Size([4, 5, 7])
        """
        super(AttentionalRNNDecoder, self).__init__()

        self.rnn_type = rnn_type
        self.attn_type = attn_type
        self.n_neurons = n_neurons
        self.attn_dim = attn_dim
        self.num_layers = num_layers
        self.scaling = scaling
        self.bias = bias
        self.dropout = dropout
        self.normalization = normalization
        self.re_init = re_init
        self.nonlinearity = nonlinearity

        # only for location-aware attention
        self.channels = channels
        self.kernel_size = kernel_size

    def _check_dim(self, tensor):
        """
        This method will check the input shape and
        calculate corresponding dimension and reshape flag.

        Arguments
        ---------
        tensor : torch.Tensor
            input tensor to be checked.
        """
        reshape = False
        if len(tensor.shape) > 3:
            reshape = True

        if len(tensor.shape) > 4:
            err_msg = (
                "Calss AttentionalRNNDecoder doesn't support tensors with more than",
                "4 dimensions. Got %i" % (str(len(tensor.shape))),
            )
            raise ValueError(err_msg)

        dim = torch.prod(torch.tensor(tensor.shape[2:]))
        return dim, reshape

    def init_params(self, first_input):
        """
        Initializes the parameters of this module.

        Arguments
        ---------
        first_input : list of tensor
            A first input used for initializing the parameters.
            The list should contain [inp_tensor, enc_states].
        """
        inp_tensor, enc_states = first_input
        device = inp_tensor.device

        self.enc_dim, self.reshape = self._check_dim(enc_states)

        # Combining the context vector and output of rnn
        self.proj = nn.Linear(
            self.n_neurons + self.attn_dim, self.n_neurons
        ).to(device)

        if self.attn_type == "content":
            self.attn = ContentBasedAttention(
                enc_dim=self.enc_dim,
                dec_dim=self.n_neurons * self.num_layers,
                attn_dim=self.attn_dim,
                output_dim=self.attn_dim,
                scaling=self.scaling,
            ).to(device)

        elif self.attn_type == "location":
            self.attn = LocationAwareAttention(
                enc_dim=self.enc_dim,
                dec_dim=self.n_neurons * self.num_layers,
                attn_dim=self.attn_dim,
                output_dim=self.attn_dim,
                conv_channels=self.channels,
                kernel_size=self.kernel_size,
                scaling=self.scaling,
            ).to(device)

        else:
            raise ValueError(f"{self.attn_type} is not implemented.")

        self.drop = nn.Dropout(p=self.dropout).to(device)

        self.rnn = RNN(
            rnn_type=self.rnn_type,
            n_neurons=self.n_neurons,
            nonlinearity=self.nonlinearity,
            normalization=self.normalization,
            num_layers=self.num_layers,
            bias=self.bias,
            dropout=self.dropout,
            re_init=self.re_init,
            bidirectional=False,
            return_hidden=True,
        )
        # The dummy context vector for initialization
        context = torch.zeros(
            inp_tensor.shape[0], inp_tensor.shape[1], self.attn_dim
        ).to(device)
        inputs = torch.cat([inp_tensor, context], dim=-1)
        self.rnn.init_params(inputs)

    def forward_step(self, inp, hs, c, enc_states, enc_len):
        """
        One step of forward pass process.

        Arguments:
        inp : torch.Tensor
            The input of current timestep.
        hs : torch.Tensor or tuple of torch.Tensor
            The cell state for RNN.
        c : torch.Tensor
            The context vector of previous timestep.
        enc_states : torch.Tensor
            The tensor generated by encoder, to be attended.
        enc_len : torch.LongTensor
            The actual length of encoder states.

        Returns:
        dec_out : torch.Tensor
        hs : torch.Tensor or tuple of torch.Tensor
            The new cell state for RNN.
        c : torch.Tensor
            The context vector of the current timestep.
        w : torch.Tensor
            The weight of attention.
        """
        cell_inp = torch.cat([inp, c], dim=-1).unsqueeze(1)
        cell_inp = self.drop(cell_inp)
        cell_out, hs = self.rnn(cell_inp, hs)
        cell_out = cell_out.squeeze(1)

        # The last layer of decoder hidden states
        if isinstance(hs, tuple):
            dec = (
                hs[0][-1]
                .reshape(-1, self.n_neurons)
            )
        else:
            dec = hs[-1].reshape(
                -1, self.n_neurons
            )

        c, w = self.attn(enc_states, enc_len, dec)
        dec_out = torch.cat([c, cell_out], dim=1)
        dec_out = self.proj(dec_out)

        return dec_out, hs, c, w

    def forward(self, inp_tensor, enc_states, wav_len, init_params=False):
        if init_params:
            self.init_params([inp_tensor, enc_states])

        if self.reshape:
            enc_states = enc_states.reshape(
                enc_states.shape[0],
                enc_states.shape[1],
                enc_states.shape[2] * enc_states.shape[3],
            )

        # calculating the actual length of enc_states
        enc_len = torch.round(enc_states.shape[1] * wav_len).long()

        # initialization
        self.attn.reset()
        c = torch.zeros(enc_states.shape[0], self.attn_dim).to(
            enc_states.device
        )
        hs = None

        # store predicted tokens
        outputs_lst, attn_lst = [], []
        for t in range(inp_tensor.shape[1]):
            outputs, hs, c, w = self.forward_step(
                inp_tensor[:, t], hs, c, enc_states, enc_len
            )
            outputs_lst.append(outputs)
            attn_lst.append(w)

        # [B, L_d, n_neurons]
        outputs = torch.stack(outputs_lst, dim=1)

        # [B, L_d, L_e]
        attn = torch.stack(attn_lst, dim=1)

        return outputs, attn


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
    nonlinearity: str
         Type of nonlinearity (tanh, relu).
    normalization: str
         Type of normalization (batchnorm, layernorm).
         Every string different from batchnorm and layernorm will result
         in no normalization.
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
        nonlinearity="relu",
        normalization="batchnorm",
        bidirectional=True,
        device="cuda",
    ):

        super().__init__()
        current_dim = int(input_size)
        self.model = torch.nn.ModuleList([])
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        for i in range(num_layers):
            rnn_lay = LiGRU_Layer(
                current_dim,
                self.hidden_size,
                self.num_layers,
                self.batch_size,
                dropout=dropout,
                nonlinearity=nonlinearity,
                normalization=normalization,
                bidirectional=self.bidirectional,
                device=device,
            )

            self.model.append(rnn_lay)

            if self.bidirectional:
                current_dim = self.hidden_size * 2
            else:
                current_dim = self.hidden_size

    @torch.jit.script_method
    def forward(self, x, hx=None):
        # type: (Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor] # noqa F821
        """Returns the output of the liGRU.

        Arguments
        ---------
        x : torch.Tensor
        """
        h = []
        if hx is not None:
            if self.bidirectional:
                hx = hx.reshape(
                    self.num_layers, self.batch_size * 2, self.hidden_size
                )

        for i, ligru_lay in enumerate(self.model):
            if hx is not None:
                x = ligru_lay(x, hx=hx[i])
            else:
                x = ligru_lay(x, hx=None)
            h.append(x[:, -1, :])
        h = torch.stack(h, dim=1)
        if self.bidirectional:
            h = h.reshape(h.shape[1] * 2, h.shape[0], self.hidden_size)
        else:
            h = h.transpose(0, 1)

        return x, h


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
    nonlinearity: str
         Type of nonlinearity (tanh, relu).
    normalization: str
         Type of normalization (batchnorm, layernorm).
         Every string different from batchnorm and layernorm will result
         in no normalization.
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
        normalization="batchnorm",
        bidirectional=False,
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

        if self.bidirectional:
            self.batch_size = self.batch_size * 2

        # Initializing batch norm
        self.normalize = False

        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(2 * self.hidden_size, momentum=0.05).to(
                device
            )
            self.normalize = True

        elif normalization == "layernorm":
            self.norm = torch.nn.LayerNorm(2 * self.hidden_size).to(device)
            self.normalize = True
        else:
            # Normalization is disabled here. self.norm is only  formally
            # initialized to avoid jit issues.
            self.norm = torch.nn.LayerNorm(2 * self.hidden_size).to(device)
            self.normalize = True

        # Initial state
        self.h_init = torch.zeros(
            1, self.hidden_size, requires_grad=False, device=self.device,
        )

        # Preloading dropout masks (gives some speed improvement)
        self._init_drop(self.batch_size)

        # Initilizing dropout
        self.drop = torch.nn.Dropout(p=self.dropout, inplace=False).to(device)

        self.drop_mask_te = torch.tensor([1.0], device=self.device).float()

        # Setting the activation function
        if nonlinearity == "tanh":
            self.act = torch.nn.Tanh().to(device)
        else:
            self.act = torch.nn.ReLU().to(device)

    @torch.jit.script_method
    def forward(self, x, hx=None):
        # type: (Tensor, Optional[Tensor]) -> Tensor # noqa F821
        """Returns the output of the liGRU layer.

        Arguments
        ---------
        x : torch.Tensor
        """
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        self._change_batch_size(x)

        # Feed-forward affine transformations (all steps in parallel)
        w = self.w(x)

        # Apply batch normalization
        if self.normalize:
            w_bn = self.norm(w.reshape(w.shape[0] * w.shape[1], w.shape[2]))
            w = w_bn.reshape(w.shape[0], w.shape[1], w.shape[2])

        # Processing time steps
        if hx is not None:
            h = self._ligru_cell(w, hx)
        else:
            h = self._ligru_cell(w, self.h_init)

        if self.bidirectional:
            h_f, h_b = h.chunk(2, dim=0)
            h_b = h_b.flip(1)
            h = torch.cat([h_f, h_b], dim=2)

        return h

    @torch.jit.script_method
    def _ligru_cell(self, w, ht):
        """Returns the hidden states for each time step.

        Arguments
        ---------
        wx : torch.Tensor
            Linearly transformed input.
        """
        hiddens = []

        # Sampling dropout mask
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

    def _init_drop(self, batch_size):
        """Initializes the recurrent dropout operation. To speed it up,
        the dropout masks are sampled in advance.
        """
        self.drop = torch.nn.Dropout(p=self.dropout, inplace=False).to(
            self.device
        )
        self.drop_mask_te = torch.tensor([1.0], device=self.device).float()

        self.N_drop_masks = 16000
        self.drop_mask_cnt = 0

        self.drop_masks = self.drop(
            torch.ones(self.N_drop_masks, self.hidden_size, device=self.device,)
        ).data

    @torch.jit.script_method
    def _sample_drop_mask(self,):
        """Selects one of the pre-defined dropout masks
        """
        if self.training:

            # Sample new masks when needed
            if self.drop_mask_cnt + self.batch_size > self.N_drop_masks:
                self.drop_mask_cnt = 0
                self.drop_masks = (
                    self.drop(torch.ones(self.N_drop_masks, self.hidden_size,))
                    .to(self.device)
                    .data
                )

            # Sampling the mask
            drop_mask = self.drop_masks[
                self.drop_mask_cnt : self.drop_mask_cnt + self.batch_size
            ]
            self.drop_mask_cnt = self.drop_mask_cnt + self.batch_size

        else:
            drop_mask = self.drop_mask_te

        return drop_mask

    @torch.jit.script_method
    def _change_batch_size(self, x):
        """This function changes the batch size when it is different from
        the one detected in the initialization method. This might happen in
        the case of multi-gpu or when we have different batch sizes in train
        and test. We also update the h_int and drop masks.
        """
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

            if self.training:
                self.drop_masks = self.drop(
                    torch.ones(
                        self.N_drop_masks, self.hidden_size, device=self.device,
                    )
                ).data


def rnn_init(module, act="tanh"):
    """
    This function is used to initialize the RNN weight.
    Recurrent connection: orthogonal initialization.


    Arguments
    ---------
    module: torch.nn.Module
        Recurrent neural network module.

    Example
    -------
    >>> inp_tensor = torch.rand([4, 10, 20])
    >>> net = RNN(rnn_type='lstm', n_neurons=5)
    >>> out_tensor = net(inp_tensor, init_params=True)
    >>> rnn_init(net)
    """
    for name, param in module.named_parameters():
        if "weight_hh" in name or ".u.weight" in name:
            nn.init.orthogonal_(param)
        elif len(param.shape) == 2:
            nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain(act))
