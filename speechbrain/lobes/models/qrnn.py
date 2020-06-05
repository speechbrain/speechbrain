"""This is a implementation for the Quasi-RNN

https://arxiv.org/pdf/1611.01576.pdf

Part of the code is adapted from:

https://github.com/salesforce/pytorch-qrnn

Authors: Jianyuan Zhong 2020

"""

import torch  # noqa: F401
import torch.nn as nn

from speechbrain.nnet.containers import Sequential


class QRNNLayer(torch.jit.ScriptModule):
    """Applies a single layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.
    Arguments
    ---------
    input_size : int
        The number of expected features in the input x.
    hidden_size : int
        The number of features in the hidden state h. If not specified, the input size is used.
    zoneout : float
        Whether to apply zoneout (i.e. failing to update elements in the hidden state) to the hidden state updates. Default: 0.
    output_gate : bool
        If True, performs QRNN-fo (applying an output gate to the output). If False, performs QRNN-f. Default: True.

    Example
    -------
    >>> import torch
    >>> model = QRNNLayer(60, 256, bidirectional=True)
    >>> a = torch.rand([10, 120, 60])
    >>> b = model(a)
    >>> print(b[0].shape)
    torch.Size([10, 120, 512])
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        bidirectional,
        zoneout=0.0,
        output_gate=True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.zoneout = zoneout
        self.output_gate = output_gate
        self.bidirectional = bidirectional

        stacked_hidden = (
            3 * self.hidden_size if self.output_gate else 2 * self.hidden_size
        )
        self.w = torch.nn.Linear(input_size, stacked_hidden, True)

        self.z_gate = nn.Tanh()
        self.f_gate = nn.Sigmoid()
        if self.output_gate:
            self.o_gate = nn.Sigmoid()

    @torch.jit.script_method
    def forgetMult(self, f, x, hidden):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor # noqa F821
        """Returns the hidden states for each time step.

        Arguments
        ---------
        wx : torch.Tensor
            Linearly transformed input.
        """
        result = []
        htm1 = hidden
        hh = f * x

        for i in range(hh.shape[0]):
            h_t = hh[i, :, :]
            ft = f[i, :, :]
            if htm1 is not None:
                h_t = h_t + (1 - ft) * htm1
            result.append(h_t)
            htm1 = h_t

        return torch.stack(result)

    @torch.jit.script_method
    def split_gate_inputs(self, y):
        # type: (Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]] # noqa F821
        if self.output_gate:
            z, f, o = y.chunk(3, dim=-1)
        else:
            z, f = y.chunk(2, dim=-1)
            o = None
        return z, f, o

    @torch.jit.script_method
    def forward(self, x, hidden=None):
        # type: (Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor] # noqa F821
        """Returns the output of the QRNN layer.

        Arguments
        ---------
        x : torch.Tensor
            input to transform linearly.
        """
        if len(x.shape) == 4:
            # if input is a 4d tensor (batch, time, channel1, channel2)
            # reshape input to (batch, time, channel)
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        # give a tensor of shape (time, batch, channel)
        x = x.permute(1, 0, 2)
        if self.bidirectional:
            x_flipped = x.flip(0)
            x = torch.cat([x, x_flipped], dim=1)

        # note: this is equivalent to doing 1x1 convolution on the input
        y = self.w(x)

        z, f, o = self.split_gate_inputs(y)

        z = self.z_gate(z)
        f = self.f_gate(f)
        if o is not None:
            o = self.o_gate(o)

        # If zoneout is specified, we perform dropout on the forget gates in F
        # If an element of F is zero, that means the corresponding neuron keeps the old value
        if self.zoneout:
            if self.training:
                mask = (
                    torch.empty(f.shape)
                    .bernoulli_(1 - self.zoneout)
                    .to(f.get_device())
                ).detach()
                f = f * mask
            else:
                f = f * (1 - self.zoneout)

        z = z.contiguous()
        f = f.contiguous()

        # Forget Mult
        c = self.forgetMult(f, z, hidden)

        # Apply output gate
        if o is not None:
            h = o * c
        else:
            h = c

        # recover shape (batch, time, channel)
        c = c.permute(1, 0, 2)
        h = h.permute(1, 0, 2)

        if self.bidirectional:
            h_fwd, h_bwd = h.chunk(2, dim=0)
            h_bwd = h_bwd.flip(1)
            h = torch.cat([h_fwd, h_bwd], dim=2)

            c_fwd, c_bwd = c.chunk(2, dim=0)
            c_bwd = c_bwd.flip(1)
            c = torch.cat([c_fwd, c_bwd], dim=2)

        return h, c[-1, :, :]


class QRNN(nn.Module):
    """Applies a multiple layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.
    Arguments
    ---------
    hidden_size :
        The number of features in the hidden state h. If not specified, the input size is used.
    num_layers :
        The number of QRNN layers to produce.
    zoneout :
        Whether to apply zoneout (i.e. failing to update elements in the hidden state) to the hidden state updates. Default: 0.
    output_gate :
        If True, performs QRNN-fo (applying an output gate to the output). If False, performs QRNN-f. Default: True.

    Example
    -------
    >>> a = torch.rand([8, 120, 40])
    >>> model = QRNN(256, 4, bidirectional=True)
    >>> b = model(a, init_params=True)
    >>> print(b.shape)
    torch.Size([8, 120, 512])
    """

    def __init__(
        self,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0,
        bidirectional=False,
        return_hidden=False,
        **kwargs,
    ):
        assert bias is True, "Removing underlying bias is not yet supported"
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.return_hidden = return_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout if dropout > 0 else None
        self.kwargs = kwargs

    def init_params(self, first_input):
        input_size = first_input.shape[-1]
        if len(first_input.shape) == 4:
            input_size = first_input.shape[-1] * first_input.shape[-2]

        layers = []
        for layer in range(self.num_layers):
            layers.append(
                QRNNLayer(
                    input_size
                    if layer == 0
                    else self.hidden_size * 2
                    if self.bidirectional
                    else self.hidden_size,
                    self.hidden_size,
                    self.bidirectional,
                    **self.kwargs,
                )
            )
        self.qrnn = Sequential(*layers)

        # for some reason, jit module cannot handle .to("cpu")...
        device = first_input.get_device()
        if device >= 0:
            self.qrnn = self.qrnn.to(device)

        if self.dropout:
            self.dropout = torch.nn.Dropout(self.dropout)

    def forward(self, x, hidden=None, init_params=False):
        if init_params:
            self.init_params(x)

        next_hidden = []

        for i, layer in enumerate(self.qrnn.layers):
            x, h = layer(x, None if hidden is None else hidden[i])

            next_hidden.append(h)

            if self.dropout and i < len(self.qrnn.layers) - 1:
                x = self.dropout(x)

        next_hidden = torch.cat(next_hidden, 0).view(
            self.num_layers, *next_hidden[0].shape[-2:]
        )

        if self.return_hidden:
            return x, next_hidden
        else:
            return x
