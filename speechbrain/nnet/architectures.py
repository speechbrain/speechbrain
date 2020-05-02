"""
Most popular neural architectures in speech and audio
"""

import math
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Sequential(torch.nn.Module):
    """A sequence of modules, implementing the `init_params` method.

    Arguments
    ---------
    *layers
        The inputs are treated as a list of layers to be
        applied in sequence. The output shape of each layer is used to
        infer the shape of the following layer.

    Example
    -------
    >>> import speechbrain.nnet.architectures
    >>> model = Sequential(
    ...     speechbrain.nnet.architectures.linear(n_neurons=100),
    ... )
    >>> inputs = torch.rand(10, 50, 40)
    >>> outputs = model(inputs, init_params=True)
    >>> outputs.shape
    torch.Size([10, 50, 100])
    """

    def __init__(
        self, *layers,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for layer in layers:
            self.layers.append(layer)

    def forward(self, x, init_params=False):
        """
        Arguments
        ---------
        x : tensor
            the input tensor to run through the network.
        """
        for layer in self.layers:
            try:
                x = layer(x, init_params=init_params)
            except TypeError:
                x = layer(x)
        return x


class linear(torch.nn.Module):
    """Computes a linear transformation y = wx + b.

    Arguments
    ---------
    n_neurons : int
        it is the number of output neurons (i.e, the dimensionality of the
        output)
    bias : bool
        if True, the additive bias b is adopted.

    Example
    -------
    >>> lin_t = linear(n_neurons=100)
    >>> inputs = torch.rand(10, 50, 40)
    >>> output = lin_t(inputs,init_params=True)
    >>> output.shape
    torch.Size([10, 50, 100])
    """

    def __init__(self, n_neurons, bias=True):
        super().__init__()
        self.n_neurons = n_neurons
        self.bias = bias

    def init_params(self, first_input):
        """
        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        fea_dim = first_input.shape[2]
        self.w = nn.Linear(fea_dim, self.n_neurons, bias=self.bias)
        self.w.to(first_input.device)

    def forward(self, x, init_params=False):
        """Returns the linear transformation of input tensor.

        Arguments
        ---------
        x : torch.Tensor
            input to transform linearly.
        """
        if init_params:
            self.init_params(x)

        # Transposing tensor (features always at the end)
        x = x.transpose(2, -1)

        wx = self.w(x)

        # Going back to the original shape format
        wx = wx.transpose(2, -1)
        return wx


class conv(nn.Module):
    """This function implements 1D, 2D, and sinc_conv (SincNet) convolutionals.

    This class implements convolutional layers:
    Conv1d is used when the specified kernel size is 1d (e.g, kernel_size=3).
    Conv2d is used when the specified kernel size is 2d (e.g, kernel_size=3,5).
    sinc_conv (SincNet) is used when sinc_conv is True.

    Args:
        out_channels: int
            It is the number of output channels.
        kernel_size: int
            It is a list containing the size of the kernels.
            For 1D convolutions, the list contains a single
            integer (convolution over the time axis), while
            for 2D convolutions the list is composed of two
            values (i.e, time and frequency kernel sizes respectively).
        stride: int
            it is a list containing the stride factors.
            For 1D convolutions, the list contains a single
            integer (stride over the time axis), while
            for 2D convolutions the list is composed of two
            values (i.e, time and frequency kernel sizes,
            respectively). When the stride factor > 1, a
            decimation (in the time or frequnecy domain) is
            implicitely performed.
        dilation: int
            it is a list containing the dilation factors.
            For 1D convolutions, the list contains a single
            integer (dilation over the time axis), while
            for 2D convolutions the list is composed of two
            values (i.e, time and frequency kernel sizes,
            respectively).
        padding: bool
            if True, zero-padding is performed.
        padding_mode: str
            This flag specifies the type of padding.
            See torch.nn documentation for more information.
        groups: int
            This option specifies the convolutional groups.
            See torch.nn documentation for more information.
        bias: bool
            If True, the additive bias b is adopted.
        sinc_conv: bool
            If True computes convolution with sinc-based filters (SincNet).
        sample_rate: int,
            Sampling rate of the input signals. It is only used for sinc_conv.
        min_low_hz: float
            Lowest possible frequency (in Hz) for a filter. It is only used for
            sinc_conv.
        min_low_hz: float
            Lowest possible value (in Hz) for a filter bandwidth.

    Example:
        >>> inp_tensor = torch.rand([10, 16000, 1])
        >>> cnn_1d = conv(out_channels=25, kernel_size=(11,))
        >>> out_tensor = cnn_1d(inp_tensor, init_params=True)
        >>> out_tensor.shape
        torch.Size([10, 15990, 25])
        >>> inp_tensor = torch.rand([10, 100, 40, 128])
        >>> cnn_2d = conv(out_channels=25, kernel_size=(11,5))
        >>> out_tensor = cnn_2d(inp_tensor, init_params=True)
        >>> out_tensor.shape
        torch.Size([10, 90, 36, 25])
        >>> inp_tensor = torch.rand([10, 4000])
        >>> sinc_conv = conv(out_channels=8,kernel_size=(129,),sinc_conv=True)
        >>> out_tensor = sinc_conv(inp_tensor, init_params=True)
        >>> out_tensor.shape
        torch.Size([10, 3872, 8])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        stride=(1, 1),
        dilation=(1, 1),
        padding=False,
        groups=1,
        bias=True,
        padding_mode="zeros",
        sinc_conv=False,
        sample_rate=16000,
        min_low_hz=50,
        min_band_hz=50,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.sinc_conv = sinc_conv
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        self.reshape_conv1d = False
        self.unsqueeze = False

        # Check the specified kernel (to decide between conv1d and conv2d)
        self._kernel_check()

    def _kernel_check(self):
        """Checks the specified kernel and decides if we have to use conv1d,
        conv2d, or sinc_conv.
        """

        self.conv1d = False
        self.conv2d = False

        if not isinstance(self.kernel_size, tuple):
            self.kernel_size = tuple(self.kernel_size,)

        # Make sure kernel_size is odd (needed for padding)
        for size in self.kernel_size:
            if size % 2 == 0:
                raise ValueError(
                    "The field kernel size must be an odd number. Got %s."
                    % (self.kernel_size)
                )

        if len(self.kernel_size) == 1:
            self.conv1d = True

        if len(self.kernel_size) == 2:
            self.conv2d = True

        if self.sinc_conv and self.conv2d:
            raise ValueError(
                "sinc_conv expects 1d kernels. Got " + len(self.kernel_size)
            )

    def init_params(self, first_input):
        """
        Initializes the parameters of the convolutional layer.

        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        self.device = first_input.device

        if self.conv1d:
            if self.sinc_conv:
                self._init_sinc_conv(first_input)
            else:
                self._init_conv1d(first_input)

        if self.conv2d:
            self._init_conv2d(first_input)

    def _init_conv1d(self, first_input):
        """
        Initializes the parameters of the conv1d layer.

        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        if len(first_input.shape) == 1:
            raise ValueError(
                "conv1d expects 2d, 3d, or 4d inputs. Got " + len(first_input)
            )

        if len(first_input.shape) == 2:
            self.unsqueeze = True
            self.in_channels = 1

        if len(first_input.shape) == 3:
            self.in_channels = first_input.shape[2]

        if len(first_input.shape) == 4:
            self.reshape_conv1d = True
            self.in_channels = first_input.shape[2] * first_input.shape[3]

        if len(first_input.shape) > 4:
            raise ValueError(
                "conv1d expects 2d, 3d, or 4d inputs. Got " + len(first_input)
            )

        self.conv = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            self.kernel_size[0],
            stride=self.stride[0],
            dilation=self.dilation[0],
            padding=0,
            groups=self.groups,
            bias=self.bias,
            padding_mode=self.padding_mode,
        ).to(first_input.device)

    def _init_conv2d(self, first_input):
        """
        Initializes the parameters of the conv2d layer.

        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        if len(first_input.shape) <= 2:
            raise ValueError(
                "conv2d expects 3d or 4d inputs. Got " + len(first_input)
            )

        if len(first_input.shape) == 3:
            self.unsqueeze = True
            self.in_channels = 1

        if len(first_input.shape) == 4:
            self.in_channels = first_input.shape[3]

        if len(first_input.shape) > 4:
            raise ValueError(
                "conv1d expects 3d or 4d inputs. Got " + len(first_input)
            )

        self.kernel_size = (self.kernel_size[1], self.kernel_size[0])
        self.stride = (self.stride[1], self.stride[0])
        self.dilation = (self.dilation[1], self.dilation[0])

        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias,
            padding_mode=self.padding_mode,
        ).to(first_input.device)

    def _init_sinc_conv(self, first_input):
        """
        Initializes the parameters of the sinc_conv layer.

        Arguments
        ---------
        first_input : tensor
            A first input used for initializing the parameters.
        """
        self._init_conv1d(first_input)

        # Initialize filterbanks such that they are equally spaced in Mel scale
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = torch.linspace(
            self._to_mel(self.min_low_hz),
            self._to_mel(high_hz),
            self.out_channels + 1,
        )

        hz = self._to_hz(mel)

        # Filter lower frequency and bands
        self.low_hz_ = hz[:-1].unsqueeze(1)
        self.band_hz_ = (hz[1:] - hz[:-1]).unsqueeze(1)

        # Maiking freq and bands learnable
        self.low_hz_ = nn.Parameter(self.low_hz_).to(self.device)
        self.band_hz_ = nn.Parameter(self.band_hz_).to(self.device)

        # Hamming window
        n_lin = torch.linspace(
            0,
            (self.kernel_size[0] / 2) - 1,
            steps=int((self.kernel_size[0] / 2)),
        )
        self.window_ = 0.54 - 0.46 * torch.cos(
            2 * math.pi * n_lin / self.kernel_size[0]
        ).to(self.device)

        # Time axis  (only half is needed due to symmetry)
        n = (self.kernel_size[0] - 1) / 2.0
        self.n_ = (
            2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate
        ).to(self.device)

    def forward(self, x, init_params=False):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : torch.Tensor
        """
        if init_params:
            self.init_params(x)

        x = x.transpose(1, -1)

        if self.reshape_conv1d:
            or_shape = x.shape
            x = x.reshape(or_shape[0], or_shape[1] * or_shape[2], or_shape[3])

        if self.unsqueeze:
            x = x.unsqueeze(1)

        if self.padding:
            x = self._manage_padding(
                x, self.kernel_size, self.dilation, self.stride
            )
        if self.sinc_conv:
            sinc_filters = self._get_sinc_filters()

            wx = F.conv1d(
                x,
                sinc_filters,
                stride=self.stride[0],
                padding=0,
                dilation=self.dilation[0],
                bias=None,
                groups=1,
            )

        else:
            wx = self.conv(x)

        # Retrieving the original shapes
        if self.unsqueeze:
            wx = wx.squeeze(1)

        if self.reshape_conv1d:
            wx = wx.reshape(or_shape[0], wx.shape[1], wx.shape[2], wx.shape[3])

        wx = wx.transpose(1, -1)

        return wx

    def _manage_padding(self, x, kernel_size, dilation, stride):
        """This function performs zero-padding on the time and frequency axis
        such that their lengths is unchanged after the convolution.

        Arguments
        ---------
        x : torch.Tensor
        kernel_size : int
        dilation : int
        stride: int
        """

        # Detecting input shape
        L_in = x.shape[-1]

        # Time padding
        padding = self._get_padding_elem(
            L_in, stride[-1], kernel_size[-1], dilation[-1]
        )

        if self.conv2d:
            padding_freq = self._get_padding_elem(
                L_in, stride[-2], kernel_size[-2], dilation[-2]
            )
            padding = padding + padding_freq

        # Applying padding
        x = nn.functional.pad(input=x, pad=tuple(padding), mode="reflect")

        return x

    def _get_sinc_filters(self,):
        """This functions creates the sinc-filters to used for sinc-conv.
        """
        # Computing the low frequencies of the filters
        low = self.min_low_hz + torch.abs(self.low_hz_)

        # Setting minimum band and minimum freq
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2,
        )
        band = (high - low)[:, 0]

        # Passing from n_ to the corresponding f_times_t domain
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        # Left part of the filters.
        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low))
            / (self.n_ / 2)
        ) * self.window_

        # Central element of the filter
        band_pass_center = 2 * band.view(-1, 1)

        # Right part of the filter (sinc filters are symmetric)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        # Combining left, central, and right part of the filter
        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right], dim=1
        )

        # Amplitude normalization
        band_pass = band_pass / (2 * band[:, None])

        # Setting up the filter coefficients
        filters = (
            (band_pass)
            .view(self.out_channels, 1, self.kernel_size[0])
            .to(self.device)
        )

        return filters

    @staticmethod
    def _get_padding_elem(L_in, stride, kernel_size, dilation):
        """This computes the number of elements to add for zero-padding.

        Arguments
        ---------
        L_in : int
        stride: int
        kernel_size : int
        dilation : int
        """
        if stride > 1:
            n_steps = math.ceil(((L_in - kernel_size * dilation) / stride) + 1)
            L_out = stride * (n_steps - 1) + kernel_size * dilation
            padding = [kernel_size // 2, kernel_size // 2]

        else:
            L_out = (L_in - dilation * (kernel_size - 1) - 1) / stride + 1
            L_out = int(L_out)

            padding = [(L_in - L_out) // 2, (L_in - L_out) // 2]
        return padding

    @staticmethod
    def _to_mel(hz):
        """Converts frequency in Hz to the mel scale.
        """
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def _to_hz(mel):
        """Converts frequency in the mel scale to Hz.
        """
        return 700 * (10 ** (mel / 2595) - 1)


class RNN(torch.nn.Module):
    """ This function implements basic RNN, LSTM and GRU models.

    This function implelent different RNN models. It accepts in input tesors
    formatted as (batch, time, fea). In the case of 4d inputs
    like (batch, time, fea, channel) the tensor ais flattened in this way:
    (batch, time, fea*channel).

    Args:
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
    Example:
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
        Initializes the parameters of the convolutional layer.

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
            self.rnn = liGRU(**kwargs)

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


class liGRU(torch.jit.ScriptModule):
    """ This function implements Light-Gated Recurrent Units (ligru).

    Ligru is a customized GRU model based on batch-norm + relu
    activations + recurrent dropout. For more info see
    "M. Ravanelli, P. Brakel, M. Omologo, Y. Bengio,
    Light Gated Recurrent Units for Speech Recognition,
    in IEEE Transactions on Emerging Topics in Computational Intelligence,
    2018"
    To speed it up, it is compiled with the torch just-in-time compiler (jit)
    right before using it.

    Args:
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

    Example:
        >>> inp_tensor = torch.rand([4, 10, 20])
        >>> net = liGRU(20,5,1,4, device='cpu')
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
            rnn_lay = liGRU_layer(
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
                current_dim == hidden_size

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


class liGRU_layer(torch.jit.ScriptModule):
    """ This function implements Light-Gated Recurrent Units (ligru) layer.

    Args:
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

        super(liGRU_layer, self).__init__()
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


class softmax(torch.nn.Module):
    """Computes the softmax of a 2d, 3d, or 4d input tensor.

    Arguments
    ---------
    softmax_type : str
        it is the type of softmax to use ('softmax', 'log_softmax')
    dim : int
        if the dimension where softmax is applied.

    Example
    -------
    >>> classifier = softmax()
    >>> inputs = torch.rand(10, 50, 40)
    >>> output = classifier(inputs)
    >>> output.shape
    torch.Size([10, 50, 40])
    """

    def __init__(self, softmax_type="log_softmax", dim=-1):
        super().__init__()

        if softmax_type == "softmax":
            self.act = torch.nn.Softmax(dim=dim)

        if softmax_type == "log_softmax":
            self.act = torch.nn.LogSoftmax(dim=dim)

    def forward(self, x):
        """Returns the softmax of the input tensor.

        Arguments
        ---------
        x : torch.Tensor
        """
        # Reshaping the tensors
        dims = x.shape

        if len(dims) == 3:
            x = x.reshape(dims[0] * dims[1], dims[2])

        if len(dims) == 4:
            x = x.reshape(dims[0] * dims[1], dims[2], dims[3])

        x_act = self.act(x)

        # Retrieving the original shape format
        if len(dims) == 3:
            x_act = x_act.reshape(dims[0], dims[1], dims[2])

        if len(dims) == 4:
            x_act = x_act.reshape(dims[0], dims[1], dims[2], dims[3])

        return x_act


class dropout(nn.Module):
    """This function implements droput.

    This function implements droput of the input tensor. In particular,
    1d dropout (nn.Dropout) is activated with 2d or 3d input tensors, while
    nn.Dropout2d is activated with 4d input tensors.


    Arguments
    ---------
    dropout_rate : float
        It is the dropout factor (between 0 and 1).
    inplace : bool
        If True, it uses inplace operations.

    Example
    -------
    >>> drop = dropout(drop_rate=0.5)
    >>> inputs = torch.rand(10, 50, 40)
    >>> drop.init_params(inputs)
    >>> output=drop(inputs)
    >>> output.shape
    torch.Size([10, 50, 40])
    """

    def __init__(
        self, drop_rate, inplace=False,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.inplace = inplace

    def init_params(self, first_input):
        """
        Arguments
        ---------
        first_input : tensor
            A dummy input of the right shape for initializing parameters.
        """

        if len(first_input.shape) <= 3:
            self.drop = nn.Dropout(p=self.drop_rate, inplace=self.inplace)

        if len(first_input.shape) == 4:
            self.drop = nn.Dropout2d(p=self.drop_rate, inplace=self.inplace)

        if len(first_input.shape) == 5:
            self.drop = nn.Dropout3d(p=self.drop_rate, inplace=self.inplace)

    def forward(self, x, init_params=False):
        """Applies dropout to the input tensor.

        Arguments
        ---------
        x : torch.Tensor
        """
        if init_params:
            self.init_params(x)

        if self.drop_rate == 0.0:
            return x

        # time must be the last
        x = x.transpose(1, 2).transpose(2, -1)

        x_drop = self.drop(x)

        # Getting original dimensionality
        x_drop = x_drop.transpose(-1, 1).transpose(2, -1)

        return x_drop


class pooling(nn.Module):
    """This function implements pooling of the input tensor

    Arguments
    ---------
    pool_type : str
        It is the type of pooling function to use ('avg','max')
    pool_axis : list
        It is a list containing the axis that will be considered
        during pooling. It must match the dimensionality of the pooling.
        If the pooling is 2D, then a list of 2 indices is expected.
    kernel_size : list
        It is the kernel size. Note that it also defines the pooling dimension.
        For instance kernel size=3 applies a 1D Pooling with a size=3,
        while kernel size=3,3 performs a 2D Pooling with a 3x3 kernel.
    stride : int
        It is the stride size.
    padding : int
        It is the numbe of padding elements to apply.
    dilation : int
        Controls the dilation factor of pooling.
    ceil_mode : int
        When True, will use ceil instead of floor to compute the output shape.

    Example
    -------
    >>> pool = pooling('max',3)
    >>> inputs = torch.rand(10, 50, 40)
    >>> pool.init_params(inputs)
    >>> output=pool(inputs)
    >>> output.shape
    torch.Size([10, 50, 38])
    """

    def __init__(
        self,
        pool_type,
        kernel_size,
        pool_axis=2,
        ceil_mode=False,
        padding=0,
        dilation=1,
        stride=1,
    ):
        super().__init__()
        self.pool_type = pool_type
        self.kernel_size = kernel_size
        self.pool_axis = pool_axis
        self.ceil_mode = ceil_mode
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.pool1d = False
        self.pool2d = False
        self.combine_batch_time = False

        if not isinstance(self.kernel_size, list):
            self.kernel_size = [self.kernel_size]
        if not isinstance(self.pool_axis, list):
            self.pool_axis = [self.pool_axis]

    def init_params(self, first_input):
        """
        Arguments
        ---------
        first_input : tensor
            A dummy input of the right shape for initializing parameters.
        """

        # Check parameters
        self.check_params(first_input)

        # Pooling initialization
        if self.pool_type == "avg":

            # Check Pooling dimension
            if self.pool1d:
                self.pool_layer = torch.nn.AvgPool1d(
                    self.kernel_size[0],
                    stride=self.stride,
                    padding=self.padding,
                    ceil_mode=self.ceil_mode,
                )

            else:
                self.pool_layer = torch.nn.AvgPool2d(
                    self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    ceil_mode=self.ceil_mode,
                )
        else:
            if self.pool1d:
                self.pool_layer = torch.nn.MaxPool1d(
                    self.kernel_size[0],
                    stride=self.stride,
                    padding=self.padding,
                    ceil_mode=self.ceil_mode,
                )

            else:
                self.pool_layer = torch.nn.MaxPool2d(
                    self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    ceil_mode=self.ceil_mode,
                )

    def check_params(self, first_input):

        # Check that enough pooling axes are specified
        if len(self.kernel_size) == 1:
            self.pool1d = True

            # In the case of a 4 dimensional input vector, we need to
            # combine the batch and time dimension together because
            # torch.nn.pool1d only accepts 3D vectors as inputs.
            if len(first_input.shape) > 3:
                self.combine_batch_time = True

            if len(self.pool_axis) != 1:
                err_msg = (
                    "pool_axes must corresponds to the pooling dimension. "
                    "The pooling dimension is 1 and %s axes are specified."
                    % (str(len(self.pool_axis)))
                )
                raise ValueError(err_msg)

            if self.pool_axis[0] >= len(first_input.shape):
                err_msg = (
                    "pool_axes is greater than the number of dimensions. "
                    "The tensor dimension is %s and the specified pooling "
                    "axis is %s."
                    % (str(len(first_input.shape)), str(self.pool_axis))
                )
                raise ValueError(err_msg)

        if len(self.kernel_size) == 2:
            self.pool2d = True

            if len(self.pool_axis) != 2:
                err_msg = (
                    "pool_axes must corresponds to the pooling dimension. "
                    "The pooling dimension is 2 and %s axes are specified."
                    % (str(len(self.pool_axis)))
                )
                raise ValueError(err_msg)

            dims = len(first_input.shape)
            if self.pool_axis[0] >= dims or self.pool_axis[1] >= dims:
                err_msg = (
                    "pool_axes is greater than the number of dimensions. "
                    "The tensor dimension is %s and the specified pooling "
                    "axis are %s."
                    % (str(len(first_input.shape)), str(self.pool_axis))
                )
                raise ValueError(err_msg)

    def forward(self, x, init_params=False):
        if init_params:
            self.init_params(x)

        # Put the pooling axes as the last dimension for torch.nn.pool
        # If the input tensor is 4 dimensional, combine the first and second
        # axes together to respect the input shape of torch.nn.pool1d
        if self.pool1d:
            x = x.transpose(-1, self.pool_axis[0])
            new_shape = x.shape
            if self.combine_batch_time:
                x = x.reshape(new_shape[0] * new_shape[1], new_shape[2], -1)

        if self.pool2d:
            x = x.transpose(-2, self.pool_axis[0]).transpose(
                -1, self.pool_axis[1]
            )

        # Apply pooling
        x = self.pool_layer(x)

        # Recover input shape
        if self.pool1d:
            if self.combine_batch_time:
                x = x.reshape(new_shape[0], new_shape[1], new_shape[2], -1)
            x = x.transpose(-1, self.pool_axis[0])

        if self.pool2d:
            x = x.transpose(-2, self.pool_axis[0]).transpose(
                -1, self.pool_axis[1]
            )

        return x
