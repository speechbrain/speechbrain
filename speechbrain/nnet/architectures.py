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
    """
     -------------------------------------------------------------------------
     nnet.architectures.linear (author: Mirco Ravanelli)

     Description:  This function implements a fully connected linear layer:
                   y = Wx + b.

     Input: - n_neurons (type: int(1,inf), mandatory):
               it is the number of output neurons (i.e, the
               dimensionality of the output)

           - bias (type: bool, Default:True):
               if True, the additive bias b is adopted.

     Example:   import torch
                from speechbrain.nnet.architectures import linear

                inp_tensor = torch.rand([4,660,190])

                # Initialization of the class
                linear_transf=linear(n_neurons=1024)

                # Executing computations
                inp_tensor = torch.rand([4,660,120])
                out_tensor = linear_transf([inp_tensor])
                print(out_tensor)
                print(out_tensor.shape)

     """

    def __init__(
        self, n_neurons, bias=True,
    ):
        super().__init__()

        self.n_neurons = n_neurons
        self.bias = bias

    def init_params(self, first_input):
        """
        Arguments
        ---------
        first_input : tensor
            A dummy input of the right shape for initializing parameters.
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

        # Apply linear transformation
        wx = self.w(x)

        # Going back to the original shape format
        wx = wx.transpose(2, -1)
        return wx


class conv(nn.Module):
    """This function implements 1D or 2D convolutional layers.

    Args:
        out_channels: it is the number of output channels.
        kernel_size: it is a list containing the size of the kernels.
            For 1D convolutions, the list contains a single
            integer (convolution over the time axis), while
            for 2D convolutions the list is composed of two
            values (i.e, time and frequenecy kernel sizes respectively).
        stride: it is a list containing the stride factors.
            For 1D convolutions, the list contains a single
            integer (stride over the time axis), while
            for 2D convolutions the list is composed of two
            values (i.e, time and frequenecy kernel sizes,
            respectively). When the stride factor > 1, a
            decimantion (in the time or frequnecy domain) is
            implicitely performed.
        dilation: it is a list containing the dilation factors.
            For 1D convolutions, the list contains a single
            integer (dilation over the time axis), while
            for 2D convolutions the list is composed of two
            values (i.e, time and frequenecy kernel sizes,
            respectively).
        padding: it is a list containing the number of elements to pad.
            For 1D convolutions, the list contains a single
            integer (padding over the time axis), while
            for 2D convolutions the list is composed of two
            values (i.e, time and frequenecy kernel sizes,
            respectively). When not specified, the padding
            is automatically performed such that the input
            and the output have the same time/frequency
            dimensionalities.
        padding_mode: This flag specifies the type of padding.
            See torch.nn documentation for more information.
        groups: This option specifies the convolutional groups.
            See torch.nn documentation for more information.
        bias: if True, the additive bias b is adopted.

    Shape (1D case):
        - x: [batch, time_steps]
        - output: [batch, out_channels, time_steps]

    Shape (2D case):
        - x: [batch, channels, time_steps]
        - output: [batch, channels, out_channels, time_steps]

    Example:
        >>> import torch
        >>> inp_tensor = torch.rand([10, 16000, 1])
        >>> cnn = conv(out_channels=25, kernel_size=11)
        >>> out_tensor = cnn(inp_tensor, init_params=True)
        >>> out_tensor.shape
        torch.Size([10, 15990, 25])

    Author:
        Mirco Ravanelli 2020
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        stride=(1, 1),
        dilation=(1, 1),
        padding=0,
        groups=1,
        bias=True,
        padding_mode="zeros",
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

        self.reshape_conv1d = False
        self.reshape_conv2d = False
        self.squeeze_conv2d = False
        self.transp_conv2d = False

        # Ensure kernel_size and padding are tuples
        if not isinstance(self.kernel_size, tuple):
            self.kernel_size = (self.kernel_size,)
        if self.padding is not None and not isinstance(self.padding, tuple):
            self.padding = (self.padding,)

        # Making sure that the kernel size is odd (if the kernel is not
        # symmetric there could a problem with the padding function)
        for size in self.kernel_size:
            if size % 2 == 0:
                raise ValueError(
                    "The field kernel size must be an odd number. Got %s."
                    % (self.kernel_size)
                )

        # Checking if 1d or 2d is specified
        self.conv1d = False
        self.conv2d = False

        if len(self.kernel_size) == 1:
            self.conv1d = True

        if len(self.kernel_size) == 2:
            self.conv2d = True

    def init_params(self, first_input):
        """
        Arguments
        ---------
        first_input : tensor
            A dummy input of the right shape for initializing parameters.
        """
        # Manage reshaping flags
        if len(first_input.shape) > 3:
            if self.conv1d:
                self.reshape_conv1d = True

        if len(first_input.shape) > 4:
            if self.conv2d:
                self.reshape_conv2d = True

        if len(first_input.shape) == 3 and self.conv2d:
            self.squeeze_conv2d = True

        if len(first_input.shape) >= 4 and self.conv2d:
            self.transp_conv2d = True

        # Detecting the number of input channels
        if self.conv1d:
            self.in_channels = first_input.shape[2]

        if self.conv2d:
            if len(first_input.shape) == 3:
                self.in_channels = 1

            elif len(first_input.shape) == 4:
                self.in_channels = first_input.shape[3]
            elif len(first_input.shape) == 5:
                self.in_channels = first_input.shape[3] * first_input.shape[4]

        # Managing 1d convolutions
        if self.conv1d:

            if self.padding is not None:
                self.padding = self.padding[0]

            # Initialization of the parameters
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

        # Managing 2d convolutions
        if self.conv2d:

            if self.padding is not None:
                self.padding = self.padding[0:-1]

            # Initialization of the parameters
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

    def forward(self, x, init_params=False):
        if init_params:
            self.init_params(x)

        # transposing input
        x = x.transpose(1, 2).transpose(2, -1)

        # Reshaping the inputs when needed
        if self.reshape_conv1d:
            or_shape = x.shape

            # revise that
            if len(or_shape) == 4:
                x = x.reshape(
                    or_shape[0] * or_shape[2], or_shape[1], or_shape[-1]
                )

            # revise that
            if len(or_shape) == 5:
                x = x.reshape(
                    or_shape[0] * or_shape[2] * or_shape[3],
                    or_shape[1],
                    or_shape[-1],
                )

        # Reshaping the inputs when needed
        if self.conv2d:

            if self.reshape_conv2d:

                or_shape = x.shape

                # revise that
                if len(or_shape) == 5:
                    x = x.reshape(
                        or_shape[0],
                        or_shape[1],
                        or_shape[2] * or_shape[3],
                        or_shape[-1],
                    )

            if self.transp_conv2d:
                x = x.transpose(1, 2)

            if self.squeeze_conv2d:
                x = x.unsqueeze(1)

        # Manage padding
        if self.padding is None:
            x = self.manage_padding(
                x, self.kernel_size, self.dilation, self.stride
            )

        # Performing convolution
        wx = self.conv(x)

        # Retrieving the original shape format when needed
        if self.conv2d:
            if self.squeeze_conv2d:
                wx = wx.squeeze(1)

            wx = wx.transpose(1, 2)

        if self.reshape_conv1d:

            if len(or_shape) == 4:
                wx = wx.reshape(
                    or_shape[0], wx.shape[1], or_shape[2], wx.shape[-1]
                )

            if len(or_shape) == 5:
                wx = wx.reshape(
                    or_shape[0],
                    wx.shape[1],
                    or_shape[2],
                    or_shape[3],
                    wx.shape[-1],
                )

        wx = wx.transpose(1, -1).transpose(2, -1)
        return wx

    @staticmethod
    def compute_conv1d_shape(L_in, kernel_size, dilation, stride):
        """
        -----------------------------------------------------------------------
        speechbrain.nnet.architectures.conv.compute_conv1d_shape (M Ravanelli)

        Description: This support function can be use to compute the output
                     dimensioality of a 1D convolutional layer. It is used
                     to detect the number of padding elements.

        Input (call):    - L_in (type: int, mandatory):
                              it is the length of the input signal

                         - kernel_size (type: int, mandatory):
                              it is the kernel size used for the convolution

                         - dilation (type: int, mandatory):
                              it is the dilation factor used for the
                              convolution.

                          - stride (type: int, mandatory):
                              it is the dilation factor used for the
                              convolution.


        Output (call):  L_out (type: integer):
                            it is the length of the output.


     Example:   import torch
                from speechbrain.nnet.architectures import conv

                inp_tensor = torch.rand([4,100,190])

                # config dictionary definition
                config={'class_name':'speechbrain.nnet.architectures.\
                    linear_combination',
                        'out_channels':'25',
                        'kernel_size': '11',
                        'stride': '2',
                        'dilation': '3'}

                # Initialization of the class
                cnn=conv(config,first_input=[inp_tensor])

                # Executing computations
                print(cnn.compute_conv1d_shape(190,11,2,3))

        -----------------------------------------------------------------------
        """
        # Output length computation
        L_out = (L_in - dilation * (kernel_size - 1) - 1) / stride + 1

        return int(L_out)

    def manage_padding(self, x, kernel_size, dilation, stride):
        """
        -----------------------------------------------------------------------
        speechbrain.nnet.architectures.conv.manage_padding
        (author: Mirco Ravanelli)

        Description: This function performs padding such that input and output
                     tensors have the same length.

        Input (call):    - x (type: torch.Tensor, mandatory):
                              it is the input (unpadded) tensor.

                         - kernel_size (type: int_list, mandatory):
                              it is the kernel size used for the convolution

                         - dilation (type: int_list, mandatory):
                              it is the dilation factor used for the
                              convolution.

                          - stride (type: int_list, mandatory):
                              it is the dilation factor used for the
                              convolution.


        Output (call):  x (type: integer):
                            it is the output (padded) tensor


     Example:   import torch
                from speechbrain.nnet.architectures import conv

                inp_tensor = torch.rand([4,100,190])

                # config dictionary definition
                config={'class_name':'speechbrain.nnet.architectures.\
                    linear_combination',
                        'out_channels':'25',
                        'kernel_size': '11',
                        'stride': '2',
                        'dilation': '3'}

                # Initialization of the class
                cnn=conv(config,first_input=[inp_tensor])

                # Executing computations
                padded_tensor=cnn.manage_padding(inp_tensor,[11],[2],[3])
                print(inp_tensor.shape)
                print(padded_tensor.shape)

        -----------------------------------------------------------------------
        """

        # Detecting input shape
        L_in = x.shape[-1]

        # Managing time padding
        if stride[-1] > 1:
            n_steps = math.ceil(
                ((L_in - kernel_size[-1] * dilation[-1]) / stride[-1]) + 1
            )
            L_out = stride[-1] * (n_steps - 1) + kernel_size[-1] * dilation[-1]
            padding = [kernel_size[-1] // 2, kernel_size[-1] // 2]

        else:
            L_in = x.shape[-1]
            L_out = self.compute_conv1d_shape(
                L_in, kernel_size[-1], dilation[-1], stride[-1]
            )
            padding = [(L_in - L_out) // 2, (L_in - L_out) // 2]

        # Managing frequency padding (for 2D CNNs)
        if self.conv2d:

            if stride[-2] > 1:
                n_steps = math.ceil(
                    ((L_in - kernel_size[-2] * dilation[-2]) / stride[-2]) + 1
                )
                L_out = (
                    stride[-2] * (n_steps - 1) + kernel_size[-2] * dilation[-2]
                )
                padding = padding + [
                    kernel_size[-2] // 2,
                    kernel_size[-2] // 2,
                ]

            else:
                L_in = x.shape[-2]
                L_out = self.compute_conv1d_shape(
                    L_in, kernel_size[-2], dilation[-2], stride[-2]
                )
                padding = padding + [(L_in - L_out) // 2, (L_in - L_out) // 2]

        # Applying padding
        x = nn.functional.pad(input=x, pad=tuple(padding), mode="reflect")
        return x


class SincConv(nn.Module):
    """
     -------------------------------------------------------------------------
     nnet.architectures.SincConv (author: Mirco Ravanelli)

     Description:  This function implements a sinc-based convolution (SincNet)

     Input (init):  - config (type, dict, mandatory):
                       it is a dictionary containing the keys described below.

                           - out_channels (type: int(1,inf), mandatory):
                               it is the number of output channels.

                           - kernel_size (type: int_list(1,inf), mandatory):
                               it is a list containing the size of the kernels.
                               For 1D convolutions, the list contains a single
                               integer (convolution over the time axis), while
                               for 2D convolutions the list is composed of two
                               values (i.e, time and frequenecy kernel sizes,
                               respectively).

                           - stride (type: int_list(1,inf), optional:
                               default: 1,1):
                               it is a list containing the stride factors.
                               For 1D convolutions, the list contains a single
                               integer (stride over the time axis), while
                               for 2D convolutions the list is composed of two
                               values (i.e, time and frequenecy kernel sizes,
                               respectively). When the stride factor > 1, a
                               decimantion (in the time or frequnecy domain) is
                               implicitely performed.

                           - dilation (type: int_list(1,inf), optional:
                               default: 1,1):
                               it is a list containing the dilation factors.
                               For 1D convolutions, the list contains a single
                               integer (dilation over the time axis), while
                               for 2D convolutions the list is composed of two
                               values (i.e, time and frequenecy kernel sizes,
                               respectively).

                           - padding (type: int_list(1,inf), optional:
                               default: None):
                               it is a list containing the number of elements
                               to pad.
                               For 1D convolutions, the list contains a single
                               integer (padding over the time axis), while
                               for 2D convolutions the list is composed of two
                               values (i.e, time and frequenecy kernel sizes,
                               respectively). When not specified, the padding
                               is automatically performed such that the input
                               and the output have the same time/frequency
                               dimensionalities.

                           - padding_mode (one_of(circular,zeros), optional:
                               default: zeros):
                               This flag specifies the type of padding.
                               See torch.nn documentation for more information.

                           - sample_rate (type: int(1,inf),
                               default: 16000):
                               it is the sampling frequency of the input
                               waveform.

                            - min_low_hz (type: float(0,inf),
                                default: 50):
                               it is the mininum frequnecy (in Hz) that a
                               learned filter can have.

                            - min_band_hz (type: float(0,inf),
                                default: 50):
                               it is the minimum band (in Hz) that a learned
                               filter can have.


                   - funct_name (type, str, default: None):
                       it is a string containing the name of the parent
                       function that has called this method.

                   - global_config (type, dict, default: None):
                       it a dictionary containing the global variables of the
                       parent config file.

                   - logger (type, logger, default: None):
                       it the logger used to write debug and error messages.
                       If logger=None and root_cfg=True, the file is created
                       from scratch.

                   - first_input (type, list, default: None)
                      this variable allows users to analyze the first input
                      given when calling the class for the first time.


     Input (call): - inp_lst(type, list, mandatory):
                       by default the input arguments are passed with a list.
                       In this case, inp is a list containing the
                       torch.tensor that corresponds to the waveform to
                       transform.
                       The tensor must be in one of the following format:
                       [batch, time].


     Output (call): - wx(type, torch.Tensor, mandatory):
                       The output is a tensor that corresponds to the convolved
                       input.


     Example:   import torch
                from speechbrain.nnet.architectures import SincConv

                inp_tensor = torch.rand([4,32000])

                # config dictionary definition
                config={'class_name':'speechbrain.nnet.architectures.\
                    linear_combination',
                        'out_channels':'25',
                        'kernel_size': '129'}

                # Initialization of the class
                sincnet=SincConv(config,first_input=[inp_tensor])

                # Executing computations
                out_tensor = sincnet([inp_tensor])
                print(out_tensor)
                print(out_tensor.shape)


    Reference:     Mirco Ravanelli, Yoshua Bengio,
                   "Speaker Recognition from raw waveform with SincNet".
                   https://arxiv.org/abs/1808.00158
     """

    def __init__(
        self,
        config,
        funct_name=None,
        global_config=None,
        functions=None,
        logger=None,
        first_input=None,
    ):
        super(SincConv, self).__init__()

        # Logger setup
        self.logger = logger

        # Here are summarized the expected options for this class
        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "out_channels": ("int(1,inf)", "mandatory"),
            "kernel_size": ("int(1,inf)", "mandatory"),
            "stride": ("int(1,inf)", "optional", "1"),
            "dilation": ("int(1,inf)", "optional", "1"),
            "padding": ("int(0,inf)", "optional", "None"),
            "sample_rate": ("int(1,inf)", "optional", "16000"),
            "min_low_hz": ("float(0,inf)", "optional", "50"),
            "min_band_hz": ("float(1,inf)", "optional", "50"),
            "padding_mode": ("one_of(zeros,circular)", "optional", "zeros"),
        }

        # FIX: Old style
        # Check, cast, and expand the options
        # self.conf = check_opts(self, self.expected_options, config, self.logger)

        # Definition of the expected input
        self.expected_inputs = ["torch.Tensor"]

        # Check the first input
        # check_inputs(
        #     self.conf, self.expected_inputs, first_input, logger=self.logger
        # )

        # FIX: The whole if block to new logger style
        # Additional check on the input shapes
        # if first_input is not None:

        #     # Shape check
        #     if len(first_input[0].shape) > 2:

        #         err_msg = (
        #             "SincConv only support one input channel (here, \
        #                 in_channels = {%i})"
        #             % (len(first_input[1:-1].shape))
        #         )

        #         logger_write(err_msg, logfile=logger)

        # Forcing the filters to be odd (i.e, perfectly symmetric)
        # FIX: Whole if block to new logging style
        # if self.kernel_size % 2 == 0:
        #     err_msg = (
        #         "The field kernel size must be and odd number. Got %s."
        #         % (self.kernel_size)
        #     )

        #     logger_write(err_msg, logfile=logger)

        # Initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(
            self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1
        )

        hz = self.to_hz(mel)

        # Filter lower frequency (out_channels, 1)
        self.low_hz_ = torch.Tensor(hz[:-1]).view(-1, 1)

        # Filter frequency band (out_channels, 1)
        self.band_hz_ = torch.Tensor(np.diff(hz)).view(-1, 1)

        # Learning parameters
        self.low_hz_ = nn.Parameter(self.low_hz_)
        self.band_hz_ = nn.Parameter(self.band_hz_)

        # Hamming window
        n_lin = torch.linspace(
            0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2))
        )  # computing only half of the window
        self.window_ = 0.54 - 0.46 * torch.cos(
            2 * math.pi * n_lin / self.kernel_size
        )

        n = (self.kernel_size - 1) / 2.0
        # Due to symmetry, I only need half of the time axes
        self.n_ = (
            2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate
        )

    def forward(self, input_lst):

        # Reading input _list
        x = input_lst[0]

        # Unsqueezing waveform
        x = x.unsqueeze(1)

        # Putting n_ and window_ on the selected device
        self.n_ = self.n_.to(x.device)
        self.window_ = self.window_.to(x.device)

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

        #  left part of the filter
        # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM
        # RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and
        # simplified the terms. This way I avoid several useless computations.
        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low))
            / (self.n_ / 2)
        ) * self.window_

        # central element of the filter
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
        self.filters = (
            (band_pass)
            .view(self.out_channels, 1, self.kernel_size)
            .to(x.device)
        )

        # Manage Padding:
        if self.padding is None:

            x = self.manage_padding(
                x, self.kernel_size, self.dilation, self.stride
            )
        else:
            self.padding = self.padding[0]

        # Performing the convolution with the sinc filters
        wx = F.conv1d(
            x,
            self.filters,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )

        return wx

    @staticmethod
    def to_mel(hz):
        """
        -----------------------------------------------------------------------
         nnet.architectures.SincConv.to_mel (author: Mirco Ravanelli)

        Description: This support function can be use to switch from the
                     standard frequency domain to the mel-frequnecy domain.

        Input (call):    - hz (type: float(0,inf), mandatory):
                              it is the current value in Hz



        Output (call):  mel_hz (type: float):
                            it is the mel-frequency that corresponds to the
                            input mel.



        Example:    import torch
                    from speechbrain.nnet.architectures import SincConv

                    inp_tensor = torch.rand([4,32000])

                    # config dictionary definition
                    config={'class_name':'speechbrain.nnet.architectures.\
                        linear_combination',
                            'out_channels':'25',
                            'kernel_size': '129'}

                    # Initialization of the class
                    sincnet=SincConv(config,first_input=[inp_tensor])

                    # Executing computations
                    print(sincnet.to_mel(4000))
        -----------------------------------------------------------------------
        """

        mel_hz = 2595 * np.log10(1 + hz / 700)

        return mel_hz

    @staticmethod
    def to_hz(mel):
        """
        -----------------------------------------------------------------------
         nnet.architectures.SincConv.to_mel (author: Mirco Ravanelli)

        Description: This support function can be use to switch from the
                     mel-frequency domain to the standard frequnecy domain.

        Input (call):    - mel (type: float(0,inf), mandatory):
                              it is the current value in the mel-frequency
                              domain.



        Output (call):  hz (type: float):
                            it is the Hz value that corresponds to the
                            input melvalue.



        Example:    import torch
                    from speechbrain.nnet.architectures import SincConv

                    inp_tensor = torch.rand([4,32000])

                    # config dictionary definition
                    config={'class_name':'speechbrain.nnet.architectures.\
                        linear_combination',
                            'out_channels':'25',
                            'kernel_size': '129'}

                    # Initialization of the class
                    sincnet=SincConv(config,first_input=[inp_tensor])

                    # Executing computations
                    print(sincnet.to_hz(2146.06452750619))
        -----------------------------------------------------------------------
        """
        hz = 700 * (10 ** (mel / 2595) - 1)

        return hz

    @staticmethod
    def compute_conv1d_shape(L_in, kernel_size, dilation, stride):
        """
        -----------------------------------------------------------------------
        speechbrain.nnet.architectures.SincConv.compute_conv1d_shape
        (M. Ravanelli)

        Description: This support function can be use to compute the output
                     dimensionality of a 1D convolutional layer. It is used
                     to detect the number of padding elements.

        Input (call):    - L_in (type: int, mandatory):
                              it is the length of the input signal

                         - kernel_size (type: int, mandatory):
                              it is the kernel size used for the convolution

                         - dilation (type: int, mandatory):
                              it is the dilation factor used for the
                              convolution.

                          - stride (type: int, mandatory):
                              it is the dilation factor used for the
                              convolution.


        Output (call):  L_out (type: integer):
                            it is the length of the output.


     Example:   import torch
                from speechbrain.nnet.architectures import SincConv

                inp_tensor = torch.rand([4,32000])

                # config dictionary definition
                config={'class_name':'speechbrain.nnet.architectures.\
                    linear_combination',
                            'out_channels':'25',
                            'kernel_size': '129'}

                # Initialization of the class
                sincnet=SincConv(config,first_input=[inp_tensor])

                # Executing computations
                print(sincnet.compute_conv1d_shape(32000,129,1,1))

        -----------------------------------------------------------------------
        """

        L_out = (L_in - dilation * (kernel_size - 1) - 1) / stride + 1

        return int(L_out)

    def manage_padding(self, x, kernel_size, dilation, stride):
        """
        -----------------------------------------------------------------------
        speechbrain.nnet.architectures.SincConv.manage_padding (M. Ravanelli)

        Description: This function performs padding such that input and output
                     tensors have the same length.

        Input (call):    - x (type: torch.Tensor, mandatory):
                              it is the input (unpadded) tensor.

                         - kernel_size (type: int(1,inf), mandatory):
                              it is the kernel size used for the convolution

                         - dilation (type: int(1,inf), mandatory):
                              it is the dilation factor used for the
                              convolution.

                          - stride (type: int(1,inf), mandatory):
                              it is the dilation factor used for the
                              convolution.


        Output (call):  x (type: integer):
                            it is the output (padded) tensor


     Example:   import torch
                from speechbrain.nnet.architectures import SincConv

                inp_tensor = torch.rand([4,32000])

                # config dictionary definition
                config={'class_name':'speechbrain.nnet.architectures.\
                    linear_combination',
                            'out_channels':'25',
                            'kernel_size': '129'}

                # Initialization of the class
                sincnet=SincConv(config,first_input=[inp_tensor])

                # Executing computations
                padded_tensor=sincnet.manage_padding(inp_tensor.unsqueeze(0),\
                    129,1,1)
                print(inp_tensor.shape)
                print(padded_tensor.shape)


}


        -----------------------------------------------------------------------
        """

        # Computing input length
        L_in = x.shape[-1]

        # Computing the number of padding elements
        if stride > 1:
            n_steps = math.ceil(((L_in - kernel_size * dilation) / stride) + 1)
            L_out = stride * (n_steps - 1) + kernel_size * dilation
            padding = [kernel_size // 2, kernel_size // 2]

        else:
            L_in = x.shape[-1]
            L_out = self.compute_conv1d_shape(
                L_in, kernel_size, dilation, stride
            )
            padding = [(L_in - L_out) // 2, (L_in - L_out) // 2]

        # Performing padding
        x = nn.functional.pad(input=x, pad=tuple(padding), mode="reflect")
        return x


class RNN_basic(torch.nn.Module):
    """
     -------------------------------------------------------------------------
     nnet.architectures.RNN_basic (author: Mirco Ravanelli)

     Description:  This function implements basic RNN, LSTM and GRU models.

     Input: - rnn_type (type:rnn,lstm,gru,ligru,qrnn, mandatory):
               it is the type of recurrent neural network to
               use.

           - n_neurons (type: int(1,inf), mandatory):
               it is the number of output neurons (i.e, the
               dimensionality of the output).

           - nonlinearity (type:tanh, relu, mandatory):
               it is the type of nonlinearity.

           - num_layers (type: int(1,inf), mandatory):
               it is the number of layers.

           - bias (type: bool, Default:True):
               if True, the additive bias b is adopted.

           - dropout (type: float(0,1), optional:0.0):
               it is the dropout factor.

           - bidirectional (type: bool,
           Default:False):
               if True, a bidirectioal model is used.

     Example:   import torch
                from speechbrain.nnet.architectures import RNN_basic

                inp_tensor = torch.rand([4,100,190])

                # config dictionary definition
                config={'class_name':'speechbrain.nnet.architectures.linear',
                        'rnn_type': 'gru',
                        'n_neurons':'512',
                        'nonlinearity': 'tanh',
                        'num_layers': '1'}

                # Initialization of the class
                RNN=RNN_basic(config,first_input=[inp_tensor])

                # Executing computations
                inp_tensor = torch.rand([4,100,140])
                out_tensor = RNN([inp_tensor])
                print(out_tensor)
                print(out_tensor.shape)


    Reference:     [1] M. Ravanelli, P. Brakel, M. Omologo, Y. Bengio,
                   "Light Gated Recurrent Units for Speech Recognition",
                   in IEEE Transactions on Emerging Topics in Computational
                   Intelligence, 2018.
                   https://arxiv.org/abs/1803.10225

                   [2] J. Bradbury, S. Merity, and C. Xiong, R. Socher,
                   "Quasi-Recurrent Neural Networks", ICLR 2017
                    https://arxiv.org/abs/1611.01576

     """

    def __init__(
        self,
        rnn_type,
        n_neurons,
        nonlinearity,
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
        Arguments
        ---------
        first_input : tensor
            A dummy input of the right shape for initializing parameters.
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

        # Vanilla LSTM
        if self.rnn_type == "lstm":
            self.rnn = torch.nn.LSTM(**kwargs)

        # Vanilla GRU
        if self.rnn_type == "gru":
            self.rnn = torch.nn.GRU(**kwargs)

        # Light-GRU
        if self.rnn_type == "ligru":
            del kwargs["bias"]
            del kwargs["batch_first"]

            kwargs["batch_size"] = first_input.shape[0]
            kwargs["device"] = first_input.device
            self.rnn = liGRU(**kwargs)

        self.rnn.to(first_input.device)

    def forward(self, x, init_params=False):
        """
        Input: - x (type: torch.Tensor, mandatory):
                   by default the input arguments are passed with a list.
                   In this case, inp is a list containing a single
                   torch.tensor that we want to transform with the RNN.
                   The tensor must be in one of the following format:
                   [batch,channels,time]. Note that we can have up to
                   three channels.

        Output: - wx (type, torch.Tensor, mandatory):
                       it is the RNN output.
        """
        if init_params:
            self.init_params(x)

        # Reshaping input tensors when needed
        if self.reshape:
            if len(x.shape) == 4:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

            if len(x.shape) == 5:
                x = x.reshape(
                    x.shape[0], x.shape[1], x.shape[2] * x.shape[3], x.shape[4]
                )

        # Computing RNN steps
        output, hn = self.rnn(x)

        return output


class liGRU(torch.jit.ScriptModule):
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

        super(liGRU, self).__init__()

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
        x = x.transpose(0, 1)
        for ligru_lay in self.model:
            x = ligru_lay(x)

        x = x.transpose(0, 1)
        return x, 0


class liGRU_layer(torch.jit.ScriptModule):
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

        self.bn_w = nn.BatchNorm1d(2 * self.hidden_size, momentum=0.05).to(
            device
        )

        self.drop = torch.nn.Dropout(p=self.dropout, inplace=False).to(device)
        self.drop_mask_te = torch.tensor([1.0], device=device).float()
        self.N_drop_masks = 100
        self.drop_mask_cnt = 0

        if self.bidirectional:
            self.h_init = torch.zeros(
                2 * self.batch_size,
                self.hidden_size,
                requires_grad=False,
                device=device,
            )
            self.drop_masks = self.drop(
                torch.ones(
                    self.N_drop_masks,
                    2 * self.batch_size,
                    self.hidden_size,
                    device=device,
                )
            ).data

        else:
            self.h_init = torch.zeros(
                self.batch_size,
                self.hidden_size,
                requires_grad=False,
                device=device,
            )
            self.drop_masks = self.drop(
                torch.ones(
                    self.N_drop_masks,
                    self.batch_size,
                    self.hidden_size,
                    device=device,
                )
            ).data

        # Setting the activation function
        self.act = torch.nn.ReLU().to(device)

    @torch.jit.script_method
    def forward(self, x):

        if self.bidirectional:
            x_flip = x.flip(0)
            x = torch.cat([x, x_flip], dim=1)

        # Feed-forward affine transformations (all steps in parallel)
        w = self.w(x)

        # Apply batch normalization
        w_bn = self.bn_w(w.view(w.shape[0] * w.shape[1], w.shape[2]))

        w = w_bn.view(w.shape[0], w.shape[1], w.shape[2])

        # Processing time steps
        h = self.ligru_cell(w)

        if self.bidirectional:
            h_f, h_b = h.chunk(2, dim=1)
            h_b = h_b.flip(0)
            h = torch.cat([h_f, h_b], dim=2)

        return h

    @torch.jit.script_method
    def ligru_cell(self, w):

        hiddens = []
        ht = self.h_init

        if self.training:

            drop_mask = self.drop_masks[self.drop_mask_cnt]
            self.drop_mask_cnt = self.drop_mask_cnt + 1

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

        for k in range(w.shape[0]):

            gates = w[k] + self.u(ht)

            at, zt = gates.chunk(2, 1)
            # ligru equation
            zt = torch.sigmoid(zt)
            hcand = self.act(at) * drop_mask
            ht = zt * ht + (1 - zt) * hcand
            hiddens.append(ht)

        # Stacking hidden states
        h = torch.stack(hiddens)
        return h


class activation(torch.nn.Module):
    """
     -------------------------------------------------------------------------
     nnet.architectures.activation (author: Mirco Ravanelli)

     Description:  This function implements a set of activation functions
                   applied element-wise to the input tensor:
                   y = act(x)

     Input: - act_type (type:relu,leaky_relu,relu_6,r_relu,p_relu,elu,selu,
                 celu,hard_shrink,soft_shrink,softplus,soft_sign,threshold,
                 hard_tanh,tanh,tanh_shrink,sigmoid,log_sigmoid,softmax,
                 log_softmax,softmax2d,softmin,linear):
               it is the type of activation function to use

            - inplace (type: bool, Default: False):
               if True, it uses inplace operations.

           - negative_slope (type: float, Default: 0.01):
               it is the negative slope for leaky_relu
               activation. Controls the angle of the negative
               slope.

           - lower (type: float, Default: 0.125):
                It is used for RReLU. It is the lower bound of
                the uniform distribution.

           - upper (type: float, Default: 0.333):
               It is used for RReLU. It is the upper bound of
               the uniform distribution.

           - min_val (type: float, Default: -1.0):
               It is used for Hardtanh. It is minimum value of
               the linear region range.

            - max_val (type: float, Default: 1.0):
               It is used for Hardtanh. It is maximum value of
               the linear region range.

            - alpha (type: float, Default: 1.0):
               It is used for elu/celu. It is alpha value
               in the elu/celu formulation.

            - beta (type: float, Default: 1.0):
               It is used for softplus. It is beta value
               in the softplus formulation.

            - threshold (type: float, Default: 20.0):
               It is used for thresold and sofplus activations.
               It is corresponds to the threshold value.

            - lambd (type: float, Default: 0.5):
               It is used for soft_shrink and hard_shrink
               activations. It is corresponds to the lamda
               value of soft_shrink/hard_shrink activations.
               See torch.nn documentation.

            - value (type: float, Default: 0.5):
               It is used for the threshold function. it is
               the value taken when x<=threshold.


           - dim (type: int(1,inf), Default: -1):
               it is used in softmax activations to determine
               the axis on which the softmax is computed.


     Example:   import torch
                from speechbrain.nnet.architectures import activation

                # config dictionary definition
                config={'class_name':'speechbrain.nnet.architectures.activation',
                        'act_type': 'relu'}

                # Initialization of the class
                inp_tensor = torch.rand([4,100,190])-0.5
                apply_act=activation(config, first_input=[inp_tensor])

                # Executing computations
                out_tensor = apply_act([inp_tensor])
                print(out_tensor)
         ----------------------------------------------------------------------
     """

    # TODO: Consider making this less complex. (So many if blocks, flake8 complains)
    def __init__(  # noqa: C901
        self,
        act_type,
        inplace=False,
        negative_slope=0.01,
        lower=0.125,
        upper=0.33333333,
        min_val=-1.0,
        max_val=1.0,
        alpha=1.0,
        beta=1.0,
        threshold=20.0,
        lambd=0.5,
        value=0.5,
        dim=-1,
    ):
        super().__init__()

        self.act_type = act_type
        self.inplace = inplace
        self.negative_slope = negative_slope
        self.lower = lower
        self.upper = upper
        self.min_val = min_val
        self.max_val = max_val
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.lambd = lambd
        self.value = value
        self.dim = dim

        # Reshaping tensors can speed up some functions (e.g softmax)
        self.reshape = False

        if self.act_type == "relu":
            self.act = torch.nn.ReLU(inplace=self.inplace)

        if self.act_type == "leaky_relu":
            self.act = torch.nn.LeakyReLU(
                negative_slope=self.negative_slope, inplace=self.inplace
            )

        if self.act_type == "relu_6":
            self.act = torch.nn.ReLU6(inplace=self.inplace)

        if self.act_type == "r_relu":
            self.act = torch.nn.RReLU(
                lower=self.lower, upper=self.upper, inplace=self.inplace
            )

        if self.act_type == "elu":
            self.act = torch.nn.ELU(alpha=self.alpha, inplace=self.inplace)

        if self.act_type == "selu":
            self.act = torch.nn.SELU(inplace=self.inplace)

        if self.act_type == "celu":
            self.act = torch.nn.CELU(alpha=self.alpha, inplace=self.inplace)

        if self.act_type == "hard_shrink":
            self.act = torch.nn.Hardshrink(lambd=self.lambd)

        if self.act_type == "soft_shrink":
            self.act = torch.nn.Softshrink(lambd=self.lambd)

        if self.act_type == "softplus":
            self.act = torch.nn.Softplus(
                beta=self.beta, threshold=self.threshold
            )

        if self.act_type == "soft_sign":
            self.act = torch.nn.SoftSign()

        if self.act_type == "threshold":
            self.act = torch.nn.Threshold(
                self.threshold, self.value, inplace=self.inplace
            )

        if self.act_type == "hard_tanh":
            self.act = torch.nn.Hardtanh(
                min_val=self.min_val,
                max_val=self.max_val,
                inplace=self.inplace,
            )

        if self.act_type == "tanh":
            self.act = torch.nn.Tanh()

        if self.act_type == "tanh_shrink":
            self.act = torch.nn.Tanhshrink()

        if self.act_type == "sigmoid":
            self.act = torch.nn.Sigmoid()

        if self.act_type == "log_sigmoid":
            self.act = torch.nn.LogSigmoid()
            self.reshape = True

        if self.act_type == "softmax":
            self.act = torch.nn.Softmax(dim=self.dim)
            self.reshape = True

        if self.act_type == "log_softmax":
            self.act = torch.nn.LogSoftmax(dim=self.dim)
            self.reshape = True

        if self.act_type == "softmax2d":
            self.act = torch.nn.Softmax2d()

        if self.act_type == "softmin":
            self.act = torch.nn.Softmin(dim=self.dim)
            self.reshape = True

    def forward(self, x):
        """
        Input: - x (type: torch.Tensor, mandatory):
                   torch.tensor that we want to transform with the RNN.
                   The tensor must be in one of the following format:
                   [batch,channels,time]. Note that we can have up to
                   three channels.

        Output: - wx (type, torch.Tensor, mandatory):
                   it is the output after applying element-wise the
                   activation function.
        """

        if self.act_type == "linear":
            return x

        # Reshaping the tensor when needed
        if self.reshape:
            dims = x.shape

            if len(dims) == 3:
                x = x.reshape(dims[0] * dims[1], dims[2])

            if len(dims) == 4:
                x = x.reshape(dims[0] * dims[1], dims[2], dims[3])

            if len(dims) == 5:
                x = x.reshape(dims[0] * dims[1], dims[2], dims[3], dims[4])

        # Applying the activation function
        x_act = self.act(x)

        # Retrieving the original shape format
        if self.reshape:

            if len(dims) == 3:
                x_act = x_act.reshape(dims[0], dims[1], dims[2])

            if len(dims) == 4:
                x_act = x_act.reshape(dims[0], dims[1], dims[2], dims[3])

            if len(dims) == 5:
                x_act = x_act.reshape(
                    dims[0], dims[1], dims[2], dims[3], dims[4]
                )

        return x_act


class dropout(nn.Module):
    """
     -------------------------------------------------------------------------
     nnet.architectures.dropout (author: Mirco Ravanelli)

     Description:  This function implements droput of the input tensor:
                   y = dropout(x).

     Input (init):  - config (type, dict, mandatory):
                       it is a dictionary containing the keys described below.

                           - act_type (type:relu,leaky_relu,relu_6,r_relu,
                                       p_relu,elu, selu,celu,hard_shrink,
                                       soft_shrink,softplus, soft_sign,
                                       threshold,hard_tanh,tanh, tanh_shrink,
                                       sigmoid,log_sigmoid,softmax,
                                       log_softmax,softmax2d,softmin,linear):
                               it is the type of activation function to use

                            - inplace (type: bool, Default:False):
                               if True, it uses inplace operations.

                            - drop_rate (type: float(0,1),
                            Default:0.0):
                               it is the dropout factor.

                   - funct_name (type, str, default: None):
                       it is a string containing the name of the parent
                       function that has called this method.

                   - global_config (type, dict, default: None):
                       it a dictionary containing the global variables of the
                       parent config file.

                   - logger (type, logger, default: None):
                       it the logger used to write debug and error messages.
                       If logger=None and root_cfg=True, the file is created
                       from scratch.

                   - first_input (type, list, default: None)
                      this variable allows users to analyze the first input
                      given when calling the class for the first time.


     Input (call): - inp_lst(type, list, mandatory):
                       by default the input arguments are passed with a list.
                       In this case, inp is a list containing a single
                       torch.tensor that we want to transform with the RNN.
                       The tensor must be in one of the following format:
                       [batch,channels,time]. Note that we can have up to
                       three channels.



     Output (call): - wx(type, torch.Tensor, mandatory):
                       it is the output after applying element-wise the
                       activation function.


     Example:   import torch
                from speechbrain.nnet.architectures import dropout

                # config dictionary definition
                config={'class_name':'speechbrain.nnet.architectures.dropout',
                        'drop_rate': '0.5'}

                # Initialization of the class
                inp_tensor = torch.rand([4,100,190])
                drop_act=dropout(config, first_input=[inp_tensor])

                # Executing computations
                out_tensor = drop_act([inp_tensor])
                print(out_tensor)

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

        # Dropout initialization
        if len(first_input.shape) <= 3:
            self.drop = nn.Dropout(p=self.drop_rate, inplace=self.inplace)

        if len(first_input.shape) == 4:
            self.drop = nn.Dropout2d(p=self.drop_rate, inplace=self.inplace)

        if len(first_input.shape) == 5:
            self.drop = nn.Dropout3d(p=self.drop_rate, inplace=self.inplace)

    def forward(self, x, init_params=False):
        if init_params:
            self.init_params(x)

        # Avoing the next steps in dropuut_rate is 0
        if self.drop_rate == 0.0:
            return x

        # time must be the last
        x = x.transpose(1, 2).transpose(2, -1)

        # Applying dropout
        x_drop = self.drop(x)

        x_drop = x_drop.transpose(-1, 1).transpose(2, -1)

        return x_drop


class pooling(nn.Module):
    """
     -------------------------------------------------------------------------
     nnet.architectures.dropout (author: Titouan Parcollet)

     Description:  This function implements pooling of the input tensor:
                   y = pooling(x).

     Input (init):  - config (type, dict, mandatory):
                       it is a dictionary containing the keys described below.

                            - pool_type (type: str, max):
                                it is the type of pooling function to use.

                            - pool_axis (type: int_list(0, 3), mandatory):
                                it is a list containing the axis that will be
                                considered during pooling. It must match the
                                dimensionality of the pooling. If the pooling
                                is 2D, then a list of 2 indices is expected.

                            - kernel_size (type: int_list, mandatory):
                                it is the kernel size. Note that it also define
                                the pooling dimension. 3 is a 1D Pooling with a
                                kernel of size 3, while 3,3 is a 2D Pooling.

                            - stride (type: int, optional, Default:1):
                                it is the stride size.

                            - padding (type: int, optional, Default:0):
                                it is the padding to apply.

                            - dilation (type: int, optional, Default:1):
                                controls the dilation of pooling.

                            - ceil_mode (type: bool, optional, Default:False):
                                when True, will use ceil instead of floor
                                to compute the output shape

                   - funct_name (type, str, optional, default: None):
                       it is a string containing the name of the parent
                       function that has called this method.

                   - global_config (type, dict, optional, default: None):
                       it a dictionary containing the global variables of the
                       parent config file.

                   - logger (type, logger, optional, default: None):
                       it the logger used to write debug and error messages.
                       If logger=None and root_cfg=True, the file is created
                       from scratch.

                   - first_input (type, list, optional, default: None)
                      this variable allows users to analyze the first input
                      given when calling the class for the first time.


     Input (call): - inp_lst(type, list, mandatory):
                       by default the input arguments are passed with a list.
                       In this case, inp is a list containing a single
                       torch.tensor that we want to transform with the pooling.
                       The tensor must be in any form since the pooling axis
                       are specified. By default, the input is of shape:
                       [batch,features,channels,time]



     Output (call): - wx(type, torch.Tensor, mandatory):
                       it is the output after applying the pooling.


     Example:   import torch
                from speechbrain.nnet.architectures import pooling

                 # config dictionary definition
                 config={'class_name':'speechbrain.nnet.architectures.pooling',
                 'pool_type': 'max',
                 'kernel_size': '2,2',
                 'stride': '2',
                 'pool_axis': '1,2'}

                 # Initialization of the class
                 inp_tensor = torch.rand([1,4,1,4])
                 pool = pooling(config, first_input=[inp_tensor])

                 # Executing computations
                 out_tensor = pool([inp_tensor])
                 print(out_tensor)
    """

    def __init__(
        self,
        pool_type,
        kernel_size,
        pool_axis=1,
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

        # Option for pooling
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

    def forward(self, x, init_params=False):
        if init_params:
            self.init_params(x)

        x = x.transpose(1, 2).transpose(2, -1)
        or_shape = x.shape

        # Put the pooling axes as the last dimension for torch.nn.pool
        # If the input tensor is 4 dimensional, combine the first and second
        # axes together to respect the input shape of torch.nn.pool1d
        if self.pool1d:
            x = x.transpose(len(or_shape) - 1, self.pool_axis[0])
            new_shape = x.shape
            if self.combine_batch_time:
                x = x.reshape(new_shape[0] * new_shape[1], new_shape[2], -1)

        if self.pool2d:
            x = x.transpose(len(or_shape) - 2, self.pool_axis[0]).transpose(
                len(or_shape) - 1, self.pool_axis[1]
            )

        # Apply pooling
        x = self.pool_layer(x)

        # Recover input shape
        if self.pool1d:
            if self.combine_batch_time:
                x = x.reshape(new_shape[0], new_shape[1], new_shape[2], -1)
            x = x.transpose(len(or_shape) - 1, self.pool_axis[0])

        if self.pool2d:
            x = x.transpose(len(or_shape) - 2, self.pool_axis[0]).transpose(
                len(or_shape) - 1, self.pool_axis[1]
            )

        x = x.transpose(-1, 1).transpose(2, -1)
        return x
