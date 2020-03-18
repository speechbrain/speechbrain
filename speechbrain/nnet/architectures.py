"""
 -----------------------------------------------------------------------------
 architectures.py

 Description: This library contains the most popular neural architectures that
              can be used to process audio and speech signals.
 -----------------------------------------------------------------------------
"""

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.data_io.data_io import recovery, initialize_with
from speechbrain.utils.input_validation import check_opts, check_inputs
from speechbrain.utils.logger import logger_write


class linear(nn.Module):
    """
     -------------------------------------------------------------------------
     nnet.architectures.linear (author: Mirco Ravanelli)

     Description:  This function implements a fully connected linear layer:
                   y = Wx + b.

     Input (init):  - config (type, dict, mandatory):
                       it is a dictionary containing the keys described below.

                           - n_neurons (type: int(1,inf), mandatory):
                               it is the number of output neurons (i.e, the
                               dimensionality of the output)

                           - bias (type: bool, optional, Default:True):
                               if True, the additive bias b is adopted.

                           - recovery (type: bool, optional, Default:True):
                               if True, the system restarts from the last
                               epoch correctly executed.

                           - initialize_with (type: str, optional, Default:\
                               None):
                               when set, this flag can be used to initialize
                               the parameters with an external pkl file. It
                               could be useful for pre-training purposes.


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
                       torch.tensor that we want to transform linearly.
                       The tensor must be in one of the following format:
                       [batch,channels,time]. Note that we can have up to
                       three channels.



     Output (call): - wx(type, torch.Tensor, mandatory):
                       The output is a tensor that corresponds to the linear
                       transformation of the input tensor.


     Example:   import torch
                from speechbrain.nnet.architectures import linear

                inp_tensor = torch.rand([4,660,190])

                # config dictionary definition
                config={'class_name':'speechbrain.nnet.architectures.linear',
                        'n_neurons':'1024'}

                # Initialization of the class
                linear_transf=linear(config,first_input=[inp_tensor])

                # Executing computations
                inp_tensor = torch.rand([4,660,120])
                out_tensor = linear_transf([inp_tensor])
                print(out_tensor)
                print(out_tensor.shape)

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
        super(linear, self).__init__()

        # Logger setup
        self.logger = logger

        # Here are summarized the expected options for this class
        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "recovery": ("bool", "optional", "True"),
            "initialize_with": ("str", "optional", "None"),
            "n_neurons": ("int(1,inf)", "mandatory"),
            "bias": ("bool", "optional", "True"),
        }

        # Check, cast, and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, self.logger
        )

        # Definition of the expected input
        self.expected_inputs = ["torch.Tensor"]

        # Check the first input
        check_inputs(
            self.conf, self.expected_inputs, first_input, logger=self.logger
        )

        # Output folder (useful for parameter saving)
        if global_config is not None:
            self.output_folder = global_config["output_folder"]
        self.funct_name = funct_name

        # Additional check on the input shapes
        if first_input is not None:

            # Shape check
            if len(first_input[0].shape) > 5 or len(first_input[0].shape) < 2:

                err_msg = (
                    'The input of "linear" must be a tensor with one of the  '
                    "following dimensions: [time] or [batch,time] or "
                    "[batch,channels,time]. Got %s "
                    % (str(first_input[0].shape))
                )

                logger_write(err_msg, logfile=logger)

        self.first_call = True

    def forward(self, input_lst):

        # Reading input _list
        x = input_lst[0]

        if self.first_call is True:
            self.first_call = False

            # Computing the dimensionality of the input
            fea_dim = x.shape[1]

            # Initialization of the parameters
            self.w = nn.Linear(fea_dim, self.n_neurons, bias=self.bias)

            # Managing initialization with an external model
            # (useful for pre-training)
            initialize_with(self)

            # Automatic recovery (when needed)
            recovery(self)

        # Transposing tensor
        x = x.transpose(1, -1)

        # Apply linear transformation
        wx = self.w(x)

        # Going back to the original shape format
        wx = wx.transpose(1, -1)

        return wx


class linear_combination(nn.Module):
    """
     -------------------------------------------------------------------------
     nnet.architectures.linear_combination (author: Mirco Ravanelli)

     Description:  This function implements a linear combination between n
                   inputs (with combination weights learned). For instance,
                   for three inputs x,y,z the output o will be:
                   o = Wx + My + z. The weights W, M are learned.
                   This function can be used to create shorcuts between
                   layers of a neural architecture (e.g. dense/skip
                   connections).

     Input (init):  - config (type, dict, mandatory):
                       it is a dictionary containing the keys described below.

                           - bias (type: bool, optional, Default:True):
                               if True, the additive bias b is adopted.

                           - recovery (type: bool, optional, Default:True):
                               if True, the system restarts from the last
                               epoch correctly executed.

                           - initialize_with (type: str, optional, Default:\
                               None):
                               when set, this flag can be used to initialize
                               the parameters with an external pkl file. It
                               could be useful for pre-training purposes.


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
                       In this case, inp is a list containing the n
                       torch.tensor elements that we want to combine.
                       The tensor must be in one of the following format:
                       [batch,channels,time]. Note that we can have up to
                       three channels.



     Output (call): - wx(type, torch.Tensor, mandatory):
                       The output is a tensor that corresponds to linear
                       combination of the n inputs.


     Example:   import torch
                from speechbrain.nnet.architectures import linear_combination

                inp1 = torch.rand([4,100,190])
                inp2 = torch.rand([4,200,190])
                inp3 = torch.rand([4,50,190])

                # config dictionary definition
                config={'class_name':'speechbrain.nnet.architectures.\
                    linear_combination'}

                # Initialization of the class
                linear_comb=linear_combination(config,\
                    first_input=[inp1,inp2,inp3])

                # Executing computations
                inp1 = torch.rand([4,100,130])
                inp2 = torch.rand([4,200,130])
                inp3 = torch.rand([4,50,130])

                out_tensor = linear_comb([inp1,inp2,inp3])
                print(out_tensor)
                print(out_tensor.shape)

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
        super(linear_combination, self).__init__()

        # Logger setup
        self.logger = logger

        # Here are summarized the expected options for this class
        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "recovery": ("bool", "optional", "True"),
            "initialize_with": ("str", "optional", "None"),
            "bias": ("bool", "optional", "True"),
        }

        # Check, cast, and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, self.logger
        )

        # Output folder (useful for parameter saving)
        if global_config is not None:
            self.output_folder = global_config["output_folder"]
        self.funct_name = funct_name

        # Additional check on the input shapes
        if first_input is not None:

            # Shape check
            if len(first_input[0].shape) > 5 or len(first_input[0].shape) < 2:

                err_msg = (
                    'The first input of "linear_combination" must be a tensor'
                    " with  one of the   following dimensions: [time] or"
                    " [batch,time] or  [batch,channels,time]. Got %s "
                    % (str(first_input[0].shape))
                )

                logger_write(err_msg, logfile=logger)

            # Shape check
            if len(first_input[1].shape) > 5 or len(first_input[1].shape) < 2:

                err_msg = (
                    'The second input of "linear_combination" must be a tensor'
                    " with  one of the   following dimensions: [time] or"
                    " [batch,time] or  [batch,channels,time]. Got %s "
                    % (str(first_input[0].shape))
                )

                logger_write(err_msg, logfile=logger)

        # Initializing the matrices
        dim_out = first_input[-1].shape[1]

        # Initialization of the parameters
        self.w = nn.ModuleList([])
        for i in range(len(first_input) - 1):
            self.w.append(
                nn.Linear(first_input[i].shape[1], dim_out, bias=self.bias)
            )

        # Managing initialization with an external model
        # (useful for pre-training)
        initialize_with(self)

        # Automatic recovery
        if global_config is not None:
            recovery(self)

    def forward(self, input_lst):

        # Reading reference input
        x = input_lst[-1]

        # Transposing the input
        x = x.transpose(1, -1)

        for i in range(len(input_lst) - 1):

            x_inp = input_lst[i].to(x.device)
            x_inp = x_inp.transpose(1, -1)

            # Apply linear transformation
            wx = self.w[i](x_inp)

            if i == 0:
                wx_tot = []

            # Managing time reduction case (when stride >1)
            if x.shape[1] != wx.shape[1]:

                time_red_factor = (wx.shape[1]) / x.shape[1]
                add_time_steps = (
                    math.ceil(time_red_factor) * x.shape[1]
                ) - wx.shape[1]

                shape_add = list(wx.shape)
                shape_add[1] = add_time_steps

                wx = torch.cat(
                    [wx, torch.zeros(tuple(shape_add)).to(x.device)], dim=1
                )

                wx = wx.reshape(x.shape[0], -1, x.shape[1], wx.shape[2])
                wx = torch.mean(wx, dim=1)

            # Appending transformation
            wx_tot.append(wx)

        # Combination of transformed inputs
        wx_tot = torch.mean(torch.stack(wx_tot), dim=0)

        # Final combination
        wx_tot = wx_tot + x

        # Going back to the original shape format
        wx = wx.transpose(1, -1)

        return wx


class conv(nn.Module):
    """
     -------------------------------------------------------------------------
     nnet.architectures.conv (author: Mirco Ravanelli)

     Description:  This function implements 1D or 2D convolutional layers.

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

                           - stride (type: int_list(1,inf), optional: \
                               default: 1,1):
                               it is a list containing the stride factors.
                               For 1D convolutions, the list contains a single
                               integer (stride over the time axis), while
                               for 2D convolutions the list is composed of two
                               values (i.e, time and frequenecy kernel sizes,
                               respectively). When the stride factor > 1, a
                               decimantion (in the time or frequnecy domain) is
                               implicitely performed.

                           - dilation (type: int_list(1,inf), optional: \
                               default: 1,1):
                               it is a list containing the dilation factors.
                               For 1D convolutions, the list contains a single
                               integer (dilation over the time axis), while
                               for 2D convolutions the list is composed of two
                               values (i.e, time and frequenecy kernel sizes,
                               respectively).

                           - padding (type: int_list(1,inf), optional: \
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

                           - padding_mode (one_of(circular,zeros), optional: \
                               default: zeros):
                               This flag specifies the type of padding.
                               See torch.nn documentation for more information.

                            - groups (type:int(1,inf), optional: \
                                default: zeros):
                               This option specifies the convolutional groups.
                               See torch.nn documentation for more information.


                           - bias (type: bool, optional, default:True):
                               if True, the additive bias b is adopted.

                           - recovery (type: bool, optional, default:True):
                               if True, the system restarts from the last
                               epoch correctly executed.

                           - initialize_with (type: str, optional, \
                               default:None):
                               when set, this flag can be used to initialize
                               the parameters with an external pkl file. It
                               could be useful for pre-training purposes.


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
                       In this case, inp is a list containing the
                       torch.tensor element that we want to transform.
                       The tensor must be in one of the following format:
                       [batch,channels,time]. Note that we can have up to
                       three channels.



     Output (call): - wx(type, torch.Tensor, mandatory):
                       The output is a tensor that corresponds to the convolved
                       input.


     Example:   import torch
                from speechbrain.nnet.architectures import conv

                inp_tensor = torch.rand([4,100,190])

                # config dictionary definition
                config={'class_name':'speechbrain.nnet.architectures.\
                    linear_combination',
                        'out_channels':'25',
                        'kernel_size': '11'}

                # Initialization of the class
                cnn=conv(config,first_input=[inp_tensor])

                # Executing computations
                inp_tensor = torch.rand([4,100,190])
                out_tensor = cnn([inp_tensor])
                print(out_tensor)
                print(out_tensor.shape)

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
        super(conv, self).__init__()

        # Logger setup
        self.logger = logger

        # Here are summarized the expected options for this class
        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "recovery": ("bool", "optional", "True"),
            "initialize_with": ("str", "optional", "None"),
            "out_channels": ("int(1,inf)", "mandatory"),
            "kernel_size": ("int_list(1,inf)", "mandatory"),
            "stride": ("int_list(1,inf)", "optional", "1,1"),
            "dilation": ("int_list(1,inf)", "optional", "1,1"),
            "padding": ("int_list(0,inf)", "optional", "None"),
            "groups": ("int(1,inf)", "optional", "1"),
            "bias": ("bool", "optional", "True"),
            "padding_mode": ("one_of(zeros,circular)", "optional", "zeros"),
        }

        # Check, cast, and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, self.logger
        )

        # Definition of the expected input
        self.expected_inputs = ["torch.Tensor"]

        # Check the first input
        check_inputs(
            self.conf, self.expected_inputs, first_input, logger=self.logger
        )

        # Output folder (useful for parameter saving)
        if global_config is not None:
            self.output_folder = global_config["output_folder"]
        self.funct_name = funct_name
        self.reshape_conv1d = False
        self.reshape_conv2d = False
        self.squeeze_conv2d = False
        self.transp_conv2d = False

        # Making sure that the kernel size is odd (if the kernel is not
        # symmetric there could a problem with the padding function)
        for size in self.kernel_size:
            if size % 2 == 0:
                err_msg = (
                    "The field kernel size must be and odd number. Got %s."
                    % (self.kernel_size)
                )

                logger_write(err_msg, logfile=logger)

        # Checking if 1d or 2d is specified
        self.conv1d = False
        self.conv2d = False

        if len(self.kernel_size) == 1:
            self.conv1d = True

        if len(self.kernel_size) == 2:
            self.conv2d = True

        # Additional check on the input shapes
        if first_input is not None:

            # Shape check
            if len(first_input[0].shape) > 5 or len(first_input[0].shape) < 2:

                err_msg = (
                    'The input of "linear" must be a tensor with one of the  '
                    "following dimensions: [time] or [batch,time] or "
                    "[batch,channels,time]. Got %s "
                    % (str(first_input[0].shape))
                )

                logger_write(err_msg, logfile=logger)

            # Manage reshaping flags
            if len(first_input[0].shape) > 3:
                if self.conv1d:
                    self.reshape_conv1d = True

            if len(first_input[0].shape) > 4:
                if self.conv2d:
                    self.reshape_conv2d = True

            if len(first_input[0].shape) == 3 and self.conv2d:
                self.squeeze_conv2d = True

            if len(first_input[0].shape) >= 4 and self.conv2d:
                self.transp_conv2d = True

            # Detecting the number of input channels
            if self.conv1d:
                self.in_channels = first_input[0].shape[1]

            if self.conv2d:
                if len(first_input[0].shape) == 3:
                    self.in_channels = 1

                elif len(first_input[0].shape) == 4:

                    self.in_channels = first_input[0].shape[2]
                elif len(first_input[0].shape) == 5:
                    self.in_channels = (
                        first_input[0].shape[2] * first_input[0].shape[3]
                    )

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
            )

        # Managing 2d convolutions
        if self.conv2d:

            if self.padding is not None:
                self.padding = self.padding[0:-1]

            # Initialization of the parameters
            self.conv = nn.Conv2d(
                self.in_channels,
                self.out_channels,
                tuple(self.kernel_size),
                stride=tuple(self.stride),
                padding=0,
                dilation=tuple(self.dilation),
                groups=self.groups,
                bias=self.bias,
                padding_mode=self.padding_mode,
            )

        # Managing initialization with an external model
        # (useful for pre-training)
        initialize_with(self)

        # Automatic recovery
        if global_config is not None:
            recovery(self)

    def forward(self, input_lst):

        # Reading input _list
        x = input_lst[0]

        # Reshaping the inputs when needed
        if self.reshape_conv1d:
            or_shape = x.shape

            if len(or_shape) == 4:
                x = x.reshape(
                    or_shape[0] * or_shape[2], or_shape[1], or_shape[-1]
                )

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

                           - sample_rate (type: int(1,inf), optional,
                               default: 16000):
                               it is the sampling frequency of the input
                               waveform.

                            - min_low_hz (type: float(0,inf), optional,
                                default: 50):
                               it is the mininum frequnecy (in Hz) that a
                               learned filter can have.

                            - min_band_hz (type: float(0,inf), optional,
                                default: 50):
                               it is the minimum band (in Hz) that a learned
                               filter can have.


                           - recovery (type: bool, optional, default:True):
                               if True, the system restarts from the last
                               epoch correctly executed.

                           - initialize_with (type: str, optional,
                           default:None):
                               when set, this flag can be used to initialize
                               the parameters with an external pkl file. It
                               could be useful for pre-training purposes.


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
            "recovery": ("bool", "optional", "True"),
            "initialize_with": ("str", "optional", "None"),
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

        # Check, cast, and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, self.logger
        )

        # Definition of the expected input
        self.expected_inputs = ["torch.Tensor"]

        # Check the first input
        check_inputs(
            self.conf, self.expected_inputs, first_input, logger=self.logger
        )

        if global_config is not None:
            self.output_folder = global_config["output_folder"]

        # Additional check on the input shapes
        if first_input is not None:

            # Shape check
            if len(first_input[0].shape) > 2:

                err_msg = (
                    "SincConv only support one input channel (here, \
                        in_channels = {%i})"
                    % (len(first_input[1:-1].shape))
                )

                logger_write(err_msg, logfile=logger)

        # Forcing the filters to be odd (i.e, perfectly symmetric)
        if self.kernel_size % 2 == 0:
            err_msg = (
                "The field kernel size must be and odd number. Got %s."
                % (self.kernel_size)
            )

            logger_write(err_msg, logfile=logger)

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

        # Managing initialization with an external model
        # (useful for pre-training)
        initialize_with(self)

        # Automatic recovery
        if global_config is not None:
            recovery(self)

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


class RNN_basic(nn.Module):
    """
     -------------------------------------------------------------------------
     nnet.architectures.RNN_basic (author: Mirco Ravanelli)

     Description:  This function implements basic RNN, LSTM and GRU models.

     Input (init):  - config (type, dict, mandatory):
                       it is a dictionary containing the keys described below.

                           - rnn_type (type:rnn,lstm,gru,ligru,qrnn,
                           mandatory):
                               it is the type of recurrent neural network to
                               use.

                           - n_neurons (type: int(1,inf), mandatory):
                               it is the number of output neurons (i.e, the
                               dimensionality of the output).

                           - nonlinearity (type:tanh, relu, mandatory):
                               it is the type of nonlinearity.

                           - num_layers (type: int(1,inf), mandatory):
                               it is the number of layers.

                           - bias (type: bool, optional, Default:True):
                               if True, the additive bias b is adopted.

                           - dropout (type: float(0,1), optional:0.0):
                               it is the dropout factor.

                           - bidirectional (type: bool, optional,
                           Default:False):
                               if True, a bidirectioal model is used.

                           - recovery (type: bool, optional, Default:True):
                               if True, the system restarts from the last
                               epoch correctly executed.

                           - initialize_with (type: str, optional,
                           Default:None):
                               when set, this flag can be used to initialize
                               the parameters with an external pkl file. It
                               could be useful for pre-training purposes.


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
                       torch.tensor that we want to transform with the RNN.
                       The tensor must be in one of the following format:
                       [batch,channels,time]. Note that we can have up to
                       three channels.



     Output (call): - wx(type, torch.Tensor, mandatory):
                       it is the RNN output.


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
        config,
        funct_name=None,
        global_config=None,
        functions=None,
        logger=None,
        first_input=None,
    ):
        super(RNN_basic, self).__init__()

        # Logger setup
        self.funct_name = funct_name
        self.logger = logger

        # Here are summarized the expected options for this class
        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "recovery": ("bool", "optional", "True"),
            "initialize_with": ("str", "optional", "None"),
            "rnn_type": ("one_of(rnn,lstm,gru,ligru,qrnn)", "mandatory"),
            "n_neurons": ("int(1,inf)", "mandatory"),
            "nonlinearity": ("one_of(tanh,relu)", "mandatory"),
            "num_layers": ("int(1,inf)", "optional", "1"),
            "bias": ("bool", "optional", "True"),
            "dropout": ("float(0,1)", "optional", "0.0"),
            "bidirectional": ("bool", "optional", "False"),
        }

        # Check, cast, and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, self.logger
        )

        # Definition of the expected input
        self.expected_inputs = ["torch.Tensor"]

        # Check the first input
        check_inputs(
            self.conf, self.expected_inputs, first_input, logger=self.logger
        )

        if global_config is not None:
            self.output_folder = global_config["output_folder"]

        self.reshape = False

        # Check input dimensionality
        if first_input is not None:

            # Shape check
            if len(first_input[0].shape) > 5 or len(first_input[0].shape) < 3:

                err_msg = (
                    'The input of "RNN_basic" must be a tensor with one of '
                    "the  following dimensions: [time] or [batch,time] or "
                    "[batch,channels,time]. Got %s "
                    % (str(first_input[0].shape))
                )

                logger_write(err_msg, logfile=logger)

                if len(first_input[0].shape) > 3:
                    self.reshape = True

        self.first_call = True

    def forward(self, input_lst):

        # Reading input _list
        x = input_lst[0]

        if self.first_call is True:
            self.first_call = False

            # Computing the feature dimensionality
            self.fea_dim = torch.prod(torch.tensor(x.shape[1:-1]))

            # Vanilla RNN
            if self.rnn_type == "rnn":
                self.rnn = torch.nn.RNN(
                    input_size=self.fea_dim,
                    hidden_size=self.n_neurons,
                    nonlinearity=self.nonlinearity,
                    num_layers=self.num_layers,
                    bias=self.bias,
                    dropout=self.dropout,
                    bidirectional=self.bidirectional,
                )
            # Vanilla LSTM
            if self.rnn_type == "lstm":
                self.rnn = torch.nn.LSTM(
                    input_size=self.fea_dim,
                    hidden_size=self.n_neurons,
                    num_layers=self.num_layers,
                    bias=self.bias,
                    dropout=self.dropout,
                    bidirectional=self.bidirectional,
                )
            # Vanilla GRU
            if self.rnn_type == "gru":
                self.rnn = torch.nn.GRU(
                    input_size=self.fea_dim,
                    hidden_size=self.n_neurons,
                    num_layers=self.num_layers,
                    bias=self.bias,
                    dropout=self.dropout,
                    bidirectional=self.bidirectional,
                )
            # Vanilla light-GRU
            if self.rnn_type == "ligru":
                self.rnn = liGRU(
                    input_size=self.fea_dim,
                    hidden_size=self.n_neurons,
                    num_layers=self.num_layers,
                    dropout=self.dropout,
                    bidirectional=self.bidirectional,
                )

            # Quasi RNN
            if self.rnn_type == "qrnn":

                # Check if qrnn (quasi-rnn) library is installed
                try:
                    from torchqrnn import QRNN
                except Exception:
                    err_msg = (
                        "QRNN is not installed. Please run "
                        "pip install cupy pynvrtc \
                            git+https://github.com/salesforce/pytorch-qrnn ."
                        "Go to https://github.com/salesforce/pytorch-qrnn \
                            for more info."
                    )
                    logger_write(err_msg, logfile=logger)

                # Needed to avoid qrnn warnings
                import warnings

                warnings.filterwarnings("ignore")

                self.rnn = QRNN(
                    input_size=self.fea_dim,
                    hidden_size=self.n_neurons,
                    num_layers=self.num_layers,
                    dropout=self.dropout,
                )

            # Managing initialization with an external model
            # (useful for pre-training)
            initialize_with(self)

            # Automatic recovery
            recovery(self)

        # Reshaping input tensors when needed
        if self.reshape:
            if len(x.shape) == 4:
                x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])

            if len(x.shape) == 5:
                x = x.reshape(
                    x.shape[0],
                    x.shape[1] * x.shape[2] * x.shape[3],
                    x.shape[4],
                )

        # Transposing input
        x = x.permute(2, 0, 1)

        # Computing RNN steps
        output, hn = self.rnn(x)

        # Tensor transpose
        output = output.permute(1, 2, 0)

        return output


class liGRU(nn.Module):
    """
     -------------------------------------------------------------------------
     nnet.architectures.liGRU (author: Mirco Ravanelli)

     Description:  This function implements the Light-GRU model.


     Input (init):
                    - input_size (type:int(1,inf), mandatory):
                        it is the dimensionality of the input.

                    - hidden_size (type: int(1,inf), mandatory):
                        it is the number of output neurons (i.e, the
                        dimensionality of the output).

                    - nonlinearity (type:tanh, relu, mandatory):
                        it is the type of nonlinearity.

                    - num_layers (type: int(1,inf), mandatory):
                        it is the number of layers.

                    - bidirectional (type: bool, optional, Default:False):
                        if True, a bidirectioal model is used.


     Input (call): - x (type, torch.Tensor, mandatory):
                       x is torch.tensor that we want to transf with the RNN.
                       The tensor must be in one of the following format:
                       [batch,channels,time]. Note that we can have up to
                       three channels.



     Output (call): - wx(type, torch.Tensor, mandatory):
                       it is the RNN output.


     Example:   import torch
                from speechbrain.nnet.architectures import liGRU

                # Initialization of the class
                model=liGRU(100,200,1,0.15,'relu',True)

                # Executing computations
                inp_tensor = torch.rand([4,100,140])
                out_tensor = model(inp_tensor.permute(2,0,1))
                print(out_tensor)
                print(out_tensor[0].shape)


    Reference:     [1] M. Ravanelli, P. Brakel, M. Omologo, Y. Bengio,
                   "Light Gated Recurrent Units for Speech Recognition",
                   in IEEE Transactions on Emerging Topics in Computational
                   Intelligence, 2018.
                   https://arxiv.org/abs/1803.10225

     """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout=0.0,
        nonlinearity="relu",
        bidirectional=False,
    ):
        super(liGRU, self).__init__()

        # Parameter list initialization
        self.wh = nn.ModuleList([])
        self.uh = nn.ModuleList([])

        # Update Gate
        self.wz = nn.ModuleList([])
        self.uz = nn.ModuleList([])

        # Layer Norm
        self.ln = nn.ModuleList([])

        # Batch Norm
        self.bn_wh = nn.ModuleList([])
        self.bn_wz = nn.ModuleList([])

        # Activations
        self.act = nn.ModuleList([])

        # RNN parameters
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.dropout = dropout

        current_dim = input_size

        # Initializing multiple layers
        for i in range(self.num_layers):

            # Setting the activation function
            if nonlinearity == "relu":
                self.act.append(torch.nn.ReLU())

            if nonlinearity == "tanh":
                self.act.append(torch.nn.Tanh())

            # Feed-forward connections
            self.wh.append(
                nn.Linear(current_dim, self.hidden_size, bias=False)
            )
            self.wz.append(
                nn.Linear(current_dim, self.hidden_size, bias=False)
            )

            # Recurrent connections
            self.uh.append(
                nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            )
            self.uz.append(
                nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            )

            # Adding orthogonal initialization for recurrent connection
            nn.init.orthogonal_(self.uh[i].weight)
            nn.init.orthogonal_(self.uz[i].weight)

            # batch norm initialization
            self.bn_wh.append(nn.BatchNorm1d(self.hidden_size, momentum=0.05))
            self.bn_wz.append(nn.BatchNorm1d(self.hidden_size, momentum=0.05))

            # updating current dimension
            current_dim = self.hidden_size

            if self.bidirectional:
                current_dim = 2 * self.hidden_size

    def forward(self, x):

        # Loop over all the layers
        for i in range(self.num_layers):

            # Initial state and concatenation
            if self.bidirectional:
                h_init = torch.zeros(2 * x.shape[1], self.hidden_size)
                x = torch.cat([x, self.flip(x, 0)], 1)
            else:
                h_init = torch.zeros(x.shape[1], self.hidden_size)

            # Dropout mask initilization (same mask for all time steps)
            if self.training:
                drop_mask = torch.bernoulli(
                    torch.Tensor(h_init.shape[0], h_init.shape[1]).fill_(
                        1.0 - self.dropout
                    )
                )
            else:
                drop_mask = torch.FloatTensor([1.0 - self.dropout])

            h_init = h_init.to(x.device)
            drop_mask = drop_mask.to(x.device)

            # Feed-forward affine transformations (all steps in parallel)
            wh = self.wh[i](x)
            wz = self.wz[i](x)

            # Apply batch normalization
            wh_bn = self.bn_wh[i](
                wh.view(wh.shape[0] * wh.shape[1], wh.shape[2])
            )
            wh = wh_bn.view(wh.shape[0], wh.shape[1], wh.shape[2])

            wz_bn = self.bn_wz[i](
                wz.view(wz.shape[0] * wz.shape[1], wz.shape[2])
            )
            wz = wz_bn.view(wz.shape[0], wz.shape[1], wz.shape[2])

            # Processing time steps
            hiddens = []
            ht = h_init
            for k in range(x.shape[0]):

                # ligru equation
                zt = torch.sigmoid(wz[k] + self.uz[i](ht))
                at = wh[k] + self.uh[i](ht)
                hcand = self.act[i](at) * drop_mask
                ht = zt * ht + (1 - zt) * hcand

                hiddens.append(ht)

            # Stacking hidden states
            h = torch.stack(hiddens)

            # Bidirectional concatenations
            if self.bidirectional:
                h_f = h[:, 0: int(x.shape[1] / 2)]
                h_b = self.flip(
                    h[:, int(x.shape[1] / 2): x.shape[1]].contiguous(), 0
                )
                h = torch.cat([h_f, h_b], 2)

            # Setup x for the next hidden layer
            x = h

        return x, ht

    @staticmethod
    def flip(x, dim):
        """
         ----------------------------------------------------------------------
         nnet.architectures.liGRU.flip (author: Mirco Ravanelli)

         Description:  This support function flips the an input tensor along
                       the specified dimension. It can be used to implement
                       bidirectional models.


         Input (init):
                        - x (type:torch.Tensor, mandatory):
                            it is the dimensionality of the input.

                        - dim (type: int(0,inf), mandatory):
                            it is the axes to flip.




         Output (call): - x(type, torch.Tensor, mandatory):
                           it is the RNN output.


         Example:   import torch
                    from speechbrain.nnet.architectures import liGRU

                    # Initialization of the class
                    model=liGRU(100,200,1,0.15,'relu',True)

                    inp_tensor = torch.rand([4,100,140])

                    print(inp_tensor)
                    print(model.flip(inp_tensor,-1))

         """
        # Computing the input shape
        xsize = x.size()

        dim = x.dim() + dim if dim < 0 else dim

        x = x.contiguous()

        x = x.view(-1, *xsize[dim:])

        # Getting the flipped version
        x = x.view(x.size(0), x.size(1), -1)[
            :,
            getattr(
                torch.arange(x.size(1) - 1, -1, -1), ("cpu", "cuda")[x.is_cuda]
            )().long(),
            :,
        ]

        return x.view(xsize)


class activation(nn.Module):
    """
     -------------------------------------------------------------------------
     nnet.architectures.activation (author: Mirco Ravanelli)

     Description:  This function implements a set of activation functions
                   applied element-wise to the input tensor:
                   y = act(x)

     Input (init):  - config (type, dict, mandatory):
                       it is a dictionary containing the keys described below.

                           - act_type (type:relu,leaky_relu,relu_6,r_relu,
                                       p_relu,elu, selu,celu,hard_shrink,
                                       soft_shrink,softplus, soft_sign,
                                       threshold,hard_tanh,tanh, tanh_shrink,
                                       sigmoid,log_sigmoid,softmax,
                                       log_softmax,softmax2d,softmin,linear):
                               it is the type of activation function to use

                            - inplace (type: bool, optional, Default:False):
                               if True, it uses inplace operations.

                           - negative_slope (type: float, optional,
                           Default:0.01):
                               it is the negative slope for leaky_relu
                               activation. Controls the angle of the negative
                               slope.

                           - lower (type: float, optional, Default:0.125):
                                It is used for RReLU. It is the lower bound of
                                the uniform distribution.

                           - upper (type: float, optional, Default:0.333):
                               It is used for RReLU. It is the upper bound of
                               the uniform distribution.

                           - min_val (type: float, optional, Default:-1.0):
                               It is used for Hardtanh. It is minimum value of
                               the linear region range.

                            - max_val (type: float, optional, Default:1.0):
                               It is used for Hardtanh. It is maximum value of
                               the linear region range.

                            - alpha (type: float, optional, Default:1.0):
                               It is used for elu/celu. It is alpha value
                               in the elu/celu formulation.

                            - beta (type: float, optional, Default:1.0):
                               It is used for softplus. It is beta value
                               in the softplus formulation.

                            - threshold (type: float, optional, Default:20.0):
                               It is used for thresold and sofplus activations.
                               It is corresponds to the threshold value.

                            - lambd (type: float, optional, Default:0.5):
                               It is used for soft_shrink and hard_shrink
                               activations. It is corresponds to the lamda
                               value of soft_shrink/hard_shrink activations.
                               See torch.nn documentation.

                            - value (type: float, optional, Default:0.5):
                               It is used for the threshold function. it is
                               the value taken when x<=threshold.


                           - dim (type: int(1,inf), optional, Default:-1):
                               it is used in softmax activations to determine
                               the axis on which the softmax is computed.


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
                       torch.tensor that we want to transform with the RNN.
                       The tensor must be in one of the following format:
                       [batch,channels,time]. Note that we can have up to
                       three channels.



     Output (call): - wx(type, torch.Tensor, mandatory):
                       it is the output after applying element-wise the
                       activation function.


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

    def __init__(
        self,
        config,
        funct_name=None,
        global_config=None,
        functions=None,
        logger=None,
        first_input=None,
    ):
        super(activation, self).__init__()

        # Logger setup
        self.logger = logger

        # Here are summarized the expected options for this class
        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "act_type": (
                "one_of(relu,leaky_relu,relu_6,r_relu,p_relu,elu,"
                + "selu,celu,hard_shrink,soft_shrink,softplus,"
                + "soft_sign,threshold,hard_tanh,tanh,"
                + "tanh_shrink,sigmoid,log_sigmoid,softmax,"
                + "log_softmax,softmax2d,softmin,linear)",
                "mandatory",
            ),
            "inplace": ("bool", "optional", "False"),
            "negative_slope": ("float", "optional", "0.01"),
            "lower": ("float", "optional", "0.125"),
            "upper": ("float", "optional", "0.33333333"),
            "min_val": ("float", "optional", "-1.0"),
            "max_val": ("float", "optional", "1.0"),
            "alpha": ("float", "optional", "1.0"),
            "beta": ("float", "optional", "1.0"),
            "threshold": ("float", "optional", "20.0"),
            "lambd": ("float", "optional", "0.5"),
            "value": ("float", "optional", "0.5"),
            "dim": ("int", "optional", "-1"),
        }

        # Check, cast, and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, self.logger
        )

        # Definition of the expected input
        self.expected_inputs = ["torch.Tensor"]

        # Check the first input
        check_inputs(
            self.conf, self.expected_inputs, first_input, logger=self.logger
        )

        # Reshaping tensors can speed up some functions (e.g softmax)
        self.reshape = False

        # Additional check on the input shapes
        if first_input is not None:

            # Shape check
            if len(first_input[0].shape) > 5 or len(first_input[0].shape) < 1:

                err_msg = (
                    'The input of "activation must be a tensor with one of '
                    "the following dimensions: [batch,time] or "
                    "[batch,channels,time]. Got %s "
                    % (str(first_input[0].shape))
                )

                logger_write(err_msg, logfile=logger)

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
            self.act = torch.nn.CELU(
                alpha=self.alpha, inplace=self.inplace
            )

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

    def forward(self, input_lst):

        # Reading input _list
        x = input_lst[0]

        if self.act_type == "linear":
            return x

        # Reshaping the tensor when needed
        if self.reshape:
            x = x.transpose(1, -1)
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

            x_act = x_act.transpose(1, -1)

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

                            - inplace (type: bool, optional, Default:False):
                               if True, it uses inplace operations.

                            - drop_rate (type: float(0,1), optional,
                            Default:0.0):
                               it is the dropout factor.

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
        self,
        config,
        funct_name=None,
        global_config=None,
        functions=None,
        logger=None,
        first_input=None,
    ):
        super(dropout, self).__init__()

        # Logger setup
        self.logger = logger

        # Here are summarized the expected options for this class
        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "drop_rate": ("float(0,1)", "mandatory"),
            "inplace": ("bool", "optional", "False"),
        }

        # Check, cast, and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, self.logger
        )

        # Definition of the expected input
        self.expected_inputs = ["torch.Tensor"]

        # Check the first input
        check_inputs(
            self.conf, self.expected_inputs, first_input, logger=self.logger
        )

        # reshaping 3d tensor in input to 1d dropput is faster
        self.reshape = False

        # Additional check on the input shapes
        if first_input is not None:

            # Shape check
            if len(first_input[0].shape) > 5 or len(first_input[0].shape) < 1:

                err_msg = (
                    'The input of "dropout" must be a tensor with one of the  '
                    "following dimensions: [batch,time] or "
                    "[batch,channels,time]. Got %s "
                    % (str(first_input[0].shape))
                )

                logger_write(err_msg, logfile=logger)

            # Dropout initialization
            if len(first_input[0].shape) <= 3:
                self.drop = nn.Dropout(p=self.drop_rate, inplace=self.inplace)

                if len(first_input[0].shape) == 3:
                    self.reshape = True

            if len(first_input[0].shape) == 4:
                self.drop = nn.Dropout2d(
                    p=self.drop_rate, inplace=self.inplace
                )

            if len(first_input[0].shape) == 5:
                self.drop = nn.Dropout3d(
                    p=self.drop_rate, inplace=self.inplace
                )

    def forward(self, input_lst):

        # Reading input _list
        x = input_lst[0]

        # Avoing the next steps in dropuut_rate is 0
        if self.drop_rate == 0.0:
            return x

        # Reshaping tensor when needed
        if self.reshape:

            x = x.transpose(1, -1)
            dims = x.shape

            x = x.reshape(dims[0] * dims[1], dims[2])

        # Applying dropout
        x_drop = self.drop(x)

        # Retrieving the original shape format
        if self.reshape:
            x_drop = x_drop.reshape(dims[0], dims[1], dims[2])
            x_drop = x_drop.transpose(1, -1)

        return x_drop
