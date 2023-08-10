"""LMDA-Net from https://doi.org/10.1016/j.neuroimage.2023.120209.
Lightweight and interpretable convolutional neural network proposed for a general decoding of single-trial EEG signals.
It was proposed for P300, and motor imagery decoding.

Authors
 * Davide Borra, 2023
"""
import torch
import speechbrain as sb


class EEGDepthAttention(torch.nn.Module):
    """
    Depth attention mechanism from https://doi.org/10.1016/j.neuroimage.2023.120209.

    Arguments
    ---------
    input_shape : tuple
        The shape of the input.
    cnn_depth_attn_kernelsize : tuple
        The kernel size of the convolutional layer. The convolution is applied along feature maps dimension.
    """

    def __init__(self, input_shape, cnn_depth_attn_kernelsize=(1, 7)):
        super(EEGDepthAttention, self).__init__()
        _, T, C, _ = input_shape
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((T, 1))
        self.conv = sb.nnet.CNN.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=cnn_depth_attn_kernelsize,
            padding="same",
            padding_mode="constant",
            bias=True,
            swap=True,
        )
        self.softmax = torch.nn.Softmax(dim=-2)

    def forward(self, x):
        """Returns the output of the EEGDepthAttention module.

        Arguments
        ---------
        x : torch.Tensor (batch, time, EEG channel, channel)
            Input to convolve. 4d tensors are expected.
        """
        D = x.shape[-1]  # num of feature maps

        y = x.transpose(-1, -3)  # (N, T, C, D) -> (N, D, C, T)
        y = y.transpose(-1, -2)
        y = self.adaptive_pool(y)
        y = y.transpose(-1, -2)
        y = y.transpose(-1, -3)  # -> (N, T, C, D)

        y = y.transpose(-2, -1)  # -> (N, T, D, C)
        y = self.conv(y)
        y = self.softmax(y)
        y = y.transpose(-2, -1)
        return y * D * x


class LMDA(torch.nn.Module):
    """
    LMDA-Net.

    Arguments
    ---------
    input_shape: tuple
        The shape of the input.
    cnn_ch_attn_kernels: int
        Number of kernels for the attention mechanism across EEG channels ("channel attention" in the paper).
    cnn_temporal_kernels: int
        Number of kernels in the temporal convolution.
    cnn_temporal_kernelsize: tuple
        Kernel size of the temporal convolution.
    cnn_spatial_kernels: int
        Number of kernels in the spatial convolution.
    cnn_depth_attn_kernelsize : tuple
        Kernel size for the attention mechanism across feature maps ("depth attention" in the paper).
    cnn_pool: tuple
        Pool size and stride.
    cnn_pool_type: string
        Pooling type.
    dropout: float
        Dropout probability.
    dense_n_neurons: int
        Number of output neurons.
    activation_type: str
        Activation function of the hidden layers.

    Example
    -------
    #>>> inp_tensor = torch.rand([1, 1000, 32, 1])
    #>>> model = LMDA(input_shape=inp_tensor.shape)
    #>>> output = model(inp_tensor)
    #>>> output.shape
    #torch.Size([1,4])
    """

    def __init__(
        self,
        input_shape=None,  # (1, T, C, 1)
        cnn_ch_attn_kernels=9,
        cnn_temporal_kernels=24,
        cnn_temporal_kernelsize=(75, 1),
        cnn_spatial_kernels=9,
        cnn_depth_attn_kernelsize=(1, 7),
        cnn_pool=(5, 1),
        cnn_pool_type="avg",
        dropout=0.65,
        dense_n_neurons=4,
        activation_type="gelu",
    ):
        super().__init__()
        if input_shape is None:
            raise ValueError("Must specify input_shape")

        if activation_type == "gelu":
            activation = torch.nn.GELU()
        elif activation_type == "elu":
            activation = torch.nn.ELU()
        elif activation_type == "relu":
            activation = torch.nn.ReLU()
        elif activation_type == "leaky_relu":
            activation = torch.nn.LeakyReLU()
        elif activation_type == "prelu":
            activation = torch.nn.PReLU()
        else:
            raise ValueError("Wrong hidden activation function")
        self.default_sf = 250  # sampling rate of the original publication (Hz)
        # T = input_shape[1]
        C = input_shape[2]
        # CONV MODULE
        # EEG channel attention mechanism
        self.channel_weight = torch.nn.Parameter(
            torch.randn(C, 1, cnn_ch_attn_kernels), requires_grad=True
        )
        torch.nn.init.xavier_uniform_(self.channel_weight.data)
        # Temporal convolutional module
        self.time_conv_module = torch.nn.Sequential()
        self.time_conv_module.add_module(
            "conv_0",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_ch_attn_kernels,
                out_channels=cnn_temporal_kernels,
                kernel_size=(1, 1),
                groups=1,
                padding="valid",
                bias=False,
                swap=True,
            ),
        )
        self.time_conv_module.add_module(
            "bnorm_0",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_temporal_kernels, momentum=0.01, affine=True,
            ),
        )
        self.time_conv_module.add_module(
            "conv_1",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_temporal_kernels,
                out_channels=cnn_temporal_kernels,
                kernel_size=cnn_temporal_kernelsize,
                groups=cnn_temporal_kernels,
                padding="valid",
                bias=False,
                swap=True,
            ),
        )
        self.time_conv_module.add_module(
            "bnorm_1",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_temporal_kernels, momentum=0.01, affine=True,
            ),
        )
        self.time_conv_module.add_module("act_1", activation)
        # Spatial convolutional module
        self.channel_conv_module = torch.nn.Sequential()
        self.channel_conv_module.add_module(
            "conv_0",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_temporal_kernels,
                out_channels=cnn_spatial_kernels,
                kernel_size=(1, 1),
                groups=1,
                padding="valid",
                bias=False,
                swap=True,
            ),
        )
        self.channel_conv_module.add_module(
            "bnorm_0",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_spatial_kernels, momentum=0.01, affine=True,
            ),
        )
        self.channel_conv_module.add_module(
            "conv_1",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_spatial_kernels,
                out_channels=cnn_spatial_kernels,
                kernel_size=(1, C),
                groups=cnn_spatial_kernels,
                padding="valid",
                bias=False,
                swap=True,
            ),
        )
        self.channel_conv_module.add_module(
            "bnorm_1",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_spatial_kernels, momentum=0.01, affine=True,
            ),
        )
        self.channel_conv_module.add_module("act_1", activation)
        self.channel_conv_module.add_module(
            "pool_1",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=cnn_pool,
                stride=cnn_pool,
                pool_axis=[1, 2],
            ),
        )
        self.channel_conv_module.add_module(
            "dropout_1", torch.nn.Dropout(p=dropout)
        )

        # Shape of intermediate feature maps
        out = torch.ones((1,) + tuple(input_shape[1:-1]) + (1,))
        out = torch.einsum("bwcd, cdh->bwch", out, self.channel_weight)
        out = self.time_conv_module(out)
        input_shape_depth_attn = (
            out.shape
        )  # storing shape of this tensor that represents the input for depth attention mechanism
        out = self.channel_conv_module(out)
        dense_input_size = self._num_flat_features(out)

        # Depth attention mechanism
        self.depth_attn = EEGDepthAttention(
            input_shape=input_shape_depth_attn,
            cnn_depth_attn_kernelsize=cnn_depth_attn_kernelsize,
        )

        # DENSE MODULE
        self.dense_module = torch.nn.Sequential()
        self.dense_module.add_module(
            "flatten", torch.nn.Flatten(),
        )
        self.dense_module.add_module(
            "fc_out",
            sb.nnet.linear.Linear(
                input_size=dense_input_size, n_neurons=dense_n_neurons,
            ),
        )
        self.dense_module.add_module("act_out", torch.nn.LogSoftmax(dim=1))

    def _num_flat_features(self, x):
        """Returns the number of flattened features from a tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input feature map.
        """

        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        """Returns the output of the model.

        Arguments
        ---------
        x : torch.Tensor (batch, time, EEG channel, channel)
            Input to convolve. 4d tensors are expected.
        """

        x = torch.einsum(
            "bwcd, cdh->bwch", x, self.channel_weight
        )  # EEG channel attention
        x = self.time_conv_module(x)  # temporal convolution
        x = self.depth_attn(x)  # depth attention
        x = self.channel_conv_module(x)  # spatial convolution
        x = self.dense_module(x)
        return x
