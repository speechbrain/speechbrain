"""ShallowConvNet from https://doi.org/10.1002/hbm.23730.
Shallow convolutional neural network proposed for motor execution and motor imagery decoding from single-trial EEG signals.
Its design is based on the filter bank common spatial pattern (FBCSP) algorithm.

Authors
 * Davide Borra, 2021
"""
import torch
import speechbrain as sb


class ShallowConvNet(torch.nn.Module):
    """ShallowConvNet.

    Arguments
    ---------
    input_shape : tuple
        The shape of the input.
    cnn_temporal_kernels : int
        Number of kernels in the 2d temporal convolution.
    cnn_temporal_kernelsize : tuple
        Kernel size of the 2d temporal convolution.
    cnn_spatial_kernels : int
        Number of kernels in the 2d spatial depthwise convolution.
    cnn_poolsize: tuple
        Pool size.
    cnn_poolstride: tuple
        Pool stride.
    cnn_pool_type: string
        Pooling type.
    dropout: float
        Dropout probability.
    dense_n_neurons: int
        Number of output neurons.
        
    Example
    -------
    >>> inp_tensor = torch.rand([1, 200, 32, 1])
    >>> model = ShallowConvNet(input_shape=inp_tensor.shape)
    >>> output = model(inp_tensor)
    >>> output.shape
    torch.Size([1,4])
    """

    def __init__(
        self,
        input_shape=None,
        cnn_temporal_kernels=40,
        cnn_temporal_kernelsize=(25, 1),
        cnn_spatial_kernels=40,
        cnn_poolsize=(75, 1),
        cnn_poolstride=(15, 1),
        cnn_pool_type="avg",
        dropout=0.5,
        dense_n_neurons=4,
    ):
        super().__init__()
        if input_shape is None:
            raise ValueError("Must specify input_shape")
        self.default_sf = 250  # sampling rate of the original publication (Hz)

        T = input_shape[1]
        C = input_shape[2]
        # CONVOLUTIONAL MODULE
        self.conv_module = torch.nn.Sequential()
        # Temporal convolution
        self.conv_module.add_module(
            "conv_0",
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=cnn_temporal_kernels,
                kernel_size=cnn_temporal_kernelsize,
                padding="valid",
                bias=True,
                swap=True,
            ),
        )

        # Spatial convolution
        self.conv_module.add_module(
            "conv_1",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_temporal_kernels,
                out_channels=cnn_spatial_kernels,
                kernel_size=(1, C),
                padding="valid",
                bias=False,
                swap=True,
            ),
        )
        self.conv_module.add_module(
            "bnorm_1",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_spatial_kernels, momentum=0.1, affine=True,
            ),
        )

        self.pool = sb.nnet.pooling.Pooling2d(
            pool_type=cnn_pool_type,
            kernel_size=cnn_poolsize,
            stride=cnn_poolstride,
            pool_axis=[1, 2],
        )

        self.dropout = torch.nn.Dropout(p=dropout)

        # DENSE MODULE
        dense_input_size = (T - cnn_temporal_kernelsize[0]) + 1
        dense_input_size = cnn_spatial_kernels * (
            int((dense_input_size - cnn_poolsize[0]) / cnn_poolstride[0]) + 1
        )
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

    def forward(self, x):
        """Returns the output of the model.

        Arguments
        ---------
        x : torch.Tensor (batch, time, EEG channel, channel)
            input to convolve. 4d tensors are expected.
        """
        x = self.conv_module(x)
        # square-pool-log-dropout module
        x = torch.square(x)  # conv non-lin
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))  # pool non-lin
        x = self.dropout(x)
        x = self.dense_module(x)
        return x
