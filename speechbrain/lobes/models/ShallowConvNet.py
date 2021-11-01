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
    sf : int
        The sampling frequency of the input EEG signals (Hz).
    cnn_temporal_kernels : int
        Number of kernels in the 2d temporal convolution.
    cnn_temporal_kernelsize : tuple
        Kernel size of the 2d temporal convolution.
    cnn_spatial_kernels : int
        Number of kernels in the 2d spatial depthwise convolution.
    poolsize: tuple
        Pool size.
    poolstride: tuple
        Pool stride.
    dropout: float
        Dropout probability.
    dense_n_neurons: int
        Number of output neurons.
    """

    def __init__(
        self,
        input_shape=None,
        sf=250,
        cnn_temporal_kernels=40,
        cnn_temporal_kernelsize=(25, 1),
        cnn_spatial_kernels=40,
        poolsize=(75, 1),
        poolstride=(15, 1),
        dropout=0.5,
        dense_n_neurons=4,
    ):
        super().__init__()
        if input_shape is None:
            raise ValueError("Must specify input_shape")
        self.default_sf = 250  # sampling rate of the original publication (Hz)
        # scaling temporal kernel and pooling sizes/strides if different sampling rate was used
        # respect to the original publication
        if sf != self.default_sf:
            tmp_cnn_temporal_kernelsize = int(
                cnn_temporal_kernelsize[0] * (sf / self.default_sf)
            )
            if tmp_cnn_temporal_kernelsize % 2 == 0:  # odd sizes
                tmp_cnn_temporal_kernelsize += 1
            cnn_temporal_kernelsize = (tmp_cnn_temporal_kernelsize, 1)
            tmp_poolsize = int(poolsize[0] * (sf / self.default_sf))
            poolsize = (tmp_poolsize, 1)
            tmp_poolstride = int(poolstride[0] * (sf / self.default_sf))
            poolstride = (tmp_poolstride, 1)

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
                transpose=True,
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
                transpose=True,
            ),
        )
        self.conv_module.add_module(
            "bnorm_1",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_spatial_kernels, momentum=0.1, affine=True,
            ),
        )

        self.pool = sb.nnet.pooling.Pooling2d(
            pool_type="avg",
            kernel_size=poolsize,
            stride=poolstride,
            pool_axis=[1, 2],
        )

        self.dropout = torch.nn.Dropout(p=dropout)

        # DENSE MODULE
        Tconv0 = (T - cnn_temporal_kernelsize[0]) + 1
        dense_input_size = cnn_spatial_kernels * (
            int((Tconv0 - poolsize[0]) / poolstride[0]) + 1
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
