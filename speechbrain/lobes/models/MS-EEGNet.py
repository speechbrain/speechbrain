"""MultiScale-EEGNet from https://doi.org/10.3389/fnhum.2021.655840.
Lightweight multi-scale convolutional neural network proposed for the decoding of single-trial EEG signals.
It was proposed for P300 decoding.

Authors
 * Davide Borra, 2021
"""
import torch
import speechbrain as sb


class MultiScaleSeparableTemporal(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        cnn_septemporal_depth_multipliers,
        cnn_septemporal_kernelsizes,
        cnn_septemporal_pool,
        activation,
        dropout,
    ):
        super().__init__()

        self.ntails = len(cnn_septemporal_kernelsizes)
        self.ms_septemporal_kernels = []
        tails = []
        for i in range(self.ntails):
            cnn_septemporal_kernelsize = cnn_septemporal_kernelsizes[i]
            cnn_septemporal_depth_multiplier = cnn_septemporal_depth_multipliers[
                i
            ]

            tail = torch.nn.Sequential()
            tail.add_module(
                "conv_0",
                sb.nnet.CNN.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=cnn_septemporal_kernelsize,
                    groups=in_channels,
                    padding="same",
                    padding_mode="constant",
                    bias=False,
                ),
            )
            cnn_septemporal_kernels = (
                in_channels * cnn_septemporal_depth_multiplier
            )
            self.ms_septemporal_kernels.append(cnn_septemporal_kernels)
            tail.add_module(
                "conv_1",
                sb.nnet.CNN.Conv2d(
                    in_channels=in_channels,
                    out_channels=cnn_septemporal_kernels,
                    kernel_size=(1, 1),
                    padding="valid",
                    bias=False,
                ),
            )

            tail.add_module(
                "bnorm_0",
                sb.nnet.normalization.BatchNorm2d(
                    input_size=cnn_septemporal_kernels,
                    momentum=0.01,
                    affine=True,
                ),
            )
            tail.add_module("act_0", activation)
            tail.add_module(
                "avg_pool_0",
                sb.nnet.pooling.Pooling2d(
                    pool_type="avg",
                    kernel_size=cnn_septemporal_pool,
                    stride=cnn_septemporal_pool,
                    pool_axis=[1, 2],
                ),
            )
            tail.add_module("dropout_0", torch.nn.Dropout(p=dropout))
            tails.append(tail)
        self.tails = torch.nn.ModuleList(tails)

    def forward(self, x):
        """Returns the output of the multi-scale feature extractor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, EEG channel, channel)
            input to convolve. 4d tensors are expected.
        """
        tail_outputs = []
        for tail in self.tails:
            tail_outputs.append(tail(x))
        return torch.cat(tail_outputs, 1)


class MSEEGNet(torch.nn.Module):
    """MS-EEGNet.

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
    cnn_spatial_depth_multiplier : int
        Depth multiplier of the 2d spatial depthwise convolution.
    cnn_spatial_max_norm : float
        Kernel max norm of the 2d spatial depthwise convolution.
    cnn_spatial_pool: tuple
        Pool size and stride after the 2d spatial depthwise convolution.
    cnn_septemporal_depth_multipliers: list
        List of depth multipliers of the 2d temporal separable convolution in the multi-scale feature extractor.
    cnn_septemporal_kernelsizes: list
        List of kernel sizes of the 2d temporal separable convolution in the multi-scale feature extractor.
    cnn_septemporal_pool: tuple
        Pool size and stride after the 2d temporal separable convolution.
    dropout: float
        Dropout probability.
    dense_max_norm: float
        Weight max norm of the fully-connected layer.
    dense_n_neurons: int
        Number of output neurons.
    activation: torch.nn.??
        Activation function of the hidden layers.
    """

    def __init__(
        self,
        input_shape=None,  # (1, T, C, 1)
        sf=128,
        cnn_temporal_kernels=8,
        cnn_temporal_kernelsize=(33, 1),
        cnn_spatial_depth_multiplier=2,
        cnn_spatial_max_norm=1.0,
        cnn_spatial_pool=(4, 1),
        cnn_septemporal_depth_multipliers=[1, 1],
        cnn_septemporal_kernelsizes=[(17, 1), (5, 1)],
        cnn_septemporal_pool=(8, 1),
        dropout=0.5,
        dense_max_norm=0.25,
        dense_n_neurons=4,
        activation=torch.nn.ELU(),
    ):
        super().__init__()
        if input_shape is None:
            raise ValueError("Must specify input_shape")
        self.default_sf = 128  # sampling rate of the original publication (Hz)
        # scaling temporal kernel and pooling sizes/strides if different sampling rate was used
        # respect to the original publication
        if sf != self.default_sf:
            tmp_cnn_temporal_kernelsize = int(
                cnn_temporal_kernelsize[0] * (sf / self.default_sf)
            )
            if tmp_cnn_temporal_kernelsize % 2 == 0:
                tmp_cnn_temporal_kernelsize += 1
            cnn_temporal_kernelsize = (tmp_cnn_temporal_kernelsize, 1)
            tmp_cnn_spatial_pool = int(
                cnn_spatial_pool[0] * (sf / self.default_sf)
            )
            cnn_spatial_pool = (tmp_cnn_spatial_pool, 1)

        T = input_shape[1]
        C = input_shape[2]
        self.conv_module = torch.nn.Sequential()
        # CONVOLUTIONAL MODULE
        # Temporal convolution
        self.conv_module.add_module(
            "conv_0",
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=cnn_temporal_kernels,
                kernel_size=cnn_temporal_kernelsize,
                padding="same",
                padding_mode="constant",
                bias=False,
            ),
        )
        self.conv_module.add_module(
            "bnorm_0",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_temporal_kernels, momentum=0.01, affine=True,
            ),
        )
        # Spatial depthwise convolution
        cnn_spatial_kernels = (
            cnn_spatial_depth_multiplier * cnn_temporal_kernels
        )
        self.conv_module.add_module(
            "conv_1",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_temporal_kernels,
                out_channels=cnn_spatial_kernels,
                kernel_size=(1, C),
                groups=cnn_temporal_kernels,
                padding="valid",
                bias=False,
                max_norm=cnn_spatial_max_norm,
            ),
        )
        self.conv_module.add_module(
            "bnorm_1",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_spatial_kernels, momentum=0.01, affine=True,
            ),
        )
        self.conv_module.add_module("act_1", activation)
        self.conv_module.add_module(
            "avg_pool_1",
            sb.nnet.pooling.Pooling2d(
                pool_type="avg",
                kernel_size=cnn_spatial_pool,
                stride=cnn_spatial_pool,
                pool_axis=[1, 2],
            ),
        )
        self.conv_module.add_module("dropout_1", torch.nn.Dropout(p=dropout))
        # Multi-scale temporal separable convolution
        ms_septemporal = MultiScaleSeparableTemporal(
            in_channels=cnn_spatial_kernels,
            cnn_septemporal_depth_multipliers=cnn_septemporal_depth_multipliers,
            cnn_septemporal_kernelsizes=cnn_septemporal_kernelsizes,
            cnn_septemporal_pool=cnn_septemporal_pool,
            activation=activation,
            dropout=dropout,
        )
        self.conv_module.add_module("ms_0", ms_septemporal)

        # DENSE MODULE
        self.dense_module = torch.nn.Sequential()
        self.dense_module.add_module(
            "flatten", torch.nn.Flatten(),
        )
        dense_input_size = torch.sum(self.ms_septemporal_kernels) * int(
            T / (cnn_spatial_pool[0] * cnn_septemporal_pool[0])
        )
        self.dense_module.add_module(
            "fc_out",
            sb.nnet.linear.Linear(
                input_size=dense_input_size,
                n_neurons=dense_n_neurons,
                max_norm=dense_max_norm,
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
        x = self.dense_module(x)
        return x
