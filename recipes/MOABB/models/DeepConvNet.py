"""DeepConvNet from https://doi.org/10.1002/hbm.23730.
Deep convolutional neural network proposed for motor execution and motor imagery decoding from single-trial EEG signals.

Authors
 * Davide Borra, 2021
"""
import torch
import speechbrain as sb


class DeepConvNet(torch.nn.Module):
    """DeepConvNet.

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
    cnn_spatial_pool : tuple
        Pool size and stride after the 2d spatial convolution.
    cnn_temporal_block_kernel0 : int
        Number of kernels in the 2d temporal convolution of the first block.
    cnn_temporal_block_kernel1 : int
        Number of kernels in the 2d temporal convolution of the second block.
    cnn_temporal_block_kernel2 : int
        Number of kernels in the 2d temporal convolution of the third block.
    cnn_temporal_block_kernelsize0 : tuple
        Kernel size of the 2d temporal convolution of the first block.
    cnn_temporal_block_kernelsize1 : tuple
        Kernel size of the 2d temporal convolution of the second block.
    cnn_temporal_block_kernelsize2 : tuple
        Kernel size of the 2d temporal convolution of the third block.
    cnn_temporal_block_pool : tuple
        Pool size and stride after each block.
    cnn_pool_type: string
        Pooling type.
    activation: torch.nn.??
        Activation function of the hidden layers.
    dropout: float
        Dropout probability.
    dense_n_neurons: int
        Number of output neurons.

    Example
    -------
    >>> inp_tensor = torch.rand([1, 1000, 32, 1])
    >>> model = DeepConvNet(input_shape=inp_tensor.shape)
    >>> output = model(inp_tensor)
    >>> output.shape
    torch.Size([1,4])
    """

    def __init__(
        self,
        input_shape=None,
        cnn_temporal_kernels=25,
        cnn_temporal_kernelsize=(10, 1),
        cnn_spatial_kernels=25,
        cnn_spatial_pool=(3, 1),
        cnn_temporal_block_kernel0=50,
        cnn_temporal_block_kernel1=100,
        cnn_temporal_block_kernel2=200,
        cnn_temporal_block_kernelsize0=(10, 1),
        cnn_temporal_block_kernelsize1=(10, 1),
        cnn_temporal_block_kernelsize2=(10, 1),
        cnn_temporal_block_pool=(3, 1),
        cnn_pool_type="max",
        activation=torch.nn.ELU(),
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
                input_size=cnn_spatial_kernels,
                affine=True,
                momentum=0.1,
                eps=1e-5,
            ),
        )
        self.conv_module.add_module("act_1", activation)

        self.conv_module.add_module(
            "pool_1",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=cnn_spatial_pool,
                stride=cnn_spatial_pool,
                pool_axis=[1, 2],
            ),
        )

        cnn_temporal_block_kernels = [
            cnn_temporal_block_kernel0,
            cnn_temporal_block_kernel1,
            cnn_temporal_block_kernel2,
        ]
        cnn_temporal_block_kernelsizes = [
            cnn_temporal_block_kernelsize0,
            cnn_temporal_block_kernelsize1,
            cnn_temporal_block_kernelsize2,
        ]
        in_channels = cnn_spatial_kernels
        # Blocks of "dropout+2d temporal conv.+batch norm+activation+pooling"
        for i in range(len(cnn_temporal_block_kernels)):
            self.conv_module.add_module(
                "dropout_{0}".format(i + 1), torch.nn.Dropout(p=dropout)
            )
            self.conv_module.add_module(
                "conv_{0}".format(i + 2),
                sb.nnet.CNN.Conv2d(
                    in_channels=in_channels,
                    out_channels=cnn_temporal_block_kernels[i],
                    kernel_size=cnn_temporal_block_kernelsizes[i],
                    padding="valid",
                    bias=False,
                    swap=True,
                ),
            )
            self.conv_module.add_module(
                "bnorm_{0}".format(i + 2),
                sb.nnet.normalization.BatchNorm2d(
                    input_size=cnn_temporal_block_kernels[i],
                    affine=True,
                    momentum=0.1,
                    eps=1e-5,
                ),
            )
            self.conv_module.add_module("act_{0}".format(i + 2), activation)
            self.conv_module.add_module(
                "pool_{0}".format(i + 2),
                sb.nnet.pooling.Pooling2d(
                    pool_type=cnn_pool_type,
                    kernel_size=cnn_temporal_block_pool,
                    stride=cnn_temporal_block_pool,
                    pool_axis=[1, 2],
                ),
            )
            in_channels = cnn_temporal_block_kernels[i]

        # DENSE MODULE
        temporal_kernel_sizes = [
            cnn_temporal_kernelsize
        ] + cnn_temporal_block_kernelsizes
        pool_sizes = [cnn_spatial_pool] + [cnn_temporal_block_pool] * len(
            cnn_temporal_block_kernelsizes
        )
        dense_input_size = T
        for i in range(len(temporal_kernel_sizes)):
            dense_input_size = (
                dense_input_size - temporal_kernel_sizes[i][0] + 1
            )
            dense_input_size = (
                int((dense_input_size - pool_sizes[i][0]) / pool_sizes[i][0])
                + 1
            )
        dense_input_size *= cnn_temporal_block_kernels[-1]

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
        x = self.dense_module(x)
        return x
