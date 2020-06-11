"""A popular speaker recognition and diarization model.

Authors
 * Nauman Dawalatabad 2020
"""

# import os
import torch  # noqa: F401
from speechbrain.nnet.containers import Sequential
from speechbrain.nnet.pooling import StatisticsPooling
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.normalization import BatchNorm1d


class Xvector(Sequential):
    """This is Xvector model used for speaker recognition and diarization.

    Arguments
    ---------
    device : str
        Device used e.g. "cpu" or "cuda"
    activation : torch class
        A class for constructing the activation layers.
    tdnn_blocks : int
        Number of time delay neural (TDNN) layers.
    tdnn_channels : int
        Output channels for TDNN layer.
    tdnn_kernel_sizes : list of ints
        List of kernel sizes for each TDNN layer.
    tdnn_dilations : list of ints
        List of dialations for kernels in each TDNN layer.
    tdnn_fin_channels : int
        The output channel size of final TDNN layer.
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.

    Example
    -------
    >>> from Xvector import Xvector
    >>> xvect_model = Xvector('cpu')
    >>> input_feats = torch.rand([5, 10, 24])
    >>> outputs = xvect_model(input_feats, init_params=True)
    >>> outputs.shape
    torch.Size([5, 1, 512])
    """

    def __init__(
        self,
        device="cpu",
        activation=torch.nn.LeakyReLU,
        tdnn_blocks=5,
        tdnn_channels=512,
        tdnn_kernel_sizes=[5, 3, 3, 1, 1],
        tdnn_dilations=[1, 2, 3, 1, 1],
        tdnn_fin_channels=1500,
        lin_blocks=2,
        lin_neurons=512,
    ):

        blocks = []

        for block_index in range(tdnn_blocks - 1):
            blocks.extend(
                [
                    Conv1d(
                        out_channels=tdnn_channels,
                        kernel_size=tdnn_kernel_sizes[block_index],
                        dilation=tdnn_dilations[block_index],
                    ),
                    activation(),
                    BatchNorm1d(),
                ]
            )

        blocks.extend(
            [
                Conv1d(
                    out_channels=tdnn_fin_channels,
                    kernel_size=tdnn_kernel_sizes[-1],
                    dilation=tdnn_dilations[-1],
                ),
                activation(),
                BatchNorm1d(),
            ]
        )

        blocks.append(StatisticsPooling(device))

        for block_index in range(lin_blocks):
            blocks.extend(
                [
                    Linear(
                        n_neurons=lin_neurons, bias=True, combine_dims=False,
                    ),
                    activation(),
                    BatchNorm1d(),
                ]
            )

        super().__init__(*blocks)
