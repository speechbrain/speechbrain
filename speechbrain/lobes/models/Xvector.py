"""A popular speaker recognition and diarization model.

Authors: Nauman Dawalatabad 2020

"""

# import os
import torch  # noqa: F401

# from speechbrain.yaml import load_extended_yaml
from speechbrain.nnet.containers import Sequential

# from speechbrain.utils.data_utils import recursive_update
from speechbrain.nnet.statistic_pooling import StatisticsPooling
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.normalization import BatchNorm1d


class Xvector(Sequential):
    """This is Xvector model used for speaker recognition and diarization.

    Arguments
    ---------
    tdnn_blocks : int
        The number of time delay neural (TDNN) blocks to include (using Conv1D).
    tdnn_overrides : mapping
        Additional parameters overriding the TDNN parameters.
    lin_blocks : int
        The number of linear neural blocks to include.
    lin_overrides : mapping
        Additional parameters overriding the linear parameters.

    TDNN Block Parameters
    --------------------
        .. include:: tdnn_block.yaml

    LINEAR Block Parameters
    --------------------
        .. include:: lin_block.yaml

    """

    def __init__(
        self,
        activation=torch.nn.LeakyReLU,
        tdnn_blocks=5,
        tdnn_channels=512,
        tdnn_kernel_sizes=[5, 3, 3, 1, 1],
        tdnn_dialations=[1, 2, 3, 1, 1],
        tdnn_fin_channels=1500,
        lin_blocks=2,
        lin_neurons=512,
    ):

        blocks = []

        for block_index in range(tdnn_blocks):
            blocks.extend(
                [
                    Conv1d(
                        out_channels=tdnn_channels,
                        kernel_size=tdnn_kernel_sizes[block_index],
                        dilation=tdnn_dialations[block_index],
                    ),
                    activation(),
                    BatchNorm1d(),
                ]
            )

        blocks.append(StatisticsPooling())

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
