"""A popular speech model.

Authors: Mirco Ravanelli 2020, Peter Plantinga 2020, Ju-Chieh Chou 2020,
    Titouan Parcollet 2020, Abdel 2020
"""
import torch
from speechbrain.nnet.RNN import LiGRU
from speechbrain.nnet.CNN import SincConv, Conv1d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.pooling import Pooling1d
from speechbrain.nnet.dropout import Dropout2d
from speechbrain.nnet.containers import Sequential
from speechbrain.nnet.normalization import BatchNorm1d


class SinCRDNN(Sequential):
    """This model is a combination of SincConv, CNNs, LiGRU, and DNNs.


    Arguments
    ---------
    activation : torch class
        A class used for constructing the activation layers. For cnn and dnn.
    dropout : float
        Neuron dropout rate, applied to cnn, rnn, and dnn.
    cnn_blocks : int
        The number of convolutional neural blocks to include.
    cnn_channels : list of ints
        A list of the number of output channels for each cnn block.
    cnn_kernelsize : tuple of ints
        The size of the convolutional kernels.
    time_pooling : bool
        Whether to pool the utterance on the time axis before the LiGRU.
    time_pooling_size : int
        The number of elements to pool on the time axis.
    time_pooling_stride : int
        The number of elements to increment by when iterating the time axis.
    rnn_layers : int
        The number of recurrent LiGRU layers to include.
    rnn_neurons : int
        Number of neurons in each layer of the LiGRU.
    rnn_bidirectional : bool
        Whether this model will process just forward or both directions.
    dnn_blocks : int
        The number of linear neural blocks to include.
    dnn_neurons : int
        The number of neurons in the linear layers.

    Example
    -------
    >>> model = SinCRDNN()
    >>> inputs = torch.rand([10, 1600])
    >>> outputs = model(inputs, init_params=True)
    >>> outputs.shape
    torch.Size([10, 10, 512])
    """

    def __init__(
        self,
        activation=torch.nn.LeakyReLU,
        dropout=0.15,
        sinc_channels=512,
        sinc_kernel_size=129,
        sinc_stride=1,
        cnn_blocks=3,
        cnn_channels=[256, 128, 128],
        cnn_kernel_sizes=[5, 5, 5],
        cnn_strides=[4, 4, 10],
        time_pooling=False,
        time_pooling_size=2,
        rnn_layers=4,
        rnn_neurons=512,
        rnn_bidirectional=True,
        dnn_blocks=2,
        dnn_neurons=512,
    ):
        blocks = []

        # SincConv Block
        blocks.extend(
            [
                SincConv(
                    out_channels=sinc_channels,
                    kernel_size=sinc_kernel_size,
                    stride=sinc_stride,
                ),
                BatchNorm1d(),
                activation(),
                Dropout2d(drop_rate=dropout),
            ]
        )

        # Conv 1d blocks
        for block_index in range(cnn_blocks):
            blocks.extend(
                [
                    Conv1d(
                        out_channels=cnn_channels[block_index],
                        kernel_size=cnn_kernel_sizes[block_index],
                        stride=cnn_strides[block_index],
                    ),
                    BatchNorm1d(),
                    activation(),
                    Dropout2d(drop_rate=dropout),
                ]
            )

        # Time pooling
        if time_pooling:
            blocks.append(
                Pooling1d(
                    pool_type="max", kernel_size=time_pooling_size, pool_axis=1,
                )
            )

        # RNN layers
        if rnn_layers > 0:
            blocks.append(
                LiGRU(
                    hidden_size=rnn_neurons,
                    num_layers=rnn_layers,
                    dropout=dropout,
                    bidirectional=rnn_bidirectional,
                )
            )

        # DNN blocks
        for block_index in range(dnn_blocks):
            blocks.extend(
                [
                    Linear(
                        n_neurons=dnn_neurons, bias=True, combine_dims=False,
                    ),
                    BatchNorm1d(),
                    activation(),
                    torch.nn.Dropout(p=dropout),
                ]
            )

        super().__init__(*blocks)
