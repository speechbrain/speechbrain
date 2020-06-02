""" A popular speech model with MATConv pooling replacing max pooling

Authors: Mirco Ravanelli 2020, Peter Plantinga 2020, Ju-Chieh Chou 2020
    Titouan Parcollet 2020, Abdel 2020, Jianyuan Zhong 2020
"""
import torch  # noqa: F401
from speechbrain.nnet.RNN import LiGRU
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.dropout import Dropout2d
from speechbrain.nnet.CNN import Conv1d, Conv2d
from speechbrain.nnet.normalization import BatchNorm1d, BatchNorm2d
from speechbrain.nnet.containers import Sequential
from speechbrain.lobes.models.MATConv import MATConvPool2d


class MRDNN(Sequential):
    """This model is a combination of CNNs, RNNs, and DNNs.

    The default CNN model is based on VGG.

    Arguments
    ---------
    output_size : int
        The length of the output (number of target classes).
    cnn_blocks : int
        The number of convolutional neural blocks to include.
    cnn_overrides : mapping
        Additional parameters overriding the CNN parameters.
    rnn_blocks : int
        The number of recurrent neural blocks to include.
    rnn_overrides : mapping
        Additional parameters overriding the RNN parameters.
    dnn_blocks : int
        The number of linear neural blocks to include.
    dnn_overrides : mapping
        Additional parameters overriding the DNN parameters.

    CNN Block Parameters
    --------------------
        .. include:: cnn_block.yaml

    RNN Block Parameters
    --------------------
        .. include:: rnn_block.yaml

    DNN Block Parameters
    --------------------
        .. include:: dnn_block.yaml

    Example
    -------
    >>> import torch
    >>> model = MRDNN()
    >>> inputs = torch.rand([10, 120, 60])
    >>> outputs = model(inputs, init_params=True)
    >>> len(outputs.shape)
    3
    """

    def __init__(
        self,
        activation=torch.nn.LeakyReLU,
        dropout=0.15,
        cnn_blocks=2,
        cnn_channels=[128, 256],
        cnn_kernelsize=(3, 3),
        frequency_striding=True,
        time_striding=True,
        matconv_outchannels=256,
        matpool_channels=256,
        matconv_kernelsize=(1, 1),
        matconv_dilations=[1, 3, 6, 9],
        rnn_layers=4,
        rnn_neurons=512,
        rnn_bidirectional=True,
        dnn_blocks=2,
        dnn_neurons=512,
        debug=False,
    ):
        # flag to debug the model
        # when being set to true it will print out the model after
        # initializing parameters
        self.debug = debug
        cnn_stride = (1, 1)
        matconv_stride = (1, 1)
        if frequency_striding:
            cnn_stride = (1, 2)
        if time_striding:
            matconv_stride = (2, 1)

        blocks = []

        self.cnn_blocks = []
        for block_index in range(cnn_blocks):

            # only apply striding on even (odd index) number of cnn layers
            stride_applied = (1, 1)
            if block_index % 2 == 1:
                stride_applied = cnn_stride

            self.cnn_blocks.extend(
                [
                    Conv2d(
                        out_channels=cnn_channels[block_index],
                        kernel_size=cnn_kernelsize,
                        stride=stride_applied,
                    ),
                    BatchNorm2d(),
                    activation(),
                ]
            )
        self.cnn_blocks.append(Dropout2d(dropout))
        blocks.extend(self.cnn_blocks)

        self.matpool_block = [
            Conv2d(
                out_channels=matconv_outchannels,
                kernel_size=matconv_kernelsize,
                stride=matconv_stride,
            ),
            BatchNorm2d(),
            activation(),
            MATConvPool2d(
                out_channels=matconv_outchannels,
                stride=(1, 1),
                matpool_channels=matpool_channels,
                activation=activation,
                dilations=matconv_dilations,
                droupout=dropout,
            ),
        ]
        blocks.extend(self.matpool_block)

        self.rnn_block = []
        if rnn_layers > 0:
            self.rnn_block.append(
                LiGRU(
                    hidden_size=rnn_neurons,
                    num_layers=rnn_layers,
                    dropout=dropout,
                    bidirectional=rnn_bidirectional,
                )
            )
        blocks.extend(self.rnn_block)

        self.dnn_block = []
        for block_index in range(dnn_blocks):
            self.dnn_block.extend(
                [
                    Linear(
                        n_neurons=dnn_neurons, bias=True, combine_dims=False,
                    ),
                    BatchNorm1d(),
                    activation(),
                    torch.nn.Dropout(p=dropout),
                ]
            )
        blocks.extend(self.dnn_block)

        super().__init__(*blocks)

    def forward(self, x, init_params=False):
        if init_params:
            output = super(MRDNN, self).forward(x, init_params)

            # initializeweights
            self._init_weight()

            # print out the model for debugging
            if self.debug:
                print(self)

            return output
        else:
            return super(MRDNN, self).forward(x, init_params)

    def _init_weight(self):
        for block in self.layers:
            if hasattr(block, "layers"):
                for layer in block.layers:
                    if isinstance(layer, Conv1d) or isinstance(layer, Conv2d):
                        torch.nn.init.kaiming_normal_(layer.conv.weight)
                    if isinstance(layer, Linear):
                        torch.nn.init.xavier_normal_(layer.w.weight)
