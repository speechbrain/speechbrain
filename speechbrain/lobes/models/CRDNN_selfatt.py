"""A popular speech model.

Authors
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
 * Ju-Chieh Chou 2020
 * Titouan Parcollet 2020
 * Abdel 2020
 * Jianyuan 2020
"""
import torch
from speechbrain.nnet.RNN import LiGRU
from speechbrain.nnet.CNN import Conv2d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.pooling import Pooling1d, Pooling2d
from speechbrain.nnet.dropout import Dropout2d
from speechbrain.nnet.containers import Sequential
from speechbrain.nnet.normalization import BatchNorm1d, LayerNorm
from speechbrain.lobes.models.transformer.Transformer import TransformerEncoder
from speechbrain.nnet.attention import PositionalwiseFeedForward


class CRDNN(Sequential):
    """This model is a combination of CNNs, RNNs, and DNNs.

    The default CNN model is based on VGG.

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
    self_attention : bool
        if set to true, self-attention layers will be placed in front of rnns
    self_attention_layers : int
        number of self-attention layers
    self_attention_num_heads : int
        number of self-attention heads
    self_attention_model_dim : int
        dimension self-attention output
    self_attention_hidden_dim : int
        hidden dimension for positionalwise feedforward
    rnn_class : torch class
        The type of rnn to use in CRDNN network (LiGRU, LSTM, GRU, RNN)
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
    >>> model = CRDNN()
    >>> inputs = torch.rand([10, 120, 60])
    >>> outputs = model(inputs, init_params=True)
    >>> outputs.shape
    torch.Size([10, 120, 512])
    """

    def __init__(
        self,
        activation=torch.nn.LeakyReLU,
        dropout=0.15,
        cnn_blocks=2,
        cnn_channels=[128, 256],
        cnn_kernelsize=(3, 3),
        time_pooling=False,
        time_pooling_size=2,
        freq_pooling_size=2,
        inter_layer_pooling_size=2,
        using_2d_pooling=False,
        self_attention=False,
        self_attention_layers=1,
        self_attention_num_heads=32,
        self_attention_model_dim=512,
        self_attention_hidden_dim=512,
        rnn_class=LiGRU,
        rnn_layers=4,
        rnn_neurons=512,
        rnn_bidirectional=True,
        rnn_re_init=False,
        dnn_postionalwise=False,
        dnn_blocks=2,
        dnn_neurons=512,
    ):

        blocks = []

        for block_index in range(cnn_blocks):
            if not using_2d_pooling:
                pooling = Pooling1d(
                    pool_type="max",
                    kernel_size=inter_layer_pooling_size,
                    pool_axis=2,
                )
            else:
                pooling = Pooling2d(
                    pool_type="max",
                    kernel_size=(
                        inter_layer_pooling_size,
                        inter_layer_pooling_size,
                    ),
                    pool_axis=(1, 2),
                )

            blocks.extend(
                [
                    Conv2d(
                        out_channels=cnn_channels[block_index],
                        kernel_size=cnn_kernelsize,
                    ),
                    LayerNorm(),
                    activation(),
                    Conv2d(
                        out_channels=cnn_channels[block_index],
                        kernel_size=cnn_kernelsize,
                    ),
                    LayerNorm(),
                    activation(),
                    # Inter-layer Pooling
                    pooling,
                    Dropout2d(drop_rate=dropout),
                ]
            )

        if time_pooling:
            blocks.extend(
                [
                    Pooling1d(
                        pool_type="max",
                        kernel_size=time_pooling_size,
                        pool_axis=1,
                    ),
                ]
            )

        if self_attention:
            blocks.append(
                TransformerEncoder(
                    num_layers=self_attention_layers,
                    nhead=self_attention_num_heads,
                    d_ffn=self_attention_num_heads,
                ),
            )

        if rnn_layers > 0:
            blocks.extend(
                [
                    rnn_class(
                        hidden_size=rnn_neurons,
                        num_layers=rnn_layers,
                        dropout=dropout,
                        bidirectional=rnn_bidirectional,
                        re_init=rnn_re_init,
                    ),
                ]
            )

        for block_index in range(dnn_blocks):
            if dnn_postionalwise:
                blocks.extend(
                    [
                        PositionalwiseFeedForward(
                            hidden_size=dnn_neurons, dropout=dropout
                        ),
                        LayerNorm(),
                        activation(),
                        torch.nn.Dropout(p=dropout),
                    ]
                )
            else:
                blocks.extend(
                    [
                        Linear(
                            n_neurons=dnn_neurons,
                            bias=True,
                            combine_dims=False,
                        ),
                        BatchNorm1d(),
                        torch.nn.LeakyReLU(),
                        torch.nn.Dropout(p=dropout),
                    ]
                )

        super().__init__(*blocks)
