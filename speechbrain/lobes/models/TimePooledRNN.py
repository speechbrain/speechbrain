"""A simple RNN-based architecture
   including layernorm and time pooling in between RNN layers.

Authors
 * Titouan Parcollet 2020
"""
import torch
from speechbrain.nnet import (
    Linear,
    Pooling1d,
    Sequential,
    LayerNorm,
    BatchNorm1d,
)


class TimePooledRNN(Sequential):
    """A simple RNN-based architecture
       including layernorm and time pooling in between RNN layers.

    Arguments
    ---------
    input_shape : tuple
        The shape of an example expected input.
    activation : torch class
        A class used for constructing the activation layers for dnn.
    time_pooling_factor_list: list of ints
        Specifies the time pooling factor in a list format. For example,
        '[1,2,2,1]' applies a time pooling of size 2 after layers 2 and 3.
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
    rnn_inter_norm : bool
        Wether layernorm is applied in between RNN layers.

    Example
    -------
    >>> inputs = torch.rand([10, 120, 60])
    >>> model = TimePooledRNN(input_shape=inputs.shape)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 120, 1024])
    """

    def __init__(
        self,
        input_shape,
        activation=torch.nn.LeakyReLU,
        dropout=0.15,
        time_pooling_factor_list=[1, 1, 1, 1],
        rnn_class=torch.nn.LSTM,
        rnn_layers=4,
        rnn_neurons=512,
        rnn_bidirectional=True,
        rnn_re_init=True,
        rnn_inter_norm=False,
        dnn_blocks=2,
        dnn_neurons=512,
    ):
        super().__init__(input_shape)

        if len(time_pooling_factor_list) != rnn_layers:

            msg = (
                "TPRNN : time_pooling_factor_list must be as long a "
                + "rnn_layers but found "
                + str(len(time_pooling_factor_list))
                + " and "
                + str(rnn_layers)
                + " respectively."
            )
            raise ValueError(msg)

        if rnn_layers > 0:
            for i in range(rnn_layers):
                self.append(
                    rnn_class,
                    hidden_size=rnn_neurons,
                    num_layers=1,
                    bidirectional=rnn_bidirectional,
                    re_init=rnn_re_init,
                )
                if time_pooling_factor_list[i] > 1:
                    self.append(
                        Pooling1d(
                            pool_type="max",
                            kernel_size=time_pooling_factor_list[i],
                            pool_axis=1,
                        )
                    )
                if rnn_inter_norm:
                    self.append(LayerNorm)
                if dropout > 0.0:
                    self.append(torch.nn.Dropout(p=dropout))

        for block_index in range(dnn_blocks):
            self.append(Linear, n_neurons=dnn_neurons, bias=True)
            self.append(BatchNorm1d)
            self.append(activation())
            self.append(torch.nn.Dropout(p=dropout))
