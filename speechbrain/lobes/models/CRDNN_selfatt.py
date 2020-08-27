"""This is an extenstion of the popular speech model with consist of
CNN -> Transformer-Encoder -> RNN -> DNN

Authors
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
 * Ju-Chieh Chou 2020
 * Titouan Parcollet 2020
 * Abdel 2020
 * Jianyuan Zhong 2020
"""
import torch
import speechbrain as sb
from speechbrain.lobes.models.transformer.Transformer import TransformerEncoder


class CRDNN(sb.nnet.Sequential):
    """This model is a combination of CNNs, RNNs, and DNNs.

    The default CNN model is based on VGG.

    Arguments
    ---------
    input_shape : tuple
        Expected shape of the input.
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
    >>> inputs = torch.rand([10, 120, 60])
    >>> model = CRDNN(input_shape=inputs.shape)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 120, 512])
    """

    def __init__(
        self,
        input_shape,
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
        rnn_class=sb.nnet.LiGRU,
        rnn_layers=4,
        rnn_neurons=512,
        rnn_bidirectional=True,
        rnn_re_init=False,
        dnn_postionalwise=False,
        dnn_blocks=2,
        dnn_neurons=512,
    ):
        super().__init__(input_shape)

        for block_index in range(cnn_blocks):
            self.append(
                sb.nnet.Conv2d,
                out_channels=cnn_channels[block_index],
                kernel_size=cnn_kernelsize,
            )
            self.append(sb.nnet.LayerNorm)
            self.append(activation())
            self.append(
                sb.nnet.Conv2d,
                out_channels=cnn_channels[block_index],
                kernel_size=cnn_kernelsize,
            )
            self.append(sb.nnet.LayerNorm)
            self.append(activation())

            if not using_2d_pooling:
                self.append(
                    sb.nnet.Pooling1d(
                        pool_type="max",
                        kernel_size=inter_layer_pooling_size,
                        pool_axis=2,
                    )
                )
            else:
                self.append(
                    sb.nnet.Pooling2d(
                        pool_type="max",
                        kernel_size=(
                            inter_layer_pooling_size,
                            inter_layer_pooling_size,
                        ),
                        pool_axis=(1, 2),
                    )
                )

            self.append(sb.nnet.Dropout2d(drop_rate=dropout)),

        if time_pooling:
            self.append(
                sb.nnet.Pooling1d(
                    pool_type="max", kernel_size=time_pooling_size, pool_axis=1,
                )
            )

        if self_attention:
            self.append(
                TransformerEncoder(
                    num_layers=self_attention_layers,
                    nhead=self_attention_num_heads,
                    d_ffn=self_attention_hidden_dim,
                )
            )

        if rnn_layers > 0:
            self.append(
                rnn_class,
                hidden_size=rnn_neurons,
                num_layers=rnn_layers,
                dropout=dropout,
                bidirectional=rnn_bidirectional,
                re_init=rnn_re_init,
            )

        for block_index in range(dnn_blocks):
            if dnn_postionalwise:
                self.append(
                    sb.nnet.PositionalwiseFeedForward(
                        hidden_size=dnn_neurons, dropout=dropout
                    )
                )
                self.append(sb.nnet.LayerNorm)
                self.append(activation())
                self.append(torch.nn.Dropout(p=dropout))
            else:
                self.append(sb.nnet.Linear, n_neurons=dnn_neurons, bias=True)
                self.append(sb.nnet.BatchNorm1d),
                self.append(activation()),
                self.append(torch.nn.Dropout(p=dropout))
