"""A popular speech model.

Authors
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
 * Ju-Chieh Chou 2020
 * Titouan Parcollet 2020
 * Abdel 2020
"""
import torch
import speechbrain as sb


class CRDNN(sb.nnet.containers.Sequential):
    """This model is a combination of CNNs, RNNs, and DNNs.

    The default CNN model is based on VGG.

    Arguments
    ---------
    input_shape : tuple
        The shape of an example expected input.
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
    using_2d_pooling: bool
        Whether using a 2D or 1D pooling after each cnn block.
    inter_layer_pooling_size : list of ints
        A list of the number of pooling for each cnn block.
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
    projection_dim : int
        The number of neurons in the projection layer.
        This layer is used to reduce the size of the flatened
        representation obtained after the CNN blocks.

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
        rnn_class=sb.nnet.RNN.LiGRU,
        inter_layer_pooling_size=[2, 2],
        using_2d_pooling=False,
        rnn_layers=4,
        rnn_neurons=512,
        rnn_bidirectional=True,
        rnn_re_init=False,
        dnn_blocks=2,
        dnn_neurons=512,
        projection_dim=-1,
    ):
        super().__init__(input_shape)

        for block_index in range(cnn_blocks):
            self.append(
                sb.nnet.CNN.Conv2d,
                out_channels=cnn_channels[block_index],
                kernel_size=cnn_kernelsize,
            )
            self.append(sb.nnet.normalization.LayerNorm)
            self.append(activation())
            self.append(
                sb.nnet.CNN.Conv2d,
                out_channels=cnn_channels[block_index],
                kernel_size=cnn_kernelsize,
            )
            self.append(sb.nnet.normalization.LayerNorm)
            self.append(activation())

            if not using_2d_pooling:
                self.append(
                    sb.nnet.pooling.Pooling1d(
                        pool_type="max",
                        kernel_size=inter_layer_pooling_size[block_index],
                        pool_axis=2,
                    )
                )
            else:
                self.append(
                    sb.nnet.pooling.Pooling2d(
                        pool_type="max",
                        kernel_size=(
                            inter_layer_pooling_size[block_index],
                            inter_layer_pooling_size[block_index],
                        ),
                        pool_axis=(1, 2),
                    )
                )

            self.append(sb.nnet.dropout.Dropout2d(drop_rate=dropout))

        if time_pooling:
            self.append(
                sb.nnet.pooling.Pooling1d(
                    pool_type="max", kernel_size=time_pooling_size, pool_axis=1,
                )
            )

        # This projection helps reducing the number of parameters
        # when using large number of CNN filters.
        # Large numbers of CNN filters + large features
        # often lead to very large flattened layers
        # This layer projects it back to something reasonable
        if projection_dim != -1:
            self.append(
                sb.nnet.linear.Linear,
                n_neurons=projection_dim,
                bias=True,
                combine_dims=True,
            )
            self.append(sb.nnet.normalization.LayerNorm)
            self.append(activation())

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
            self.append(sb.nnet.linear.Linear, n_neurons=dnn_neurons, bias=True)
            self.append(sb.nnet.normalization.BatchNorm1d)
            self.append(activation())
            self.append(torch.nn.Dropout(p=dropout))
