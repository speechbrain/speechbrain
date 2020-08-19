"""Vanilla Neural Network for simple tests

Authors
* Elena Rastorgueva 2020
"""
import torch
import speechbrain as sb


class VanillaNN(sb.nnet.Sequential):
    """ A simple vanilla Deep Neural Network.

    Arguments
    ---------
    activation : torch class
        A class used for constructing the activation layers.
    dnn_blocks : int
        The number of linear neural blocks to include.
    dnn_neurons : int
        The number of neurons in the linear layers.

    Example
    -------
    >>> model = VanillaNN()
    >>> inputs = torch.rand([10, 120, 60])
    >>> outputs = model(inputs, init_params = True)
    >>> outputs.shape
    torch.Size([10, 120, 512])
    """

    def __init__(
        self, activation=torch.nn.LeakyReLU, dnn_blocks=2, dnn_neurons=512,
    ):
        blocks = []

        for block_index in range(dnn_blocks):
            blocks.extend(
                [
                    sb.nnet.Linear(
                        n_neurons=dnn_neurons, bias=True, combine_dims=False,
                    ),
                    activation(),
                ]
            )
        super().__init__(*blocks)
