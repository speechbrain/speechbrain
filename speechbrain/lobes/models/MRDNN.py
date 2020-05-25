""" A popular speech model with MATConv pooling replacing max pooling

Authors: Mirco Ravanelli 2020, Peter Plantinga 2020, Ju-Chieh Chou 2020
    Titouan Parcollet 2020, Abdel 2020, Jianyuan Zhong 2020
"""
import os
import torch  # noqa: F401
from speechbrain.nnet.sequential import Sequential
from speechbrain.lobes.models.CRDNN import NeuralBlock


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
    >>> model = MRDNN(cnn_overrides={'pooling':{'out_channels':10}})
    >>> inputs = torch.rand([10, 120, 60])
    >>> outputs = model(inputs, init_params=True)
    >>> len(outputs.shape)
    3
    """

    def __init__(
        self,
        cnn_blocks=1,
        cnn_overrides={},
        rnn_blocks=1,
        rnn_overrides={},
        dnn_blocks=1,
        dnn_overrides={},
    ):
        blocks = []

        current_dir = os.path.dirname(os.path.abspath(__file__))
        for i in range(cnn_blocks):
            blocks.append(
                NeuralBlock(
                    block_index=i + 1,
                    param_file=os.path.join(current_dir, "matconv_block.yaml"),
                    overrides=cnn_overrides,
                )
            )

        for i in range(rnn_blocks):
            blocks.append(
                NeuralBlock(
                    block_index=i + 1,
                    param_file=os.path.join(current_dir, "rnn_block.yaml"),
                    overrides=rnn_overrides,
                )
            )

        for i in range(dnn_blocks):
            blocks.append(
                NeuralBlock(
                    block_index=i + 1,
                    param_file=os.path.join(current_dir, "dnn_block.yaml"),
                    overrides=dnn_overrides,
                )
            )

        super().__init__(*blocks)
