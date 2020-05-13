"""A popular speech model.

Authors: Mirco Ravanelli 2020, Peter Plantinga 2020, Ju-Chieh Chou 2020,
    Titouan Parcollet 2020, Abdel 2020
"""
import os
import torch  # noqa: F401
from speechbrain.yaml import load_extended_yaml
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.sequential import Sequential
from speechbrain.nnet.activations import Softmax
from speechbrain.nnet.pooling import Pooling
from speechbrain.utils.data_utils import recursive_update


class CRDNN(Sequential):
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
    >>> model = CRDNN(output_size=40)
    >>> inputs = torch.rand([10, 120, 60])
    >>> outputs = model(inputs, init_params=True)
    >>> outputs.shape
    torch.Size([10, 116, 40])
    """

    def __init__(
        self,
        output_size,
        cnn_param_file="cnn_block.yaml",
        cnn_blocks=1,
        cnn_overrides={},
        rnn_param_file="rnn_block.yaml",
        rnn_blocks=1,
        rnn_overrides={},
        dnn_param_file="dnn_block.yaml",
        dnn_blocks=1,
        dnn_overrides={},
        time_pooling=False,
        time_pooling_stride=2,
        time_pooling_size=2,
    ):
        blocks = []

        current_dir = os.path.dirname(os.path.abspath(__file__))
        for i in range(cnn_blocks):
            blocks.append(
                NeuralBlock(
                    block_index=i + 1,
                    param_file=os.path.join(current_dir, cnn_param_file),
                    overrides=cnn_overrides,
                )
            )

        if time_pooling:
            blocks.append(
                Pooling(
                    pool_type="max",
                    stride=time_pooling_stride,
                    kernel_size=time_pooling_size,
                    pool_axis=1,
                )
            )

        for i in range(rnn_blocks):
            blocks.append(
                NeuralBlock(
                    block_index=i + 1,
                    param_file=os.path.join(current_dir, rnn_param_file),
                    overrides=rnn_overrides,
                )
            )

        for i in range(dnn_blocks):
            blocks.append(
                NeuralBlock(
                    block_index=i + 1,
                    param_file=os.path.join(current_dir, dnn_param_file),
                    overrides=dnn_overrides,
                )
            )

        blocks.append(Linear(output_size, bias=False))
        blocks.append(Softmax(apply_log=True))

        super().__init__(*blocks)


class NeuralBlock(Sequential):
    """A block of neural network layers.

    This module loads a parameter file and constructs a model based on the
    stored hyperparameters. Two hyperparameters are treated specially:

    * `constants.block_index`: This module overrides this parameter with
        the value that is passed to the constructor.
    * `constants.sequence`: This indicates the order of applying layers.
        If it doesn't exist, the layers are applied in the order they
        appear in the file.

    Arguments
    ---------
    block_index : int
        The index of this block in the network (starting from 1).
    param_file : str
        The location of the file storing the parameters for this block.
    overrides : mapping
        Parameters to change from the defaults listed in yaml.

    Example
    -------
    >>> inputs = torch.rand([10, 50, 40])
    >>> param_file = 'speechbrain/lobes/models/dnn_block.yaml'
    >>> dnn = NeuralBlock(1, param_file)
    >>> outputs = dnn(inputs, init_params=True)
    >>> outputs.shape
    torch.Size([10, 50, 512])
    """

    def __init__(self, block_index, param_file, overrides={}):
        block_override = {"block_index": block_index}
        recursive_update(overrides, block_override)
        layers = load_extended_yaml(open(param_file), overrides)

        super().__init__(*[getattr(layers, op) for op in layers.sequence])
