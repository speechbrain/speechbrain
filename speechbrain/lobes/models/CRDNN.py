"""A popular speech model.

Authors: Mirco Ravanelli 2020, Peter Plantinga 2020, Ju-Chieh Chou 2020,
    Titouan Parcollet 2020, Abdel 2020
"""
import torch
from speechbrain.nnet.architectures import linear, activation, Sequential
from speechbrain.utils.data_utils import load_extended_yaml, recursive_update


class CRDNN(torch.nn.Module):
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

    Example
    -------
    >>> import torch
    >>> model = CRDNN(output_len=40)
    >>> inputs = torch.rand([10, 60, 120])
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 120, 40])
    """
    def __init__(
        self,
        output_size,
        cnn_blocks=1,
        cnn_overrides={},
        rnn_blocks=1,
        rnn_overrides={},
        dnn_blocks=1,
        dnn_overrides={},
    ):
        super().__init__()

        blocks = []

        for i in range(cnn_blocks):
            blocks.append(NeuralBlock(
                block_index=i + 1,
                param_file='speechbrain/lobes/models/cnn_block.yaml',
                overrides=cnn_overrides,
            ))

        for i in range(rnn_blocks):
            blocks.append(NeuralBlock(
                block_index=i + 1,
                param_file='speechbrain/lobes/models/rnn_block.yaml',
                overrides=rnn_overrides,
            ))

        for i in range(dnn_blocks):
            blocks.append(NeuralBlock(
                block_index=i + 1,
                param_file='speechbrain/lobes/models/dnn_block.yaml',
                overrides=dnn_overrides,
            ))

        blocks.append(linear(output_size, bias=False))
        blocks.append(activation('log_softmax'))

        self.blocks = Sequential(blocks)

    def init_params(self, dummy_input):
        self.blocks.init_params(dummy_input)

    def forward(self, features):
        """Returns the output of the model.

        Arguments
        ---------
        features : tensor
            The input features to the network.
        """
        return self.blocks(features)


class NeuralBlock(torch.nn.Module):
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
    >>> inputs = torch.rand([10, 40, 200])
    >>> param_file = 'speechbrain/lobes/models/rnn_block.yaml'
    >>> cnn = NeuralBlock(1, param_file, ['rnn'])
    >>> outputs = cnn(inputs)
    >>> outputs.shape
    torch.Size([10, 40, 128, 196])
    """
    def __init__(self, block_index, param_file, overrides={}):
        """"""
        super().__init__()

        block_override = {'constants': {'block_index': block_index}}
        recursive_update(overrides, block_override)
        layers = load_extended_yaml(open(param_file), overrides)
        sequence = layers['constants']['sequence']

        self.block = Sequential(layers[op] for op in sequence)

    def init_params(self, dummy_input):
        self.block.init_params(dummy_input)

    def forward(self, x):
        """Returns the output of the neural operations.

        Arguments
        ---------
        x : tensor
            The tensor to perform neural operations on.
        """
        return self.block(x)
