import torch
from speechbrain.nnet.normalization import normalize
from speechbrain.nnet.architectures import (
    conv,
    RNN_basic,
    linear,
    activation,
    pooling,
    dropout,
)


class VGG2_BLSTM_MLP(torch.nn.Module):
    """This model is a combination of VGG2, bi-directional LSTM, and MLP

    Args:
        output_len (int): The length of the output (number of classes)
        vgg_blocks (int): The number of vgg blocks to include
        n_neurons (int): The number of neurons in the RNN and MLP
        mlp_blocks (int): The number of mlp blocks to include
        activation (str): The activation function to use on VGG and MLP
        drop_rate (float): The rate of dropping neurons

    Example:
        >>> import torch
        >>> model = VGG2_BLSTM_MLP(output_len=40)
        >>> inputs = torch.rand([10, 60, 120])
        >>> outputs = model(inputs)
        >>> outputs.shape
        torch.Size([10, 120, 40])
    """
    def __init__(
        self,
        output_len,
        vgg_blocks=2,
        n_neurons=512,
        mlp_blocks=2,
        activation_fn='leaky_relu',
        drop_rate=0.15,
    ):
        super().__init__()
        self.output_len = output_len
        self.vgg_blocks = vgg_blocks
        self.n_neurons = n_neurons
        self.mlp_blocks = mlp_blocks
        self.activation_fn = activation_fn
        self.drop_rate = drop_rate

        blocks = []

        for i in range(vgg_blocks):
            blocks.append(self._vgg_block(i + 1))

        blocks.append(RNN_basic(
            rnn_type='lstm',
            n_neurons=n_neurons,
            nonlinearity='tanh',
            num_layers=4,
            dropout=drop_rate,
            bidirectional=True,
        ))

        for i in range(mlp_blocks):
            blocks.append(self._mlp_block())

        blocks.append(linear(self.output_len, bias=False))
        blocks.append(activation('log_softmax'))

        self.blocks = torch.nn.Sequential(*blocks)

    def forward(self, features):
        return self.blocks(features)

    def _vgg_block(self, block_index):
        return torch.nn.Sequential(
            conv(block_index * 128, kernel_size=[3, 3]),
            normalize('batchnorm'),
            activation(self.activation_fn),
            conv(block_index * 128, kernel_size=[3, 3]),
            normalize('batchnorm'),
            activation(self.activation_fn),
            pooling('max', kernel_size=2, stride=2),
            dropout(self.drop_rate),
        )

    def _mlp_block(self):
        return torch.nn.Sequential(
            linear(self.n_neurons),
            normalize('batchnorm'),
            activation(self.activation_fn),
            dropout(self.drop_rate),
        )
