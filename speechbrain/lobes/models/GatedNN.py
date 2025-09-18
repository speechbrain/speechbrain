"""Gated Neural Network variant of VanillaNN.

Authors
-------
 * Adel Moumen 2025
"""

import torch

import speechbrain as sb

class GatedNNBlock(torch.nn.Module):
    """
    """
    def __init__(
        self,
        n_neurons,
        input_shape=None,
        input_size=None,
        activation=sb.nnet.activations.GeLU,
        bias=False,
        combine_dims=False,
    ):
        super().__init__()
        self.combine_dims = combine_dims

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]
            if len(input_shape) == 4 and self.combine_dims:
                input_size = input_shape[2] * input_shape[3]

        self.fc1 = torch.nn.Linear(input_size, n_neurons, bias=bias)
        self.fc2 = torch.nn.Linear(input_size, n_neurons, bias=bias)
        self.fc3 = torch.nn.Linear(n_neurons, input_size, bias=bias)
        self.activation = activation()
        
    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x_act = self.activation(x_fc1) * x_fc2
        x_fc3 = self.fc3(x_act)
        return x_fc3
        
class GatedNN(sb.nnet.containers.Sequential):
    """
    """

    def __init__(
        self,
        input_shape,
        activation=sb.nnet.activations.GeLU,
        blocks=2,
        neurons=512,
        bias=False,
        combine_dims=False,
    ):
        super().__init__(input_shape=input_shape)

        for _ in range(blocks):
            self.append(
                GatedNNBlock,
                n_neurons=neurons,
                activation=activation,
                bias=bias,
                combine_dims=combine_dims,
                layer_name="gated_nn_block",
            )
