"""Gated Neural Network variant of ``VanillaNN`` for simple feed-forward tests.

Authors
-------
 * Adel Moumen 2025
"""

import torch

import speechbrain as sb


class GatedNNBlock(torch.nn.Module):
    """Single gated feed-forward block used in :class:`GatedNN`.

    This block applies two parallel linear projections to the input and combines
    them with an element-wise product after passing one branch through a
    non-linear activation. A final linear layer projects the gated representation
    back to the original input dimensionality.

    Arguments
    ---------
    n_neurons : int
        Number of neurons in the hidden (gated) representation.
    input_shape : tuple or None
        Shape of the input tensor. Used to infer ``input_size`` when not given.
    input_size : int or None
        Flattened size of the last (or spatially combined) input dimension.
        One of ``input_shape`` or ``input_size`` must be provided.
    activation : torch.nn.Module or callable
        Activation class used in the gated branch (default: ``torch.nn.GELU``).
    bias : bool
        If True, use bias terms in the linear layers.
    combine_dims : bool
        If True and the input is 4D, combines the last two dimensions before
        applying the linear layers.
    """

    def __init__(
        self,
        n_neurons,
        input_shape=None,
        input_size=None,
        activation=torch.nn.GELU,
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
        """Returns the output of the GatedNNBlock.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        x : torch.Tensor
            Output tensor.
        """
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x_act = self.activation(x_fc1) * x_fc2
        x_fc3 = self.fc3(x_act)
        return x_fc3


class GatedNN(sb.nnet.containers.Sequential):
    """A simple stacked Gated Neural Network for feed-forward modeling.

    This model stacks multiple :class:`GatedNNBlock` modules on top of each
    other, keeping the same input and output dimensionality while increasing
    representational power through gated non-linear transformations.

    Arguments
    ---------
    input_shape : tuple
        Expected shape of the input tensors.
    activation : torch.nn.Module or callable
        Activation class used inside each gated block (default: ``torch.nn.GELU``).
    blocks : int
        Number of stacked gated blocks.
    neurons : int
        Number of neurons in the hidden (gated) representation of each block.
    bias : bool
        If True, use bias terms in the linear layers.
    combine_dims : bool
        If True and the input is 4D, combines the last two dimensions before
        applying the linear layers in each block.

    Example
    -------
    >>> inputs = torch.rand([10, 120, 60])
    >>> model = GatedNN(input_shape=inputs.shape, blocks=2, neurons=512)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 120, 60])
    """

    def __init__(
        self,
        input_shape,
        activation=torch.nn.GELU,
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
