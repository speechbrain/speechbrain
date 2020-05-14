"""Library for implementing cascade (sequences) of different neural modules.

Author
    Peter Plantinga 2020
"""

import torch
import logging
from speechbrain.yaml import load_extended_yaml

logger = logging.getLogger(__name__)


class Sequential(torch.nn.Module):
    """A sequence of modules which may use the `init_params=True` argument in `forward()` for initialization.

    Arguments
    ---------
    *layers
        The inputs are treated as a list of layers to be
        applied in sequence. The output shape of each layer is used to
        infer the shape of the following layer.

    Example
    -------
    >>> from speechbrain.nnet.linear import Linear
    >>> model = Sequential(
    ...     Linear(n_neurons=100),
    ...     Linear(n_neurons=200),
    ... )
    >>> inputs = torch.rand(10, 50, 40)
    >>> outputs = model(inputs, init_params=True)
    >>> outputs.shape
    torch.Size([10, 50, 200])
    """

    def __init__(
        self, *layers,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for layer in layers:
            self.layers.append(layer)

    def forward(self, x, init_params=False):
        """
        Arguments
        ---------
        x : tensor
            the input tensor to run through the network.
        """
        for layer in self.layers:
            try:
                x = layer(x, init_params=init_params)
            except TypeError:
                x = layer(x)
        return x


class Nerve(torch.nn.Module):
    """Replicate one block of modules from a yaml file with shortcuts.

    Arguments
    ---------
    param_file : str
        The location of the yaml file to use for a block. This file is
        expected to have an item called "block" defining the block,
        and can assume that the item "block_index" will be overridden.
    overrides : dict
        A set of overrides to apply to the parameters file.
    copies : int
        The number of times to replicate the layers listed in the yaml.
    shortcuts : str
        One of "residual", "dense", or "skip"
    combine_fn : function
        A function that takes the shortcut and next input, and combines them.
    """

    def __init__(
        self,
        param_file,
        overrides={},
        copies=1,
        shortcuts="",
        combine_fn=torch.add,
    ):
        super().__init__()
        self.blocks = torch.nn.ModuleList()
        self.shortcuts = shortcuts
        self.combine_fn = combine_fn

        # One-based indexing for blocks, so it can be used as a multiplier
        for block_index in range(1, copies + 1):

            # Override the block_index in the yaml file
            overrides["block_index"] = block_index
            with open(param_file) as f:
                params = load_extended_yaml(f, overrides)
                self.blocks.append(params.block)

    def forward(self, inputs, init_params=True):
        """
        Arguments
        ---------
        inputs : torch.Tensor
            The inputs to the nerve
        init_params : bool
            Whether to initialize the parameters of the blocks
        """
        shortcut_inputs = inputs

        # For residual and skip connections, the last block has
        # different inputs than the rest of the blocks.
        for block in self.blocks[:-1]:
            outputs = block(inputs, init_params)

            # Compute inputs for next block
            if self.shortcuts in ["dense", "skip"]:
                shortcut_inputs = self.combine_fn(shortcut_inputs, outputs)
            if self.shortcuts == "dense":
                inputs = shortcut_inputs
            else:
                inputs = outputs

        # Last block gets different input in some cases
        if self.shortcuts == "residual":
            inputs = self.combine_fn(shortcut_inputs, outputs)
        elif self.shortcuts == "skip":
            inputs = shortcut_inputs

        # Return the output of the last block
        return self.blocks[-1](inputs, init_params)
