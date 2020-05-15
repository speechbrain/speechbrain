"""Library for implementing cascade (sequences) of different neural modules.

Author
    Peter Plantinga 2020
"""

import torch
import logging
from speechbrain.yaml import load_extended_yaml
from speechbrain.nnet.linear import Linear

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


def torch_add_init_ignore(x, y, init_params=False):
    """Torch add, but ignoring the init_params argument"""
    return torch.add(x, y)


def project_add(x, y, init_params=False):
    """Project the shortcut before adding"""
    projection = Linear(y.size(-1))
    return torch.add(projection(x, init_params), y)


class ReplicateBlock(torch.nn.Module):
    """Replicate one block of modules from a yaml file with shortcuts.

    Note: all shortcuts start from the output of the first block,
    since the first block may change the shape significantly.

    Arguments
    ---------
    param_file : str
        The location of the yaml file to use for a block. This file is
        expected to have an item called "block" defining the block,
        and can assume that the item "block_index" will be overridden.
    overrides : dict
        A set of overrides to apply to the parameters file.
    copies : int
        The number of times to replicate the block listed in the yaml.
    shortcuts : str
        One of "", "residual", "dense", or "skip"
        * "residual" - first block output passed to final output
        * "dense" - input of each block is from all previous blocks
        * "skip" - output of each block is passed to final output
    combine_fn : function
        A function that takes the shortcut and next input, and combines them.
        This function is also passed `init_params` in case parameters need to
        be initialized inside of the function.
    """

    def __init__(
        self,
        param_file,
        overrides={},
        copies=1,
        shortcuts="",
        combine_fn=torch_add_init_ignore,
    ):
        super().__init__()
        self.blocks = torch.nn.ModuleList()
        self.shortcuts = shortcuts
        self.combine_fn = combine_fn

        # One-based indexing for blocks, so it can be used as a multiplier
        for block_index in range(1, copies + 1):

            # Override the block_index in the yaml file
            overrides["block_index"] = block_index
            self.blocks.append(NeuralBlock(param_file, overrides))

    def forward(self, inputs, init_params=True):
        """
        Arguments
        ---------
        inputs : torch.Tensor
            The inputs to the nerve
        init_params : bool
            Whether to initialize the parameters of the blocks.
        """
        # Don't include first block in shortcut, since it may change
        # the shape in significant ways.
        inputs = self.blocks[0](inputs, init_params)
        shortcut_inputs = inputs

        for block in self.blocks[1:]:
            outputs = block(inputs, init_params)

            if self.shortcuts in ["dense", "skip"]:
                shortcut_inputs = self.combine_fn(
                    shortcut_inputs, outputs, init_params=init_params,
                )
            if self.shortcuts == "dense":
                inputs = shortcut_inputs
            else:
                inputs = outputs

        if self.shortcuts == "residual":
            shortcut_inputs = self.combine_fn(
                shortcut_inputs, outputs, init_params=init_params,
            )

        if self.shortcuts in ["residual", "dense", "skip"]:
            outputs = shortcut_inputs
        else:
            outputs = inputs

        return outputs


class NeuralBlock(torch.nn.Module):
    """Construct a block of pytorch modules from a parameter file.

    Arguments
    ---------
    param_file : str
        A parameter file to use for constructing a block. This file
        is expected to define a parameter called "block" with a single
        module containing the computation of the block.
    overrides : dict
        A set of overrides to use when parsing the parameter file.
    """

    def __init__(self, param_file, overrides={}):
        super().__init__()
        with open(param_file) as f:
            params = load_extended_yaml(f, overrides)
        self.block = params.block

    def forward(self, x, init_params=False):
        return self.block(x, init_params)
