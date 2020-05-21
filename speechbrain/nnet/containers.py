"""Library for implementing cascade (sequences) of different neural modules.

Author
    Peter Plantinga 2020
"""

import torch
import logging
import functools
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


def ignore_init(function):
    """Wrapper function to ignore the init_params argument"""
    return lambda x, y, init_params=False: function(x, y)


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
    yaml_overrides : dict
        A set of yaml overrides to apply to the parameters file.
    replication_count : int
        The number of times to replicate the block listed in the yaml.
    shortcut_type : str
        One of:
        * "residual" - first block output passed to final output
        * "dense" - input of each block is from all previous blocks
        * "skip" - output of each block is passed to final output
    shortcut_projection : bool
        Only has an effect if `shortcut_type` is passed. Whether to add a
        linear projection layer to the shortcut connection before combining
        with the output, to handle different sizes.
    shortcut_combine_fn : str or function
        Either a pre-defined function (one of "add", "sub", "mul", "div",
        "avg", "cat") or a user-defined function that takes the shortcut
        and next input, and combines them, as well as `init_params`
        in case parameters need to be initialized inside of the function.
    """

    def __init__(
        self,
        param_file,
        yaml_overrides={},
        replication_count=1,
        shortcut_type=None,
        shortcut_projection=False,
        shortcut_combine_fn="add",
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        if shortcut_type not in [None, "residual", "dense", "skip"]:
            raise ValueError(
                "'shortcuts' must be one of 'residual', 'dense', or 'skip'"
            )
        self.shortcut_type = shortcut_type
        self.shortcut_projection = shortcut_projection

        # Define combination options
        combine_functions = {
            "add": ignore_init(torch.add),
            "sub": ignore_init(functools.partial(torch.add, alpha=-1)),
            "mul": ignore_init(torch.mul),
            "div": ignore_init(torch.div),
            "avg": ignore_init(lambda x, y: torch.add(x, y) / 2),
            "cat": ignore_init(functools.partial(torch.cat, axis=-1)),
        }
        if isinstance(shortcut_combine_fn, str):
            if shortcut_combine_fn not in combine_functions:
                raise ValueError(
                    "'shortcut_combine_fn' must be one of %s or a user-"
                    "defined function that combines two arguments and takes "
                    "`init_params` as an argument." % combine_functions.keys()
                )

            self.shortcut_combine_fn = combine_functions[shortcut_combine_fn]
        else:
            self.shortcut_combine_fn = shortcut_combine_fn

        # Initialize containers
        self.layers = torch.nn.ModuleList()
        self.blocks = []
        self.projections = []

        # One-based indexing for blocks, so it can be used as a multiplier
        for block_index in range(1, replication_count + 1):

            # Override the block_index in the yaml file
            yaml_overrides["block_index"] = block_index
            layers = load_yaml_modules(param_file, yaml_overrides)

            # Store the layers as blocks for easy access, and
            # register as layers for a flatter structure
            self.blocks.append(layers)
            self.layers.extend(layers)

    def forward(self, x, init_params=True):
        """
        Arguments
        ---------
        x : torch.Tensor
            The inputs to the replicated modules.
        init_params : bool
            Whether to initialize the parameters of the blocks.
        """
        # Don't include first block in shortcut, since it may change
        # the shape in significant ways.
        for layer in self.blocks[0]:
            x = self._apply_layer(layer, x, init_params)
        shortcut = x

        for i, block in enumerate(self.blocks[1:]):
            for layer in block:
                x = self._apply_layer(layer, x, init_params)

            # Record outputs and combine with current shortcut
            if self.shortcut_type in ["dense", "skip"]:
                shortcut = self._combine(shortcut, x, init_params, i)

            # Update inputs to next layer
            if self.shortcut_type == "dense":
                x = shortcut

        # Apply residual connection to final output
        if self.shortcut_type == "residual":
            shortcut = self._combine(shortcut, x, init_params, 0)

        if self.shortcut_type in ["residual", "dense", "skip"]:
            x = shortcut

        return x

    def _combine(self, shortcut, x, init_params=False, block_index=0):
        """Handle combining shortcut with outputs."""

        # Initialize projection if necessary
        if init_params and self.shortcut_projection:
            projection = Linear(x.size(-1), bias=False, combine_dims=True)

            # Store and register projection
            self.projections.append(projection)
            self.layers.append(projection)

        # Apply projection
        if self.shortcut_projection:
            shortcut = self.projections[block_index](shortcut)

        return self.shortcut_combine_fn(shortcut, x, init_params=init_params)

    def _apply_layer(self, layer, x, init_params):
        """Apply a function handling cases with no init_params"""
        try:
            return layer(x, init_params=init_params)
        except TypeError:
            return layer(x)


def load_yaml_modules(param_file, overrides={}):
    """Construct a block of pytorch modules from a parameter file.

    Arguments
    ---------
    param_file : str
        A parameter file to use for constructing a block.
    overrides : dict
        A set of overrides to use when parsing the parameter file.

    Returns
    -------
    A list of all top-level torch.nn.Modules.
    """

    with open(param_file) as f:
        params = load_extended_yaml(f, overrides, return_dict=True)

    return [v for v in params.values() if isinstance(v, torch.nn.Module)]
