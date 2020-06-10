"""Library for implementing cascade (sequences) of different neural modules.

Authors
 * Peter Plantinga 2020
"""

import torch
import logging
import operator
import functools
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


class ConnectBlocks(torch.nn.Module):
    """Connect a sequence of blocks with shortcut connections.

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

    Example
    -------
    >>> block1 = Linear(n_neurons=10)
    >>> block2 = Linear(n_neurons=10)
    >>> block3 = Linear(n_neurons=10)
    >>> inputs = torch.rand(10, 100, 20)
    >>> model = ConnectBlocks([block1, block2, block3])
    >>> outputs = model(inputs, init_params=True)
    >>> outputs.shape
    torch.Size([10, 100, 10])
    """

    def __init__(
        self,
        blocks,
        shortcut_type="residual",
        shortcut_projection=False,
        shortcut_combine_fn="add",
    ):
        super().__init__()
        self.blocks = torch.nn.ModuleList(blocks)
        if shortcut_type not in [None, "residual", "dense", "skip"]:
            raise ValueError(
                "'shortcuts' must be one of 'residual', 'dense', or 'skip'"
            )
        self.shortcut_type = shortcut_type
        self.shortcut_projection = shortcut_projection
        self.projections = []

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
        x = _apply_block(self.blocks[0], x, init_params)
        shortcut = x

        for i, block in enumerate(self.blocks[1:]):
            x = _apply_block(block, x, init_params)

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
            projection_size = functools.reduce(operator.mul, x.shape[2:], 1)
            projection = Linear(projection_size, bias=False, combine_dims=True)

            # Store and register projection
            self.projections.append(projection)
            self.layers.append(projection)

        # Apply projection
        if self.shortcut_projection:
            shortcut = self.projections[block_index](
                shortcut, init_params=init_params
            )
            shortcut = shortcut.reshape(x.shape)

        return self.shortcut_combine_fn(shortcut, x, init_params=init_params)


def _apply_block(block, x, init_params):
    """Apply a function handling cases with no init_params"""
    try:
        return block(x, init_params=init_params)
    except TypeError:
        return block(x)
