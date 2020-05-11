import os
import torch  # noqa: F401
from speechbrain.yaml import load_extended_yaml
from speechbrain.nnet.sequential import Sequential
from speechbrain.utils.data_utils import recursive_update


class Xvector(Sequential):
    def __init__(
        self,
        output_size,
        tdnn_blocks=1,
        tdnn_overrides={},
        tdnn_stats_pool_blocks=1,
        tdnn_stats_pool_overrides={},
        tdnn_lin_blocks=1,
        tdnn_lin_overrides={},
    ):
        blocks = []

        current_dir = os.path.dirname(os.path.abspath(__file__))
        for i in range(tdnn_blocks):
            blocks.append(
                NeuralBlock(
                    block_index=i + 1,
                    param_file=os.path.join(current_dir, "tdnn_block.yaml"),
                    overrides=tdnn_overrides,
                )
            )

        for i in range(tdnn_stats_pool_blocks):
            blocks.append(
                NeuralBlock(
                    block_index=i + 1,
                    param_file=os.path.join(
                        current_dir, "tdnn_stats_pool_block.yaml"
                    ),
                    overrides=tdnn_stats_pool_overrides,
                )
            )

        for i in range(tdnn_lin_blocks):
            blocks.append(
                NeuralBlock(
                    block_index=i + 1,
                    param_file=os.path.join(current_dir, "tdnn_lin_block.yaml"),
                    overrides=tdnn_lin_overrides,
                )
            )

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
