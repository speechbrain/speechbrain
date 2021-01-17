"""
Embedding part of enhancement model, simple 1D CNN with normalization
and activation on each layer. LayerNorm and LeakyReLU are used.

Authors:
 * ChienFeng Liao 2020
 * Peter Plantinga 2020
 """
import torch
import speechbrain as sb


class CNNEmbedding(sb.nnet.containers.Sequential):
    """Embedding model for enhancement, used with transformer model.

    Arguments
    ---------
    input_shape : tuple of int
        The expected shape of the input. Used to infer layer sizes.
    base_channels : int
        Channels in the first 1D CNN layer, subsequent layers are reduced
        in size by half.
    blocks : int
        Number of 1D CNN layers in the model.
    """

    def __init__(self, input_shape, base_channels, blocks=4):
        super().__init__(input_shape=input_shape)

        for i in range(blocks):
            out_channels = max(base_channels // 2 ** i, 256)
            self.append(
                sb.nnet.CNN.Conv1d,
                out_channels=out_channels,
                kernel_size=3,
                padding="same",
            )
            self.append(sb.nnet.normalization.LayerNorm)
            self.append(torch.nn.LeakyReLU())
