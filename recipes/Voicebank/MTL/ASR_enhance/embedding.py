import torch
import speechbrain as sb


class CNNEmbedding(sb.nnet.containers.Sequential):
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
