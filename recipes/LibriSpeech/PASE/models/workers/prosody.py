
import torch
import torch.nn as nn

import speechbrain as sb
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.normalization import BatchNorm1d


class ProsodyWorker(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_blocks=1,
        hidden_channels=256,
        hidden_kernel_size=1,
        out_channels=4,
        activation=torch.nn.PReLU,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()

        for block_index in range(hidden_blocks):
            self.blocks.extend(
                [
                    Conv1d(
                        in_channels=in_channels, out_channels=hidden_channels, kernel_size=hidden_kernel_size,
                    ),
                    activation(hidden_channels),
                ],
            )
            in_channels = hidden_channels

        self.blocks.append(
            Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        )

    def forward(self, x, *args, **kwargs):
        x = x[0]
        for layer in self.blocks:
            try:
                if layer._get_name() == 'PReLU':
                    x = x.transpose(1, -1)
                x = layer(x, *args, **kwargs)
                if layer._get_name() == 'PReLU':
                    x = x.transpose(1, -1)
            except TypeError:
                x = layer(x)
        return x
