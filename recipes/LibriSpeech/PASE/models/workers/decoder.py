import torch
import torch.nn as nn

import speechbrain as sb
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.CNN import Conv1d, ConvTranspose1d
from speechbrain.nnet.normalization import BatchNorm1d


class WaveformWorker(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        decoder_blocks=3,
        decoder_channels=[512, 256, 128],
        decoder_kernel_sizes=[30, 30, 30],
        decoder_strides=[4, 4, 10],
        lin_neurons=256,
        activation=torch.nn.PReLU,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()

        for block_index in range(decoder_blocks):
            out_channels = decoder_channels[block_index]
            padding = max(0, (decoder_kernel_sizes[block_index] - decoder_strides[block_index]) // 2)
            self.blocks.extend(
                [
                    ConvTranspose1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=decoder_kernel_sizes[block_index],
                        stride=decoder_strides[block_index],
                        padding=padding,
                    ),
                    BatchNorm1d(input_size=out_channels),
                    activation(out_channels, init=0),
                ]
            )
            in_channels = decoder_channels[block_index]

        self.blocks.extend(
            [
                Conv1d(
                    in_channels=in_channels,
                    out_channels=lin_neurons,
                    kernel_size=1,
                    padding='valid',
                ),
                nn.PReLU(lin_neurons),
                Conv1d(in_channels=lin_neurons, out_channels=1, kernel_size=1, padding='valid')
            ],
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
