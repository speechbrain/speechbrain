import torch
import torch.nn as nn

import speechbrain as sb
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.normalization import BatchNorm1d


class GIMWorker(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_blocks=1,
        hidden_channels=256,
        hidden_kernel_size=1,
        hidden_activation=torch.nn.PReLU,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()

        for block_index in range(hidden_blocks):
            self.blocks.extend(
                [
                    Conv1d(
                        in_channels=in_channels, out_channels=hidden_channels, kernel_size=hidden_kernel_size,
                    ),
                    hidden_activation(hidden_channels),
                ],
            )
            in_channels = hidden_channels

        self.blocks.extend(
            [
                Conv1d(in_channels=in_channels, out_channels=1, kernel_size=1),
            ],
        )

    def make_samples(self, embeddings):
        embedding_sig, embedding_pos, embedding_neg = embeddings
        x_pos_lim = torch.cat((embedding_sig, embedding_pos), dim=2)
        x_neg_lim = torch.cat((embedding_sig, embedding_neg), dim=2)

        x_pos_gim = torch.mean(x_pos_lim, dim=1).unsqueeze(1)    # (batch, 1, 200)
        x_neg_gim = torch.mean(x_neg_lim, dim=1).unsqueeze(1)    # (batch, 1, 200)

        return torch.cat((x_pos_gim, x_neg_gim), dim=0)     # (2*batch, 1, 200)

    def forward(self, embeddings, *args, **kwargs):
        x = self.make_samples(embeddings)
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
