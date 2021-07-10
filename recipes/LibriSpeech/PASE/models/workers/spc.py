
import numpy as np
import torch
import torch.nn as nn

import speechbrain as sb
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.normalization import BatchNorm1d


class SPCWorker(torch.nn.Module):
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

    def select_chunk_SPC(self, input_tensor):
    # select samples for SPC
    # input_tensor: (batch, number_of_frames, embedding_size)

      batch_size = input_tensor.shape[0]
      framnum = input_tensor.shape[1]

      idx_ls1 = np.arange(30, framnum - 30)

      anchor_idx = np.random.choice(idx_ls1)

      idx_lsp = np.arange(anchor_idx + 15, np.minimum(framnum - 5, anchor_idx + 50))
      idx_lsn = np.arange(np.maximum(5, anchor_idx-50), anchor_idx - 15)

      xp_id = np.random.choice(idx_lsp)
      xn_id = np.random.choice(idx_lsn)

      xp_ids =[xp_id - i for i in range(4, -1, -1)]
      xn_ids = [xn_id - i for i in range(4, -1, -1)]

      x_anchor = input_tensor[:,[anchor_idx],:].contiguous().view(batch_size,1, -1)#(batch, 1,embedding_size)
      x_pos = input_tensor[:,xp_ids,:].contiguous().view(batch_size,1, -1)#(batch, 1,5*embedding_size)
      x_neg = input_tensor[:,xn_ids,:].contiguous().view(batch_size,1, -1)#(batch, 1,5*embedding_size)
      return x_anchor, x_pos, x_neg


    def make_samples(self, embeddings):
        embedding_si = embeddings[0]#(batch, number_of_frames, embedding_size)

        x_anchor, x_pos_spc, x_neg_spc = self.select_chunk_SPC(embedding_si)

        x_pos = torch.cat((x_anchor, x_pos_spc), dim=-1)#(batch,1, 6*embedding_size)
        x_neg = torch.cat((x_anchor, x_neg_spc), dim=-1)#(batch,1, 6*embedding_size)

        return torch.cat((x_pos, x_neg), dim=0)#(2*batch, 1,6*embedding_size)

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
