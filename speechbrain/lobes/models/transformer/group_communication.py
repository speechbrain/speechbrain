import torch
import torch.nn as nn
import torch.nn.functional as F

from speechbrain.nnet.group_linear import GroupLinear


class GroupCommunication(nn.Module):
    def __init__(self, dim, n_blocks):
        super(GroupCommunication, self).__init__()

        # self.n_heads = n_heads
        self.n_heads = 2
        self.n_blocks = n_blocks
        # self.head_dim = self.block_dim // self.n_heads
        self.head_dim = 32
        self.scale = self.head_dim ** -0.5

    def init_params(self, first_input):
        self.dim = first_input.shape[-1]
        self.block_dim = self.dim // self.n_blocks

        self.emb_dim = self.head_dim * self.n_heads * self.n_blocks

        self.query_net = GroupLinear(self.dim, self.emb_dim, self.n_blocks).to(
            first_input.device
        )
        self.key_net = GroupLinear(self.dim, self.emb_dim, self.n_blocks).to(
            first_input.device
        )
        self.value_net = GroupLinear(self.dim, self.emb_dim, self.n_blocks).to(
            first_input.device
        )
        self.final = GroupLinear(self.emb_dim, self.dim, self.n_blocks).to(
            first_input.device
        )

    def forward(self, x, init_params=False):
        if init_params:
            self.init_params(x)

        bsz, seq_len, _ = x.shape
        # x = x.view(seq_len, bsz, self.n_blocks, self.block_dim)

        q = self.query_net(x).view(
            bsz, seq_len, self.n_blocks, self.n_heads, self.head_dim
        )
        k = self.key_net(x).view(
            bsz, seq_len, self.n_blocks, self.n_heads, self.head_dim
        )
        v = self.value_net(x).view(
            bsz, seq_len, self.n_blocks, self.n_heads, self.head_dim
        )

        q = q.transpose(2, 3) * self.scale
        k = k.transpose(2, 3)
        v = v.transpose(2, 3)

        score = torch.matmul(q, k.transpose(3, 4))
        score = F.softmax(score, dim=-1)
        out = torch.matmul(score, v).transpose(2, 3)
        score = score.mean(dim=2)

        out = out.reshape(
            bsz, seq_len, self.n_blocks * self.head_dim * self.n_heads
        )
        out = self.final(out)
        out = out.view(bsz, seq_len, self.dim)

        return out
