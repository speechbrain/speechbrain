import torch
import torch.nn as nn
import torch.nn.functional as F

from speechbrain.nnet.group_linear import GroupLinear

class GroupCommunication(nn.Module):
    def __init__(self, dim, n_blocks):
        super(GroupCommunication, self).__init__()

        #self.n_heads = n_heads
        self.n_heads = 2
        self.n_blocks = n_blocks
        self.dim = dim
        self.block_dim = dim // self.n_blocks
        #self.head_dim = self.block_dim // self.n_heads
        self.head_dim = 32
        self.scale = self.head_dim ** -0.5

        emb_dim = self.head_dim * self.n_heads * n_blocks

        self.query_net = GroupLinear(dim, emb_dim, n_blocks)
        self.key_net = GroupLinear(dim, emb_dim, n_blocks)
        self.value_net = GroupLinear(dim, emb_dim, n_blocks)
        self.final = GroupLinear(emb_dim, dim, n_blocks)

    def forward(self, x):
        seq_len, bsz, _ = x.shape
        #x = x.view(seq_len, bsz, self.n_blocks, self.block_dim)

        q = self.query_net(x).view(seq_len, bsz, self.n_blocks, self.n_heads, self.head_dim)
        k = self.key_net(x).view(seq_len, bsz, self.n_blocks, self.n_heads, self.head_dim)
        v = self.value_net(x).view(seq_len, bsz, self.n_blocks, self.n_heads, self.head_dim)

        q = q.transpose(2,3) * self.scale
        k = k.transpose(2,3)
        v = v.transpose(2,3)

        score = torch.matmul(q, k.transpose(3,4))
        score = F.softmax(score, dim=-1)
        out = torch.matmul(score, v).transpose(2,3)
        score = score.mean(dim=2)

        out = out.reshape(seq_len, bsz, self.n_blocks * self.head_dim * self.n_heads)
        out = self.final(out)
        out = out.view(seq_len, bsz, self.dim)

        return out



