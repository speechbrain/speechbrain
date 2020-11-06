import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules.group_linear_layer import GroupLinearLayer

class MemoryAttention(nn.Module):
    def __init__(self, n_blocks_query, n_blocks_val, dim_query, dim_val, n_heads=8):
        super(MemoryAttention, self).__init__()

        self.n_heads = n_heads
        self.n_blocks_val = n_blocks_val
        self.dim_val = dim_val
        self.block_dim_val = dim_val // self.n_blocks_val

        self.n_blocks_query = n_blocks_query
        self.dim_query = dim_query
        self.block_dim_query = dim_query // self.n_blocks_query

        self.head_dim = 64
        self.scale = self.head_dim ** -0.5

        self.query_net = GroupLinearLayer(self.block_dim_query, self.head_dim * self.n_heads, n_blocks_query)
        self.key_net = GroupLinearLayer(self.block_dim_val, self.head_dim * self.n_heads, n_blocks_val)
        self.value_net = GroupLinearLayer(self.block_dim_val, self.head_dim * self.n_heads, n_blocks_val)
        self.final = GroupLinearLayer(self.head_dim * self.n_heads, self.block_dim_query, n_blocks_query)

    def forward(self, q, kv):

        #comes in as: bs, pos*emb.  
        #positions_attend x T*bs x emb


        #q = q.permute(1,0,2)
        #kv = kv.permute(1,0,2)

        #print('kv shape after permute', kv.shape)

        seq_len_q,bsz,_ = q.shape
        seq_len_v,bsz,_ = kv.shape

        q = q.reshape((seq_len_q, bsz, self.n_blocks_query * self.block_dim_query))
        
        kv = kv.reshape((seq_len_v, bsz, self.n_blocks_val * self.block_dim_val))

        q = self.query_net(q).view(seq_len_q, bsz, self.n_blocks_query, self.n_heads, self.head_dim)
        k = self.key_net(kv).view(seq_len_v, bsz, self.n_blocks_val, self.n_heads, self.head_dim)
        v = self.value_net(kv).view(seq_len_v, bsz, self.n_blocks_val, self.n_heads, self.head_dim)

        q = q.transpose(2,3) * self.scale
        k = k.transpose(2,3)
        v = v.transpose(2,3)
        score = torch.matmul(q, k.transpose(3,4))
        #print('score shape', score.shape)
        score = F.softmax(score, dim=-1)
        out = torch.matmul(score, v).transpose(2,3)
        #print('out shape', out.shape)
        score = score.mean(dim=2)

        out = out.reshape(seq_len_q, bsz, self.n_blocks_query * self.head_dim * self.n_heads)
        out = self.final(out)
        out = out.view(seq_len_q, bsz, self.dim_query)


        return out, score

