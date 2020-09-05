import torch.nn as nn

class GroupLayerNorm(nn.Module):
    def __init__(self, dim, nb, eps=1e-6, export=False):
        super(GroupLayerNorm, self).__init__()

        self.num_rims = nb
        self.dim = dim

        self.weight = nn.Parameter(torch.ones(1,1,dim*num_rims,))
        self.bias = nn.Parameter(torch.zeros(1,1,dim*num_rims,))

        self.norm = nn.LayerNorm(dim, eps=eps, export=export, elementwise_affine=False)

    def forward(self, x):
        seq_len, bsz, _ = x.shape
        x = x.view(seq_len, bsz, self.num_rims, self.dim)

        x = self.norm(x)

        x = x.view(seq_len, bsz, self.num_rims * self.dim)

        weight_use = self.weight.repeat(seq_len, bsz, 1)
        bias_use = self.bias.repeat(seq_len, bsz, 1)

        x = x * weight_use + bias_use

        return x


