import torch
import torch.nn as nn


class GroupLayerNorm(nn.Module):
    def __init__(
        self, dim, nb, eps=1e-6,
    ):
        super(GroupLayerNorm, self).__init__()

        self.num_rims = nb
        self.dim = dim
        self.eps = eps

    def init_params(self, first_input):
        self.dim = first_input.shape[-1]

        self.weight = nn.Parameter(
            torch.ones([1, 1, self.dim], device=first_input.device)
        )
        self.bias = nn.Parameter(
            torch.zeros([1, 1, self.dim], device=first_input.device)
        )

        self.norm = nn.LayerNorm(
            self.dim // self.num_rims, eps=self.eps, elementwise_affine=False
        ).to(first_input.device)

    def forward(self, x, init_params=False):
        if init_params:
            self.init_params(x)

        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.num_rims, self.dim // self.num_rims)

        x = self.norm(x)

        x = x.view(bsz, seq_len, self.dim)

        # weight_use = self.weight.repeat(bsz, seq_len, 1)
        # bias_use = self.bias.repeat(bsz, seq_len, 1)

        x = x * self.weight + self.bias

        return x
