"""
Gumbel Softmax implementation with multiple groups possible.

Authors
 * Rudolf A. Braun 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelVectorQuantizer(nn.Module):
    """Vector quantization using gumbel softmax. Copied from fairseq implementation.
    Arguments
    ---------
        input_dim: int
            Input dimension (channels).
        num_vars: int
            Number of quantized vectors per group.
        temp_tuple: float
            Temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor).
        groups: int
            Number of groups for vector quantization.
        vq_dim: int
            Dimensionality of the resulting quantized vector.

    Example
    -------
    >>> quantiser = GumbelVectorQuantizer(128, 100, (2.0, 0.25, 0.999995,), 2, 50 )
    >>> inputs = torch.rand(10, 12, 128)
    >>> output = quantiser(inputs)
    >>> output["x"].shape
    torch.Size([10, 12, 50])
    """

    def __init__(self, input_dim, num_vars, temp_tuple, groups, vq_dim):
        super().__init__()

        self.groups = groups
        self.input_dim = input_dim
        self.num_vars = num_vars
        self.vq_dim = vq_dim

        assert (
            vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        var_dim = vq_dim // groups

        self.vars = nn.Parameter(
            torch.FloatTensor(1, groups * num_vars, var_dim)
        )
        nn.init.uniform_(self.vars)

        self.weight_proj = nn.Linear(self.input_dim, groups * num_vars)
        nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
        nn.init.zeros_(self.weight_proj.bias)

        assert len(temp_tuple) == 3, temp_tuple

        self.max_temp, self.min_temp, self.temp_decay = temp_tuple
        self.curr_temp = self.max_temp
        self.max_ent = nn.Parameter(
            torch.log(torch.tensor(float(self.num_vars * self.groups))),
            requires_grad=False,
        )

    def update_temp(self, steps):
        """ Update the temperature given the current step """
        self.curr_temp = max(
            self.max_temp * self.temp_decay ** steps, self.min_temp
        )

    def forward(self, x):
        """ Forward the latent vector to obtain a quantised output """

        result = {
            "num_vars": self.num_vars * self.groups,
            "temp": self.curr_temp,
        }

        bsz, tsz, fsz = x.shape
        x = x.reshape(-1, fsz)
        x = self.weight_proj(x)
        x = x.view(bsz * tsz * self.groups, -1)

        _, k = x.max(-1)
        hard_x = (
            x.new_zeros(*x.shape)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(bsz * tsz, self.groups, -1)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)
        result["code_perplexity"] = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        ).sum()

        avg_probs = torch.softmax(
            x.view(bsz * tsz, self.groups, -1).float(), dim=-1
        ).mean(dim=0)
        result["prob_perplex"] = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        ).sum()

        result["temp"] = self.curr_temp

        if self.training:
            x = F.gumbel_softmax(
                x.float(), tau=self.curr_temp, hard=True
            ).type_as(x)
        else:
            x = hard_x

        x = x.view(bsz * tsz, -1)

        vars = self.vars
        x = x.unsqueeze(-1) * vars
        x = x.view(bsz * tsz, self.groups, self.num_vars, -1)
        x = x.sum(-2)
        x = x.view(bsz, tsz, -1)
        result["x"] = x
        return result
