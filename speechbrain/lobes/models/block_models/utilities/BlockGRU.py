"""
Goal1: a GRU where the weight matrices have a block structure so that information flow is constrained

Data is assumed to come in [block1, block2, ..., block_n].

Goal2: Dynamic parameter sharing between blocks (RIMs)

"""

import torch
import torch.nn as nn
from .GroupLinearLayer import GroupLinearLayer
from .sparse_attn import Sparse_attention


"""
Given an N x N matrix, and a grouping of size, set all elements off the block diagonal to 0.0
"""


def zero_matrix_elements(matrix, k):
    assert matrix.shape[0] % k == 0
    assert matrix.shape[1] % k == 0
    g1 = matrix.shape[0] // k
    g2 = matrix.shape[1] // k
    new_mat = torch.zeros_like(matrix)
    for b in range(0, k):
        new_mat[b * g1 : (b + 1) * g1, b * g2 : (b + 1) * g2] += matrix[
            b * g1 : (b + 1) * g1, b * g2 : (b + 1) * g2
        ]

    matrix *= 0.0
    matrix += new_mat


class BlockGRU(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ninp, nhid, k):
        super(BlockGRU, self).__init__()

        assert ninp % k == 0
        assert nhid % k == 0

        self.k = k
        self.gru = nn.GRUCell(ninp, nhid)
        self.nhid = nhid
        self.ninp = ninp

    def blockify_params(self):
        pl = self.gru.parameters()

        for p in pl:
            p = p.data
            if p.shape == torch.Size([self.nhid * 3]):
                pass
                """biases, don't need to change anything here"""
            if p.shape == torch.Size(
                [self.nhid * 3, self.nhid]
            ) or p.shape == torch.Size([self.nhid * 3, self.ninp]):
                for e in range(0, 4):
                    zero_matrix_elements(
                        p[self.nhid * e : self.nhid * (e + 1)], k=self.k
                    )

    def forward(self, input, h):

        # self.blockify_params()

        hnext = self.gru(input, h)

        return hnext, None


class SharedBlockGRU(nn.Module):
    """Dynamic sharing of parameters between blocks(RIM's)"""

    def __init__(self, ninp, nhid, k, n_templates):
        super(SharedBlockGRU, self).__init__()

        assert ninp % k == 0
        assert nhid % k == 0

        self.k = k
        self.m = nhid // self.k

        self.n_templates = n_templates
        self.templates = nn.ModuleList(
            [nn.GRUCell(ninp, self.m) for _ in range(0, self.n_templates)]
        )
        self.nhid = nhid

        self.ninp = ninp

        self.gll_write = GroupLinearLayer(self.m, 16, self.n_templates)
        self.gll_read = GroupLinearLayer(self.m, 16, 1)
        self.sa = Sparse_attention(1)
        print("Using Gumble sparsity")

    def blockify_params(self):

        return

    def forward(self, input, h):

        # self.blockify_params()
        bs = h.shape[0]
        h = h.reshape((h.shape[0], self.k, self.m)).reshape(
            (h.shape[0] * self.k, self.m)
        )

        input = input.reshape(input.shape[0], 1, input.shape[1])
        input = input.repeat(1, self.k, 1)
        input = input.reshape(input.shape[0] * self.k, input.shape[2])

        h_read = self.gll_read((h * 1.0).reshape((h.shape[0], 1, h.shape[1])))

        hnext_stack = []

        for template in self.templates:
            hnext_l = template(input, h)
            hnext_l = hnext_l.reshape((hnext_l.shape[0], 1, hnext_l.shape[1]))
            hnext_stack.append(hnext_l)

        hnext = torch.cat(hnext_stack, 1)

        write_key = self.gll_write(hnext)

        """
        sm = nn.Softmax(2)
        att = sm(torch.bmm(h_read, write_key.permute(0, 2, 1))).squeeze(1)
        att = self.sa(att).unsqueeze(1)
        """

        att = torch.nn.functional.gumbel_softmax(
            torch.bmm(h_read, write_key.permute(0, 2, 1)), tau=0.5, hard=True
        )

        # att = att*0.0 + 0.25

        # print('hnext shape before att', hnext.shape)
        hnext = torch.bmm(att, hnext)
        hnext = hnext.mean(dim=1)
        hnext = hnext.reshape((bs, self.k, self.m)).reshape(
            (bs, self.k * self.m)
        )
        # print('shapes', hnext.shape, cnext.shape)

        return hnext, att.data.reshape(bs, self.k, self.n_templates)


if __name__ == "__main__":

    Blocks = BlockGRU(2, 6, k=2)
    opt = torch.optim.Adam(Blocks.parameters())

    pl = Blocks.gru.parameters()

    inp = torch.randn(100, 2)
    h = torch.randn(100, 6)

    h2 = Blocks(inp, h)

    L = h2.sum() ** 2

    # L.backward()
    # opt.step()
    # opt.zero_grad()

    pl = Blocks.gru.parameters()
    for p in pl:
        print(p.shape)
        # print(torch.Size([Blocks.nhid*4]))
        if p.shape == torch.Size([Blocks.nhid * 3]):
            print(p.shape, "a")
            # print(p)
            """biases, don't need to change anything here"""
        if p.shape == torch.Size(
            [Blocks.nhid * 3, Blocks.nhid]
        ) or p.shape == torch.Size([Blocks.nhid * 3, Blocks.ninp]):
            print(p.shape, "b")
            for e in range(0, 4):
                print(p[Blocks.nhid * e : Blocks.nhid * (e + 1)])
