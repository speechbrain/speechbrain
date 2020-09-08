import torch.nn.functional as F
import torch
import torch.nn as nn
import math


class GroupLinear(nn.Module):
    def __init__(self, din, dout, nb, bias=True, a=None):
        super(GroupLinear, self).__init__()
        self.nb = nb

        din = din // nb
        dout = dout // nb

        self.dout = dout

        if a is None:
            a = 1.0 / math.sqrt(dout)

        # gain = 1.0 / math.sqrt(2)
        # a = gain * math.sqrt(6.0 / (din + dout))

        self.weight = nn.Parameter(
            torch.FloatTensor(nb, din, dout).uniform_(-a, a)
        )

        self.bias = bias

        if bias is True:
            self.bias = nn.Parameter(
                torch.FloatTensor(nb, dout).uniform_(-a, a)
            )
            # self.bias = nn.Parameter(torch.zeros(dout*nb))
        else:
            self.bias = None

    def forward(self, x):

        # input: ts x bs x blocks*nhid
        # ts*bs , blocks, nhid
        # blocks, ts*bs, nhid
        bs, ts, m = x.shape

        x = x.reshape((bs * ts, self.nb, m // self.nb))
        x = x.permute(1, 0, 2)
        x = torch.bmm(x, self.weight)
        x = x.permute(1, 0, 2)

        if self.bias is not None:
            x = x + self.bias

        x = x.reshape((bs, ts, self.dout * self.nb))

        # if not self.bias is None:
        #    x += self.bias

        return x


class GroupMLP(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, din, dout, nb, dropout=0.1):
        super(GroupMLP, self).__init__()

        self.w_1 = nn.Parameter(0.01 * torch.randn(nb, din, dout))
        self.w_2 = nn.Parameter(0.01 * torch.randn(nb, dout, din))

        self.layer_norm = nn.LayerNorm(din)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x * 1.0
        x = x.permute(1, 0, 2)
        x = torch.bmm(F.relu(torch.bmm(x, self.w_1)), self.w_2)
        x = x.permute(1, 0, 2)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)

        return x


if __name__ == "__main__":

    GLN = GroupLinear(512, 1024, 2, bias=True)

    # bs, blocks, nhid
    x = torch.randn(64, 12, 2 * 512)

    print(GLN(x).shape)

    # for p in GLN.parameters():
    #    print(p.shape)
