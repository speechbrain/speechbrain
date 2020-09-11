import torch
import torch.nn as nn


class GroupLinearLayer(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, din, dout, num_blocks):
        super(GroupLinearLayer, self).__init__()

        # gain = 1.0 / math.sqrt(2)
        # a = gain * math.sqrt(6.0 / (din + dout))

        a = 0.1
        self.weight = nn.Parameter(
            torch.FloatTensor(num_blocks, din, dout).uniform_(-a, a)
        )

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = torch.bmm(x, self.weight)
        x = x.permute(1, 0, 2)

        return x


class SharedGroupLinearLayer(nn.Module):
    """All the parameters are shared using soft attention this layer is used for sharing Q,K,V parameters of MHA"""

    def __init__(self, din, dout, num_blocks, ns=None, bias=True, a=None):
        super(SharedGroupLinearLayer, self).__init__()

        if ns is None:
            ns = num_blocks

        if num_blocks == 1:
            n_templates = 1
        else:
            n_templates = ns

        self.w = nn.ModuleList(
            [nn.Linear(din, dout) for _ in range(0, n_templates)]
        )
        self.gll_write = GroupLinearLayer(dout, 16, n_templates)
        self.gll_read = GroupLinearLayer(din, 16, 1)
        self.nb = num_blocks

        self.weight = self.w[0].weight
        self.bias = None

    def forward(self, x):
        T, bsz, _ = x.shape
        x = x.reshape((x.shape[0] * x.shape[1], self.nb, x.shape[2] // self.nb))
        # input size (bs,num_blocks,din), required matching num_blocks vs n_templates
        bs_size = x.shape[0]
        k = x.shape[1]
        x = x.reshape(k * bs_size, -1)
        x_read = self.gll_read((x * 1.0).reshape((x.shape[0], 1, x.shape[1])))
        x_next = []
        for layer in self.w:
            x_next_l = layer(x)
            x_next.append(x_next_l)
        x_next = torch.stack(x_next, 1)  # (k*bs,n_templates,dout)

        x_write = self.gll_write(x_next)

        sm = nn.Softmax(2)
        att = sm(torch.bmm(x_read, x_write.permute(0, 2, 1)))

        x_next = torch.bmm(att, x_next)

        x_next = x_next.mean(dim=1).reshape(bs_size, k, -1)

        x_next = x_next.reshape((T, bsz, self.nb * x_next.shape[2]))

        return x_next


if __name__ == "__main__":

    nb = 4
    ns = 2
    din = 512
    dout = 512
    GLN = SharedGroupLinearLayer(din, dout, nb, ns)

    # for g in GLN.parameters():
    #    print(g.shape)

    print("params", sum(g.numel() for g in GLN.parameters()))

    T = 10
    bs = 10

    x = torch.randn(T, bs, nb * din)
    print("x shape", x.shape)
    # x = torch.randn(bs,nb,128)

    print(GLN(x).shape)
