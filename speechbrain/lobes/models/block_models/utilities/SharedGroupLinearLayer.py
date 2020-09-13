import torch
import torch.nn as nn
from .GroupLinearLayer import GroupLinearLayer


class SharedGroupLinearLayer(nn.Module):
    """All the parameters are shared using soft attention this layer is used for sharing Q,K,V parameters of MHA"""

    def __init__(self, din, dout, n_templates):
        super(SharedGroupLinearLayer, self).__init__()

        self.w = nn.ModuleList(
            [nn.Linear(din, dout, bias=False) for _ in range(0, n_templates)]
        )
        self.gll_write = GroupLinearLayer(dout, 16, n_templates)
        self.gll_read = GroupLinearLayer(din, 16, 1)
        # self.register_buffer(self.w)

    def forward(self, x):
        # input size (bs,num_blocks,din), required matching num_blocks vs n_templates
        bs_size = x.shape[0]
        k = x.shape[1]
        x = x.reshape(k * bs_size, -1)
        x_read = self.gll_read((x * 1.0).reshape((x.shape[0], 1, x.shape[1])))
        x_next = []
        for mod in self.w:
            x_next_l = mod(x)
            x_next.append(x_next_l)
        x_next = torch.stack(x_next, 1)  # (k*bs,n_templates,dout)

        x_write = self.gll_write(x_next)
        sm = nn.Softmax(2)
        att = sm(torch.bmm(x_read, x_write.permute(0, 2, 1)))

        x_next = torch.bmm(att, x_next)

        x_next = x_next.mean(dim=1).reshape(bs_size, k, -1)

        return x_next


if __name__ == "__main__":

    GLN = SharedGroupLinearLayer(25, 22, 6)

    x = torch.randn(64, 12, 25)

    print(GLN(x).shape)

    for p in GLN.parameters():
        print(p.shape)
