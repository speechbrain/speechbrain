import torch
import torch.nn as nn


class GroupLinearLayer(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, din, dout, num_blocks):
        super(GroupLinearLayer, self).__init__()

        self.w = nn.Parameter(0.01 * torch.randn(num_blocks, din, dout))

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = torch.bmm(x, self.w)
        return x.permute(1, 0, 2)


if __name__ == "__main__":

    GLN = GroupLinearLayer()

    x = torch.randn(64, 12, 25)

    print(GLN(x).shape)

    for p in GLN.parameters():
        print(p.shape)
