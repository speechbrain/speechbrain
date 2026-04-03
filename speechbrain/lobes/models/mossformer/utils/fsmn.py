import torch as th
import torch.nn as nn
import torch.nn.functional as F


class UniDeepFsmn(nn.Module):
    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None):
        super(UniDeepFsmn, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if lorder is None:
            return

        self.lorder = lorder
        self.hidden_size = hidden_size

        self.linear = nn.Linear(input_dim, hidden_size)

        self.project = nn.Linear(hidden_size, output_dim, bias=False)

        self.conv1 = nn.Conv2d(
            output_dim,
            output_dim,
            [lorder + lorder - 1, 1],
            [1, 1],
            groups=output_dim,
            bias=False,
        )

    def forward(self, input):
        f1 = F.relu(self.linear(input))

        p1 = self.project(f1)

        x = th.unsqueeze(p1, 1)

        x_per = x.permute(0, 3, 2, 1)

        y = F.pad(x_per, [0, 0, self.lorder - 1, self.lorder - 1])

        out = x_per + self.conv1(y)

        out1 = out.permute(0, 3, 2, 1)

        return input + out1.squeeze()


class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, lorder=20, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        self.twidth = lorder * 2 - 1
        self.kernel_size = (self.twidth, 1)
        for i in range(self.depth):
            dil = 2**i
            pad_length = lorder + (dil - 1) * (lorder - 1) - 1
            setattr(
                self,
                f"pad{i + 1}",
                nn.ConstantPad2d((0, 0, pad_length, pad_length), value=0.0),
            )
            setattr(
                self,
                f"conv{i + 1}",
                nn.Conv2d(
                    self.in_channels * (i + 1),
                    self.in_channels,
                    kernel_size=self.kernel_size,
                    dilation=(dil, 1),
                    groups=self.in_channels,
                    bias=False,
                ),
            )
            setattr(
                self,
                f"norm{i + 1}",
                nn.InstanceNorm2d(in_channels, affine=True),
            )
            setattr(self, f"prelu{i + 1}", nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, f"pad{i + 1}")(skip)
            out = getattr(self, f"conv{i + 1}")(out)
            out = getattr(self, f"norm{i + 1}")(out)
            out = getattr(self, f"prelu{i + 1}")(out)
            skip = th.cat([out, skip], dim=1)
        return out


class UniDeepFsmn_dilated(nn.Module):
    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None):
        super(UniDeepFsmn_dilated, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if lorder is None:
            return

        self.lorder = lorder
        self.hidden_size = hidden_size

        self.linear = nn.Linear(input_dim, hidden_size)

        self.project = nn.Linear(hidden_size, output_dim, bias=False)

        self.conv = DilatedDenseNet(
            depth=2, lorder=lorder, in_channels=output_dim
        )

    def forward(self, input):
        f1 = F.relu(self.linear(input))

        p1 = self.project(f1)

        x = th.unsqueeze(p1, 1)

        x_per = x.permute(0, 3, 2, 1)

        out = self.conv(x_per)

        out1 = out.permute(0, 3, 2, 1)

        return input + out1.squeeze()
