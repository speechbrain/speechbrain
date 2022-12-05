import torch.nn as nn
import torch.nn.functional as F
import torch


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.norm_type = norm_type

        if norm_type == "bn":
            self.norm1 = nn.BatchNorm2d(out_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        elif norm_type == "in":
            self.norm1 = nn.InstanceNorm2d(
                out_channels, affine=True, track_running_stats=True
            )
            self.norm2 = nn.InstanceNorm2d(
                out_channels, affine=True, track_running_stats=True
            )
        elif norm_type == "ln":
            self.norm1 = nn.GroupNorm(1, out_channels)
            self.norm2 = nn.GroupNorm(1, out_channels)
        else:
            raise ValueError("Unknown norm type {}".format(norm_type))

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.norm1)
        init_bn(self.norm2)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):

        x = input
        x = F.relu_(self.norm1(self.conv1(x)))
        x = F.relu_(self.norm2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x


class Cnn14(nn.Module):
    def __init__(self, mel_bins, emb_dim, norm_type="bn", interpret=False):

        super(Cnn14, self).__init__()
        self.interpret = interpret

        self.norm_type = norm_type
        if norm_type == "bn":
            self.norm0 = nn.BatchNorm2d(mel_bins)
        elif norm_type == "in":
            self.norm0 = nn.InstanceNorm2d(
                mel_bins, affine=True, track_running_stats=True
            )
        elif norm_type == "ln":
            self.norm0 = nn.GroupNorm(1, mel_bins)
        else:
            raise ValueError("Unknown norm type {}".format(norm_type))

        self.conv_block1 = ConvBlock(
            in_channels=1, out_channels=64, norm_type=norm_type
        )
        self.conv_block2 = ConvBlock(
            in_channels=64, out_channels=128, norm_type=norm_type
        )
        self.conv_block3 = ConvBlock(
            in_channels=128, out_channels=256, norm_type=norm_type
        )
        self.conv_block4 = ConvBlock(
            in_channels=256, out_channels=512, norm_type=norm_type
        )
        self.conv_block5 = ConvBlock(
            in_channels=512, out_channels=1024, norm_type=norm_type
        )
        self.conv_block6 = ConvBlock(
            in_channels=1024, out_channels=emb_dim, norm_type=norm_type
        )

        self.init_weight()

    def init_weight(self):
        init_bn(self.norm0)

    def forward(self, x, mixup_lambda=None):
        """
        Input: (B, 1, T, M)"""

        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.transpose(1, 3)
        x = self.norm0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x3_out = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x3_out, p=0.2, training=self.training)
        x2_out = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x2_out, p=0.2, training=self.training)
        x1_out = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x1_out, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        if not self.interpret:
            return x.unsqueeze(1)  # [B, 1, D]

        return x.unsqueeze(1), (x1_out, x2_out, x3_out)
