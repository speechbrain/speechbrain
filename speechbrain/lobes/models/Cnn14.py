""" This file implements the CNN14 model from https://arxiv.org/abs/1912.10211

 Authors
 * Cem Subakan 2022
 * Francesco Paissan 2022
"""

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
    """This class implements the convolutional block used in CNN14

    Arguments
    ---------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    norm_type : str in ['bn', 'in', 'ln']
        The type of normalization

    Example:
    --------
    >>> convblock = ConvBlock(10, 20, 'ln')
    >>> x = torch.rand(5, 10, 20, 30)
    >>> y = convblock(x)
    >>> print(y.shape)
    torch.Size([5, 20, 10, 15])
    """

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
        """
        Initializes the model convolutional layers and the batchnorm layers
        """
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.norm1)
        init_bn(self.norm2)

    def forward(self, x, pool_size=(2, 2), pool_type="avg"):
        """The forward pass for convblocks in CNN14

        Arguments:
        ----------
        x : torch.Tensor
            input tensor with shape B x C_in x D1 x D2
        where B = Batchsize
              C_in = Number of input channel
              D1 = Dimensionality of the first spatial dim
              D2 = Dimensionality of the second spatial dim
        pool_size : tuple with integer values
            Amount of pooling at each layer
        pool_type : str in ['max', 'avg', 'avg+max']
            The type of pooling
        """

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
            raise Exception("Incorrect pooling type!")
        return x


class Cnn14(nn.Module):
    """This class implements the Cnn14 model from https://arxiv.org/abs/1912.10211

    Arguments
    ---------
    mel_bins : int
        Number of mel frequency bins in the input
    emb_dim : int
        The dimensionality of the output embeddings
    norm_type: str in ['bn', 'in', 'ln']
        The type of normalization
    return_reps: bool (default=False)
        If True the model returns intermediate representations as well for interpretation

    Example:
    --------
    >>> cnn14 = Cnn14(120, 256)
    >>> x = torch.rand(3, 400, 120)
    >>> h = cnn14.forward(x)
    >>> print(h.shape)
    torch.Size([3, 1, 256])
    """

    def __init__(self, mel_bins, emb_dim, norm_type="bn", return_reps=False):
        super(Cnn14, self).__init__()
        self.return_reps = return_reps

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
        """
        Initializes the model batch norm layer
        """
        init_bn(self.norm0)

    def forward(self, x):
        """
        The forward pass for the CNN14 encoder

        Arguments:
        ----------
        x : torch.Tensor
            input tensor with shape B x C_in x D1 x D2
        where B = Batchsize
              C_in = Number of input channel
              D1 = Dimensionality of the first spatial dim
              D2 = Dimensionality of the second spatial dim
        """

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

        # [B x 1 x emb_dim]
        if not self.return_reps:
            return x.unsqueeze(1)

        return x.unsqueeze(1), (x1_out, x2_out, x3_out)
