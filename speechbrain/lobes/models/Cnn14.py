""" This file implements the CNN14 model from https://arxiv.org/abs/1912.10211

 Authors
 * Cem Subakan 2022
 * Francesco Paissan 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    Example
    -------
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

        Arguments
        ---------
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

        Returns
        -------
        The output of one conv block
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
    l2i : bool
        If True, remove one of the outputs.

    Example
    -------
    >>> cnn14 = Cnn14(120, 256)
    >>> x = torch.rand(3, 400, 120)
    >>> h = cnn14.forward(x)
    >>> print(h.shape)
    torch.Size([3, 1, 256])
    """

    def __init__(
        self, mel_bins, emb_dim, norm_type="bn", return_reps=False, l2i=False
    ):
        super(Cnn14, self).__init__()
        self.return_reps = return_reps
        self.l2i = l2i

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

        Arguments
        ---------
        x : torch.Tensor
            input tensor with shape B x C_in x D1 x D2
            where B = Batchsize
                  C_in = Number of input channel
                  D1 = Dimensionality of the first spatial dim
                  D2 = Dimensionality of the second spatial dim

        Returns
        -------
        Outputs of CNN14 encoder
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
        x4_out = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x4_out, p=0.2, training=self.training)
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

        if self.l2i:
            return x.unsqueeze(1), (x1_out, x2_out, x3_out)
        else:
            return x.unsqueeze(1), (x1_out, x2_out, x3_out, x4_out)


class CNN14PSI(nn.Module):
    """
    This class estimates a mel-domain saliency mask

    Arguments
    ---------
    dim : int
        Dimensionality of the embeddings

    Returns
    -------
        Estimated saliency map (before sigmoid)

    Example
    -------
    >>> from speechbrain.lobes.models.Cnn14 import Cnn14
    >>> classifier_embedder = Cnn14(mel_bins=80, emb_dim=2048, return_reps=True)
    >>> x = torch.randn(2, 201, 80)
    >>> _, hs = classifier_embedder(x)
    >>> psimodel = CNN14PSI(2048)
    >>> xhat = psimodel.forward(hs)
    >>> print(xhat.shape)
    torch.Size([2, 1, 201, 80])
    """

    def __init__(
        self,
        dim=128,
    ):
        super().__init__()

        self.convt1 = nn.ConvTranspose2d(dim, dim, 3, (2, 2), 1)
        self.convt2 = nn.ConvTranspose2d(dim // 2, dim, 3, (2, 2), 1)
        self.convt3 = nn.ConvTranspose2d(dim, dim, (7, 4), (2, 4), 1)
        self.convt4 = nn.ConvTranspose2d(dim // 4, dim, (5, 4), (2, 2), 1)
        self.convt5 = nn.ConvTranspose2d(dim, dim, (3, 3), (2, 2), 1)
        self.convt6 = nn.ConvTranspose2d(dim // 8, dim, (3, 3), (2, 2), 1)
        self.convt7 = nn.ConvTranspose2d(dim, dim, (4, 3), (2, 2), 0)
        self.convt8 = nn.ConvTranspose2d(dim, 1, (3, 4), (2, 2), 0)

        self.nonl = nn.ReLU(True)

    def forward(self, hs, labels=None):
        """
        Forward step. Given the classifier representations estimates a saliency map.

        Arguments
        ---------
        hs : torch.Tensor
            Classifier's representations.
        labels : None
            Unused

        Returns
        -------
        xhat : torch.Tensor
            Estimated saliency map (before sigmoid)
        """

        h1 = self.convt1(hs[0])
        h1 = self.nonl(h1)

        h2 = self.convt2(hs[1])
        h2 = self.nonl(h2)
        h = h1 + h2

        h3 = self.convt3(h)
        h3 = self.nonl(h3)

        h4 = self.convt4(hs[2])
        h4 = self.nonl(h4)
        h = h3 + h4

        h5 = self.convt5(h)
        h5 = self.nonl(h5)

        h6 = self.convt6(hs[3])
        h6 = self.nonl(h6)
        h = h5 + h6

        h = self.convt7(h)
        h = self.nonl(h)

        xhat = self.convt8(h)
        return xhat


class CNN14PSI_stft(nn.Module):
    """
    This class estimates a saliency map on the STFT domain, given classifier representations.

    Arguments
    ---------
    dim : int
        Dimensionality of the input representations.
    outdim : int
        Defines the number of output channels in the saliency map.

    Example
    -------
    >>> from speechbrain.lobes.models.Cnn14 import Cnn14
    >>> classifier_embedder = Cnn14(mel_bins=80, emb_dim=2048, return_reps=True)
    >>> x = torch.randn(2, 201, 80)
    >>> _, hs = classifier_embedder(x)
    >>> psimodel = CNN14PSI_stft(2048, 1)
    >>> xhat = psimodel.forward(hs)
    >>> print(xhat.shape)
    torch.Size([2, 1, 201, 513])
    """

    def __init__(self, dim=128, outdim=1):
        super().__init__()

        self.convt1 = nn.ConvTranspose2d(dim, dim, 3, (2, 4), 1)
        self.convt2 = nn.ConvTranspose2d(dim // 2, dim, 3, (2, 4), 1)
        self.convt3 = nn.ConvTranspose2d(dim, dim, (7, 4), (2, 4), 1)
        self.convt4 = nn.ConvTranspose2d(dim // 4, dim, (5, 4), (2, 4), 1)
        self.convt5 = nn.ConvTranspose2d(dim, dim // 2, (3, 5), (2, 2), 1)
        self.convt6 = nn.ConvTranspose2d(dim // 8, dim // 2, (3, 3), (2, 4), 1)
        self.convt7 = nn.ConvTranspose2d(
            dim // 2, dim // 4, (4, 3), (2, 2), (0, 5)
        )
        self.convt8 = nn.ConvTranspose2d(
            dim // 4, dim // 8, (3, 4), (2, 2), (0, 2)
        )
        self.convt9 = nn.ConvTranspose2d(dim // 8, outdim, (1, 5), (1, 4), 0)

        self.nonl = nn.ReLU(True)

    def forward(self, hs):
        """
        Forward step to estimate the saliency map

        Arguments
        --------
        hs : torch.Tensor
            Classifier's representations.

        Returns
        --------
        xhat : torch.Tensor
            An Estimate for the saliency map
        """

        h1 = self.convt1(hs[0])
        h1 = self.nonl(h1)

        h2 = self.convt2(hs[1])
        h2 = self.nonl(h2)
        h = h1 + h2

        h3 = self.convt3(h)
        h3 = self.nonl(h3)

        h4 = self.convt4(hs[2])
        h4 = self.nonl(h4)
        h = h3 + h4

        h5 = self.convt5(h)
        h5 = self.nonl(h5)

        h6 = self.convt6(hs[3])
        h6 = self.nonl(h6)

        h = h5 + h6

        h = self.convt7(h)
        h = self.nonl(h)

        h = self.convt8(h)
        xhat = self.convt9(h)

        return xhat
