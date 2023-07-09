"""ResNet PreActived for speaker verification

Authors
 * Mickael Rouvier 2022
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def conv3x3(in_planes, out_planes, stride=1):
    """2D convolution with kernel_size = 3"""

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """2D convolution with kernel_size = 1"""

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SEBlock(nn.Module):
    """An implementation of Squeeze-and-Excitation Block.

    Arguments
    ---------
    channels : int
        The number of channels.
    reduction : int
        The reduction factor of channels.

    Example
    -------
    >>> inp_tensor = torch.rand([1, 64, 80, 40])
    >>> se_layer = ResNet.SEBlock(64)
    >>> out_tensor = se_layer(inp_tensor)
    >>> out_tensor.shape
    torch.Size([1, 64, 80, 40])
    """

    def __init__(self, channels, reduction=1, activation=nn.ReLU):
        super(SEBlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            activation(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y



class BasicBlock(nn.Module):
    """An implementation of ResNet Block.

    Arguments
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        The number of output channels.
    stride : int
        Factor that reduce the spatial dimensionality
    downsample : torch function
        A function for downsample the identity of block when stride != 1
    activation : torch class
        A class for constructing the activation layers.

    Example
    -------
    >>> inp_tensor = torch.rand([1, 64, 80, 40])
    >>> layer = BasicBlock(64, 64, stride=1)
    >>> out_tensor = layer(inp_tensor)
    >>> out_tensor.shape
    torch.Size([1, 64, 80, 40])
    """

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, activation=nn.ReLU):
        super(BasicBlock, self).__init__()
        self.activation = activation()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = conv3x3(in_channels, out_channels, stride)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)

        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv1x1(out_channels, out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.activation(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out



class SEBasicBlock(nn.Module):
    """An implementation of Squeeze-and-Excitation ResNet Block.

    Arguments
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        The number of output channels.
    stride : int
        Factor that reduce the spatial dimensionality
    downsample : torch function
        A function for downsample the identity of block when stride != 1
    activation : torch class
        A class for constructing the activation layers.

    Example
    -------
    >>> inp_tensor = torch.rand([1, 64, 80, 40])
    >>> layer = BasicBlock(64, 64, stride=1)
    >>> out_tensor = layer(inp_tensor)
    >>> out_tensor.shape
    torch.Size([1, 64, 80, 40])
    """

    def __init__(self, in_channels, out_channels, reduction=1, stride=1, downsample=None, activation=nn.ReLU):
        super(SEBasicBlock, self).__init__()
        self.activation = activation()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = conv3x3(in_channels, out_channels, stride)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)

        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv1x1(out_channels, out_channels)

        self.downsample = downsample
        self.stride = stride

        self.se = SEBlock(out_channels, reduction)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.activation(out)
        out = self.conv3(out)

        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out




class ResNet(nn.Module):
    """An implementation of ResNet

    Arguments
    ---------
    device : str
        Device used, e.g., "cpu" or "cuda".
    activation : torch class
        A class for constructing the activation layers.
    channels : list of ints
        List of number of channels used per stage.
    block_sizes : list of ints
        List of number of groups created per stage.
    strides : list of ints
        List of stride per stage.
    lin_neurons : int
        Number of neurons in linear layers.

    Example
    -------
    >>> input_feats = torch.rand([2, 1, 400, 80])
    >>> compute_embedding = ResNet(lin_neurons=256)
    >>> outputs = compute_embedding(input_feats)
    >>> outputs.shape
    torch.Size([2, 256])
    """


    def __init__(
        self,
        input_size=80,
        device="cpu",
        activation=torch.nn.ReLU,
        channels=[128, 128, 256, 256],
        block_sizes=[3, 4, 6, 3],
        strides=[1, 2, 2, 2],
        lin_neurons=256,
    ):

        super().__init__()

        assert len(channels) == 4
        assert len(block_sizes) == 4
        assert len(strides) == 4

        input_out = math.ceil(input_size/(strides[0]*strides[1]*strides[2]*strides[3]))

        self.conv1 = nn.Conv2d(1, channels[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.activation1 = activation()

        self.layer1 = self._make_layer_se(channels[0], channels[0], block_sizes[0], stride=strides[0])
        self.layer2 = self._make_layer_se(channels[0], channels[1], block_sizes[1], stride=strides[1])
        self.layer3 = self._make_layer(channels[1], channels[2], block_sizes[2], stride=strides[2])
        self.layer4 = self._make_layer(channels[2], channels[3], block_sizes[3], stride=strides[3])

        self.norm_stats = torch.nn.BatchNorm1d(2*input_out*channels[-1])

        self.attention = nn.Sequential(
                nn.Conv1d(channels[-1]*input_out, 128, kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Conv1d(128, channels[-1]*input_out, kernel_size=1),
                nn.Softmax(dim=2)
        )

        self.fc_embed = nn.Linear(2*input_out*channels[-1], lin_neurons)
        self.norm_embed = torch.nn.BatchNorm1d(lin_neurons)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer_se(self, in_channels, out_channels, block_num, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(out_channels)
                        )

        layers = []
        layers.append(SEBasicBlock(in_channels, out_channels, 1, stride, downsample))

        for i in range(1, block_num):
            layers.append(SEBasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)



    def _make_layer(self, in_channels, out_channels, block_num, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(out_channels)
                        )

        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))

        for i in range(1, block_num):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.transpose(2,3)
        x = x.flatten(1,2)

        w = self.attention(x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-5) )
        x = torch.cat([mu,sg], dim=1)
        x = self.norm_stats(x)

        x = self.fc_embed(x)
        x = self.norm_embed(x)

        return x

class Classifier(torch.nn.Module):
    """This class implements the cosine similarity on the top of features.

    Arguments
    ---------
    device : str
        Device used, e.g., "cpu" or "cuda".
    lin_neurons : int
        Number of neurons in linear layers.
    out_neurons : int
        Number of classes.
    """

    def __init__(
        self,
        device="cpu",
        lin_neurons=256,
        out_neurons=5994,
    ):

        super().__init__()
        self.W = nn.Parameter(torch.FloatTensor(out_neurons, lin_neurons))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x):
        x = F.normalize(x)
        W = F.normalize(self.W)

        return F.linear(x, W)
