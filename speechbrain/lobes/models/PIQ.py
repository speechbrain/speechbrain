import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from speechbrain.lobes.models.dual_path import (
    Encoder,
    SBTransformerBlock,
    Dual_Path_Model,
    Decoder,
)


class custom_classifier(nn.Module):
    def __init__(self, dim=128, num_classes=50):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, num_classes)

    def forward(self, z):
        z = F.relu(self.lin1(z))
        yhat = (self.lin2(z)).unsqueeze(1) 
        return yhat

class Conv2dEncoder_v2(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        # self.encoder = nn.Sequential(
        self.conv1 = nn.Conv2d(1, dim, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(dim)
        # nn.ReLU(True),
        # nn.LeakyReLU(),
        # nn.Conv2d(dim, dim, 4, 2, 1),
        # nn.BatchNorm2d(dim),
        # nn.ReLU(True),
        self.conv2 = nn.Conv2d(dim, dim, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(dim)
        # nn.ReLU(True),
        # nn.LeakyReLU(),
        self.conv3 = nn.Conv2d(dim, dim, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(dim)
        # nn.ReLU(True),
        # nn.LeakyReLU(),
        self.conv4 = nn.Conv2d(dim, dim, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(dim)

        self.resblock = ResBlockAudio(dim)
        # self.resblock2 = ResBlock(dim)

        self.nonl = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        h1 = self.conv1(x)
        h1 = self.bn1(h1)
        h1 = self.nonl(h1)

        h2 = self.conv2(h1)
        h2 = self.bn2(h2)
        h2 = self.nonl(h2)

        h3 = self.conv3(h2)
        h3 = self.bn3(h3)
        h3 = self.nonl(h3)

        h4 = self.conv4(h3)
        h4 = self.bn4(h4)
        h4 = self.nonl(h4)

        h4 = self.resblock(h4)

        return h4

class ResBlockAudio(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            # nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)
