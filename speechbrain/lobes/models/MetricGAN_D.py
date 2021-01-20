"""Discriminator used in MetricGAN

WORK-IN-PROGRESS 

Authors
* Szu-Wei Fu 2020
"""
import torch
from torch import nn
from torch.nn.utils import spectral_norm


class MetricGAN_D(nn.Module):
    def __init__(
        self, kernel_size=(5, 5), base_channels=15, activation=nn.LeakyReLU,
    ):
        super().__init__()

        self.activation = activation(negative_slope=0.3)

        # self.IN = nn.InstanceNorm2d(num_features=2, affine=True, track_running_stats=True)
        self.BN = nn.BatchNorm2d(num_features=2)

        self.conv1 = spectral_norm(
            nn.Conv2d(
                in_channels=2,
                kernel_size=kernel_size,
                out_channels=base_channels,
            )
        )
        self.conv2 = spectral_norm(
            nn.Conv2d(
                in_channels=base_channels,
                kernel_size=kernel_size,
                out_channels=base_channels,
            )
        )
        self.conv3 = spectral_norm(
            nn.Conv2d(
                in_channels=base_channels,
                kernel_size=kernel_size,
                out_channels=base_channels,
            )
        )
        self.conv4 = spectral_norm(
            nn.Conv2d(
                in_channels=base_channels,
                kernel_size=kernel_size,
                out_channels=base_channels,
            )
        )
        self.Linear1 = spectral_norm(
            nn.Linear(in_features=base_channels, out_features=50)
        )
        self.Linear2 = spectral_norm(nn.Linear(in_features=50, out_features=10))
        self.Linear3 = spectral_norm(nn.Linear(in_features=10, out_features=1))

    def forward(self, x):
        out = self.BN(x)

        out = self.conv1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.activation(out)

        out = self.conv4(out)
        out = self.activation(out)

        out = torch.mean(out, (2, 3))

        out = self.Linear1(out)
        out = self.activation(out)

        out = self.Linear2(out)
        out = self.activation(out)

        out = self.Linear3(out)

        return out
      
      
#    def __init__(
#        self, kernel_size=(5, 5), base_channels=15, activation=nn.LeakyReLU,
#    ):
#        super().__init__()
#
#        self.IN = InstanceNorm2d(input_size=2, affine=True)
#        self.activation = activation()
#        self.conv1 = Conv2d(
#            in_channels=2, kernel_size=kernel_size, out_channels=base_channels
#        )
#        self.conv2 = Conv2d(
#            in_channels=base_channels,
#            kernel_size=kernel_size,
#            out_channels=base_channels,
#        )
#        self.conv3 = Conv2d(
#            in_channels=base_channels,
#            kernel_size=kernel_size,
#            out_channels=base_channels,
#        )
#        self.conv4 = Conv2d(
#            in_channels=base_channels,
#            kernel_size=kernel_size,
#            out_channels=base_channels,
#        )
#        self.liner1 = Linear(50, input_size=base_channels)
#        self.liner2 = Linear(10, input_size=50)
#        self.liner3 = Linear(1, input_size=10)


#    def forward(self, x):
#        out = self.IN(x)
#
#        out = self.conv1(out)
#        out = self.activation(out)
#
#        out = self.conv2(out)
#        out = self.activation(out)
#
#        out = self.conv3(out)
#        out = self.activation(out)
#
#        out = self.conv4(out)
#        out = self.activation(out)
#
#        out = torch.mean(out, (1, 2))
#
#        out = self.liner1(out)
#        out = self.activation(out)
#
#        out = self.liner2(out)
#        out = self.activation(out)
#
#        out = self.liner3(out)
#
#        return out
