"""Discriminator used in MetricGAN

Authors
* Szu-Wei Fu 2020
"""
import torch
from torch import nn
from torch.nn.utils import spectral_norm


class generator(nn.Module):
    def __init__(
        self,
        input_size=257,
        hidden_size=200,
        num_layers=2,
        dropout=0,
        activation=nn.LeakyReLU,
    ):
        super().__init__()
        self.activation = activation(negative_slope=0.3)

        self.blstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0 except for forget gate
        """
        for name, param in self.blstm.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
                param.data[hidden_size : 2 * hidden_size] = 1
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

        self.linear1 = nn.Linear(400, 300, bias=True)
        nn.init.xavier_uniform_(self.linear1.weight, gain=1.0)
        nn.init.zeros_(self.linear1.bias)

        self.linear2 = nn.Linear(300, 257, bias=True)
        nn.init.xavier_uniform_(self.linear2.weight, gain=1.0)
        nn.init.zeros_(self.linear2.bias)

    def my_sigmoid(self, x):
        return 1.2 / (1 + torch.exp(-(1 / 1.6) * x))

    def forward(self, x):
        out, _ = self.blstm(x)

        out = self.linear1(out)
        out = self.activation(out)

        out = self.linear2(out)
        out = self.my_sigmoid(out)

        return out


class discriminator(nn.Module):
    def __init__(
        self, kernel_size=(5, 5), base_channels=15, activation=nn.LeakyReLU,
    ):
        super().__init__()

        self.activation = activation(negative_slope=0.3)

        self.IN = nn.InstanceNorm2d(
            num_features=2,
            momentum=0.01,
            affine=False,
            track_running_stats=False,
        )
        # self.BN = nn.BatchNorm2d(num_features=2, momentum=0.01)

        self.conv1 = spectral_norm(
            nn.Conv2d(
                in_channels=2,
                kernel_size=kernel_size,
                out_channels=base_channels,
            )
        )
        nn.init.xavier_uniform_(self.conv1.weight, gain=1.0)
        nn.init.zeros_(self.conv1.bias)

        self.conv2 = spectral_norm(
            nn.Conv2d(
                in_channels=base_channels,
                kernel_size=kernel_size,
                out_channels=base_channels,
            )
        )
        nn.init.xavier_uniform_(self.conv2.weight, gain=1.0)
        nn.init.zeros_(self.conv2.bias)

        self.conv3 = spectral_norm(
            nn.Conv2d(
                in_channels=base_channels,
                kernel_size=kernel_size,
                out_channels=base_channels,
            )
        )
        nn.init.xavier_uniform_(self.conv3.weight, gain=1.0)
        nn.init.zeros_(self.conv3.bias)

        self.conv4 = spectral_norm(
            nn.Conv2d(
                in_channels=base_channels,
                kernel_size=kernel_size,
                out_channels=base_channels,
            )
        )
        nn.init.xavier_uniform_(self.conv4.weight, gain=1.0)
        nn.init.zeros_(self.conv4.bias)

        self.Linear1 = spectral_norm(
            nn.Linear(in_features=base_channels, out_features=50)
        )
        nn.init.xavier_uniform_(self.Linear1.weight, gain=1.0)
        nn.init.zeros_(self.Linear1.bias)

        self.Linear2 = spectral_norm(nn.Linear(in_features=50, out_features=10))
        nn.init.xavier_uniform_(self.Linear2.weight, gain=1.0)
        nn.init.zeros_(self.Linear2.bias)

        self.Linear3 = spectral_norm(nn.Linear(in_features=10, out_features=1))
        nn.init.xavier_uniform_(self.Linear3.weight, gain=1.0)
        nn.init.zeros_(self.Linear3.bias)

    def forward(self, x):
        out = self.IN(x)

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
