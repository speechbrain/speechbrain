import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys

import argparse
import copy


class TcResNet(torch.nn.Module):
    def __init__(self, input_size=24):
        super().__init__()

        n_channels = [16, 24, 32, 48]
        # n_channels = config['n_channels']
        # dilation = [2, 4, 8]
        n_blocks = len(n_channels) - 1

        self.n_layers = n_blocks

        self.layers = nn.ModuleDict()
        self.skip_layers = nn.ModuleDict()

        n_label = 2
        # width_multiplier = config['width_multiplier']
        width_multiplier = 1
        n_channels = [int(x * width_multiplier) for x in n_channels]

        self.dropout = nn.Dropout(0.2)

        self.layers["conv_0"] = nn.Conv2d(input_size, n_channels[0], kernel_size=(3, 1), stride=1, padding=(1,0), bias=False)

        for i, n in zip(range(1, n_blocks+1), n_channels[1:]):

            self.layers[f"conv_{i}_0"] = nn.Conv2d(n_channels[i-1], n, (9, 1), stride=2, padding=(4,0), bias=False)
            self.layers[f"bn_{i}_0"] = nn.BatchNorm2d(n, affine=False)

            self.layers[f"conv_{i}_1"] = nn.Conv2d(n, n, (9, 1), stride=1, padding=(5,0), bias=False)
            self.layers[f"bn_{i}_1"] = nn.BatchNorm2d(n, affine=False)

            self.skip_layers[f"conv_{i}"] = nn.Conv2d(n_channels[i-1], n, (1, 1), stride=2, padding=(2,0), bias=False)
            self.skip_layers[f"bn_{i}"] = nn.BatchNorm2d(n, affine=False)

        self.layers["conv_fc"] = nn.Conv2d(n_channels[-1], n_label, (1, 1), stride=1, bias=False)

        self.layers["output"] = nn.Linear(n_channels[-1], n_label, bias=False)

        self.activations = nn.ModuleDict({
            "relu": nn.ReLU()
        })

    def forward(self, x):

        x = torch.unsqueeze(x, 1)

        x = x.view(x.size(0), x.size(3), x.size(2), x.size(1))  # [N,C,H,W] to [N,W,H,C]
        # x = x.unsqueeze(1)
        x = self.layers["conv_0"](x)


        prev_x = x
        for i in range(1, self.n_layers + 1):
            for j in range(2):
                x = self.layers[f"conv_{i}_{j}"](x)
                x = self.layers[f"bn_{i}_{j}"](x)
                if j==1:
                    prev_x = self.skip_layers[f"conv_{i}"](prev_x)
                    prev_x = self.skip_layers[f"bn_{i}"](prev_x)
                    prev_x = self.activations["relu"](prev_x)
                    x = x + prev_x
                    prev_x = x
                x = self.activations["relu"](x)

        # x = self.layers["pool"](x)
        x = nn.AvgPool2d(kernel_size=x.shape[2:4], stride=1)(x)
        x = self.dropout(x)

        x = self.layers["conv_fc"](x)

        x = x.view(x.size(0), -1)
        x = F.log_softmax(x, dim=-1)
        x = torch.unsqueeze(x, 1)

        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")

    parser.add_argument("--config", default="./config/himia/tcres8.json", type=str,
                        help="path to config file")

    args = parser.parse_args()

    model = TcResNet()

    batch = 2

    input_data = torch.randn((batch, 100, 24))
    output = model(input_data)
    print(output.size())
