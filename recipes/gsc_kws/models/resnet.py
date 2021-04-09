import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys

import argparse
import copy

class ResNet(torch.nn.Module):
    def __init__(self, input_size=24):
        super().__init__()
        config = {'n_layers':6, 'n_feature_maps':19, 'dropout': 0.2, 'n_labels':12, 'pool':1}
        self.n_layers = config["n_layers"]
        n_maps = config["n_feature_maps"]
        dropout_rate = config["dropout"]

        # self.config = {'n_layers':6, 'n_feature_maps':19, 'dropout': 0.2, 'n_labels':12, 'pool':1}
        # self.n_layers = 6
        # n_maps = 19
        # dropout_rate = 0.2


        self.layers = nn.ModuleDict()

        self.layers["conv_0"] = nn.Conv2d(1, n_maps, (3, 3), padding=1, bias=False)

        for i in range(1, self.n_layers + 1):
            if True:
                padding_size = int(2**((i-1) // 3))
                dilation_size = int(2**((i-1) // 3))
                self.layers[f"conv_{i}"] = nn.Conv2d(n_maps, n_maps, (3, 3), padding=padding_size, dilation=dilation_size, bias=False)
            else:
                self.layers[f"conv_{i}"] = nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, bias=False)
            self.layers[f"bn_{i}"] = nn.BatchNorm2d(n_maps, affine=True)

        if "pool" in config:
            self.layers["pool"] = nn.AvgPool2d(config["pool"])

        self.layers["output"] = nn.Linear(n_maps, config["n_labels"])

        self.activations = nn.ModuleDict({
            "relu": nn.ReLU()
        })
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layers["conv_0"](x)
        x = self.activations["relu"](x)

        if "pool" in self.layers:
            x = self.layers["pool"](x)

        prev_x = x
        for i in range(1, self.n_layers + 1):
            x = self.layers[f"conv_{i}"](x)
            # x = self.layers[f"bn_{i}"](x)
            x = self.activations["relu"](x)

            if i % 2 == 0:
                x = x + prev_x
                prev_x = x

            x = self.layers[f"bn_{i}"](x)
            # x = self.activations["relu"](x)
        x = self.dropout(x)
        # x = x.view(x.size(0), -1)

        x = x.view(x.size(0), x.size(1), -1) # shape: (batch, features, o3)
        x = x.mean(2)
        x = self.layers["output"](x)
        # x = self.layers['softmax'](x)
        x = F.log_softmax(x, dim=-1)
        x = torch.unsqueeze(x, 1)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")

    model = ResNet()
    batch = 64
    input_data = torch.randn((batch, 151, 24))
    output = model(input_data)
    print(output.size())
