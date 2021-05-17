import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseSettings
from typing import Tuple
import argparse

class Res8Settings(BaseSettings):
    num_labels: int = 2
    pooling: Tuple[int, int] = (3, 4)
    num_maps: int = 45

class Res8(torch.nn.Module):
    def __init__(self, num_labels=4, config=Res8Settings()):
        super().__init__()
        n_maps = config.num_maps
        self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
        self.pool = nn.AvgPool2d(config.pooling)  # flipped -- better for 80 log-Mels

        self.n_layers = n_layers = 6
        self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, bias=False) for _ in range(n_layers)]
        for i, conv in enumerate(self.convs):
            self.add_module(f'bn{i + 1}', nn.BatchNorm2d(n_maps, affine=False))
            self.add_module(f'conv{i + 1}', conv)
        self.output = nn.Linear(n_maps, num_labels)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        # x = x[:, :1]  # log-Mels only
        # x = x.permute(0, 1, 3, 2).contiguous()  # Original res8 uses (time, frequency) format
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, f'conv{i}')(x))
            if i == 0:
                if hasattr(self, 'pool'):
                    y = self.pool(y)
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, f'bn{i}')(x)
        x = x.view(x.size(0), x.size(1), -1)  # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        output = self.output(x)
        output = torch.unsqueeze(output, 1)
        output = F.log_softmax(output, dim=-1)

        return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Feature Extractor")

    model = Res8()
    batch = 2
    input_data = torch.randn((batch, 1, 151, 40))
    output = model(input_data)
    print(output.size())


    from torchsummary import summary
    summary(model, (1, 151, 40))
