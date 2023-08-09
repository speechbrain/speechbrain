"""Bioformer from http://arxiv.org/abs/2111.03842.

Authors
 * Francesco Paissan, 2022
"""
from speechbrain.lobes.models.transformer.Transformer import (
    TransformerEncoderLayer,
)
from speechbrain import nnet
import torch.nn as nn
import torch

torch.autograd.set_detect_anomaly(True)


class BioformerMHSA(nn.Module):
    def __init__(self, att_heads=2, d_ffn=32, dropout=0.2):
        super().__init__()

        self.mhsa = TransformerEncoderLayer(
            d_ffn=d_ffn,
            nhead=att_heads,
            d_model=64,
            normalize_before=True,
            activation=nn.GELU,
        )

        self.model = nnet.containers.Sequential(
            nnet.linear.Linear(n_neurons=128, input_size=64, bias=False),
            nn.GELU(),
            nnet.dropout.Dropout2d(drop_rate=dropout),
            nnet.linear.Linear(64, input_size=128, bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.mhsa(x)[0] + x  # reduntant skip ? ...just did multiplication
        x = self.model(x) + x

        return x


class Bioformer(nn.Module):
    def __init__(self, C=22, att_heads=2, d_ffn=32, classes=4):
        super().__init__()

        self.class_token = nn.Parameter(torch.randn(1, 1, 64))
        self.class_token.requires_grad = True

        # encode input signal in T // kernel tokens of 64 elements
        self.cnn = nnet.containers.Sequential(
            nnet.CNN.Conv1d(64, 10, in_channels=22, stride=10, padding="same")
        )

        self.att = nnet.containers.Sequential(
            BioformerMHSA(att_heads, d_ffn, dropout=0.5),
            BioformerMHSA(att_heads, d_ffn, dropout=0.5),
        )

        # classifier
        self.dnn = nnet.containers.Sequential(
            nnet.linear.Linear(n_neurons=4, input_size=64), nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = torch.squeeze(x, -1)

        # remove channels dim, performing 1D conv

        x = self.cnn(x)

        # append class token
        x = torch.cat((x, self.class_token.repeat(x.shape[0], 1, 1)), dim=1)
        x = self.att(x)

        x = self.dnn(x[:, -1, :])

        return x.squeeze(1)
