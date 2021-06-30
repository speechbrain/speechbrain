"""CRDNN
"""
from speechbrain.lobes.models.CRDNN import CRDNN
import torch
import torch.nn as nn


class CRDNN_C(nn.Module):
    def __init__(
        self,
        nclass=10,
        input_size=64,
        activation=nn.ReLU(),
        dropout=0.5,
        cnn_blocks=3,
        cnn_channels=(64, 64, 64),
        cnn_kernelsize=(3, 3, 3),
        time_pooling=True,
        time_pooling_size=2,
        using_2d_pooling=True,
        inter_layer_pooling_size=[2, 2, 2],
        stride=[1, 1, 1],
        rnn_layers=2,
        rnn_neurons=64,
        dnn_blocks=2,
        dnn_neurons=64,
    ):
        super(CRDNN_C, self).__init__()
        self.crdnn = CRDNN(
            input_size=input_size,
            dnn_neurons=dnn_neurons,
            cnn_blocks=cnn_blocks,
            cnn_channels=cnn_channels,
            rnn_layers=rnn_layers,
            using_2d_pooling=using_2d_pooling,
            inter_layer_pooling_size=[2, 2],
            time_pooling=time_pooling,
            time_pooling_size=time_pooling_size,
        )
        self.dense = nn.Linear(dnn_neurons, nclass)
        self.sigmoid = nn.Sigmoid()
        self.dense_softmax = nn.Linear(dnn_neurons, nclass)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        x = self.crdnn(x)
        strong = self.dense(x)  # [bs, frames, nclass]
        strong = self.sigmoid(strong)
        sof = self.dense_softmax(x)  # [bs, frames, nclass]
        sof = self.softmax(sof)
        sof = torch.clamp(sof, min=1e-7, max=1)
        weak = (strong * sof).sum(1) / sof.sum(1)  # [bs, nclass]

        return strong, weak
