"""The SpeechBrain implementation of ContextNet by
https://arxiv.org/pdf/2005.03191.pdf
"""
import torch
from torch.nn import Dropout
from speechbrain.nnet.CNN import DepthwiseSeparableConv1d, Conv1d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.pooling import AdaptivePool
from speechbrain.nnet.containers import Sequential
from speechbrain.nnet.normalization import BatchNorm1d
from speechbrain.nnet.activations import Swish


class ContextNet(Sequential):
    """This class implements the ContextNet
    https://arxiv.org/pdf/2005.03191.pdf

    Example
    -------
    >>> block = ContextNet()
    >>> inp = torch.randn([8, 120, 40])
    >>> out = block(inp, True)
    >>> print(out.shape)
    torch.Size([8, 15, 640])
    """

    def __init__(
        self,
        out_channels=640,
        conv_channels=None,
        kernel_size=3,
        strides=None,
        num_blocks=21,
        num_layers=5,
        inner_dim=12,
        alpha=1,
        beta=1,
        dropout=0.15,
        activation=Swish,
        norm=BatchNorm1d,
        residuals=None,
    ):
        if conv_channels is None:
            conv_channels = [*[256] * 10, *[512] * 11]
        if strides is None:
            strides = [
                1,
                1,
                2,
                1,
                1,
                1,
                2,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ]
        if residuals is None:
            residuals = [True] * num_blocks

        blocks = [
            DepthwiseSeparableConv1d(conv_channels[-1], kernel_size,),
            norm(),
        ]

        for i in range(num_blocks):
            channels = conv_channels[i] * alpha
            blocks.append(
                ContextNetBlock(
                    out_channels=channels,
                    kernel_size=kernel_size,
                    num_layers=num_layers,
                    inner_dim=inner_dim,
                    stride=strides[i],
                    beta=beta,
                    dropout=dropout,
                    activation=activation,
                    norm=norm,
                    residual=residuals[i],
                )
            )

        blocks.extend(
            [DepthwiseSeparableConv1d(out_channels, kernel_size,), norm()]
        )
        super().__init__(*blocks)


class SEmodule(torch.nn.Module):
    """ This class implements the Squeeze-and-excitation module
    """

    def __init__(self, inner_dim):
        super().__init__()
        self.inner_dim = inner_dim

    def init_params(self, first_input):
        bz, t, chn = first_input.shape

        self.avg_pool = AdaptivePool(1)
        self.bottleneck = Sequential(
            Linear(n_neurons=self.inner_dim), Linear(n_neurons=chn)
        )

    def forward(self, x, init_params=False):
        if init_params:
            self.init_params(x)

        bz, t, chn = x.shape

        avg = self.avg_pool(x)
        avg = self.bottleneck(avg, init_params)
        context = avg.repeat(1, t, 1)

        return x * context


class ContextNetBlock(torch.nn.Module):
    """ This class implements a block in ContextNet

    Example
    -------
    >>> block = ContextNetBlock(256, 3, 5, 12, 2)
    >>> inp = torch.randn([8, 120, 40])
    >>> out = block(inp, True)
    >>> print(out.shape)
    torch.Size([8, 60, 640])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        num_layers,
        inner_dim,
        stride=1,
        beta=1,
        dropout=0.15,
        activation=Swish,
        norm=BatchNorm1d,
        residual=True,
    ):
        super().__init__()
        self.residual = residual

        blocks = []

        for i in range(num_layers):
            blocks.extend(
                [
                    DepthwiseSeparableConv1d(
                        out_channels,
                        kernel_size,
                        stride=stride if i == num_layers - 1 else 1,
                    ),
                    norm(),
                ]
            )

        self.Convs = Sequential(*blocks)
        self.SE = SEmodule(inner_dim)
        self.drop = Dropout(dropout)
        self.reduced_cov = None
        if residual:
            self.reduced_cov = Sequential(
                Conv1d(out_channels, kernel_size=1, stride=stride), norm()
            )

    def forward(self, x, init_params=False):
        out = self.Convs(x, init_params)
        out = self.SE(out, init_params)
        if self.reduced_cov:
            out = out + self.reduced_cov(x, init_params)
        return self.drop(out)
