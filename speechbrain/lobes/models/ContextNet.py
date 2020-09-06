"""The SpeechBrain implementation of ContextNet by
https://arxiv.org/pdf/2005.03191.pdf

Authors
 * Jianyuan Zhong 2020
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

    Arguements
    ----------
    out_channels: int
        number of output channels of this model (default 640)
    conv_channels: Optional(list[int])
        number of output channels for each of the contextnet block. If not provided, it will be initialize as the defualt setting of https://arxiv.org/pdf/2005.03191.pdf
    kernel_size: int
        kernel size of convolution layers (default 3)
    strides: Optional(list[int])
        striding factor for each context block, this stride is applied at the last convolution layer at each context block. If not provided, it will be initialize as the defualt setting of https://arxiv.org/pdf/2005.03191.pdf
    num_blocks: int
        number of context block (default 21)
    num_layers: int
        number of depthwise convolution layers for each context block (default 5)
    inner_dim: int
        inner dimension of bottle-neck network of the SE Module (default 12)
    alpha: float
        the factor to scale the output channel of the network (default 1)
    beta: float
        beta to scale the Swish activation (default 1)
    dropout: float
        dropout (default 0.15)
    activation: torch class
        activation function for each context block (default Swish)
    se_activation: torch class
        activation function for SE Module (default torch.nn.Sigmoid)
    norm: torch class
        normalization to regularize the model (default BatchNorm1d)
    residuals: Optional(list[bool])
        whether apply residual connection at each context block (default None)


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
        se_activation=torch.nn.Sigmoid,
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
            DepthwiseSeparableConv1d(conv_channels[0], kernel_size,),
            norm(),
            activation(beta) if isinstance(activation, Swish) else activation(),
        ]

        for i in range(num_blocks):
            channels = int(conv_channels[i] * alpha)
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
                    se_activation=se_activation,
                    norm=norm,
                    residual=residuals[i],
                )
            )

        blocks.extend(
            [
                DepthwiseSeparableConv1d(out_channels, kernel_size,),
                norm(),
                activation(beta)
                if isinstance(activation, Swish)
                else activation(),
            ]
        )
        super().__init__(*blocks)


class SEmodule(torch.nn.Module):
    """ This class implements the Squeeze-and-excitation module

    Arguements
    ----------
    inner_dim: int
        inner dimension of bottle-neck network of the SE Module (default 12)
    activation: torch class
        activation function for SE Module (default torch.nn.Sigmoid)
    norm: torch class
        normalization to regularize the model (default BatchNorm1d)

    Example
    -------
    >>> net = SEmodule(64)
    >>> inp = torch.randn([8, 120, 40])
    >>> out = net(inp, True)
    >>> print(out.shape)
    torch.Size([8, 120, 40])
    """

    def __init__(
        self, inner_dim, activation=torch.nn.Sigmoid, norm=BatchNorm1d,
    ):
        super().__init__()
        self.inner_dim = inner_dim
        self.norm = norm
        self.activation = activation

    def init_params(self, first_input):
        bz, t, chn = first_input.shape
        self.conv = Sequential(
            DepthwiseSeparableConv1d(chn, 1, 1), self.norm(), self.activation()
        )
        self.avg_pool = AdaptivePool(1)
        self.bottleneck = Sequential(
            Linear(n_neurons=self.inner_dim),
            self.activation(),
            Linear(n_neurons=chn),
            self.activation(),
        )

    def forward(self, x, init_params=False):
        if init_params:
            self.init_params(x)

        bz, t, chn = x.shape

        x = self.conv(x, init_params)
        avg = self.avg_pool(x)
        avg = self.bottleneck(avg, init_params)
        context = avg.repeat(1, t, 1)
        return x * context


class ContextNetBlock(torch.nn.Module):
    """ This class implements a block in ContextNet

    Arguements
    ----------
    out_channels: int
        number of output channels of this model (default 640)
    kernel_size: int
        kernel size of convolution layers (default 3)
    strides: int
        striding factor for this context block (default 1)
    num_layers: int
        number of depthwise convolution layers for this context block (default 5)
    inner_dim: int
        inner dimension of bottle-neck network of the SE Module (default 12)
    beta: float
        beta to scale the Swish activation (default 1)
    dropout: float
        dropout (default 0.15)
    activation: torch class
        activation function for this context block (default Swish)
    se_activation: torch class
        activation function for SE Module (default torch.nn.Sigmoid)
    norm: torch class
        normalization to regularize the model (default BatchNorm1d)
    residuals: bool
        whether apply residual connection at this context block (default None)

    Example
    -------
    >>> block = ContextNetBlock(256, 3, 5, 12, 2)
    >>> inp = torch.randn([8, 120, 40])
    >>> out = block(inp, True)
    >>> print(out.shape)
    torch.Size([8, 60, 256])
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
        se_activation=torch.nn.Sigmoid,
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
        self.SE = SEmodule(inner_dim, activation=se_activation, norm=norm)
        self.drop = Dropout(dropout)
        self.reduced_cov = None
        if residual:
            self.reduced_cov = Sequential(
                Conv1d(out_channels, kernel_size=3, stride=stride), norm()
            )

        if isinstance(activation, Swish):
            self.activation = activation(beta)
        else:
            self.activation = activation()

    def forward(self, x, init_params=False):
        out = self.Convs(x, init_params)
        out = self.SE(out, init_params)
        if self.reduced_cov:
            out = out + self.reduced_cov(x, init_params)
        out = self.activation(out)

        if init_params:
            self._reset_params()

        return self.drop(out)

    def _reset_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p)
