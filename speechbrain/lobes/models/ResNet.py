"""This is a module to ResNet-based encoder
Mostly modified from https://github.com/FrancescoSaverioZuppichini/ResNet
Also refer to https://github.com/KaimingHe/resnet-1k-layers

Authors
 * Hwidong Na 2020
"""
import torch
from collections import OrderedDict
from speechbrain.nnet.linear import Linear


class Conv2dAuto(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (
            self.kernel_size[0] // 2,
            self.kernel_size[1] // 2,
        )  # dynamic add padding based on the kernel_size


def conv_bn(in_channels, out_channels, *args, **kwargs):
    return torch.nn.Sequential(
        OrderedDict(
            {
                "conv": Conv2dAuto(in_channels, out_channels, *args, **kwargs),
                "bn": torch.nn.BatchNorm2d(out_channels),
            }
        )
    )


class ResNetBlock(torch.nn.Module):
    """An abstract implementation of ResNet block
    Let x be input, y be output
    y = x + block(x) if x.shape == y.shape
    y = shorcut(x) + block(x) otherwise


    Arguements
    ----------
    in_channels: int
        number of input channels of this model
    out_channels: int
        number of output channels of this model
    expansion: int
        number of expansion of output channels
    downsampling: int
        stride for convolutions
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        expansion=1,
        downsampling=1,
        *args,
        **kwargs,
    ):
        super(ResNetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        self.downsampling = downsampling
        self.shortcut = (
            conv_bn(
                self.in_channels,
                self.expanded_channels,
                kernel_size=1,
                stride=self.downsampling,
                bias=False,
            )
            if self.should_apply_shortcut
            else None
        )

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


class ResNetBasicBlock(ResNetBlock):
    """An implementation of ResNet basic block without expansion, i.e.
    Conv3x3-BN-ReLU-Conv3x3-BN.

    Arguements
    ----------
    in_channels: int
        number of input channels of this model
    out_channels: int
        number of output channels of this model
    activation : torch class
        A class for constructing the activation layers.

    Example
    -------
    >>> x = torch.rand((8, 64, 120, 40))
    >>> conv = ResNetBasicBlock(64, 64)
    >>> conv.shortcut == None
    True
    >>> out = conv(x)
    >>> out.shape
    torch.Size([8, 64, 120, 40])
    >>> conv = ResNetBasicBlock(64, 128)
    >>> conv.shortcut == None
    False
    >>> out = conv(x)
    >>> out.shape
    torch.Size([8, 128, 120, 40])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        activation=torch.nn.ReLU,
        *args,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = torch.nn.Sequential(
            conv_bn(
                self.in_channels,
                self.out_channels,
                kernel_size=3,
                bias=False,
                stride=self.downsampling,
            ),
            activation(),
            conv_bn(
                self.out_channels,
                self.expanded_channels,
                kernel_size=3,
                bias=False,
            ),
        )


class ResNetBottleNeckBlock(ResNetBlock):
    """An implementation of ResNet basic block with expansion, i.e.
    Conv1x1-BN-ReLU-Conv3x3-BN-ReLU-Conv1x1(x4)-BN.

    Arguements
    ----------
    in_channels: int
        number of input channels of this model
    out_channels: int
        number of output channels of this model
    activation : torch class
        A class for constructing the activation layers.

    Example
    -------
    >>> x = torch.rand((8, 64, 120, 40))
    >>> conv = ResNetBottleNeckBlock(64, 64)
    >>> conv.shortcut == None
    False
    >>> out = conv(x)
    >>> out.shape
    torch.Size([8, 256, 120, 40])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        activation=torch.nn.ReLU,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_channels, out_channels, expansion=4, *args, **kwargs
        )
        self.blocks = torch.nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, kernel_size=1),
            activation(),
            conv_bn(
                self.out_channels,
                self.out_channels,
                kernel_size=3,
                stride=self.downsampling,
            ),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, kernel_size=1,),
        )


# Pre-activation version
def pre_act(in_channels, out_channels, activation, *args, **kwargs):
    return torch.nn.Sequential(
        OrderedDict(
            {
                "bn": torch.nn.BatchNorm2d(in_channels),
                "activation": activation(),
                "conv": Conv2dAuto(in_channels, out_channels, *args, **kwargs),
            }
        )
    )


class PreActResNetBasicBlock(ResNetBlock):
    """An implementation of pre-activation ResNet basic block without
    expansion, i.e.  BN-ReLU-Conv3x3-BN-ReLU-Conv3x3.

    Arguements
    ----------
    in_channels: int
        number of input channels of this model
    out_channels: int
        number of output channels of this model
    activation : torch class
        A class for constructing the activation layers.

    Example
    -------
    >>> x = torch.rand((8, 64, 120, 40))
    >>> conv = PreActResNetBasicBlock(64, 64)
    >>> conv.shortcut == None
    True
    >>> out = conv(x)
    >>> out.shape
    torch.Size([8, 64, 120, 40])
    >>> conv = PreActResNetBasicBlock(64, 128)
    >>> conv.shortcut == None
    False
    >>> out = conv(x)
    >>> out.shape
    torch.Size([8, 128, 120, 40])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        activation=torch.nn.ReLU,
        *args,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = torch.nn.Sequential(
            pre_act(
                self.in_channels,
                self.out_channels,
                activation,
                kernel_size=3,
                bias=False,
                stride=self.downsampling,
            ),
            pre_act(
                self.out_channels,
                self.expanded_channels,
                activation,
                kernel_size=3,
                bias=False,
            ),
        )


class PreActResNetBottleNeckBlock(ResNetBlock):
    """An implementation of pre-activation ResNet basic block with expansion,
    i.e.  BN-ReLU-Conv1x1-BN-ReLU-Conv3x3-BN-ReLU-Conv1x1(x4).
    Shortcut is a Conv1x1 from input channels to expanded channels

    Arguements
    ----------
    in_channels: int
        number of input channels of this model
    out_channels: int
        number of output channels of this model
    activation : torch class
        A class for constructing the activation layers.

    Example
    -------
    >>> x = torch.rand((8, 64, 120, 40))
    >>> conv = PreActResNetBottleNeckBlock(64, 64)
    >>> conv.shortcut == None
    False
    >>> out = conv(x)
    >>> out.shape
    torch.Size([8, 256, 120, 40])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        activation=torch.nn.ReLU,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_channels, out_channels, expansion=4, *args, **kwargs
        )
        self.shortcut = (
            Conv2dAuto(
                self.in_channels,
                self.expanded_channels,
                kernel_size=1,
                stride=self.downsampling,
                bias=False,
            )
            if self.should_apply_shortcut
            else None
        )
        self.blocks = torch.nn.Sequential(
            pre_act(
                self.in_channels, self.out_channels, activation, kernel_size=1,
            ),
            pre_act(
                self.out_channels,
                self.out_channels,
                activation,
                kernel_size=3,
                stride=self.downsampling,
            ),
            pre_act(
                self.out_channels,
                self.expanded_channels,
                activation=activation,
                kernel_size=1,
            ),
        )


class ResNetLayer(torch.nn.Module):
    """An implementation of ResNet layer, consisting of multiple ResNetBlocks,

    Arguements
    ----------
    in_channels: int
        number of input channels of this model
    out_channels: int
        number of output channels of this model
    num_blocks : int
        number of blocks in layer
    block : {PreAct,}{ResNetBasicBlock, ResNetBottleNeckBlock}
        A class for constructing the residual block.

    Example
    -------
    >>> x = torch.rand((8, 64, 120, 40))
    >>> conv = ResNetLayer(64, 64, 2, ResNetBasicBlock)
    >>> conv.expansion
    1
    >>> out = conv(x)
    >>> out.shape
    torch.Size([8, 64, 120, 40])
    >>> conv = ResNetLayer(64, 64, 2, ResNetBottleNeckBlock)
    >>> conv.expansion
    4
    >>> out = conv(x)
    >>> out.shape
    torch.Size([8, 256, 120, 40])
    >>> conv = ResNetLayer(64, 64, 2, PreActResNetBasicBlock)
    >>> conv.expansion
    1
    >>> out = conv(x)
    >>> out.shape
    torch.Size([8, 64, 120, 40])
    >>> conv = ResNetLayer(64, 64, 2, PreActResNetBottleNeckBlock)
    >>> conv.expansion
    4
    >>> out = conv(x)
    >>> out.shape
    torch.Size([8, 256, 120, 40])
    """

    def __init__(
        self, in_channels, out_channels, num_blocks, block, *args, **kwargs,
    ):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        # Instanciate to get expansion
        self.block = block(
            in_channels,
            out_channels,
            *args,
            **kwargs,
            downsampling=downsampling,
        )

        self.blocks = torch.nn.Sequential(
            self.block,
            *[
                block(
                    out_channels * self.expansion,
                    out_channels,
                    downsampling=1,
                    *args,
                    **kwargs,
                )
                for _ in range(num_blocks - 1)
            ],
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

    @property
    def expansion(self):
        return self.block.expansion


class ResNetEncoder(torch.nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    Beginning with Conv7x7(stride 2)-MaxPool3x3(strides 2), which results the 1/4
    size of the original feature size.

    Arguements
    ----------
    in_channels: int
        number of input channels of this model
    blocks_sizes: list
        number of output channels for each block of this model
    depths: list
        number of blocks for each layer of this model
    block : {PreAct,}{ResNetBasicBlock, ResNetBottleNeckBlock}
        A class for constructing the residual block.
    activation : torch class
        A class for constructing the activation layers.

    Example
    -------
    >>> x = torch.rand((8, 64, 120, 40))
    >>> conv = ResNetEncoder(64, [64], [2], ResNetBasicBlock)
    >>> conv.expansion
    1
    >>> out = conv(x)
    >>> out.shape
    torch.Size([8, 64, 30, 10])
    >>> conv = ResNetEncoder(64, [64], [2], ResNetBottleNeckBlock)
    >>> conv.expansion
    4
    >>> out = conv(x)
    >>> out.shape
    torch.Size([8, 256, 30, 10])
    >>> conv = ResNetEncoder(64, [64], [2], PreActResNetBasicBlock)
    >>> conv.expansion
    1
    >>> out = conv(x)
    >>> out.shape
    torch.Size([8, 64, 30, 10])
    >>> conv = ResNetEncoder(64, [64], [2], PreActResNetBottleNeckBlock)
    >>> conv.expansion
    4
    >>> out = conv(x)
    >>> out.shape
    torch.Size([8, 256, 30, 10])
    """

    def __init__(
        self,
        in_channels=3,
        blocks_sizes=[64, 128, 256, 512],
        depths=[2, 2, 2, 2],
        block=ResNetBasicBlock,
        activation=torch.nn.ReLU,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.blocks_sizes = blocks_sizes

        self.gate = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels,
                self.blocks_sizes[0],
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            torch.nn.BatchNorm2d(self.blocks_sizes[0]),
            activation(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        # Instanciate to get expansion
        self.layer = ResNetLayer(
            blocks_sizes[0],
            blocks_sizes[0],
            depths[0],
            block,
            activation=activation,
            *args,
            **kwargs,
        )
        self.blocks = torch.nn.ModuleList(
            [
                self.layer,
                *[
                    ResNetLayer(
                        in_channels * self.expansion,
                        out_channels,
                        num_blocks,
                        block,
                        activation=activation,
                        *args,
                        **kwargs,
                    )
                    for (in_channels, out_channels), num_blocks in zip(
                        self.in_out_block_sizes, depths[1:]
                    )
                ],
            ]
        )

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x

    @property
    def expansion(self):
        return self.layer.expansion


class ResNet(torch.nn.Module):
    """This model extracts embedding for speaker recognition and diarization.
    After the ResNet model defined by blocks_size, depths and block, AvgPool1x1
    is followed by a linear transformation for later stage.

    Arguments
    ---------
    device : str
        Device used e.g. "cpu" or "cuda"
    in_channels : int
        Number of channels in the input features.
    lin_neurons : int
        Number of neurons in linear layers.

    Example
    -------
    >>> input_feats = torch.rand([8, 1, 120, 40])
    >>> spk_emb = ResNet('cpu', 1, 256, block=ResNetBasicBlock)
    >>> spk_emb.encoder.expansion
    1
    >>> outputs = spk_emb(input_feats, init_params=True)
    >>> outputs.shape
    torch.Size([8, 1, 256])
    >>> spk_emb = ResNet('cpu', 1, 256, block=ResNetBottleNeckBlock)
    >>> spk_emb.encoder.expansion
    4
    >>> outputs = spk_emb(input_feats, init_params=True)
    >>> outputs.shape
    torch.Size([8, 1, 256])
    >>> spk_emb = ResNet('cpu', 1, 256, block=PreActResNetBasicBlock)
    >>> spk_emb.encoder.expansion
    1
    >>> outputs = spk_emb(input_feats, init_params=True)
    >>> outputs.shape
    torch.Size([8, 1, 256])
    >>> spk_emb = ResNet('cpu', 1, 256, block=PreActResNetBottleNeckBlock)
    >>> spk_emb.encoder.expansion
    4
    >>> outputs = spk_emb(input_feats, init_params=True)
    >>> outputs.shape
    torch.Size([8, 1, 256])
    """

    def __init__(
        self, device="cpu", in_channels=1, lin_neurons=512, *args, **kwargs
    ):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.fc = Linear(n_neurons=lin_neurons, bias=True)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.to(device)

    def forward(self, x, init_params=False):
        if init_params:
            self._reset_params()
        x = self.encoder(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], 1, -1)
        x = self.fc(x, init_params)
        return x

    def _reset_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p)


# Helper classes for popular architectures
class ResNet18(ResNet):
    def __init__(self, device="cpu", in_channels=1, lin_neurons=512):
        super().__init__(
            device,
            in_channels,
            lin_neurons,
            block=ResNetBasicBlock,
            deepths=[2, 2, 2, 2],
        )


class ResNet34(ResNet):
    def __init__(self, device="cpu", in_channels=1, lin_neurons=512):
        super().__init__(
            device,
            in_channels,
            lin_neurons,
            block=ResNetBasicBlock,
            deepths=[3, 4, 6, 3],
        )


class ResNet50(ResNet):
    def __init__(self, device="cpu", in_channels=1, lin_neurons=512):
        super().__init__(
            device,
            in_channels,
            lin_neurons,
            block=ResNetBottleNeckBlock,
            deepths=[3, 4, 6, 3],
        )


class ResNet101(ResNet):
    def __init__(self, device="cpu", in_channels=1, lin_neurons=512):
        super().__init__(
            device,
            in_channels,
            lin_neurons,
            block=ResNetBottleNeckBlock,
            deepths=[3, 4, 23, 3],
        )


class ResNet152(ResNet):
    def __init__(self, device="cpu", in_channels=1, lin_neurons=512):
        super().__init__(
            device,
            in_channels,
            lin_neurons,
            block=ResNetBottleNeckBlock,
            deepths=[3, 8, 36, 3],
        )


class PreAct18(ResNet):
    def __init__(self, device="cpu", in_channels=1, lin_neurons=512):
        super().__init__(
            device,
            in_channels,
            lin_neurons,
            block=PreActResNetBasicBlock,
            deepths=[2, 2, 2, 2],
        )


class PreAct34(ResNet):
    def __init__(self, device="cpu", in_channels=1, lin_neurons=512):
        super().__init__(
            device,
            in_channels,
            lin_neurons,
            block=PreActResNetBasicBlock,
            deepths=[3, 4, 6, 3],
        )


class PreAct50(ResNet):
    def __init__(self, device="cpu", in_channels=1, lin_neurons=512):
        super().__init__(
            device,
            in_channels,
            lin_neurons,
            block=PreActResNetBottleNeckBlock,
            deepths=[3, 4, 6, 3],
        )


class PreAct101(ResNet):
    def __init__(self, device="cpu", in_channels=1, lin_neurons=512):
        super().__init__(
            device,
            in_channels,
            lin_neurons,
            block=PreActResNetBottleNeckBlock,
            deepths=[3, 4, 23, 3],
        )


class PreAct152(ResNet):
    def __init__(self, device="cpu", in_channels=1, lin_neurons=512):
        super().__init__(
            device,
            in_channels,
            lin_neurons,
            block=PreActResNetBottleNeckBlock,
            deepths=[3, 8, 36, 3],
        )
