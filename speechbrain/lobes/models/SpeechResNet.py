"""This is a module to ResNet-based encoder varaints
Mostly modified from https://github.com/clovaai/voxceleb_trainer/

Authors
 * Hwidong Na 2020
"""
import torch
import torch.nn.functional as F
from speechbrain.nnet.linear import Linear
from speechbrain.lobes.models.ResNet import Conv2dAuto
from speechbrain.lobes.models.ResNet import conv_bn
from speechbrain.lobes.models.ResNet import pre_act
from speechbrain.lobes.models.ResNet import PreActResNetBlock
from speechbrain.lobes.models.ResNet import ResNetBlock
from speechbrain.lobes.models.ResNet import ResNetLayer


class GatingBlock(torch.nn.Module):
    """An implementation of gating mechanism is useful for speaker embedding
    """

    def __init__(self, channel, reduction=8):
        super(GatingBlock, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, channel // reduction),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channel // reduction, channel),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpeechResNetBasicBlock(ResNetBlock):
    """An implementation of ResNet basic block with gatting mechanism, i.e.
    Conv3x3-BN-ReLU-Conv3x3-BN-Gating.

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
    >>> conv = SpeechResNetBasicBlock(64, 64)
    >>> conv.shortcut == None
    True
    >>> out = conv(x)
    >>> out.shape
    torch.Size([8, 64, 120, 40])
    >>> conv = SpeechResNetBasicBlock(64, 128)
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
            GatingBlock(self.expanded_channels),
        )


class SpeechResNetBottleNeckBlock(ResNetBlock):
    """An implementation of ResNet basic block with expansion and gating, i.e.
    Conv1x1-BN-ReLU-Conv3x3-BN-ReLU-Conv1x1(x4)-BN-Gating.

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
    >>> conv = SpeechResNetBottleNeckBlock(64, 64)
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
            GatingBlock(self.expanded_channels),
        )


class SpeechPreActBasicBlock(PreActResNetBlock):
    """An implementation of pre-activation ResNet basic block with gating, i.e.
    BN-ReLU-Conv3x3-BN-ReLU-Conv3x3-Gating.

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
    >>> conv = SpeechPreActBasicBlock(64, 64)
    >>> conv.shortcut == None
    True
    >>> out = conv(x)
    >>> out.shape
    torch.Size([8, 64, 120, 40])
    >>> conv = SpeechPreActBasicBlock(64, 128)
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
            GatingBlock(self.expanded_channels),
        )


class SpeechPreActBottleNeckBlock(PreActResNetBlock):
    """An implementation of pre-activation ResNet basic block with expansion
    and gating i.e.  BN-ReLU-Conv1x1-BN-ReLU-Conv3x3-BN-ReLU-Conv1x1(x4)-Gating.
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
    >>> conv = SpeechPreActBottleNeckBlock(64, 64)
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
            GatingBlock(self.expanded_channels),
        )


class SpeechResNetEncoder(torch.nn.Module):
    """
    The major difference from the original ResNet is stride. At the begining,
    along with the temporal dimension, not the fequency one. From the second
    block, 2x2 strides reduce the feature size. In a consequence, the temporal
    dimension shrinks to 1/32 of the original one. The default block sizes are
    1/4 of the original structure, which also reduce memory footprint.

    Arguements
    ----------
    in_channels: int
        number of input channels of this model
    blocks_sizes: list
        number of output channels for each block of this model
    depths: list
        number of blocks for each layer of this model
    block : SpeechResNetBasicBlock
        A class for constructing the residual block.
    activation : torch class
        A class for constructing the activation layers.

    Example
    -------
    >>> x = torch.rand((8, 1, 256, 40))
    >>> conv = SpeechResNetEncoder(1, [16, 32, 64, 128], [2, 2, 2, 2])
    >>> conv.expansion
    1
    >>> out = conv(x)
    >>> out.shape
    torch.Size([8, 128, 8, 5])
    >>> conv = SpeechResNetEncoder(1, [16, 32, 64, 128], [3, 4, 6, 3])
    >>> conv.expansion
    1
    >>> out = conv(x)
    >>> out.shape
    torch.Size([8, 128, 8, 5])
    """

    def __init__(
        self,
        in_channels=1,
        blocks_sizes=[16, 32, 64, 128],
        depths=[2, 2, 2, 2],
        block=SpeechResNetBasicBlock,
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
                stride=(2, 1),
                padding=3,
                bias=False,
            ),
            torch.nn.BatchNorm2d(self.blocks_sizes[0]),
            activation(),
            torch.nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1),
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
        self.out_channels = blocks_sizes[-1]  # for later stage

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x

    @property
    def expansion(self):
        return self.layer.expansion


class SpeechResNet(torch.nn.Module):
    """This model extracts embedding for speaker recognition and diarization.
    After the ResNet model defined by blocks_size, depths and block, attentive
    pooling is followed by a linear transformation for later stage.

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
    >>> spk_emb = SpeechResNet('cpu', 1, 256, block=SpeechResNetBasicBlock)
    >>> spk_emb.encoder.expansion
    1
    >>> outputs = spk_emb(input_feats, init_params=True)
    >>> outputs.shape
    torch.Size([8, 1, 256])
    """

    def __init__(
        self, device="cpu", in_channels=1, lin_neurons=512, *args, **kwargs
    ):
        super().__init__()

        self.encoder = SpeechResNetEncoder(in_channels, *args, **kwargs)
        # self.pool = torch.nn.AvgPool2d((9,1), stride=1)
        self.fc = Linear(n_neurons=lin_neurons, bias=True)
        out_channels = self.encoder.out_channels
        self.sap_linear = Linear(out_channels)
        self.att_vector = torch.nn.Parameter(torch.FloatTensor(out_channels))
        self.to(device)

    def _reset_params(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Parameter):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x, init_params=False):
        if init_params:
            self._reset_params()
        x = self.encoder(x)
        # x = self.pool(x) # potentially harmful
        N, C, T, D = x.shape
        x = x.reshape(N, C, T * D).permute(0, 2, 1)
        h = torch.tanh(self.sap_linear(x, init_params))
        w = torch.matmul(h, self.att_vector)
        w = F.softmax(w, dim=1).reshape(N, T * D, 1)
        x = torch.sum(x * w, dim=1)
        x = x.reshape(x.shape[0], 1, -1)
        x = self.fc(x, init_params)
        return x


# Helper classes for popular architectures
class ResNet18(SpeechResNet):
    def __init__(self, device="cpu", in_channels=1, lin_neurons=512):
        super().__init__(
            device,
            in_channels,
            lin_neurons,
            block=SpeechResNetBasicBlock,
            blocks_sizes=[16, 32, 64, 128],
            depths=[2, 2, 2, 2],
        )


class ResNet34(SpeechResNet):
    def __init__(self, device="cpu", in_channels=1, lin_neurons=512):
        super().__init__(
            device,
            in_channels,
            lin_neurons,
            block=SpeechResNetBasicBlock,
            blocks_sizes=[16, 32, 64, 128],
            depths=[3, 4, 6, 3],
        )


class ResNet50(SpeechResNet):
    def __init__(self, device="cpu", in_channels=1, lin_neurons=512):
        super().__init__(
            device,
            in_channels,
            lin_neurons,
            block=SpeechResNetBottleNeckBlock,
            blocks_sizes=[16, 32, 64, 128],
            depths=[3, 4, 6, 3],
        )


class ResNet101(SpeechResNet):
    def __init__(self, device="cpu", in_channels=1, lin_neurons=512):
        super().__init__(
            device,
            in_channels,
            lin_neurons,
            block=SpeechResNetBottleNeckBlock,
            blocks_sizes=[16, 32, 64, 128],
            depths=[3, 4, 23, 3],
        )


class ResNet152(SpeechResNet):
    def __init__(self, device="cpu", in_channels=1, lin_neurons=512):
        super().__init__(
            device,
            in_channels,
            lin_neurons,
            block=SpeechResNetBottleNeckBlock,
            blocks_sizes=[16, 32, 64, 128],
            depths=[3, 8, 36, 3],
        )


class PreAct18(SpeechResNet):
    def __init__(self, device="cpu", in_channels=1, lin_neurons=512):
        super().__init__(
            device,
            in_channels,
            lin_neurons,
            block=SpeechPreActBasicBlock,
            blocks_sizes=[16, 32, 64, 128],
            depths=[2, 2, 2, 2],
        )


class PreAct34(SpeechResNet):
    def __init__(self, device="cpu", in_channels=1, lin_neurons=512):
        super().__init__(
            device,
            in_channels,
            lin_neurons,
            block=SpeechPreActBasicBlock,
            blocks_sizes=[16, 32, 64, 128],
            depths=[3, 4, 6, 3],
        )


class PreAct50(SpeechResNet):
    def __init__(self, device="cpu", in_channels=1, lin_neurons=512):
        super().__init__(
            device,
            in_channels,
            lin_neurons,
            block=SpeechPreActBottleNeckBlock,
            blocks_sizes=[16, 32, 64, 128],
            depths=[3, 4, 6, 3],
        )


class PreAct101(SpeechResNet):
    def __init__(self, device="cpu", in_channels=1, lin_neurons=512):
        super().__init__(
            device,
            in_channels,
            lin_neurons,
            block=SpeechPreActBottleNeckBlock,
            blocks_sizes=[16, 32, 64, 128],
            depths=[3, 4, 23, 3],
        )


class PreAct152(SpeechResNet):
    def __init__(self, device="cpu", in_channels=1, lin_neurons=512):
        super().__init__(
            device,
            in_channels,
            lin_neurons,
            block=SpeechPreActBottleNeckBlock,
            blocks_sizes=[16, 32, 64, 128],
            depths=[3, 8, 36, 3],
        )
