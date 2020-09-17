"""A popular speaker recognition and diarization model.

Authors
 * Nauman Dawalatabad 2020
 * Mirco Ravanelli 2020
"""

# import os
import torch  # noqa: F401
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.data_io.data_io import length_to_mask
from speechbrain.nnet.CNN import Conv1d as _Conv1d
from speechbrain.nnet.normalization import BatchNorm1d as _BatchNorm1d
from speechbrain.nnet.linear import Linear


# Skip transpose as much as possible for efficiency
class Conv1d(_Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(skip_transpose=True, *args, **kwargs)


class BatchNorm1d(_BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(skip_transpose=True, *args, **kwargs)


class TDNNBlock(nn.Module):
    """An implementation of TDNN

    Arguements
    ----------
    out_channels: int
        The number of output channels
    kernel_size: int
        The kernel size of the TDNN blocks
    dialation: int
        The dialation of the Res2Net block
    activation : torch class
        A class for constructing the activation layers.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1,2)
    >>> layer = TDNNBlock(64, kernel_size=3, dilation=1)
    >>> out_tensor = layer(inp_tensor, init_params=True).transpose(1,2)
    >>> out_tensor.shape
    torch.Size([8, 120, 64])
    """

    def __init__(self, out_channels, kernel_size, dilation, activation=nn.ReLU):
        super(TDNNBlock, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.extend(
            [
                Conv1d(out_channels, kernel_size, dilation=dilation),
                activation(),
                BatchNorm1d(),
            ]
        )

    def forward(self, x, init_params=False):
        for layer in self.blocks:
            try:
                x = layer(x, init_params=init_params)
            except TypeError:
                x = layer(x)

        return x


class Res2NetBlock(torch.nn.Module):
    """An implementation of Res2NetBlock w/ dialation

    Arguments
    ---------
    out_channels: int
        The number of output channels
    scale: int
        The scale of the Res2Net block
    dialation: int
        The dialation of the Res2Net block

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1,2)
    >>> layer = Res2NetBlock(64, scale=4, dilation=3)
    >>> out_tensor = layer(inp_tensor, init_params=True).transpose(1,2)
    >>> out_tensor.shape
    torch.Size([8, 120, 64])
    """

    def __init__(self, out_channels, scale=8, dilation=1):
        super(Res2NetBlock, self).__init__()
        assert out_channels % scale == 0
        self.blocks = nn.ModuleList()

        hidden_channel = out_channels // scale
        self.blocks.extend(
            [
                TDNNBlock(hidden_channel, kernel_size=3, dilation=dilation)
                for i in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, x, init_params=False):
        y = []
        for i, x_i in enumerate(torch.chunk(x, self.scale, dim=1)):
            if i == 0:
                y_i = x_i
            elif i == 1:
                y_i = self.blocks[i - 1](x_i, init_params=init_params)
            else:
                y_i = self.blocks[i - 1](x_i + y_i, init_params=init_params)
            y.append(y_i)
        y = torch.cat(y, dim=1)
        return y


class SEBlock(nn.Module):
    """An implementation of sqeeuze-and-excitation block

    Arguments
    ---------
    channels: int
        The number of input channels

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1,2)
    >>> se_layer = SEBlock(16, 64)
    >>> lengths = torch.randint(1, 120, (8,))
    >>> out_tensor = se_layer(inp_tensor, lengths, True).transpose(1,2)
    >>> out_tensor.shape
    torch.Size([8, 120, 64])
    """

    def __init__(self, se_channels, out_channels):
        super(SEBlock, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.extend(
            [
                Conv1d(se_channels, kernel_size=1),
                torch.nn.ReLU(inplace=True),
                Conv1d(out_channels, kernel_size=1),
                torch.nn.Sigmoid(),
            ]
        )

    def forward(self, x, lengths=None, init_params=False):
        L = x.shape[-1]
        if lengths is not None:
            mask = length_to_mask(lengths, max_len=L, device=x.device)
            mask = mask.unsqueeze(1)
            total = mask.sum(dim=2, keepdim=True)
            s = (x * mask).sum(dim=2, keepdim=True) / total
        else:
            s = x.mean(dim=2, keepdim=True)
        for layer in self.blocks:
            try:
                s = layer(s, init_params=init_params)
            except TypeError:
                s = layer(s)
        return s * x


class AttentiveStatisticsPooling(nn.Module):
    """This class implements a attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of input tensor

    Arguments
    ---------
    channels: int
        The number of input channels
    attention_channels: int
        The number of attention channels

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1,2)
    >>> asp_layer = AttentiveStatisticsPooling(64)
    >>> lengths = torch.randint(1, 120, (8,))
    >>> out_tensor = asp_layer(inp_tensor, lengths, True).transpose(1,2)
    >>> out_tensor.shape
    torch.Size([8, 1, 128])
    """

    def __init__(self, channels, attention_channels=128, global_context=True):
        super().__init__()

        self.eps = 1e-12
        self.global_context = global_context
        self.blocks = nn.ModuleList()
        self.blocks.extend(
            [
                TDNNBlock(attention_channels, kernel_size=1, dilation=1),
                nn.Tanh(),
                Conv1d(channels, kernel_size=1),
            ]
        )

    def forward(self, x, lengths=None, init_params=False):
        """Calculates mean and std for a batch (input tensor).

        Arguments
        ---------
        x : torch.Tensor
            of shape [N, C, L]
        """
        L = x.shape[-1]

        def _compute_statistics(x, m, dim=2, eps=self.eps):
            mean = (m * x).sum(dim)
            std = torch.sqrt(
                (m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim) + eps
            )
            return mean, std

        if lengths is None:
            lengths = torch.ones(x.shape[0], device=x.device) * L

        # Make binary mask of shape [N, 1, L]
        mask = length_to_mask(lengths, max_len=L, device=x.device)
        mask = mask.unsqueeze(1)

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        if self.global_context:
            # torch.std is unstable for backward computation
            # https://github.com/pytorch/pytorch/issues/4320
            total = mask.sum(dim=2, keepdim=True).float()
            mean, std = _compute_statistics(x, mask / total)
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x

        for layer in self.blocks:
            try:
                attn = layer(attn, init_params)
            except TypeError:
                attn = layer(attn)

        # Filter out zero-paddings
        attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=2)
        mean, std = _compute_statistics(x, attn)
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)

        return pooled_stats


class SERes2NetBlock(nn.Module):
    """An implementation of building block in ECAPA-TDNN, i.e.
    TDNN-Res2Net-TDNN-SEBlock.

    Arguements
    ----------
    out_channels: int
        The number of output channels
    res2net_scale: int
        The scale of the Res2Net block
    kernel_size: int
        The kernel size of the TDNN blocks
    dialation: int
        The dialation of the Res2Net block
    activation : torch class
        A class for constructing the activation layers.

    Example
    -------
    >>> x = torch.rand((8, 120, 64)).transpose(1,2)
    >>> conv = SERes2NetBlock(64, res2net_scale=4)
    >>> out = conv(x, init_params=True).transpose(1,2)
    >>> out.shape
    torch.Size([8, 120, 64])
    """

    def __init__(
        self,
        out_channels,
        res2net_scale=8,
        se_channels=128,
        kernel_size=1,
        dilation=1,
        activation=torch.nn.ReLU,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.blocks = nn.ModuleList()
        self.blocks.extend(
            [
                TDNNBlock(
                    out_channels,
                    kernel_size=1,
                    dilation=1,
                    activation=activation,
                ),
                Res2NetBlock(out_channels, res2net_scale, dilation),
                TDNNBlock(
                    out_channels,
                    kernel_size=1,
                    dilation=1,
                    activation=activation,
                ),
                SEBlock(se_channels, out_channels),
            ]
        )
        self.shortcut = None

    def forward(self, x, lengths=None, init_params=False):
        if init_params and x.shape[1] != self.out_channels:
            self.shortcut = Conv1d(self.out_channels, kernel_size=1)

        residual = x
        if self.shortcut:
            residual = self.shortcut(x, init_params=init_params)
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lengths, init_params=init_params)
            except TypeError:
                x = layer(x, init_params=init_params)
        x += residual
        return x


class ECAPA_TDNN(torch.nn.Module):
    """An implementation of the speaker embedding model in a recent paper.
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://arxiv.org/abs/2005.07143)

    Arguments
    ---------
    device : str
        Device used e.g. "cpu" or "cuda"
    activation : torch class
        A class for constructing the activation layers.
    channels : list of ints
        Output channels for TDNN/SERes2Net layer.
    kernel_sizes : list of ints
        List of kernel sizes for each layer.
    dilations : list of ints
        List of dilations for kernels in each layer.
    lin_neurons : int
        Number of neurons in linear layers.

    Example
    -------
    >>> input_feats = torch.rand([5, 120, 80])
    >>> compute_embedding = ECAPA_TDNN(lin_neurons=192)
    >>> outputs = compute_embedding(input_feats, init_params=True)
    >>> outputs.shape
    torch.Size([5, 1, 192])
    """

    def __init__(
        self,
        device="cpu",
        lin_neurons=192,
        activation=torch.nn.ReLU,
        channels=[512, 512, 512, 512, 1536],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        attention_channels=128,
        res2net_scale=8,
        se_channels=128,
        global_context=True,
    ):

        super().__init__()
        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(dilations)
        self.channels = channels
        self.blocks = nn.ModuleList()

        # The initial TDNN layer
        self.blocks.append(
            TDNNBlock(channels[0], kernel_sizes[0], dilations[0], activation)
        )

        # SE-Res2Net layers
        for i in range(1, len(channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    activation=activation,
                )
            )

        # Multi-layer feature aggregation
        self.mfa = TDNNBlock(
            channels[-1], kernel_sizes[-1], dilations[-1], activation
        )

        # Attantitve Statistical pooling
        self.asp = AttentiveStatisticsPooling(
            channels[-1],
            attention_channels=attention_channels,
            global_context=global_context,
        )
        self.asp_bn = BatchNorm1d()

        # Final linear transformation
        self.fc = Conv1d(lin_neurons, kernel_size=1)

    def forward(self, x, lengths=None, init_params=False):
        """Returns the embedding vector.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
        """
        # Minimize transpose for efficiency
        x = x.transpose(1, 2)

        xl = []
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lengths, init_params=init_params)
            except TypeError:
                try:
                    x = layer(x, init_params=init_params)
                except TypeError:
                    x = layer(x)
            xl.append(x)

        # Multi-layer feature aggregation
        x = torch.cat(xl[1:], dim=1)
        x = self.mfa(x, init_params=init_params)

        # Attantitve Statistical pooling
        x = self.asp(x, lengths=lengths, init_params=init_params)
        x = self.asp_bn(x, init_params=init_params)

        # Final linear transformation
        x = self.fc(x, init_params=init_params)

        x = x.transpose(1, 2)
        return x


class Classifier(torch.nn.Module):
    """This class implements the cosine similarity on the top of features.

    Arguments
    ---------
    device : str
        Device used e.g. "cpu" or "cuda"
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.
    out_neurons : int
        Number of classes

    Example
    -------
    >>> classify = Classifier('cpu', out_neurons=2)
    >>> outputs = torch.tensor([ [1., -1.], [-9., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> outupts = outputs.unsqueeze(1)
    >>> cos = classify(outputs, init_params=True)
    >>> (cos < -1.0).long().sum()
    tensor(0)
    >>> (cos > 1.0).long().sum()
    tensor(0)
    """

    def __init__(
        self, device="cpu", lin_blocks=1, lin_neurons=192, out_neurons=1211,
    ):

        super().__init__()
        self.blocks = nn.ModuleList()

        for block_index in range(lin_blocks):
            self.blocks.extend([_BatchNorm1d(), Linear(n_neurons=lin_neurons)])

        # Final Layer
        self.weight = nn.Parameter(
            torch.FloatTensor(out_neurons, lin_neurons).to(device)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, init_params=False):
        """Returns the output probabilities over speakers.

        Arguments
        ---------
        x : torch.Tensor
        """
        for layer in self.blocks:
            try:
                x = layer(x, init_params=init_params)
            except TypeError:
                x = layer(x)

        # Need to be normalized
        x = F.linear(F.normalize(x.squeeze(1)), F.normalize(self.weight))
        return x.unsqueeze(1)
