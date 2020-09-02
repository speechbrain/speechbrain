import torch
import torch.nn as nn
import torch.nn.functional as F

from speechbrain.processing.signal_processing import overlap_and_add
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.containers import Sequential

EPS = 1e-8


class Encoder(nn.Module):
    """This class learns the adaptive frontend for the ConvTasnet model

    Arguments:
    L: The filter kernel size, needs to an odd number
    N: number of dimensions at the output of the adaptive front end.

    Example:
    ----------
    >>> inp = torch.rand(10, 100)
    >>> encoder = Encoder(11, 20)
    >>> h = encoder(inp)
    >>> h.shape
    torch.Size([10, 20, 20])
    """

    def __init__(self, L, N):
        super(Encoder, self).__init__()
        # Hyper-parameter
        self.L, self.N = L, N
        # Components
        # 50% overlap
        self.conv1d_U = Conv1d(N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture, init_params=True):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            mixture_w: [M, K, N], where K = (T-L)/(L/2)+1 = 2T/L-1
        """
        mixture = torch.unsqueeze(mixture, -1)  # [M, T, 1]

        if init_params:
            self.conv1d_U(mixture, init_params=True)

        conv_out = self.conv1d_U(mixture)
        mixture_w = F.relu(conv_out)  # [M, K, N]
        return mixture_w


class Decoder(nn.Module):
    """
    This class implements the decoder for the ConvTasnet.
    The seperated source embeddings are fed to the decoder to reconstruct the estimated sources in the time domain.

    Argument:
    L: Number of bases to use when reconstructing

    Example:
    ---------
    >>> L, C = 8, 2
    >>> mixture_w = torch.randn(10, 100, 8)
    >>> est_mask = torch.randn(10, 100, C, 8)
    >>> Decoder = Decoder(L)
    >>> mixture_hat = Decoder(mixture_w, est_mask)
    >>> mixture_hat.shape
    torch.Size([10, 404, 2])
    """

    def __init__(self, L):
        super(Decoder, self).__init__()
        # Hyper-parameter
        self.L = L
        # Components
        self.basis_signals = Linear(L, bias=False)

    def forward(self, mixture_w, est_mask, init_params=True):
        """
        Args:
            mixture_w: [M, K, N]
            est_mask: [M, K, C, N]
        Returns:
            est_source: [M, T, C]
        """

        if init_params:
            source_w = torch.unsqueeze(mixture_w, 2) * est_mask  # [M, K, C, N]
            source_w = source_w.permute(0, 2, 1, 3)  # [M, C, K, N]

            self.basis_signals(source_w, init_params=True)

        # D = W * M
        source_w = torch.unsqueeze(mixture_w, 2) * est_mask  # [M, K, C, N]
        source_w = source_w.permute(0, 2, 1, 3)  # [M, C, K, N]
        # S = DV
        est_source = self.basis_signals(source_w)  # [M, C, K, L]
        est_source = overlap_and_add(est_source, self.L // 2)  # M x C x T

        return est_source.permute(0, 2, 1)  # M x T x C


class TemporalBlocksSequential(Sequential):
    """
    A wrapper for the temporalblock layer to replicate it

    Arguments:
    B: the number of input channels, and the number of output channels
    H: the number of intermediate channels
    P: the kernel size in the convolutions
    R: the number of times to replicate the multilayer Temporal Blocks
    X: The number of layers of Temporal Blocks with different dilations
    norm type: the type of normalization, in ['gLN', 'cLN']
    causal: to use causal or non-causal convolutions, in [True, False]

    Example:
    ---------
    >>> B, H, P, R, X = 10, 10, 5, 2, 3
    >>> TemporalBlocks = TemporalBlocksSequential(B, H, P, R, X, 'gLN', False)
    >>> x = torch.randn(14, 100, 10)
    >>> y = TemporalBlocks(x, init_params=True)
    >>> y.shape
    torch.Size([14, 100, 10])
    """

    def __init__(self, B, H, P, R, X, norm_type, causal):
        repeats = []
        for r in range(R):
            blocks = []
            for x in range(X):
                dilation = 2 ** x
                blocks += [
                    TemporalBlock(
                        B,
                        H,
                        P,
                        stride=1,
                        padding="same",
                        dilation=dilation,
                        norm_type=norm_type,
                        causal=causal,
                    )
                ]
            repeats.extend(blocks)

        super().__init__(*repeats)


class MaskNet(Sequential):
    def __init__(
        self,
        N,
        B,
        H,
        P,
        X,
        R,
        C,
        norm_type="gLN",
        causal=False,
        mask_nonlinear="relu",
    ):
        """
        Args:
            N: Number of filters in autoencoder
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask, in ['softmax', 'relu']

        Example:
        ---------
        >>> N, B, H, P, X, R, C = 11, 12, 2, 5, 6, 1, 2
        >>> MaskNet = MaskNet(N, B, H, P, X, R, C)
        >>> mixture_w = torch.randn(10, 100, 11)
        >>> est_mask = MaskNet(mixture_w)
        >>> est_mask.shape
        torch.Size([10, 100, 2, 11])
        """
        super(MaskNet, self).__init__()
        # Hyper-parameter
        self.C = C
        self.mask_nonlinear = mask_nonlinear
        # Components
        # [M, K, N] -> [M, K, N]
        self.layer_norm = ChannelwiseLayerNorm(N)
        # [M, K, N] -> [M, K, B]
        self.bottleneck_conv1x1 = Conv1d(B, 1, bias=False)
        # [M, K, B] -> [M, K, B]

        self.temporal_conv_net = TemporalBlocksSequential(
            B, H, P, R, X, norm_type, causal
        )
        # [M, K, B] -> [M, K, C*N]
        self.mask_conv1x1 = Conv1d(C * N, 1, bias=False)

    def forward(self, mixture_w, init_params=True):
        """
        Keep this API same with TasNet
        Args:
            mixture_w: [M, K, N], M is batch size
        returns:
            est_mask: [M, K, C, N]
        """

        if init_params:
            y = self.layer_norm(mixture_w)
            y = self.bottleneck_conv1x1(y, init_params=True)

            y = self.temporal_conv_net(y, init_params=True)
            y = self.mask_conv1x1(y, init_params=True)

        M, K, N = mixture_w.size()
        y = self.layer_norm(mixture_w)
        y = self.bottleneck_conv1x1(y)
        y = self.temporal_conv_net(y)
        score = self.mask_conv1x1(y)

        # score = self.network(mixture_w)  # [M, K, N] -> [M, K, C*N]
        score = score.contiguous().reshape(
            M, K, self.C, N
        )  # [M, K, C*N] -> [M, K, C, N]
        if self.mask_nonlinear == "softmax":
            est_mask = F.softmax(score, dim=2)
        elif self.mask_nonlinear == "relu":
            est_mask = F.relu(score)
        else:
            raise ValueError("Unsupported mask non-linear function")
        return est_mask


class TemporalBlock(Sequential):
    """
    The conv1d compound layers used in Masknet

    Arguments:
    in_channels: the number of input channels, and the number of output channels
    out_channels: the number of intermediate channels
    kernel_size: the kernel size in the convolutions
    stride: convolution stride in convolutional layers
    padding: the type of padding in the convolutional layers,
            (same, valid, causal). If "valid", no padding is performed.
    dilation: amount of dilation in convolutional layers
    norm type: the type of normalization, in ['gLN', 'cLN']
    causal: to use causal or non-causal convolutions, in [True, False]

    Example:
    ---------
    >>> in_channels = 10
    >>> TemporalBlock = TemporalBlock(in_channels, 10, 11, 1, 'same', 1)
    >>> x = torch.randn(14, 100, 10)
    >>> y = TemporalBlock(x, init_params=True)
    >>> y.shape
    torch.Size([14, 100, 10])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        norm_type="gLN",
        causal=False,
    ):
        # [M, K, B] -> [M, K, H]
        conv1x1 = Conv1d(out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = choose_norm(norm_type, out_channels)
        # [M, K, H] -> [M, K, B]
        dsconv = DepthwiseSeparableConv(
            out_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            norm_type,
            causal,
        )
        # Put together
        net = [conv1x1, prelu, norm, dsconv]
        super().__init__(*net)

    def forward(self, x, init_params=False):
        """
        Arguments
            x: [M, K, B]
        Returns:
            [M, K, B]
        """
        residual = x
        for layer in self.layers:
            try:
                x = layer(x, init_params=init_params)
            except TypeError:
                x = layer(x)
        return x + residual


class DepthwiseSeparableConv(Sequential):
    """
    Building block for the Temporal Blocks of Masknet in ConvTasNet

    Arguments:
    in_channels: number of input channels
    out_channels: number of output channels
    kernel_size: the kernel size in the convolutions
    stride: convolution stride in convolutional layers
    padding: the type of padding in the convolutional layers,
            (same, valid, causal). If "valid", no padding is performed.
    dilation: amount of dilation in convolutional layers
    norm type: the type of normalization, in ['gLN', 'cLN']
    causal: to use causal or non-causal convolutions, in [True, False]

    Example:
    ---------
    >>> in_channels = 10
    >>> DSconv = DepthwiseSeparableConv(in_channels, 10, 11, 1, 'same', 1)
    >>> x = torch.randn(14, 100, 10)
    >>> y = DSconv(x, init_params=True)
    >>> y.shape
    torch.Size([14, 100, 10])

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        norm_type="gLN",
        causal=False,
    ):
        # [M, K, H] -> [M, K, H]
        depthwise_conv = Conv1d(
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        if causal:
            chomp = Chomp1d(padding)
        prelu = nn.PReLU()
        norm = choose_norm(norm_type, in_channels)
        # [M, K, H] -> [M, K, B]
        pointwise_conv = Conv1d(out_channels, 1, bias=False)
        # Put together
        if causal:
            net = [depthwise_conv, chomp, prelu, norm, pointwise_conv]
        else:
            net = [depthwise_conv, prelu, norm, pointwise_conv]
        super().__init__(*net)


class Chomp1d(nn.Module):
    """This class cuts out a portion of the signal from the end.
    It is written as a class to be able to incorporate it inside a sequential wrapper.

    Argument:
    chomp_size: The size of the portion to discard. (in samples)

    Example:
    ----------
    >>> x = torch.randn(10, 110, 5)
    >>> chomp = Chomp1d(10)
    >>> x_chomped = chomp(x)
    >>> x_chomped.shape
    torch.Size([10, 100, 5])
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Args:
            x: [M, Kpad, H ]
        Returns:
            [M, K, H]
        """
        return x[:, : -self.chomp_size, :].contiguous()


def choose_norm(norm_type, channel_size):
    """
    This function returns the chosen normalization type

    Arguments:
    norm_type: in ['gLN', 'cLN', 'batchnorm']
    channel_size: number of channels (integer)

    Example:
    >>> norm = choose_norm('gLN', 10)
    >>> print(norm)
    GlobalLayerNorm()
    """
    if norm_type == "gLN":
        return GlobalLayerNorm(channel_size)
    elif norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size)
    else:
        return nn.BatchNorm1d(channel_size)


class ChannelwiseLayerNorm(nn.Module):
    """
    Channel-wise Layer Normalization (cLN)

    Arguments:
    Channel_size: number of channels in the normalization dimension (the third dimension)

    Example:
    ---------
    >>> x = torch.randn(2, 3, 3)
    >>> norm_func = ChannelwiseLayerNorm(3)
    >>> x_normalized = norm_func(x)
    >>> x.shape
    torch.Size([2, 3, 3])
    """

    def __init__(self, channel_size):
        super(ChannelwiseLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, 1, channel_size))  # [1, 1, N]
        self.beta = nn.Parameter(torch.Tensor(1, 1, channel_size))  # [1, 1, N]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, K, N], M is batch size, N is channel size, K is length
        Returns:
            cLN_y: [M, K, N]
        """
        mean = torch.mean(y, dim=2, keepdim=True)  # [M, K, 1]
        var = torch.var(y, dim=2, keepdim=True, unbiased=False)  # [M, K, 1]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return cLN_y


class GlobalLayerNorm(nn.Module):
    """
    Global Layer Normalization (gLN)

    Arguments:
    Channel_size: number of channels in the third dimension

    Example:
    ---------
    >>> x = torch.randn(2, 3, 3)
    >>> norm_func = GlobalLayerNorm(3)
    >>> x_normalized = norm_func(x)
    >>> x.shape
    torch.Size([2, 3, 3])
    """

    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, 1, channel_size))  # [1, 1, N]
        self.beta = nn.Parameter(torch.Tensor(1, 1, channel_size))  # [1, 1, N]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, K, N], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, K. N]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(
            dim=2, keepdim=True
        )  # [M, 1, 1]
        var = (
            (torch.pow(y - mean, 2))
            .mean(dim=1, keepdim=True)
            .mean(dim=2, keepdim=True)
        )
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y
