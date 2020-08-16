import torch
import torch.nn as nn
import torch.nn.functional as F

# import math
# import itertools
from speechbrain.processing.signal_processing import overlap_and_add
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.containers import Sequential

EPS = 1e-8


class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
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


class ConvTasNet(Sequential):
    def __init__(
        self,
        N,
        L,
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
            L: Length of the filters (in samples)
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
        """
        super(ConvTasNet, self).__init__()
        # Hyper-parameter
        self.N, self.L, self.B, self.H, self.P, self.X, self.R, self.C = (
            N,
            L,
            B,
            H,
            P,
            X,
            R,
            C,
        )
        self.norm_type = norm_type
        self.causal = causal
        self.mask_nonlinear = mask_nonlinear
        # Components
        self.encoder = Encoder(L, N)
        self.separator = MaskNet(
            N, B, H, P, X, R, C, norm_type, causal, mask_nonlinear
        )
        self.decoder = Decoder(N, L)
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(
        self, mixture,
    ):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            est_source: [M, C, T]
        """

        mixture_w = self.encoder(mixture)
        est_mask = self.separator(mixture_w)
        est_source = self.decoder(mixture_w, est_mask)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))
        return est_source


class TemporalBlocksSequential(Sequential):
    def __init__(self, B, H, P, R, X, norm_type, causal):
        repeats = []
        for r in range(R):
            blocks = []
            for x in range(X):
                dilation = 2 ** x
                # padding = (
                #    (P - 1) * dilation if causal else (P - 1) * dilation // 2
                # )
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
            mask_nonlinear: use which non-linear function to generate mask
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
        # Put together
        # self.network = nn.Sequential(
        #    layer_norm , bottleneck_conv1x1, #temporal_conv_net, mask_conv1x1
        # )

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
        score = score.reshape(M, K, self.C, N)  # [M, K, C*N] -> [M, K, C, N]
        if self.mask_nonlinear == "softmax":
            est_mask = F.softmax(score, dim=2)
        elif self.mask_nonlinear == "relu":
            est_mask = F.relu(score)
        else:
            raise ValueError("Unsupported mask non-linear function")
        return est_mask


class TemporalBlock(Sequential):
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
        # super(TemporalBlock, self).__init__()
        # [M, B, K] -> [M, H, K]
        conv1x1 = Conv1d(out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, out_channels)
        # [M, H, K] -> [M, B, K]
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
        ---------
        x : tensor
            the input tensor to run through the network.
        """
        residual = x
        for layer in self.layers:
            try:
                x = layer(x, init_params=init_params)
            except TypeError:
                x = layer(x)
        return x + residual

    # def forward(self, x):
    #    """
    #    Args:
    #        x: [M, K, B]
    #    Returns:
    #        [M, K, B]
    #    """
    #    residual = x
    #    out = self(x)
    #    # TODO: when P = 3 here works fine, but when P = 2 maybe need to pad?
    #    return out + residual  # look like w/o F.relu is better than w/ F.relu
    #    # return F.relu(out + residual)


class DepthwiseSeparableConv(Sequential):
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
        # super(DepthwiseSeparableConv, self).__init__()
        # Use `groups` option to implement depthwise convolution
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
        norm = chose_norm(norm_type, in_channels)
        # [M, K, H] -> [M, K, B]
        pointwise_conv = Conv1d(out_channels, 1, bias=False)
        # Put together
        if causal:
            net = [depthwise_conv, chomp, prelu, norm, pointwise_conv]
        else:
            net = [depthwise_conv, prelu, norm, pointwise_conv]
        super().__init__(*net)


class Chomp1d(nn.Module):
    """To ensure the output length is the same as the input.
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


def chose_norm(norm_type, channel_size):
    """The input of normlization will be (M, K, C), where M is batch size,
       C is channel size and K is sequence length.
    """
    if norm_type == "gLN":
        return GlobalLayerNorm(channel_size)
    elif norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size)
    else:  # norm_type == "BN":
        # Given input (M, K, C), nn.BatchNorm1d(C) will accumulate statics
        # along M and K, so this BN usage is right.
        return nn.BatchNorm1d(channel_size)


# TODO: Use nn.LayerNorm to impl cLN to speed up
class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN)"""

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
    """Global Layer Normalization (gLN)"""

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
