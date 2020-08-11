"""The SpeechBrain implementation of ContextNet by
https://arxiv.org/pdf/2005.03191.pdf

Authors
 * Jianyuan Zhong 2020
"""
import torch
import torch.nn as nn
from speechbrain.nnet.complex_networks.CNN import ComplexConv2d
from speechbrain.nnet.complex_networks.linear import ComplexLinear
from speechbrain.nnet.complex_networks.normalization import ComplexBatchNorm
from speechbrain.nnet.complex_networks.RNN import ComplexLiGRU
from speechbrain.nnet.containers import Sequential


class DCCRN(nn.Module):
    def __init__(
        self,
        conv_channels=[32, 64, 128, 128, 256],
        kernel_size=[3, 5],
        strides=[2, 1],
        rnn_size=128,
        rnn_layers=2,
        padding="same",
    ):
        super().__init__()
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers
        self.padding = padding
        self.encoder_outputs = []

    def init_params(self, first_input):
        N, T, F, C = first_input.shape
        self.device = first_input.device

        self.encoder_convs = [
            Encoder_layer(
                c, self.kernel_size, self.strides, padding=self.padding
            )
            for c in self.conv_channels
        ]
        self.decoder_convs = [
            Decoder_layer(c, self.kernel_size, padding=self.padding)
            for c in self.conv_channels[:-1][::-1]
        ]
        self.decoder_convs += [
            Decoder_layer(
                1, self.kernel_size, padding=self.padding, use_norm_act=False
            )
        ]
        self.rnn = ComplexLiGRU(
            self.rnn_size,
            num_layers=self.rnn_layers,
            normalization="batchnorm",
        )

        encoder = Sequential(*self.encoder_convs)
        encoder_out = encoder(first_input, init_params=True)
        self.linear_dim = encoder_out.shape[2] * encoder_out.shape[3] // 2
        self.linear_trans = ComplexLinear(self.linear_dim)

    def forward(self, x, init_params=False):
        if init_params:
            self.init_params(x)

        self.encoder_outputs.append(x)
        for conv in self.encoder_convs:
            x = conv(x, init_params=False)
            self.encoder_outputs.append(x)

        # Apply RNN and linear transform back to 4-D
        rnn_out = self.rnn(x, init_params=init_params)
        rnn_out = self.linear_trans(rnn_out, init_params=init_params)

        # Split then reshape is needed instead of directly reshape
        rnn_out_r, rnn_out_i = torch.split(rnn_out, self.linear_dim, -1)
        rnn_out_r = rnn_out_r.reshape(
            x.shape[0], x.shape[1], x.shape[2], x.shape[3] // 2
        )
        rnn_out_i = rnn_out_i.reshape(
            x.shape[0], x.shape[1], x.shape[2], x.shape[3] // 2
        )
        rnn_out = torch.cat([rnn_out_r, rnn_out_i], dim=3)

        decoder_out = rnn_out
        for i, conv in enumerate(self.decoder_convs):
            # TODO: change to concat
            decoder_out += self.encoder_outputs[-(i + 1)]
            upsample_size = self.encoder_outputs[-(i + 2)].shape[2]
            decoder_out = conv(
                decoder_out,
                [decoder_out.shape[1], upsample_size],
                init_params=init_params,
            )

        return decoder_out


class Encoder_layer(Sequential):
    def __init__(
        self,
        channels,
        kernel_size,
        strides,
        activation=nn.LeakyReLU,
        norm=ComplexBatchNorm,
        padding="same",
    ):
        blocks = [
            ComplexConv2d(channels, kernel_size, strides, padding=padding),
            norm(),
            activation(),
        ]
        super().__init__(*blocks)


class Decoder_layer(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        activation=nn.LeakyReLU,
        norm=ComplexBatchNorm,
        padding="same",
        use_norm_act=True,
    ):
        super().__init__()
        self.use_norm_act = use_norm_act
        self.conv = ComplexConv2d(channels, kernel_size, padding=padding)
        if self.use_norm_act:
            self.activation = activation()
            self.norm = ComplexBatchNorm()

    def forward(self, x, upsample_size, init_params=False):
        # Upsample to the expected size
        x = x.permute([0, 3, 1, 2])
        up = nn.Upsample(
            size=upsample_size, mode="bilinear", align_corners=True
        )
        x = up(x)
        x = x.permute([0, 2, 3, 1])

        x = self.conv(x, init_params=init_params)
        if self.use_norm_act:
            x = self.norm(x, init_params=init_params)
            x = self.activation(x)

        return x
