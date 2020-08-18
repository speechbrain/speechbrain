"""The SpeechBrain implementation of DCCRN by
https://arxiv.org/pdf/2008.00264.pdf

Authors
 * Chien-Feng Liao 2020
"""
import torch  # noqa F4001
import torch.nn as nn
from speechbrain.nnet.complex_networks.CNN import ComplexConv2d
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.complex_networks.normalization import ComplexBatchNorm
from speechbrain.nnet.RNN import LSTM
from speechbrain.nnet.complex_networks.complex_ops import complex_concat


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

        self.encoder_convs = nn.ModuleList(
            [
                Encoder_layer(
                    c, self.kernel_size, self.strides, padding=self.padding
                )
                for c in self.conv_channels
            ]
        )

        x = first_input
        # Get the output size of each encoder layers
        self.encoder_size = []
        for conv in self.encoder_convs:
            x = conv(x, init_params=True)
            self.encoder_size.append(x.shape[2])

        self.rnn = LSTM(
            self.rnn_size,
            num_layers=self.rnn_layers,
            # normalization="batchnorm",
        )

        # Linear layer to transform rnn output back to 4-D
        self.linear_dim = self.encoder_size[-1] * self.conv_channels[-1] * 2
        self.linear_trans = Linear(self.linear_dim)

        self.decoder_convs = nn.ModuleList(
            [
                Decoder_layer(
                    c,
                    self.kernel_size,
                    output_size=[T, u],
                    padding=self.padding,
                )
                for c, u in zip(
                    self.conv_channels[:-1][::-1], self.encoder_size[:-1][::-1]
                )
            ]
        )
        self.decoder_convs.append(
            Decoder_layer(
                1,
                self.kernel_size,
                output_size=[T, F],
                padding=self.padding,
                use_norm_act=False,
            )
        )

    def forward(self, x, init_params=False):
        if init_params:
            self.init_params(x)

        encoder_outputs = [x]
        for conv in self.encoder_convs:
            x = conv(x, init_params=False)
            encoder_outputs.append(x)

        # Apply RNN and linear transform back to 4-D
        rnn_out = self.rnn(x, init_params=init_params)
        rnn_out = self.linear_trans(rnn_out, init_params=init_params)

        # If use complex RNN + complex Linear:
        # split then reshape is needed instead of directly reshape
        # rnn_out_r, rnn_out_i = torch.split(rnn_out, self.linear_dim, -1)
        # rnn_out_r = rnn_out_r.reshape(
        #     x.shape[0], x.shape[1], x.shape[2], x.shape[3] // 2
        # )
        # rnn_out_i = rnn_out_i.reshape(
        #     x.shape[0], x.shape[1], x.shape[2], x.shape[3] // 2
        # )
        # rnn_out = torch.cat([rnn_out_r, rnn_out_i], dim=3)

        # Directly reshape
        rnn_out = rnn_out.reshape(
            x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        )

        decoder_out = rnn_out
        for i, conv in enumerate(self.decoder_convs):
            # Concat last layer output with encoder output
            skip_ = encoder_outputs[-(i + 1)]
            decoder_out = complex_concat(
                [skip_, decoder_out], input_type="convolution", channels_axis=3
            )
            decoder_out = conv(decoder_out, init_params=init_params)

        return decoder_out


class Encoder_layer(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        strides,
        activation=nn.LeakyReLU,
        norm=ComplexBatchNorm,
        padding="same",
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.norm = norm
        self.padding = padding

    def init_params(self, first_input):
        self.device = first_input.device
        self.conv = ComplexConv2d(
            self.channels, self.kernel_size, self.strides, padding=self.padding
        ).to(self.device)
        self.norm = self.norm().to(self.device)
        self.activation = self.activation().to(self.device)

    def forward(self, x, init_params=False):
        if init_params:
            self.init_params(x)

        x = self.conv(x, init_params=init_params)
        x = self.norm(x, init_params=init_params)
        x = self.activation(x)

        return x


class Decoder_layer(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        output_size,
        activation=nn.LeakyReLU,
        norm=ComplexBatchNorm,
        padding="same",
        use_norm_act=True,
    ):
        super().__init__()
        self.use_norm_act = use_norm_act
        self.conv = ComplexConv2d(channels, kernel_size, padding=padding)
        self.up = nn.Upsample(
            size=output_size, mode="bilinear", align_corners=True
        )
        if self.use_norm_act:
            self.activation = activation()
            self.norm = ComplexBatchNorm()

    def forward(self, x, init_params=False):
        # Upsample to the expected size
        x = x.permute([0, 3, 1, 2])
        x = self.up(x)
        x = x.permute([0, 2, 3, 1])

        x = self.conv(x, init_params=init_params)
        if self.use_norm_act:
            x = self.norm(x, init_params=init_params)
            x = self.activation(x)

        return x
