"""The SpeechBrain implementation of DCCRN by
https://arxiv.org/pdf/2008.00264.pdf

Authors
 * Chien-Feng Liao 2020
"""
import torch  # noqa F4001
import torch.nn as nn
from speechbrain.nnet.complex_networks.CNN import ComplexConv2d
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.complex_networks.normalization import ComplexBatchNorm
from speechbrain.nnet.RNN import LSTM
from speechbrain.nnet.complex_networks.complex_ops import complex_concat


class DCCRN(nn.Module):
    def __init__(
        self,
        input_shape,
        conv_channels=[32, 64, 128, 128, 256],
        kernel_size=[3, 5],
        rnn_size=128,
        rnn_layers=2,
        padding="same",
        norm=ComplexBatchNorm,
    ):
        super().__init__()
        N, T, F, C = input_shape

        # Encoder layers
        self.encoder_convs = nn.ModuleList()
        prev_c = C
        encoder_size = []
        for c in conv_channels:
            self.encoder_convs.append(
                Encoder_layer(
                    c,
                    kernel_size,
                    stride=[1, 2],  # halve frequency
                    input_size=prev_c,
                    padding=padding,
                    norm=norm,
                )
            )
            prev_c = c * 2
            F = self.get_output_size(F, 2, kernel_size[1], 1)
            encoder_size.append(F)

        # Bottleneck layers
        # Conv1d layer to transform rnn output back to 4-D
        # and add semi-causal
        self.rnn = LSTM(
            rnn_size, input_shape=[N, T, F, prev_c], num_layers=rnn_layers
        )
        self.linear_trans = Conv1d(
            F * prev_c,
            3,
            input_shape=[N, T, rnn_size],
            stride=1,
            padding="valid",
        )

        # Decoder layers
        self.decoder_convs = nn.ModuleList()
        for c, u in zip(conv_channels[:-1][::-1], encoder_size[:-1][::-1]):
            self.decoder_convs.append(
                Decoder_layer(
                    c,
                    kernel_size,
                    input_size=prev_c * 2,
                    up_size=u,
                    padding=padding,
                    norm=norm,
                )
            )
            prev_c = c * 2

        self.decoder_convs.append(
            Decoder_layer(
                1,
                kernel_size,
                prev_c,
                input_shape[2],
                padding=padding,
                use_norm_act=False,
            )
        )

    def get_output_size(
        self, L_in: int, stride: int, kernel_size: int, dilation: int
    ):
        L_out = (
            L_in + 2 * kernel_size // 2 - dilation * (kernel_size - 1) - 1
        ) / stride + 1
        return int(L_out)

    def forward(self, x):
        encoder_outputs = [x]
        for conv in self.encoder_convs:
            x = conv(x)
            encoder_outputs.append(x)

        # Apply RNN and reshape back to 4-D
        rnn_out, _ = self.rnn(x)
        # Semi-causal padding on time axis, kernel size is fixed to 3
        rnn_out = torch.nn.functional.pad(rnn_out, (0, 0, 1, 1))
        rnn_out = self.linear_trans(rnn_out)
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
            decoder_out = conv(decoder_out)

        return decoder_out


class Encoder_layer(nn.Module):
    def __init__(
        self,
        out_channels,
        kernel_size,
        stride,
        input_size,
        activation=nn.PReLU,
        norm=ComplexBatchNorm,
        padding="same",
    ):
        super().__init__()

        self.conv = ComplexConv2d(
            out_channels,
            kernel_size,
            input_size=input_size,
            stride=stride,
            padding=padding,
        )
        self.norm = norm(input_size=out_channels * 2)
        self.activation = activation()

    def forward(self, x):

        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)

        return x


class Decoder_layer(nn.Module):
    def __init__(
        self,
        out_channels,
        kernel_size,
        input_size,
        up_size,
        activation=nn.PReLU,
        norm=ComplexBatchNorm,
        padding="same",
        use_norm_act=True,
    ):
        super().__init__()
        self.use_norm_act = use_norm_act
        self.conv = ComplexConv2d(
            out_channels, kernel_size, input_size=input_size, padding=padding
        )
        self.up = nn.Upsample(
            size=[up_size, input_size], mode="bilinear", align_corners=True
        )
        if self.use_norm_act:
            self.activation = activation()
            self.norm = norm(input_size=out_channels * 2)

    def forward(self, x):
        # Upsample to the expected size
        x = self.up(x)
        x = self.conv(x)

        if self.use_norm_act:
            x = self.norm(x)
            x = self.activation(x)

        return x
