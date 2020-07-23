"""CNN Transformer model for SE in the SpeechBrain style

Authors
* Chien-Feng Liao 2020
"""
import torch  # noqa E402
from torch import nn
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.nnet.containers import Sequential
from speechbrain.lobes.models.transformer.Transformer import (
    TransformerInterface,
    get_lookahead_mask,
)


class CNNTransformerSE(TransformerInterface):
    """This is an implementation of transformer model with CNN pre-encoder for SE

    Arguements
    ----------
    output_size: int
        the number of expected neurons in the output layer.
    output_activation: torch class
        the activation function of output layer, (default=ReLU).
    nhead: int
        the number of heads in the multiheadattention models (default=8).
    num_layers: int
        the number of sub-layers in the transformer (default=8).
    d_ffn: int
        the number of expected features in the encoder layers (default=512).
    dropout: int
        the dropout value (default=0.1).
    activation: torch class
        the activation function of intermediate layers (default=LeakyReLU).
    causal: bool
        True for causal setting, the model is forbidden to see future frames. (default=True)
    cnn_channels: list
        A list for the number of channels in the CNN encoder.
    cnn_kernelsize: list
        A list for the kernel sizes in the CNN encoder.
    Example
    -------
    >>> src = torch.rand([8, 120, 60])
    >>> net = CNNTransformerSE(257)
    >>> out = net.forward(src, init_params=True)
    >>> print(out.shape)
    torch.Size([8, 120, 257])
    """

    def __init__(
        self,
        output_size,
        output_activation=nn.ReLU,
        nhead=8,
        num_layers=8,
        d_ffn=512,
        dropout=0.1,
        activation=nn.LeakyReLU,
        causal=True,
        cnn_channels=[1024, 512, 128, 256],
        cnn_kernelsize=[3, 3, 3, 3],
    ):
        super().__init__(
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=0,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            return_attention=False,
            positional_encoding=False,
        )

        self.custom_src_module = CNNencoder(
            activation=activation,
            cnn_channels=cnn_channels,
            cnn_kernelsize=cnn_kernelsize,
            causal=causal,
        )

        self.causal = causal
        self.output_layer = Linear(output_size, bias=False)
        self.output_activation = output_activation()

    def forward(self, x, init_params=False):
        if self.causal:
            self.attn_mask = get_lookahead_mask(x)
        else:
            self.attn_mask = None

        src = self.custom_src_module(x, init_params)
        encoder_output = self.encoder(
            src=src,
            src_mask=self.attn_mask,
            src_key_padding_mask=None,
            init_params=init_params,
        )

        output = self.output_layer(encoder_output, init_params)
        output = self.output_activation(output)

        return output


class CNNencoder(Sequential):
    def __init__(
        self,
        activation=nn.LeakyReLU,
        cnn_channels=[1024, 512, 128, 256],
        cnn_kernelsize=[3, 3, 3, 3],
        causal=True,
    ):
        if causal:
            pad = "causal"
        else:
            pad = "same"

        blocks = []

        for c, k in zip(cnn_channels, cnn_kernelsize):
            blocks.extend(
                [
                    Conv1d(out_channels=c, kernel_size=k, padding=pad),
                    LayerNorm(),
                    activation(),
                ]
            )

        super().__init__(*blocks)
