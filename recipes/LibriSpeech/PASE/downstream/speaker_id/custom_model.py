"""
This file contains a very simple PASE encoder module to use for speaker-id.

To replace this model, change the `!new:` tag in the hyperparameter file
to refer to a built-in SpeechBrain model or another file containing
a custom PyTorch module.

Authors
 * Eshwanth Baskaran 2021
"""

import torch  # noqa: F401
import torch.nn as nn
import speechbrain as sb
from speechbrain.nnet.pooling import StatisticsPooling
from speechbrain.nnet.CNN import Conv1d, SincConv
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.normalization import BatchNorm1d

from qrnn import QRNN


class ConvBlock(torch.nn.Module):

    """A convolution block used in the PASE encoder

    Attributes:
        conv_layer (torch.nn.Module): Convolution layer
        norm_layer (torch.nn.Module): Normalization layer
        activation (torch.nn.Module): Activation layer
    """

    def __init__(self,
                 conv_layer_cls,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 activation=torch.nn.PReLU):
        super().__init__()
        self.conv_layer = conv_layer_cls(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                        )
        self.norm_layer = BatchNorm1d(input_size=out_channels)
        self.activation = activation()

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.norm_layer(x)
        x = self.activation(x)
        return x


class PASEEncoder(torch.nn.Module):

    """Encoder model class for PASE/PASE+ system

    Attributes:
        conv_blocks (torch.nn.ModuleList): List of conv layers (except last layer)
        final_conv_block (torch.nn.ModuleList): Final conv layer (applied after QRNN block)
        final_embed_dim (int): Final embedding dimension depending on whether skip connection was used
        final_norm_layer (torch.nn.Module or None): Normalization layer applied if skip connections results are appended
        qrnn_block (torch.nn.Module or None): QRNN block
        skip_conns (torch.nn.ModuleList): conv layers for skip connections
    """

    def __init__(
        self,
        activation=torch.nn.PReLU,
        use_sincnet=True,
        use_qrnn=False,
        use_skip_conns=False,
        in_channels=1,
        blocks_channels=[64,64,128,128,256,256,512,512,100],
        blocks_kernel_sizes=[251,20,11,11,11,11,11,11,1],
        blocks_strides=[1,10,2,1,2,1,2,2,1],
        qrnn_hidden_neurons=512,
    ):
        """PASE encoder model initialization

        Args:
            activation (TYPE, optional): Activation function to use in encoder
            use_sincnet (bool, optional): Whether to use SincNet. Default is Conv1d
            use_qrnn (bool, optional): Whether to use QRNN in the encoder
            use_skip_conns (bool, optional): Whether to use skip connections and add them to final output
            in_channels (int, optional): Number of channels in the input
            blocks_channels (list, optional): List of values for channels in conv blocks
            blocks_kernel_sizes (list, optional): List of size values for kernels in conv blocks
            blocks_strides (list, optional): List of stride values in conv blocks
            qrnn_hidden_neurons (int, optional): Number of hidden neurons in QRNN
        """
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        self.skip_conns = nn.ModuleList() if use_skip_conns else None
        self.qrnn_block = None
        self.final_conv_block = nn.ModuleList()
        self.final_norm_layer = None

        # Conv blocks
        for block_index in range(len(blocks_channels)-1):
            conv_layer_cls = SincConv if (block_index == 0 and use_sincnet) else Conv1d
            out_channels = blocks_channels[block_index]
            self.conv_blocks.append(
                ConvBlock(
                    conv_layer_cls=conv_layer_cls,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=blocks_kernel_sizes[block_index],
                    stride=blocks_strides[block_index],
                    activation=activation,
                )
            )

            # skip connections
            if use_skip_conns:
                self.skip_conns.append(
                    Conv1d(
                        in_channels=out_channels,
                        out_channels=blocks_channels[-1],
                        kernel_size=1,
                        bias=False,
                    )
                )
            in_channels = blocks_channels[block_index]

        # Q-RNN block
        if use_qrnn:
            self.qrnn_block = QRNN(
                            input_size=in_channels,
                            hidden_size=qrnn_hidden_neurons,
                            num_layers=1,
                            dropout=0,
                            window=2,
                        )
            in_channels = qrnn_hidden_neurons

        # Final convolution block
        self.final_conv_block = Conv1d(
                                    in_channels=in_channels,
                                    out_channels=blocks_channels[block_index + 1],
                                    kernel_size=blocks_kernel_sizes[block_index + 1],
                                    stride=blocks_strides[block_index + 1],
                                )

        # Final normalization layer
        self.final_norm_layer = BatchNorm1d(input_size=blocks_channels[block_index + 1], affine=False)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """Forward function of module

        Args:
            x (torch.Tensor): Input tensor
            *args: Additional args (if any)
            **kwargs: Additional keyword args (if any)

        Returns:
            torch.Tensor: The output from the encoder
        """
        skip_conn_res = []

        for i, layer in enumerate(self.conv_blocks):
            x = layer(x)
            if self.skip_conns:
                skip_conn_res.append(self.skip_conns[i](x))

        if self.qrnn_block:
            in_ = x.transpose(0, 1)
            out, _ = self.qrnn_block(in_)
            x = out.transpose(0, 1)

        x = self.final_conv_block(x)

        for res in skip_conn_res:
            x = self._fuse_skip(x, res)

        if self.final_norm_layer:
            x = self.final_norm_layer(x)

        return x

    def _fuse_skip(self, x: torch.Tensor, skip_res: torch.Tensor):
        """Fuse skip connection values with final result"""
        dfactor = skip_res.shape[1] // x.shape[1]
        if dfactor > 1:
            # Incase first dimension (feature) values dont match
            maxlen = x.shape[1] * dfactor
            skip_res = skip_res[:, :maxlen, :]
            bsz, slen, feats = skip_res.shape
            skip_re = skip_res.reshape(bsz, slen // dfactor, feats, dfactor)
            skip_res = torch.mean(skip_re, dim=3)
        return x + skip_res


class Classifier(torch.nn.Module):
    def __init__(
        self,
        in_channels=100,
        activation=torch.nn.LeakyReLU,
        lin_blocks=1,
        lin_neurons=100,
        out_neurons=28,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()

        for block_index in range(lin_blocks):
            self.blocks.extend(
                [
                    sb.nnet.linear.Linear(input_size=in_channels, n_neurons=lin_neurons),
                    activation(),
                    sb.nnet.normalization.BatchNorm1d(input_size=lin_neurons),
                ],
            )
        self.blocks.extend(
            [
                torch.nn.Flatten(),
                sb.nnet.linear.Linear(input_size=lin_neurons*lin_neurons, n_neurons=out_neurons),
                sb.nnet.activations.Softmax(apply_log=True),
            ],
        )

    def forward(self, x, *args, **kwargs):
        for layer in self.blocks:
            try:
                x = layer(x, *args, **kwargs)
            except TypeError:
                x = layer(x)
        return x
