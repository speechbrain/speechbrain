import torch
from torch import nn
from speechbrain.nnet import CNN
from speechbrain.nnet.normalization import BatchNorm1d
from speechbrain.nnet.activations import Softmax
from speechbrain.nnet.containers import Sequential


class ResidualConvBlock(nn.Module):
    """
    A one-dimensional convolutional residual block

    Arguments
    ---------
    out_channels: int
        the number of output channels for convolutional layers
    in_channels: int
        the number of input channels for convolutional layers
    kernel_size: int
        the kernel size for convolutional layers
    stride: int
        the stride for convolutional layers
    dilation: int
        the dilation for convolutional layers
    """

    def __init__(
        self,
        out_channels,
        in_channels,
        kernel_size,
        stride=1,
        dilation=1,
        padding="same",
        input_shape=None,
    ):
        super().__init__()
        self.batch_norm = BatchNorm1d(input_size=in_channels)
        self.relu = nn.ReLU()
        self.conv = CNN.Conv1d(
            out_channels=out_channels,
            kernel_size=kernel_size,
            in_channels=in_channels,
            stride=stride,
            dilation=dilation,
            input_shape=input_shape,
            padding=padding,
        )
        self.residual_conv = CNN.Conv1d(
            out_channels=out_channels,
            kernel_size=kernel_size,
            in_channels=in_channels,
            stride=stride,
            dilation=dilation,
            input_shape=input_shape,
            padding=padding,
        )

    def forward(self, inputs):
        """
        Computes the forward pass

        Arguments
        ---------
        inputs: torch.Tensor
            the input tensor
        """
        x = inputs
        x = self.batch_norm(x)
        x = self.relu(x)
        x_straight = self.conv(x)
        x_residual = self.residual_conv(inputs)
        x = x_straight + x_residual
        return x


class Encoder(nn.Module):
    def __init__(self, conv_layers, input_shape=None):
        super().__init__()
        self.conv_sequence = Sequential(input_shape=input_shape, *conv_layers)
        self.relu = nn.ReLU()
        num_channels = self._get_output_channels()
        self.batch_norm = BatchNorm1d(input_size=num_channels)
        self.softmax = Softmax(apply_log=False)

    def _get_output_channels(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 10, self.conv_sequence.input_shape[-1])
            dummy_output = self.conv_sequence(dummy_input)
        return dummy_output.shape[-1]

    def forward(self, inputs):
        x = inputs
        x = self.conv_sequence(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.softmax(x)
        return x, None
