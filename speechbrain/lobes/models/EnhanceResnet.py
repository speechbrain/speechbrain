"""Wide ResNet for Speech Enhancement.

Author
 * Peter Plantinga 2022
"""
import torch
import speechbrain as sb
from speechbrain.processing.features import STFT, ISTFT, spectral_magnitude


class EnhanceResnet(torch.nn.Module):
    """Model for enhancement based on Wide ResNet.

    Full model description at: https://arxiv.org/pdf/2112.06068.pdf

    Arguments
    ---------
    n_fft : int
        Number of points in the fourier transform, see ``speechbrain.processing.features.STFT``
    win_length : int
        Length of stft window in ms, see ``speechbrain.processing.features.STFT``
    hop_length : int
        Time between windows in ms, see ``speechbrain.processing.features.STFT``
    sample_rate : int
        Number of samples per second of input audio.
    channel_counts : list of ints
        Number of output channels in each CNN block. Determines number of blocks.
    dense_count : int
        Number of dense layers.
    dense_nodes : int
        Number of nodes in the dense layers.
    activation : function
        Function to apply before convolution layers.
    normalization : class
        Name of class to use for constructing norm layers.
    dropout : float
        Portion of layer outputs to drop during training (between 0 and 1).
    mask_weight : float
        Amount of weight to give mask. 0 - no masking, 1 - full masking.

    Example
    -------
    >>> inputs = torch.rand([10, 16000])
    >>> model = EnhanceResnet()
    >>> outputs, feats = model(inputs)
    >>> outputs.shape
    torch.Size([10, 15872])
    >>> feats.shape
    torch.Size([10, 63, 257])
    """

    def __init__(
        self,
        n_fft=512,
        win_length=32,
        hop_length=16,
        sample_rate=16000,
        channel_counts=[128, 128, 256, 256, 512, 512],
        dense_count=2,
        dense_nodes=1024,
        activation=torch.nn.GELU(),
        normalization=sb.nnet.normalization.BatchNorm2d,
        dropout=0.1,
        mask_weight=0.99,
    ):
        super().__init__()

        self.mask_weight = mask_weight

        # First, convert time-domain to log spectral magnitude inputs
        self.stft = STFT(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            sample_rate=sample_rate,
        )

        # CNN takes log spectral mag inputs
        self.CNN = sb.nnet.containers.Sequential(
            input_shape=[None, None, n_fft // 2 + 1]
        )
        for channel_count in channel_counts:
            self.CNN.append(
                ConvBlock,
                channels=channel_count,
                activation=activation,
                normalization=normalization,
                dropout=dropout,
            )

        # Fully connected layers
        self.DNN = sb.nnet.containers.Sequential(
            input_shape=self.CNN.get_output_shape()
        )
        for _ in range(dense_count):
            self.DNN.append(
                sb.nnet.linear.Linear, n_neurons=dense_nodes, combine_dims=True,
            )
            self.DNN.append(activation)
            self.DNN.append(sb.nnet.normalization.LayerNorm)
            self.DNN.append(torch.nn.Dropout(p=dropout))

        # Output layer produces real mask that is applied to complex inputs
        self.DNN.append(sb.nnet.linear.Linear, n_neurons=n_fft // 2 + 1)

        # Convert back to time domain
        self.istft = ISTFT(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            sample_rate=sample_rate,
        )

    def forward(self, x):
        """Processes the input tensor and outputs the enhanced speech."""

        # Generate features
        noisy_spec = self.stft(x)
        log_mag = self.extract_feats(noisy_spec)

        # Generate mask
        mask = self.DNN(self.CNN(log_mag))
        mask = mask.clamp(min=0, max=1).unsqueeze(-1)

        # Apply mask
        masked_spec = self.mask_weight * mask * noisy_spec
        masked_spec += (1 - self.mask_weight) * noisy_spec

        # Extract feats for loss computation
        enhanced_features = self.extract_feats(masked_spec)

        # Return resynthesized waveform
        return self.istft(masked_spec), enhanced_features

    def extract_feats(self, x):
        """Takes the stft output and produces features for computation."""
        return torch.log1p(spectral_magnitude(x, power=0.5))


class ConvBlock(torch.nn.Module):
    """Convolution block, including squeeze-and-excitation.

    Arguments
    ---------
    input_shape : tuple of ints
        The expected size of the inputs.
    channels : int
        Number of output channels.
    activation : function
        Function applied before each block.
    normalization : class
        Name of a class to use for constructing norm layers.
    dropout : float
        Portion of block outputs to drop during training.

    Example
    -------
    >>> inputs = torch.rand([10, 20, 30, 128])
    >>> block = ConvBlock(input_shape=inputs.shape, channels=256)
    >>> outputs = block(inputs)
    >>> outputs.shape
    torch.Size([10, 20, 15, 256])
    """

    def __init__(
        self,
        input_shape,
        channels,
        activation=torch.nn.GELU(),
        normalization=sb.nnet.normalization.LayerNorm,
        dropout=0.1,
    ):
        super().__init__()
        self.activation = activation
        self.downsample = sb.nnet.CNN.Conv2d(
            input_shape=input_shape,
            out_channels=channels,
            kernel_size=3,
            stride=(2, 1),
        )
        self.conv1 = sb.nnet.CNN.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3
        )
        self.norm1 = normalization(input_size=channels)
        self.conv2 = sb.nnet.CNN.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3,
        )
        self.norm2 = normalization(input_size=channels)
        self.dropout = sb.nnet.dropout.Dropout2d(drop_rate=dropout)

        self.se_block = SEblock(input_size=channels)

    def forward(self, x):
        """Processes the input tensor with a convolutional block."""
        x = self.downsample(x)
        residual = self.activation(x)
        residual = self.norm1(residual)
        residual = self.dropout(residual)
        residual = self.conv1(residual)
        residual = self.activation(residual)
        residual = self.norm2(residual)
        residual = self.dropout(residual)
        residual = self.conv2(residual)
        residual *= self.se_block(residual)
        return x + residual


class SEblock(torch.nn.Module):
    """Squeeze-and-excitation block.

    Defined: https://arxiv.org/abs/1709.01507

    Arguments
    ---------
    input_size : tuple of ints
        Expected size of the input tensor

    Example
    -------
    >>> inputs = torch.rand([10, 20, 30, 256])
    >>> se_block = SEblock(input_size=inputs.shape[-1])
    >>> outputs = se_block(inputs)
    >>> outputs.shape
    torch.Size([10, 1, 1, 256])
    """

    def __init__(self, input_size):
        super().__init__()
        self.linear1 = sb.nnet.linear.Linear(
            input_size=input_size, n_neurons=input_size
        )
        self.linear2 = sb.nnet.linear.Linear(
            input_size=input_size, n_neurons=input_size
        )

    def forward(self, x):
        """Processes the input tensor with a squeeze-and-excite block."""
        # torch.mean causes weird inplace error
        # x = torch.mean(x, dim=(1, 2), keepdim=True)
        count = x.size(1) * x.size(2)
        x = torch.sum(x, dim=(1, 2), keepdim=True) / count
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        return torch.sigmoid(x)
