import torch
import speechbrain as sb


class EnhanceResnet(torch.nn.Module):
    def __init__(
        self,
        n_fft=512,
        win_length=32,
        hop_length=16,
        sample_rate=16000,
        channel_counts=[128, 128, 256, 256, 512, 512],
        dense_count=2,
        dense_nodes=1024,
        activation=torch.nn.GELU,
        normalization=sb.nnet.normalization.BatchNorm2d,
        dropout=0.1,
        mask_weight=0.99,
    ):
        super().__init__()

        self.mask_weight = mask_weight

        # First, convert time-domain to log spectral magnitude inputs
        self.stft = sb.processing.features.STFT(
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
        self.istft = sb.processing.features.ISTFT(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            sample_rate=sample_rate,
        )

    def forward(self, x):

        # Generate features
        noisy_spec = self.stft(x)
        log_mag = self.extract_feats(noisy_spec)

        # Generate mask
        mask = self.DNN(self.CNN(log_mag))
        mask = mask.clamp(min=0, max=1).unsqueeze(1)

        # Apply mask
        masked_spec = self.mask_weight * mask * noisy_spec
        masked_spec += (1 - self.mask_weight) * noisy_spec

        # Extract feats for loss computation
        enhanced_features = self.extract_feats(masked_spec)

        # Return resynthesized waveform
        return self.istft(masked_spec), enhanced_features

    def extract_feats(self, x):
        """Takes the stft output and produces features for computation."""
        x = sb.processing.features.spectral_magnitude(x, power=0.5)
        return torch.log1p(x)


class ConvBlock(torch.nn.Module):
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
    def __init__(self, input_size):
        super().__init__()
        self.linear1 = sb.nnet.linear.Linear(
            input_size=input_size, n_neurons=input_size
        )
        self.linear2 = sb.nnet.linear.Linear(
            input_size=input_size, n_neurons=input_size
        )

    def forward(self, x):
        x = torch.mean(x, dim=(1, 2), keepdim=True)
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        return torch.sigmoid(x)
