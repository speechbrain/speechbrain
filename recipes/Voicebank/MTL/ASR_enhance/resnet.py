import torch
import speechbrain as sb


@torch.jit.script
def mish(x):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    """
    return x * torch.tanh(torch.nn.functional.softplus(x))


class Mish(torch.nn.Module):
    def forward(self, x):
        return mish(x)


class EnhanceResnet(sb.nnet.containers.Sequential):
    def __init__(
        self,
        input_shape,
        output_size,
        channel_counts=[128, 128, 256, 256],
        time_downsample=False,
        dense_count=2,
        dense_nodes=512,
        activation=Mish(),
        normalization=sb.nnet.normalization.LayerNorm,
        dropout=0.1,
        use_se=True,
    ):
        super().__init__(input_shape=input_shape)

        if not isinstance(time_downsample, list):
            time_downsample = [time_downsample for _ in channel_counts]
        self.append(sb.nnet.containers.Sequential, layer_name="CNN")
        for channel_count, time_down in zip(channel_counts, time_downsample):
            self.CNN.append(
                ConvBlock,
                channels=channel_count,
                time_downsample=time_down,
                activation=activation,
                normalization=normalization,
                dropout=dropout,
                use_se=use_se,
            )
            if time_down:
                self.CNN.append(
                    sb.nnet.pooling.Pooling1d(
                        pool_type="max",
                        input_dims=4,
                        kernel_size=2,
                        pool_axis=1,
                    )
                )

        for _ in range(dense_count):
            self.append(
                sb.nnet.linear.Linear, n_neurons=dense_nodes, combine_dims=True,
            )
            self.append(activation)
            self.append(sb.nnet.normalization.LayerNorm)
            self.append(torch.nn.Dropout(p=dropout))

        self.append(sb.nnet.linear.Linear, n_neurons=output_size)


class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        input_shape,
        channels,
        time_downsample=False,
        activation=torch.nn.GELU(),
        normalization=sb.nnet.normalization.LayerNorm,
        dropout=0.1,
        use_se=True,
    ):
        super().__init__()
        self.activation = activation
        self.time_downsample = time_downsample
        self.use_se = use_se
        self.downsample = sb.nnet.CNN.Conv2d(
            input_shape=input_shape,
            out_channels=channels,
            kernel_size=3,
            stride=(2, 1),
            # stride=(2, 2 if time_downsample else 1),
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

        if use_se:
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
        if self.use_se:
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
