"""A PyTorch implementation of the ECAPA2 architecture.

This module implements the ECAPA2 model and Sub-Center ArcFace Classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from speechbrain.dataio.dataio import length_to_mask


class SubCenterClassifier(nn.Module):
    """Sub-Center ArcFace Classifier.

    Arguments
    ---------
    input_size : int
        Expected size of the input dimension.
    out_neurons : int
        Number of output classes.
    k_subcenters : int
        Number of sub-centers per class.
    margin : float
        Angular margin penalty.
    scale : float
        Feature scale multiplier.

    Example
    -------
    >>> classify = SubCenterClassifier(input_size=192, out_neurons=10)
    >>> x = torch.randn(8, 192)
    >>> output = classify(x)
    >>> output.shape
    torch.Size([8, 10])
    """

    def __init__(
        self,
        input_size,
        out_neurons,
        k_subcenters=3,
        margin=0.2,
        scale=32.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.out_neurons = out_neurons
        self.k = k_subcenters
        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(
            torch.randn(out_neurons * k_subcenters, input_size)
        )
        nn.init.xavier_uniform_(self.weight)

        m = torch.tensor(margin)
        pi = torch.tensor(torch.pi)

        self.register_buffer("cos_m", torch.cos(m))
        self.register_buffer("sin_m", torch.sin(m))
        self.register_buffer("th", torch.cos(pi - m))
        self.register_buffer("mm", torch.sin(pi - m) * m)

    def forward(self, x, label=None):
        """Processes the input tensor x and returns output probabilities."""
        if x.dim() == 3:
            x = x.squeeze(1)

        embeddings = F.normalize(x)
        weights = F.normalize(self.weight)

        cosine_sim = F.linear(embeddings, weights)
        cosine_sim = cosine_sim.view(-1, self.out_neurons, self.k)
        cosine_sim, _ = torch.max(cosine_sim, dim=2)

        if label is None:
            return cosine_sim

        sine_sim = torch.sqrt(1.0 - torch.clamp(cosine_sim**2, 0, 1))

        phi = cosine_sim * self.cos_m - sine_sim * self.sin_m
        phi = torch.where(cosine_sim > self.th, phi, cosine_sim - self.mm)

        one_hot = torch.zeros_like(cosine_sim)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = one_hot * phi + (1.0 - one_hot) * cosine_sim
        output *= self.scale

        return output.unsqueeze(1)


class ECAPA2DownsampleBlock(nn.Module):
    """Downsampling block for ECAPA2."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """Processes the input tensor."""
        return self.bn(self.conv(x))


class ECAPA2SEBlock2d(nn.Module):
    """2D Squeeze-and-Excitation block."""

    def __init__(self, in_channels, se_channels=128):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_channels, se_channels)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(se_channels)
        self.fc2 = nn.Linear(se_channels, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Processes the input tensor."""
        b, c, _, _ = x.size()
        w = self.avg_pool(x).view([b, c])
        w = self.fc1(w)
        w = self.relu(w)
        w = self.bn1(w)
        w = self.fc2(w)
        w = self.sigmoid(w)
        w = w.view([b, c, 1, 1])
        return x * w


class ECAPA2ConvBlock(nn.Module):
    """Convolutional block with SE processing for ECAPA2 frontend."""

    def __init__(
        self, in_channels, out_channels, stride=(1, 1), se_channels=128
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=True,
        )
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.se = ECAPA2SEBlock2d(out_channels, se_channels=se_channels)

    def forward(self, x):
        """Processes the input tensor."""
        out = self.bn1(self.relu1(self.conv1(x)))
        out = self.bn2(self.relu2(self.conv2(out)))
        out = self.bn3(self.relu3(self.conv3(out)))
        out = self.se(out)
        return out


class ECAPA2SEBlock1d(nn.Module):
    """1D Squeeze-and-Excitation block."""

    def __init__(self, in_channels, se_channels=128):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, se_channels)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(se_channels)
        self.fc2 = nn.Linear(se_channels, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Processes the input tensor."""
        y = torch.mean(x, dim=2)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.bn1(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.unsqueeze(2)
        return x * y


class ECAPA2Res2NetConv1d(nn.Module):
    """Res2Net convolutional block for 1D features."""

    def __init__(
        self, channels, kernel_size=3, padding=2, dilation=2, se_channels=128
    ):
        super().__init__()
        self.n_groups = 8
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(channels)

        self.conv_filters = nn.ModuleList(
            [
                nn.Conv1d(
                    channels // self.n_groups,
                    channels // self.n_groups,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                    padding_mode="reflect",
                    bias=True,
                )
                for _ in range(self.n_groups - 1)
            ]
        )

        self.batch_norms = nn.ModuleList(
            [
                nn.BatchNorm1d(channels // self.n_groups)
                for _ in range(self.n_groups - 1)
            ]
        )

        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(channels)
        self.se = ECAPA2SEBlock1d(channels, se_channels=se_channels)

    def forward(self, x):
        """Processes the input tensor."""
        x = self.bn1(self.relu1(self.conv1(x)))
        chunks = torch.chunk(x, self.n_groups, dim=1)
        outputs = [chunks[0]]
        for i in range(1, self.n_groups):
            if i == 1:
                sp = self.conv_filters[i - 1](chunks[i])
            else:
                sp = self.conv_filters[i - 1](chunks[i] + outputs[i - 1])
            sp = F.relu(sp)
            sp = self.batch_norms[i - 1](sp)
            outputs.append(sp)
        x = torch.cat(outputs, dim=1)
        x = self.bn2(self.relu2(self.conv2(x)))
        x = self.se(x)
        return x


class ECAPA2TDNNBlock(nn.Module):
    """TDNN block for ECAPA2."""

    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=True,
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """Processes the input tensor."""
        return self.bn(self.relu(self.conv(x)))


class ECAPA2DenseBlock(nn.Module):
    """Dense convolutional block for ECAPA2."""

    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, bias=True
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        """Processes the input tensor."""
        return self.relu(self.conv(x))


class ECAPA2AttentiveStatPoolingBlock(nn.Module):
    """Attentive Statistics Pooling for ECAPA2."""

    def __init__(self, in_channels, attention_channels=128, out_channels=3072):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(
                in_channels * 3, attention_channels, kernel_size=1, bias=True
            ),
            nn.ReLU(),
            nn.BatchNorm1d(attention_channels),
            nn.Conv1d(
                attention_channels, in_channels, kernel_size=1, bias=True
            ),
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, lengths=None):
        """Processes the input tensor with optional lengths mask."""
        L = x.shape[-1]
        if lengths is None:
            mask = torch.ones(x.shape[0], 1, L, device=x.device)
        else:
            mask = length_to_mask(
                lengths * L, max_len=L, device=x.device
            ).unsqueeze(1)

        total = mask.sum(dim=2, keepdim=True).clamp(min=1.0)
        mean_x = (x * mask).sum(dim=2, keepdim=True) / total
        var_x = ((x - mean_x) * mask).pow(2).sum(dim=2, keepdim=True) / total
        std_x = torch.sqrt(var_x + 1e-05)

        mean_x_exp = mean_x.expand_as(x)
        std_x_exp = std_x.expand_as(x)
        concatenated = torch.cat([x, mean_x_exp, std_x_exp], dim=1)

        attention_logits = self.attention(concatenated)
        attention_logits = attention_logits.masked_fill(
            mask == 0, float("-inf")
        )
        attention_weights = F.softmax(attention_logits, dim=2)

        x_scaled = x * attention_weights
        mu = torch.sum(x_scaled, dim=2)
        sum_squared = torch.sum(x_scaled * x, dim=2)
        variance_estimate = torch.abs(sum_squared - torch.pow(mu, 2))
        sigma = torch.sqrt(variance_estimate + 1e-05)

        pool = torch.cat([mu, sigma], dim=1)
        return self.bn(pool)


class ECAPA2EmbeddingBlock(nn.Module):
    """Kiwano-style Embedding Block: BN -> Linear -> BN."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm_stats = nn.BatchNorm1d(in_channels)
        self.fc = nn.Linear(in_channels, out_channels)
        self.norm_embed = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """Processes the input tensor."""
        x = self.norm_stats(x)
        x = self.fc(x)
        x = self.norm_embed(x)
        return x


class ECAPA2Frontend(nn.Module):
    """2D Convolutional Frontend for ECAPA2."""

    def __init__(self, channels1, channels2, attention_channels=128):
        super().__init__()

        c1 = channels1
        c2 = channels2

        self.conv_1 = ECAPA2ConvBlock(1, c1, se_channels=attention_channels)
        self.conv_2 = ECAPA2ConvBlock(c1, c1, se_channels=attention_channels)
        self.conv_3 = ECAPA2ConvBlock(c1, c1, se_channels=attention_channels)

        self.downsample_1 = ECAPA2DownsampleBlock(c1, c1, stride=(2, 1))
        self.conv_4 = ECAPA2ConvBlock(
            c1, c1, stride=(2, 1), se_channels=attention_channels
        )
        self.conv_5 = ECAPA2ConvBlock(c1, c1, se_channels=attention_channels)
        self.conv_6 = ECAPA2ConvBlock(c1, c1, se_channels=attention_channels)
        self.conv_7 = ECAPA2ConvBlock(c1, c1, se_channels=attention_channels)

        self.downsample_2 = ECAPA2DownsampleBlock(c1, c1, stride=(2, 1))
        self.conv_8 = ECAPA2ConvBlock(
            c1, c1, stride=(2, 1), se_channels=attention_channels
        )
        self.conv_9 = ECAPA2ConvBlock(c1, c1, se_channels=attention_channels)
        self.conv_10 = ECAPA2ConvBlock(c1, c2, se_channels=attention_channels)
        self.conv_11 = ECAPA2ConvBlock(c2, c2, se_channels=attention_channels)

        self.downsample_3 = ECAPA2DownsampleBlock(c2, c2, stride=(2, 1))
        self.conv_12 = ECAPA2ConvBlock(
            c2, c2, stride=(2, 1), se_channels=attention_channels
        )
        self.conv_13 = ECAPA2ConvBlock(c2, c2, se_channels=attention_channels)
        self.conv_14 = ECAPA2ConvBlock(c2, c2, se_channels=attention_channels)
        self.conv_15 = ECAPA2ConvBlock(c2, c2, se_channels=attention_channels)

        self.downsample_4 = ECAPA2DownsampleBlock(c2, c2, stride=(2, 1))
        self.conv_16 = ECAPA2ConvBlock(
            c2, c2, stride=(2, 1), se_channels=attention_channels
        )
        self.conv_17 = ECAPA2ConvBlock(c2, c2, se_channels=attention_channels)
        self.conv_18 = ECAPA2ConvBlock(c2, c2, se_channels=attention_channels)
        self.conv_19 = ECAPA2ConvBlock(c2, c2, se_channels=attention_channels)
        self.conv_20 = ECAPA2ConvBlock(c2, c2, se_channels=attention_channels)

    def forward(self, x):
        """Processes the input tensor."""
        x = self.conv_1(x)
        x = self.conv_2(x) + x
        x = self.conv_3(x) + x

        x = self.conv_4(x) + self.downsample_1(x)
        x = self.conv_5(x) + x
        x = self.conv_6(x) + x
        x = self.conv_7(x) + x

        x = self.conv_8(x) + self.downsample_2(x)
        x = self.conv_9(x) + x
        x = self.conv_10(x)
        x = self.conv_11(x) + x

        x = self.conv_12(x) + self.downsample_3(x)
        x = self.conv_13(x) + x
        x = self.conv_14(x) + x
        x = self.conv_15(x) + x

        x = self.conv_16(x) + self.downsample_4(x)
        x = self.conv_17(x) + x
        x = self.conv_18(x) + x
        x = self.conv_19(x) + x
        x = self.conv_20(x) + x

        return x


class ECAPA2(nn.Module):
    """An implementation of the ECAPA2 speaker embedding model.

    Arguments
    ---------
    input_size : int
        Expected size of the input dimension.
    lin_neurons : int
        Number of neurons in the output embedding layer.
    channels : list of ints
        Output channels for internal network blocks.
    attention_channels: int
        The number of attention channels in SE and pooling layers.
    activation : torch class
        A class for constructing the activation layers.

    Example
    -------
    >>> input_feats = torch.rand([5, 120, 80])
    >>> compute_embedding = ECAPA2(input_size=80, lin_neurons=192)
    >>> outputs = compute_embedding(input_feats)
    >>> outputs.shape
    torch.Size([5, 1, 192])
    """

    def __init__(
        self,
        input_size,
        lin_neurons=192,
        channels=[164, 192, 1024, 1536],
        attention_channels=128,
        activation=torch.nn.ReLU,
        **kwargs,
    ):
        super().__init__()

        c1 = channels[0]
        c2 = channels[1]
        tdnn_c = channels[2]
        pool_c = channels[3]

        self.frontend = ECAPA2Frontend(
            c1, c2, attention_channels=attention_channels
        )

        final_freq_dim = input_size // 16
        if final_freq_dim == 0:
            raise ValueError(
                f"Input size {input_size} is too small for ECAPA2 (requires >= 16)."
            )

        flattened_dim = c2 * final_freq_dim

        self.tdnn_1 = ECAPA2TDNNBlock(flattened_dim, tdnn_c, kernel_size=1)
        self.tdnn_2 = ECAPA2Res2NetConv1d(
            tdnn_c, se_channels=attention_channels
        )
        self.dense_1 = ECAPA2DenseBlock(tdnn_c, pool_c)

        self.pooling_1 = ECAPA2AttentiveStatPoolingBlock(
            pool_c,
            attention_channels=attention_channels,
            out_channels=pool_c * 2,
        )
        self.dense_2 = ECAPA2EmbeddingBlock(pool_c * 2, lin_neurons)

    def forward(self, x, lengths=None):
        """Returns the embedding vector.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, channel).
        lengths : torch.Tensor
            Corresponding relative lengths of inputs.

        Returns
        -------
        x : torch.Tensor
            Embedding vector.
        """
        x = x.transpose(1, 2).unsqueeze(1)

        x = self.frontend(x)
        x = x.flatten(1, 2)

        x = self.tdnn_1(x)
        x = self.tdnn_2(x) + x
        x = self.dense_1(x)

        x = self.pooling_1(x, lengths=lengths)
        x = self.dense_2(x)

        return x.unsqueeze(1)
