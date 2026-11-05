"""A PyTorch implementation of the ECAPA2 architecture.

This module implements the ECAPA2 model and Sub-Center ArcFace Classifier.

Authors
 * Othman Istaiteh 2026

Acknowledgment:
The source code is derived from the Kiwano toolkit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from speechbrain.dataio.dataio import length_to_mask


class SubCenterClassifier(nn.Module):
    """Speaker classifier with an additive angular margin and sub-centers
    (Sub-center ArcFace).

    Each class is represented by ``k_subcenters`` learnable weight vectors
    instead of one. The score of a class is the cosine similarity between the
    normalised embedding and the closest of its sub-centers, so noisy or
    atypical utterances of a speaker can attach to a secondary sub-center
    without pulling the main one away.

    During training (when ``label`` is given), the additive angular margin of
    ArcFace (https://arxiv.org/abs/1801.07698) is applied to the target class:
    its angle ``theta`` becomes ``theta + margin``, and all the logits are
    multiplied by ``scale``. Without ``label``, the raw cosine similarities
    are returned, e.g. for inference.

    Reference: Deng et al., "Sub-center ArcFace: Boosting Face Recognition by
    Large-scale Noisy Web Faces", ECCV 2020.

    Arguments
    ---------
    input_size : int
        Size of the input embeddings.
    out_neurons : int
        Number of classes (speakers).
    k_subcenters : int
        Number of sub-centers per class.
    margin : float
        Additive angular margin, in radians, applied to the target class.
    scale : float
        Scale applied to the logits when the margin is applied.

    Example
    -------
    >>> classify = SubCenterClassifier(input_size=192, out_neurons=10)
    >>> x = torch.randn(8, 192)
    >>> output = classify(x)
    >>> output.shape
    torch.Size([8, 10])
    >>> labels = torch.randint(0, 10, (8,))
    >>> output = classify(x, labels)
    >>> output.shape
    torch.Size([8, 1, 10])
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
        """Computes the class scores of the embeddings.

        Arguments
        ---------
        x : torch.Tensor
            Embeddings, of shape [batch, input_size] or [batch, 1, input_size].
        label : torch.Tensor
            Class indices of shape [batch] or [batch, 1]. If given, the
            angular margin and the scale are applied.

        Returns
        -------
        torch.Tensor
            Cosine similarities of shape [batch, out_neurons] if ``label`` is
            None, otherwise scaled logits with margin of shape
            [batch, 1, out_neurons].
        """
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
    """Projection shortcut of the ECAPA2 frontend: a convolution followed by
    batch normalisation.

    It is used on the residual path of the frontend stages that downsample
    the frequency axis, so that the shortcut matches the shape of the output
    of the strided ``ECAPA2ConvBlock``.

    Arguments
    ---------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple
        Size of the convolution kernel.
    stride : int or tuple
        Stride of the convolution, as (frequency, time).

    Example
    -------
    >>> block = ECAPA2DownsampleBlock(16, 16, stride=(2, 1))
    >>> x = torch.rand(4, 16, 80, 100)
    >>> block(x).shape
    torch.Size([4, 16, 40, 100])
    """

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
    """Squeeze-and-Excitation block for 2D feature maps.

    The input is averaged over frequency and time, passed through a
    bottleneck (Linear, ReLU, BatchNorm, Linear, Sigmoid), and the resulting
    per-channel weights rescale the input channels.

    Arguments
    ---------
    in_channels : int
        Number of input (and output) channels.
    se_channels : int
        Number of channels of the bottleneck.

    Example
    -------
    >>> block = ECAPA2SEBlock2d(16, se_channels=8)
    >>> x = torch.rand(4, 16, 80, 100)
    >>> block(x).shape
    torch.Size([4, 16, 80, 100])
    """

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
    """Convolutional block of the ECAPA2 frontend.

    Three 3x3 convolutions, each followed by ReLU and batch normalisation,
    then a Squeeze-and-Excitation block. The first convolution uses
    ``stride``, which downsamples the frequency axis when set to (2, 1).

    Arguments
    ---------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : tuple
        Stride of the first convolution, as (frequency, time).
    se_channels : int
        Number of channels of the Squeeze-and-Excitation bottleneck.

    Example
    -------
    >>> block = ECAPA2ConvBlock(1, 16, stride=(2, 1), se_channels=8)
    >>> x = torch.rand(4, 1, 80, 100)
    >>> block(x).shape
    torch.Size([4, 16, 40, 100])
    """

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
    """Squeeze-and-Excitation block for 1D feature maps.

    The input is averaged over time, passed through a bottleneck (Linear,
    ReLU, BatchNorm, Linear, Sigmoid), and the resulting per-channel weights
    rescale the input channels.

    Arguments
    ---------
    in_channels : int
        Number of input (and output) channels.
    se_channels : int
        Number of channels of the bottleneck.

    Example
    -------
    >>> block = ECAPA2SEBlock1d(64, se_channels=16)
    >>> x = torch.rand(4, 64, 100)
    >>> block(x).shape
    torch.Size([4, 64, 100])
    """

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
    """Res2Net block with dilated convolutions for 1D features.

    A 1x1 convolution is followed by a split of the channels into 8 groups.
    The first group is kept as is, and each following group goes through a
    dilated convolution after adding the output of the previous group, which
    gives the block several receptive fields. The groups are concatenated,
    passed through a second 1x1 convolution and a Squeeze-and-Excitation
    block. Every convolution is followed by ReLU and batch normalisation.

    Arguments
    ---------
    channels : int
        Number of input (and output) channels. Must be divisible by 8.
    kernel_size : int
        Kernel size of the dilated convolutions.
    padding : int
        Padding of the dilated convolutions. Use
        ``dilation * (kernel_size - 1) // 2`` to keep the number of frames.
    dilation : int
        Dilation of the dilated convolutions.
    se_channels : int
        Number of channels of the Squeeze-and-Excitation bottleneck.

    Example
    -------
    >>> block = ECAPA2Res2NetConv1d(64, se_channels=16)
    >>> x = torch.rand(4, 64, 100)
    >>> block(x).shape
    torch.Size([4, 64, 100])
    """

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
    """Time-delay (1D convolution) block: Conv1d, ReLU and batch
    normalisation.

    The convolution has no padding, so a kernel size larger than 1 reduces
    the number of frames.

    Arguments
    ---------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Kernel size of the convolution.
    dilation : int
        Dilation of the convolution.

    Example
    -------
    >>> block = ECAPA2TDNNBlock(80, 64)
    >>> x = torch.rand(4, 80, 100)
    >>> block(x).shape
    torch.Size([4, 64, 100])
    """

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
    """Frame-level projection block: Conv1d followed by ReLU, without
    normalisation.

    In ECAPA2 it maps the output of the Res2Net block to the dimension used
    by the attentive statistics pooling.

    Arguments
    ---------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Kernel size of the convolution.

    Example
    -------
    >>> block = ECAPA2DenseBlock(64, 128)
    >>> x = torch.rand(4, 64, 100)
    >>> block(x).shape
    torch.Size([4, 128, 100])
    """

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
    """Channel-wise attentive statistics pooling, followed by batch
    normalisation.

    Each frame is concatenated with the global mean and standard deviation of
    the utterance. A small network computes, for every channel, attention
    weights over time, which give a weighted mean and a weighted standard
    deviation. Padded frames are masked using ``lengths``. The output is the
    concatenation of the weighted mean and standard deviation.

    Arguments
    ---------
    in_channels : int
        Number of input channels.
    attention_channels : int
        Number of channels of the attention network.
    out_channels : int
        Size of the output, which must be ``2 * in_channels``.

    Example
    -------
    >>> block = ECAPA2AttentiveStatPoolingBlock(
    ...     64, attention_channels=16, out_channels=128
    ... )
    >>> x = torch.rand(4, 64, 100)
    >>> block(x, lengths=torch.tensor([1.0, 0.8, 0.6, 0.5])).shape
    torch.Size([4, 128])
    """

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
        """Pools the frames into one vector per utterance.

        Arguments
        ---------
        x : torch.Tensor
            Frame-level features of shape [batch, channels, time].
        lengths : torch.Tensor
            Relative lengths of the utterances. If None, all the frames are
            used.

        Returns
        -------
        torch.Tensor
            Pooled statistics of shape [batch, 2 * channels].
        """
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
    """Embedding layer of ECAPA2: batch normalisation, linear projection and
    batch normalisation.

    Arguments
    ---------
    in_channels : int
        Size of the pooled statistics.
    out_channels : int
        Size of the speaker embedding.

    Example
    -------
    >>> block = ECAPA2EmbeddingBlock(128, 32)
    >>> x = torch.rand(4, 128)
    >>> block(x).shape
    torch.Size([4, 32])
    """

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
    """2D convolutional frontend of ECAPA2.

    A stack of 20 residual ``ECAPA2ConvBlock`` in five stages. The last four
    stages start with a block of stride (2, 1), with a projection shortcut,
    so the frequency axis is divided by 16 while the time axis is kept.

    Arguments
    ---------
    channels1 : int
        Number of channels of the first stages.
    channels2 : int
        Number of channels of the last stages.
    attention_channels : int
        Number of channels of the Squeeze-and-Excitation bottlenecks.

    Example
    -------
    >>> frontend = ECAPA2Frontend(8, 16, attention_channels=8)
    >>> x = torch.rand(2, 1, 80, 50)
    >>> frontend(x).shape
    torch.Size([2, 16, 5, 50])
    """

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
