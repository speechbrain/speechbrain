"""EEGConformer from https://doi.org/10.1109/TNSRE.2022.3230250.
Compact convolutional transformer with a shallow convolutional module used to learn low-level local features and a
self-attention module to learn global correlation within the local temporal features. It was proposed for motor imagery
and emotion decoding.

This code is a Speechbrain porting of the published code released by EEGConformer's authors, available at:
https://github.com/eeyhsong/EEG-Conformer.git.

Authors
 * Davide Borra, 2023
"""
import torch
import speechbrain as sb


class EEGConformer(torch.nn.Module):
    """EEGConformer.

    Arguments
    ---------
    input_shape: tuple
        The shape of the input.
    cnn_temporal_kernels: int
        Number of kernels in the 2d temporal convolution in the convolutional module.
    cnn_spatial_kernels: int
        Number of kernels in the 2d spatial convolution in the convolutional module.
    cnn_temporal_kernelsize: tuple
        Kernel size of the 2d temporal convolution in the convolutional module.
    cnn_poolsize: tuple
        Pool size in the convolutional module.
    cnn_poolstride: tuple
        Pool stride in the convolutional module.
    cnn_pool_type: string
        Pooling type in the convolutional module.
    cnn_dropout: float
        Dropout probability in the convolutional module.
    cnn_activation_type: str
        Activation function of hidden layers in the convolutional module.
    attn_depth: int
        Depth of the transformer module.
    attn_heads: int
        Number of heads in the transformer module.
    attn_dropout: float
        Dropout probability for the transformer module.
    dense_n_neurons: int
        Number of output neurons.

    Example
    -------
    #>>> inp_tensor = torch.rand([1, 200, 32, 1])
    #>>> model = EEGConformer(input_shape=inp_tensor.shape)
    #>>> output = model(inp_tensor)
    #>>> output.shape
    #torch.Size([1,4])
    """

    def __init__(
        self,
        input_shape=None,  # (1, T, C, 1)
        cnn_temporal_kernels=40,
        cnn_spatial_kernels=40,
        cnn_temporal_kernelsize=(13, 1),
        cnn_poolsize=(38, 1),
        cnn_poolstride=(17, 1),
        cnn_pool_type="avg",
        cnn_dropout=0.5,
        cnn_activation_type="elu",
        attn_depth=2,
        attn_heads=2,
        attn_dropout=0.5,
        dense_n_neurons=4,
    ):
        super().__init__()
        if input_shape is None:
            raise ValueError("Must specify input_shape")

        C = input_shape[-2]
        T = input_shape[-3]

        # EMBEDDING MODULE (CONVOLUTIONAL MODULE)
        self.emb_module = PatchEmbedding(
            cnn_temporal_kernels=cnn_temporal_kernels,
            cnn_spatial_kernels=cnn_spatial_kernels,
            cnn_temporal_kernelsize=cnn_temporal_kernelsize,
            cnn_spatial_kernelsize=(1, C),
            cnn_poolsize=cnn_poolsize,
            cnn_poolstride=cnn_poolstride,
            cnn_pool_type=cnn_pool_type,
            dropout=cnn_dropout,
            activation_type=cnn_activation_type,
        )
        # TRANSFORMER MODULE
        self.transformer_module = TransformerEncoder(
            attn_depth=attn_depth,
            emb_size=self.emb_module.emb_size,
            attn_heads=attn_heads,
            dropout=attn_dropout,
        )

        # Shape of intermediate feature maps
        out = self.emb_module(torch.ones((1, T, C, 1)))
        out = self.transformer_module(out)
        dense_input_size = self._num_flat_features(out)
        # DENSE MODULE
        self.dense_module = torch.nn.Sequential()
        self.dense_module.add_module(
            "flatten", torch.nn.Flatten(),
        )
        self.dense_module.add_module(
            "fc_out",
            sb.nnet.linear.Linear(
                input_size=dense_input_size, n_neurons=dense_n_neurons,
            ),
        )
        self.dense_module.add_module("act_out", torch.nn.LogSoftmax(dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the output of the model.

        Arguments
        ---------
        x : torch.Tensor (batch, time, EEG channel, channel)
            Input to convolve. 4d tensors are expected.
        """

        x = self.emb_module(x)  # (batch, time_, EEG channel, channel)
        x = self.transformer_module(x)
        x = self.dense_module(x)
        return x

    def _num_flat_features(self, x):
        """Returns the number of flattened features from a tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input feature map.
        """

        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PatchEmbedding(torch.nn.Module):
    """Class that defines the convolutional feature extractor based on a shallow CNN, to be used in EEGConformer.

    Arguments
    ---------
    cnn_temporal_kernels: int
        Number of kernels in the 2d temporal convolution in the convolutional module.
    cnn_spatial_kernels: int
        Number of kernels in the 2d spatial convolution in the convolutional module.
    cnn_temporal_kernelsize: tuple
        Kernel size of the 2d temporal convolution in the convolutional module.
    cnn_poolsize: tuple
        Pool size in the convolutional module.
    cnn_poolstride: tuple
        Pool stride in the convolutional module.
    cnn_pool_type: string
        Pooling type in the convolutional module.
    dropout: float
        Dropout probability in the convolutional module.
    activation_type: str
        Activation function of hidden layers in the convolutional module.
    """

    def __init__(
        self,
        cnn_temporal_kernels,
        cnn_spatial_kernels,
        cnn_temporal_kernelsize,
        cnn_spatial_kernelsize,
        cnn_poolsize,
        cnn_poolstride,
        cnn_pool_type,
        dropout,
        activation_type,
    ):
        super().__init__()
        if activation_type == "gelu":
            activation = torch.nn.GELU()
        elif activation_type == "elu":
            activation = torch.nn.ELU()
        elif activation_type == "relu":
            activation = torch.nn.ReLU()
        elif activation_type == "leaky_relu":
            activation = torch.nn.LeakyReLU()
        elif activation_type == "prelu":
            activation = torch.nn.PReLU()
        else:
            raise ValueError("Wrong hidden activation function")

        self.emb_size = cnn_spatial_kernels

        self.shallownet = torch.nn.Sequential(
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=cnn_temporal_kernels,
                kernel_size=cnn_temporal_kernelsize,  # (1, kernel_temp_conv),
                padding="valid",
                bias=True,
                swap=True,
            ),
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_temporal_kernels,
                out_channels=cnn_spatial_kernels,
                kernel_size=cnn_spatial_kernelsize,  # (C, 1),
                padding="valid",
                bias=True,
                swap=True,
            ),
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_spatial_kernels, momentum=0.01, affine=True,
            ),
            activation,
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=cnn_poolsize,  # (1, kernel_avg_pool),
                stride=cnn_poolstride,  # (1, stride_avg_pool),
                pool_axis=[1, 2],
            ),
            torch.nn.Dropout(dropout),
        )

        self.projection = sb.nnet.CNN.Conv2d(
            in_channels=cnn_spatial_kernels,
            out_channels=cnn_spatial_kernels,
            kernel_size=(1, 1),
            padding="valid",
            bias=True,
            swap=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the output of the convolutional feature extractor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, EEG channel, channel)
            Input to convolve. 4d tensors are expected.
        """
        x = self.shallownet(
            x
        )  # (batch, time_, 1, channel) was (batch, channel, EEG channel, time_)
        x = self.projection(x)  # (batch, time_, 1, channel)
        x = x.reshape(
            x.shape[0], x.shape[1] * x.shape[2], x.shape[-1]
        )  # (batch, time_, emb_size=channel*1) #ok
        return x


class MultiHeadAttention(torch.nn.Module):
    """Class that defines a multi-head attention mechanism.

    Arguments
    ---------
    emb_size: int
        Number of features from the embedding module.
    num_heads: int
        Number of heads in the transformer module.
    dropout: float
        Dropout probability for the transformer module.
    """

    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads

        self.keys = sb.nnet.linear.Linear(
            input_size=emb_size, n_neurons=emb_size, bias=True,
        )
        self.queries = sb.nnet.linear.Linear(
            input_size=emb_size, n_neurons=emb_size, bias=True,
        )
        self.values = sb.nnet.linear.Linear(
            input_size=emb_size, n_neurons=emb_size, bias=True,
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.projection = sb.nnet.linear.Linear(
            input_size=emb_size, n_neurons=emb_size, bias=True,
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        queries = self.queries(x)
        queries = queries.reshape(
            queries.shape[:2]
            + (self.num_heads, int(queries.shape[-1] / self.num_heads))
        )
        queries = queries.transpose(-2, -3)

        keys = self.keys(x)
        keys = keys.reshape(
            keys.shape[:2]
            + (self.num_heads, int(keys.shape[-1] / self.num_heads))
        )
        keys = keys.transpose(-2, -3)

        values = self.values(x)
        values = values.reshape(
            values.shape[:2]
            + (self.num_heads, int(values.shape[-1] / self.num_heads))
        )
        values = values.transpose(-2, -3)

        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = torch.nn.functional.softmax(energy / scaling, dim=-1)
        att = self.dropout(att)
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)

        out = out.transpose(1, 2)  # b h n d-> b n h d
        out = out.reshape(
            out.shape[:2] + (out.shape[2] * out.shape[3],)
        )  # b n h d -> b n h*d
        out = self.projection(out)
        return out


class ResidualAdd(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(torch.nn.Sequential):
    def __init__(self, emb_size, expansion, dropout):
        super().__init__(
            sb.nnet.linear.Linear(
                input_size=emb_size, n_neurons=expansion * emb_size, bias=True
            ),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            sb.nnet.linear.Linear(
                input_size=expansion * emb_size, n_neurons=emb_size, bias=True
            ),
        )


class TransformerEncoderBlock(torch.nn.Sequential):
    """Class that defines a single block of the transformer.

    Arguments
    ---------
    emb_size: int
        Number of features from the embedding module.
    attn_heads: int
        Number of heads in the transformer module.
    dropout: float
        Dropout probability for the transformer module.
    forward_expansion: int
    """

    def __init__(self, emb_size, attn_heads, dropout, forward_expansion=4):
        super().__init__(
            ResidualAdd(
                torch.nn.Sequential(
                    torch.nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, attn_heads, dropout),
                    torch.nn.Dropout(dropout),
                )
            ),
            ResidualAdd(
                torch.nn.Sequential(
                    torch.nn.LayerNorm(emb_size),
                    FeedForwardBlock(
                        emb_size, expansion=forward_expansion, dropout=dropout
                    ),
                    torch.nn.Dropout(dropout),
                )
            ),
        )


class TransformerEncoder(torch.nn.Sequential):
    """Class that defines the transformer module, to be used in EEGConformer.

    Arguments
    ---------
    attn_depth: int
        Depth of the transformer module.
    emb_size: int
        Number of features from the embedding module.
    attn_heads: int
        Number of heads in the transformer module.
    dropout: float
        Dropout probability for the transformer module.
    """

    def __init__(self, attn_depth, emb_size, attn_heads, dropout):
        super().__init__(
            *[
                TransformerEncoderBlock(emb_size, attn_heads, dropout)
                for _ in range(attn_depth)
            ]
        )
