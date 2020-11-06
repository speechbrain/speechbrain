import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from performer_pytorch import Performer

from speechbrain.nnet.linear import Linear
from speechbrain.lobes.models.transformer.Transformer import TransformerEncoder
from speechbrain.lobes.models.transformer.Transformer import PositionalEncoding
from speechbrain.lobes.models.transformer.TransformerSE import CNNTransformerSE
import speechbrain.nnet.RNN as SBRNN


EPS = 1e-8


class GlobalLayerNorm(nn.Module):
    """
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size) –
          input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True,
          this module has learnable per-element affine parameters
          initialized to ones (for weights) and zeros (for biases).
    """

    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = (
                    self.weight * (x - mean) / torch.sqrt(var + self.eps)
                    + self.bias
                )
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = (
                    self.weight * (x - mean) / torch.sqrt(var + self.eps)
                    + self.bias
                )
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    """
       Calculate Cumulative Layer Normalization
       dim: you want to norm dim
       elementwise_affine: learnable per-element affine parameters
    """

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine, eps=1e-8
        )

    def forward(self, x):
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            # N x K x S x C == only channel norm
            x = super().forward(x)
            # N x C x K x S
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim, shape):
    if norm == "gln":
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == "cln":
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == "ln":
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)


class Encoder(nn.Module):
    """
       Conv-Tasnet Encoder part
       kernel_size: the length of filters
       out_channels: the number of filters
    """

    def __init__(self, kernel_size=2, out_channels=64, in_channels=1):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )
        self.in_channels = in_channels

    def forward(self, x, init_params=True):
        """
          Input:
              x: [B, T], B is batch size, T is times
          Returns:
              x: [B, C, T_out]
              T_out is the number of time steps
        """
        # B x T -> B x 1 x T
        if self.in_channels == 1:
            x = torch.unsqueeze(x, dim=1)
        # B x 1 x T -> B x C x T_out
        x = self.conv1d(x)
        x = F.relu(x)

        return x


class TransformerBasedEncoder(nn.Module):
    """
       The transformer encoder we discussed on slack
       kernel_size: the length of filters
       out_channels: the number of filters
    """

    def __init__(
        self,
        kernel_size=2,
        out_channels=64,
        in_channels=1,
        d_ffn=1024,
        nhead=8,
        num_layers=1,
        version=1,
        use_positional_encoding=False,
    ):
        super(TransformerBasedEncoder, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.version = version
        if version == 1:
            self.conv1d = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                groups=1,
                bias=False,
            )
            self.pooling = Linear(1)
        elif version == 2:
            self.conv1d = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=kernel_size // 2,
                groups=1,
                bias=False,
            )
        else:
            raise ValueError("No such version!")

        self.transformer = SBTransformerBlock(
            num_layers=num_layers,
            nhead=nhead,
            dropout=0,
            d_ffn=d_ffn,
            use_positional_encoding=use_positional_encoding,
        )

    def forward(self, x, init_params=True):
        """
          Input:
              x: [B, T], B is batch size, T is times
          Returns:
              x: [B, C, T_out]
              T_out is the number of time steps
        """
        # B x T -> B x 1 x T
        batchsize = x.size(0)
        if self.in_channels == 1:
            x = torch.unsqueeze(x, dim=1)

        x = self.conv1d(x)
        if self.version == 1:
            x, _ = self._Segmentation(x, self.kernel_size)
            x = x.permute(0, 3, 2, 1)
            x = x.reshape(-1, x.size(-2), x.size(-1))
            x = self.transformer(x, init_params=init_params)
            x = x.reshape(batchsize, -1, x.size(-2), x.size(-1))
            x = x.permute(0, 1, 3, 2)
            x = self.pooling(x, init_params=init_params)
            x = x.squeeze(-1).permute(0, 2, 1)
        else:
            x, gap = self._Segmentation(x, self.kernel_size)
            x = x.permute(0, 3, 2, 1)
            x = x.reshape(-1, x.size(-2), x.size(-1))
            x = self.transformer(x, init_params=init_params)
            x = x.reshape(batchsize, -1, x.size(-2), x.size(-1))
            x = x.permute(0, 3, 2, 1)
            x = self._over_add(x, gap)

        x = F.relu(x)

        return x

    def _padding(self, input, K):
        """
           padding the audio times
           K: chunks of length
           P: hop size
           input: [B, N, L]
        """
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        """
           the segmentation stage splits
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, S]
        """
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = (
            torch.cat([input1, input2], dim=3).view(B, N, -1, K).transpose(2, 3)
        )

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        """
           Merge sequence
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        """
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input


class Decoder(nn.ConvTranspose1d):
    """
        Decoder of the TasNet
        This module can be seen as the gradient of Conv1d with respect to its input.
        It is also known as a fractionally-strided convolution
        or a deconvolution (although it is not an actual deconvolution operation).
    """

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x, init_params=True):
        """
        x: [B, N, L]
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError(
                "{} accept 3/4D tensor as input".format(self.__name__)
            )
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


class Decoder_twolayer(nn.Module):
    """
        Two layer Decoder
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, bias=False
    ):
        super(Decoder_twolayer, self).__init__()

        self.convt1 = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )
        self.convt2 = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            bias=bias,
        )

    def forward(self, x, init_params=True):
        """
        x: [B, N, L]
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError(
                "{} accept 3/4D tensor as input".format(self.__name__)
            )
        x = x if x.dim() == 3 else torch.unsqueeze(x, 1)

        x = self.convt1(x)
        x = self.convt2(x)

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


class Dual_RNN_Block(nn.Module):
    """
       Implementation of the intra-RNN and the inter-RNN
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs
                     of each LSTM layer except the last layer,
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
    """

    def __init__(
        self,
        out_channels,
        hidden_channels,
        rnn_type="LSTM",
        norm="ln",
        dropout=0,
        bidirectional=False,
        num_spks=2,
    ):
        super(Dual_RNN_Block, self).__init__()
        # RNN model
        self.intra_rnn = getattr(nn, rnn_type)(
            out_channels,
            hidden_channels,
            1,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.inter_rnn = getattr(nn, rnn_type)(
            out_channels,
            hidden_channels,
            1,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        # Norm
        self.intra_norm = select_norm(norm, out_channels, 4)
        self.inter_norm = select_norm(norm, out_channels, 4)
        # Linear
        self.intra_linear = nn.Linear(
            hidden_channels * 2 if bidirectional else hidden_channels,
            out_channels,
        )
        self.inter_linear = nn.Linear(
            hidden_channels * 2 if bidirectional else hidden_channels,
            out_channels,
        )

    def forward(self, x):
        """
           x: [B=batch size, N=channels, K=chunk size, S=number of chunks]
           out: [Spks, B, N, K, S]
        """

        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra_rnn = x.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)
        # [BS, K, H]
        intra_rnn, _ = self.intra_rnn(intra_rnn)
        # [BS, K, N]
        intra_rnn = self.intra_linear(
            intra_rnn.contiguous().view(B * S * K, -1)
        ).view(B * S, K, -1)
        # [B, S, K, N]
        intra_rnn = intra_rnn.view(B, S, K, N)
        # [B, N, K, S]
        intra_rnn = intra_rnn.permute(0, 3, 2, 1).contiguous()
        intra_rnn = self.intra_norm(intra_rnn)

        # [B, N, K, S]
        intra_rnn = intra_rnn + x

        # inter RNN
        # [BK, S, N]
        inter_rnn = intra_rnn.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)
        # [BK, S, H]
        inter_rnn, _ = self.inter_rnn(inter_rnn)
        # [BK, S, N]
        inter_rnn = self.inter_linear(
            inter_rnn.contiguous().view(B * S * K, -1)
        ).view(B * K, S, -1)
        # [B, K, S, N]
        inter_rnn = inter_rnn.view(B, K, S, N)
        # [B, N, K, S]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous()
        inter_rnn = self.inter_norm(inter_rnn)
        # [B, N, K, S]
        out = inter_rnn + intra_rnn

        return out


class Dual_Path_RNN(nn.Module):
    """
       Implementation of the Dual-Path-RNN model
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs
                     of each LSTM layer except the last layer,
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
            num_layers: number of Dual-Path-Block
            K: the length of chunk
            num_spks: the number of speakers
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        rnn_type="LSTM",
        norm="ln",
        dropout=0,
        bidirectional=False,
        num_layers=4,
        K=200,
        num_spks=2,
    ):
        super(Dual_Path_RNN, self).__init__()
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.dual_rnn = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_rnn.append(
                Dual_RNN_Block(
                    out_channels,
                    hidden_channels,
                    rnn_type=rnn_type,
                    norm=norm,
                    dropout=dropout,
                    bidirectional=bidirectional,
                )
            )

        self.conv2d = nn.Conv2d(
            out_channels, out_channels * num_spks, kernel_size=1
        )
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        # gated output layer
        self.output = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid()
        )

    def forward(self, x, init_params=True):
        """
           x: [B, N, L]
        """
        # [B, N, L]
        x = self.norm(x)
        # [B, N, L]
        x = self.conv1d(x)
        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)
        # [B, N*spks, K, S]
        for i in range(self.num_layers):
            x = self.dual_rnn[i](x)
        x = self.prelu(x)
        x = self.conv2d(x)
        # [B*spks, N, K, S]
        B, _, K, S = x.shape
        x = x.view(B * self.num_spks, -1, K, S)
        # [B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x) * self.output_gate(x)
        # [spks*B, N, L]
        x = self.end_conv1x1(x)
        # [B*spks, N, L] -> [B, spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)
        # [spks, B, N, L]
        x = x.transpose(0, 1)

        return x

    def _padding(self, input, K):
        """
           padding the audio times
           K: chunks of length
           P: hop size
           input: [B, N, L]
        """
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        """
           the segmentation stage splits
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, S]
        """
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = (
            torch.cat([input1, input2], dim=3).view(B, N, -1, K).transpose(2, 3)
        )

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        """
           Merge sequence
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        """
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input


class IdentityBlock:
    def _init__(self, **kwargs):
        pass

    def __call__(self, x, init_params=True):
        return x


class FastTransformerBlock(nn.Module):
    """
    This block is used to implement fast transformer models with efficient attention.
    The implementations are taken from https://fast-transformers.github.io/
    """

    def __init__(
        self,
        attention_type,
        out_channels,
        num_layers=6,
        nhead=8,
        d_ffn=1024,
        dropout=0,
        activation="relu",
        reformer_bucket_size=32,
    ):
        super(FastTransformerBlock, self).__init__()
        from fast_transformers.builders import TransformerEncoderBuilder

        # cem: there is another way of building the transformer.. I think this one is NOT the most flexible way, but it's easier.

        builder = TransformerEncoderBuilder.from_kwargs(
            attention_type=attention_type,
            n_layers=num_layers,
            n_heads=nhead,
            feed_forward_dimensions=d_ffn,
            query_dimensions=out_channels
            // nhead,  # @ Mirko: do these make sense?
            value_dimensions=out_channels // nhead,
            dropout=dropout,
            attention_dropout=dropout,
            chunk_size=reformer_bucket_size,
        )
        self.mdl = builder.get()

        self.attention_type = attention_type
        self.reformer_bucket_size = reformer_bucket_size

    def forward(self, x, init_params=False):
        if self.attention_type == "reformer":

            # pad zeros at the end
            pad_size = (self.reformer_bucket_size * 2) - (
                x.shape[1] % (self.reformer_bucket_size * 2)
            )
            device = x.device
            x_padded = torch.cat(
                [x, torch.zeros(x.size(0), pad_size, x.size(-1)).to(device)],
                dim=1,
            )

            # apply the model
            x_padded = self.mdl(x_padded)

            # get rid of zeros at the end
            return x_padded[:, :-pad_size, :]
        else:
            return self.mdl(x)


class PyTorchPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PyTorchPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class PytorchTransformerBlock(nn.Module):
    def __init__(
        self,
        out_channels,
        num_layers=6,
        nhead=8,
        d_ffn=2048,
        dropout=0.1,
        activation="relu",
        use_positional_encoding=True,
    ):
        super(PytorchTransformerBlock, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_channels,
            nhead=nhead,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation=activation,
        )
        # cem :this encoder thing has a normalization component. we should look at that probably also.
        self.mdl = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if use_positional_encoding:
            self.pos_encoder = PyTorchPositionalEncoding(out_channels)
        else:
            self.pos_encoder = None

    def forward(self, x, init_params=False):
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        return self.mdl(x)


class PerformerBlock(nn.Module):
    def __init__(
        self,
        out_channels,
        num_layers=6,
        nhead=8,
        ff_mult=4,
        use_positional_encoding=True,
    ):
        super(PerformerBlock, self).__init__()

        self.encoder_layer = Performer(
            dim=out_channels, heads=nhead, depth=num_layers, ff_mult=ff_mult
        )

        if use_positional_encoding:
            self.pos_enc = PositionalEncoding()

    def forward(self, x, init_params=False):
        if self.pos_enc is not None:
            pos_enc = self.pos_enc(x, init_params=init_params)
            x = x + pos_enc
        return self.encoder_layer(x)


class SBTransformerBlock(nn.Module):
    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn=2048,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation="relu",
        return_attention=False,
        num_modules=1,
        use_group_comm=False,
        use_positional_encoding=False,
        norm_before=False,
    ):
        super(SBTransformerBlock, self).__init__()
        self.use_positional_encoding = use_positional_encoding

        if activation == "relu":
            activation = nn.ReLU
        elif activation == "gelu":
            activation = nn.GELU
        else:
            raise ValueError("unknown activation")

        self.mdl = TransformerEncoder(
            num_layers,
            nhead,
            d_ffn,
            kdim,
            vdim,
            dropout,
            activation,
            return_attention,
            num_modules,
            use_group_comm,
            norm_before,
        )

        if use_positional_encoding:
            self.pos_enc = PositionalEncoding()

    def forward(self, x, init_params=False):
        if self.use_positional_encoding:
            pos_enc = self.pos_enc(x, init_params=init_params)
            return self.mdl(x + pos_enc, init_params=init_params)
        else:
            return self.mdl(x, init_params=init_params)


class SBRNNBlock(nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_layers,
        rnn_type="LSTM",
        dropout=0,
        bidirectional=True,
    ):
        super(SBRNNBlock, self).__init__()

        self.mdl = getattr(SBRNN, rnn_type)(
            hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x, init_params=False):
        return self.mdl(x, init_params=init_params)


class PTRNNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers,
        rnn_type="LSTM",
        dropout=0,
        bidirectional=True,
    ):
        super(PTRNNBlock, self).__init__()
        from torch.nn.modules.rnn import LSTM

        self.mdl = LSTM(
            in_channels,
            hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, x, init_params=False):
        return self.mdl(x)[0]


class ModularTransformerBlock(nn.Module):
    def __init__(
        self,
        out_channels,
        nhead,
        num_layers,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.LeakyReLU,
        output_activation=nn.ReLU,
        causal=False,
        custom_emb_module=None,
        num_modules=1,
        use_group_comm=False,
    ):
        super(ModularTransformerBlock, self).__init__()

        self.mdl = CNNTransformerSE(
            out_channels,
            output_activation,
            nhead,
            num_layers,
            d_ffn,
            dropout,
            activation,
            causal,
            custom_emb_module,
            num_modules,
            use_group_comm,
        )

    def forward(self, x, init_params=False):
        return self.mdl(x, init_params=init_params)


class DPTNetBlock(nn.Module):
    """
    The DPT Net block
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(
        self, d_model, nhead, dim_feedforward=256, dropout=0, activation="relu"
    ):

        from torch.nn.modules.activation import MultiheadAttention
        from torch.nn.modules.normalization import LayerNorm
        from torch.nn.modules.dropout import Dropout
        from torch.nn.modules.rnn import LSTM
        from torch.nn.modules.linear import Linear

        super(DPTNetBlock, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        # self.linear1 = Linear(d_model, dim_feedforward)
        self.rnn = LSTM(d_model, d_model * 2, 1, bidirectional=True)
        self.dropout = Dropout(dropout)
        # self.linear2 = Linear(dim_feedforward, d_model)
        self.linear2 = Linear(d_model * 2 * 2, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(DPTNetBlock, self).__setstate__(state)

    def forward(self, src, init_params=True):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(
            src, src, src, attn_mask=None, key_padding_mask=None
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = self.rnn(src)[0]
        src2 = self.activation(src2)
        src2 = self.dropout(src2)
        src2 = self.linear2(src2)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu


class Dual_Computation_Block(nn.Module):
    """
#            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
    """

    def __init__(
        self,
        intra_mdl,
        inter_mdl,
        out_channels,
        norm="ln",
        skip_around_intra=True,
        linear_layer_after_inter_intra=True,
    ):
        super(Dual_Computation_Block, self).__init__()

        self.intra_mdl = intra_mdl
        self.inter_mdl = inter_mdl
        self.skip_around_intra = skip_around_intra
        self.linear_layer_after_inter_intra = linear_layer_after_inter_intra

        # Norm
        self.norm = norm
        if norm is not None:
            self.intra_norm = select_norm(norm, out_channels, 4)
            self.inter_norm = select_norm(norm, out_channels, 4)

        # Linear
        if linear_layer_after_inter_intra:
            self.intra_linear = Linear(out_channels)
            self.inter_linear = Linear(out_channels)

    def forward(self, x, init_params=True):
        """
           x: [B, N, K, S]
           out: [Spks, B, N, K, S]
        """
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra = x.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)
        # [BS, K, H]

        intra = self.intra_mdl(intra, init_params=init_params)

        # [BS, K, N]
        if self.linear_layer_after_inter_intra:
            intra = self.intra_linear(
                intra.contiguous().view(B * S * K, -1), init_params=init_params
            ).view(B * S, K, -1)
        # [B, S, K, N]
        intra = intra.view(B, S, K, N)
        # [B, N, K, S]
        intra = intra.permute(0, 3, 2, 1).contiguous()
        if self.norm is not None:
            intra = self.intra_norm(intra)

        # [B, N, K, S]
        if self.skip_around_intra:
            intra = intra + x

        # inter RNN
        # [BK, S, N]
        inter = intra.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)
        # [BK, S, H]
        inter = self.inter_mdl(inter, init_params=init_params)

        # [BK, S, N]
        if self.linear_layer_after_inter_intra:
            inter = self.inter_linear(
                inter.contiguous().view(B * S * K, -1), init_params=init_params
            ).view(B * K, S, -1)
        # [B, K, S, N]
        inter = inter.view(B, K, S, N)
        # [B, N, K, S]
        inter = inter.permute(0, 3, 1, 2).contiguous()
        if self.norm is not None:
            inter = self.inter_norm(inter)
        # [B, N, K, S]
        out = inter + intra

        return out


class Dual_Path_Model(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        intra_model,
        inter_model,
        num_layers=1,
        norm="ln",
        K=200,
        num_spks=2,
        skip_around_intra=True,
        linear_layer_after_inter_intra=True,
        use_global_pos_enc=False,
        max_length=20000,
    ):
        super(Dual_Path_Model, self).__init__()
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.use_global_pos_enc = use_global_pos_enc

        if self.use_global_pos_enc:
            self.pos_enc = PositionalEncoding(max_length)

        self.dual_mdl = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_mdl.append(
                copy.deepcopy(
                    Dual_Computation_Block(
                        intra_model,
                        inter_model,
                        out_channels,
                        norm,
                        skip_around_intra=skip_around_intra,
                        linear_layer_after_inter_intra=linear_layer_after_inter_intra,
                    )
                )
            )

        self.conv2d = nn.Conv2d(
            out_channels, out_channels * num_spks, kernel_size=1
        )
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        # gated output layer
        self.output = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid()
        )

    def forward(self, x, init_params=True):
        """
           x: [B, N, L]
        """

        # [B, N, L]
        x = self.norm(x)
        # [B, N, L]
        x = self.conv1d(x)
        if self.use_global_pos_enc:
            x = self.pos_enc(x.transpose(1, -1), init_params).transpose(
                1, -1
            ) + x * (x.size(1) ** 0.5)

        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)
        # [B, N*spks, K, S]
        for i in range(self.num_layers):
            x = self.dual_mdl[i](x, init_params=init_params)

        # self.dual_mdl[1].inter_mdl.mdl.layers[0].linear1.weight to see the weights

        x = self.prelu(x)
        x = self.conv2d(x)
        # [B*spks, N, K, S]
        B, _, K, S = x.shape
        x = x.view(B * self.num_spks, -1, K, S)
        # [B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x) * self.output_gate(x)
        # [spks*B, N, L]
        x = self.end_conv1x1(x)
        # [B*spks, N, L] -> [B, spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)
        # [spks, B, N, L]
        x = x.transpose(0, 1)

        return x

    def _padding(self, input, K):
        """
           padding the audio times
           K: chunks of length
           P: hop size
           input: [B, N, L]
        """
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        """
           the segmentation stage splits
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, S]
        """
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = (
            torch.cat([input1, input2], dim=3).view(B, N, -1, K).transpose(2, 3)
        )

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        """
           Merge sequence
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        """
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input
