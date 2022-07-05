"""Library for the Reseource-Efficient Sepformer.

Authors
 * Cem Subakan 2022
"""

import torch
import torch.nn as nn
from speechbrain.lobes.models.dual_path import select_norm
from speechbrain.lobes.models.transformer.Transformer import (
    TransformerEncoder,
    PositionalEncoding,
    get_lookahead_mask,
)
import speechbrain.nnet.RNN as SBRNN
import copy


EPS = torch.finfo(torch.get_default_dtype()).eps


class MemLSTM(nn.Module):
    """the Mem-LSTM of SkiM --
    Note: This is taken from the SkiM implementation in ESPNet toolkit and modified for compability with SpeechBrain.

    Arguments:
    ---------
    hidden_size: int,
        Dimension of the hidden state.
    dropout: float,
        dropout ratio. Default is 0.
    bidirectional: bool,
        Whether the LSTM layers are bidirectional.
        Default is False.
    mem_type: 'hc', 'h', 'c', or 'id'.
        This controls whether the hidden (or cell) state of
        SegLSTM will be processed by MemLSTM.
        In 'id' mode, both the hidden and cell states will
        be identically returned.
    norm_type: 'gln', 'cln'
        This selects the type of normalization
        cln is for causal implemention

    Example
    ---------
    >>> x = (torch.randn(1, 5, 64), torch.randn(1, 5, 64))
    >>> block = MemLSTM(64)
    >>> x = block(x, 5)
    >>> x[0].shape
    torch.Size([1, 5, 64])
    """

    def __init__(
        self,
        hidden_size,
        dropout=0.0,
        bidirectional=False,
        mem_type="hc",
        norm_type="cln",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.input_size = (int(bidirectional) + 1) * hidden_size
        self.mem_type = mem_type

        assert mem_type in [
            "hc",
            "h",
            "c",
            "id",
        ], f"only support 'hc', 'h', 'c' and 'id', current type: {mem_type}"

        if mem_type in ["hc", "h"]:
            self.h_net = SBRNNBlock(
                input_size=self.input_size,
                hidden_channels=self.hidden_size,
                num_layers=1,
                outsize=self.input_size,
                rnn_type="LSTM",
                dropout=dropout,
                bidirectional=bidirectional,
            )

            self.h_norm = select_norm(
                norm=norm_type, dim=self.input_size, shape=3, eps=EPS
            )
        if mem_type in ["hc", "c"]:
            self.c_net = SBRNNBlock(
                input_size=self.input_size,
                hidden_channels=self.hidden_size,
                num_layers=1,
                outsize=self.input_size,
                rnn_type="LSTM",
                dropout=dropout,
                bidirectional=bidirectional,
            )

            self.c_norm = select_norm(
                norm=norm_type, dim=self.input_size, shape=3, eps=EPS
            )

    def forward(self, hc, S):
        """The forward function for the memory RNN

        Arguments
        ---------
        hc : torch.Tensor
            (h, c), tuple of hidden and cell states from SegLSTM

            shape of h and c: (d, B*S, H)
                where d is the number of directions
                      B is the batchsize
                      S is the number chunks
                      H is the latent dimensionality
        S : int
            S is the number of chunks
        """
        if self.mem_type == "id":
            ret_val = hc
        else:
            h, c = hc
            d, BS, H = h.shape
            B = BS // S
            h = h.transpose(1, 0).contiguous().view(B, S, d * H)  # B, S, dH
            c = c.transpose(1, 0).contiguous().view(B, S, d * H)  # B, S, dH
            if self.mem_type == "hc":
                h = h + self.h_norm(self.h_net(h).permute(0, 2, 1)).permute(
                    0, 2, 1
                )
                c = c + self.c_norm(self.c_net(c).permute(0, 2, 1)).permute(
                    0, 2, 1
                )
            elif self.mem_type == "h":
                h = h + self.h_norm(self.h_net(h).permute(0, 2, 1)).permute(
                    0, 2, 1
                )
                c = torch.zeros_like(c)
            elif self.mem_type == "c":
                h = torch.zeros_like(h)
                c = c + self.c_norm(self.c_net(c).permute(0, 2, 1)).permute(
                    0, 2, 1
                )

            h = h.view(B * S, d, H).transpose(1, 0).contiguous()
            c = c.view(B * S, d, H).transpose(1, 0).contiguous()
            ret_val = (h, c)

        if not self.bidirectional:
            # for causal setup
            causal_ret_val = []
            for x in ret_val:
                x_ = torch.zeros_like(x)
                x_[:, 1:, :] = x[:, :-1, :]
                causal_ret_val.append(x_)
            ret_val = tuple(causal_ret_val)

        return ret_val


class SegLSTM(nn.Module):
    """the Segment-LSTM of SkiM
    Note: This is taken from the SkiM implementation in ESPNet toolkit and modified for compatibility with SpeechBrain.

    Arguments:
    ----------
    input_size: int,
        dimension of the input feature.
        The input should have shape (batch, seq_len, input_size).
    hidden_size: int,
        dimension of the hidden state.
    dropout: float,
        dropout ratio. Default is 0.
    bidirectional: bool,
        whether the LSTM layers are bidirectional.
        Default is False.
    norm_type: gln, cln.
        This selects the type of normalization
        cln is for causal implementation.

    Example
    ---------
    >>> x = torch.randn(3, 20, 64)
    >>> hc = None
    >>> seglstm = SegLSTM(64, 64)
    >>> y = seglstm(x, hc)
    >>> y[0].shape
    torch.Size([3, 20, 64])
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        dropout=0.0,
        bidirectional=False,
        norm_type="cLN",
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)
        self.norm = select_norm(
            norm=norm_type, dim=input_size, shape=3, eps=EPS
        )

    def forward(self, input, hc):
        """The forward function of the Segment LSTM

        Arguments
        ---------
        input : torch.Tensor of size [B*S, T, H]
            where B is the batchsize
                  S is the number of chunks
                  T is the chunks size
                  H is the latent dimensionality

            (h, c), tuple of hidden and cell states from SegLSTM

            shape of h and c: (d, B*S, H)
                where d is the number of directions
                      B is the batchsize
                      S is the number chunks
                      H is the latent dimensionality
        """
        B, T, H = input.shape

        if hc is None:
            # In fist input SkiM block, h and c are not available
            d = self.num_direction
            h = torch.zeros(d, B, self.hidden_size).to(input.device)
            c = torch.zeros(d, B, self.hidden_size).to(input.device)
        else:
            h, c = hc

        output, (h, c) = self.lstm(input, (h, c))
        output = self.dropout(output)
        output = self.proj(output.contiguous().view(-1, output.shape[2])).view(
            input.shape
        )
        output_norm = self.norm(output.permute(0, 2, 1)).permute(0, 2, 1)

        output = input + output_norm
        return output, (h, c)


class SBRNNBlock(nn.Module):
    """RNNBlock with output layer.

    Arguments
    ---------
    input_size : int
        Dimensionality of the input features.
    hidden_channels : int
        Dimensionality of the latent layer of the rnn.
    num_layers : int
        Number of the rnn layers.
    out_size : int
        Number of dimensions at the output of the linear layer
    rnn_type : str
        Type of the the rnn cell.
    dropout : float
        Dropout rate
    bidirectional : bool
        If True, bidirectional.

    Example
    ---------
    >>> x = torch.randn(10, 100, 64)
    >>> rnn = SBRNNBlock(64, 100, 1, 128, bidirectional=True)
    >>> x = rnn(x)
    >>> x.shape
    torch.Size([10, 100, 128])
    """

    def __init__(
        self,
        input_size,
        hidden_channels,
        num_layers,
        outsize,
        rnn_type="LSTM",
        dropout=0,
        bidirectional=True,
    ):
        super(SBRNNBlock, self).__init__()

        self.mdl = getattr(SBRNN, rnn_type)(
            hidden_channels,
            input_size=input_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        rnn_outsize = 2 * hidden_channels if bidirectional else hidden_channels
        self.out = nn.Linear(rnn_outsize, outsize)

    def forward(self, x):
        """Returns the transformed output.

        Arguments
        ---------
        x : torch.Tensor
            [B, L, N]
            where, B = Batchsize,
                   N = number of filters
                   L = time points
        """
        rnn_out = self.mdl(x)[0]
        out = self.out(rnn_out)
        return out


class SBTransformerBlock_wnormandskip(nn.Module):
    """A wrapper for the SpeechBrain implementation of the transformer encoder.

    Arguments
    ---------
    num_layers : int
        Number of layers.
    d_model : int
        Dimensionality of the representation.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Dimensionality of positional feed forward.
    input_shape : tuple
        Shape of input.
    kdim : int
        Dimension of the key (Optional).
    vdim : int
        Dimension of the value (Optional).
    dropout : float
        Dropout rate.
    activation : str
        Activation function.
    use_positional_encoding : bool
        If true we use a positional encoding.
    norm_before: bool
        Use normalization before transformations.

    Example
    ---------
    >>> x = torch.randn(10, 100, 64)
    >>> block = SBTransformerBlock_wnormandskip(1, 64, 8)
    >>> x = block(x)
    >>> x.shape
    torch.Size([10, 100, 64])
    """

    def __init__(
        self,
        num_layers,
        d_model,
        nhead,
        d_ffn=2048,
        input_shape=None,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation="relu",
        use_positional_encoding=False,
        norm_before=False,
        attention_type="regularMHA",
        causal=False,
        use_norm=True,
        use_skip=True,
        norm_type="gln",
    ):
        super(SBTransformerBlock_wnormandskip, self).__init__()
        self.use_positional_encoding = use_positional_encoding

        if activation == "relu":
            activation = nn.ReLU
        elif activation == "gelu":
            activation = nn.GELU
        else:
            raise ValueError("unknown activation")

        self.causal = causal

        self.mdl = TransformerEncoder(
            num_layers=num_layers,
            nhead=nhead,
            d_ffn=d_ffn,
            input_shape=input_shape,
            d_model=d_model,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            activation=activation,
            normalize_before=norm_before,
            causal=causal,
            attention_type=attention_type,
        )

        self.use_norm = use_norm
        self.use_skip = use_skip

        if use_norm:
            self.norm = select_norm(
                norm=norm_type, dim=d_model, shape=3, eps=EPS
            )

        if use_positional_encoding:
            self.pos_enc = PositionalEncoding(
                input_size=d_model, max_len=100000
            )

    def forward(self, x):
        """Returns the transformed output.

        Arguments
        ---------
        x : torch.Tensor
            Tensor shape [B, L, N],
            where, B = Batchsize,
                   L = time points
                   N = number of filters
        """
        src_mask = get_lookahead_mask(x) if self.causal else None

        if self.use_positional_encoding:
            pos_enc = self.pos_enc(x)
            out = self.mdl(x + pos_enc, src_mask=src_mask)[0]
        else:
            out = self.mdl(x, src_mask=src_mask)[0]

        if self.use_norm:
            out = self.norm(out.permute(0, 2, 1)).permute(0, 2, 1)
        if self.use_skip:
            out = out + x

        return out


class ResourceEfficientSeparationPipeline(nn.Module):
    """ Resource Efficient Separation Pipeline Used for RE-SepFormer and SkiM

    Note: This implementation is a generalization of the ESPNET implementation of SkiM

    Arguments:
    ----------
    input_size: int,
        Dimension of the input feature.
        Input shape shoud be (batch, length, input_size)
    hidden_size: int,
        Dimension of the hidden state.
    output_size: int,
        Dimension of the output size.
    dropout: float,
        Dropout ratio. Default is 0.
    num_blocks: int
        Number of basic SkiM blocks
    segment_size: int
        Segmentation size for splitting long features
    bidirectional: bool,
        Whether the RNN layers are bidirectional.
    mem_type: 'hc', 'h', 'c', 'id' or None.
        This controls whether the hidden (or cell) state of SegLSTM
        will be processed by MemLSTM.
        In 'id' mode, both the hidden and cell states will
        be identically returned.
        When mem_type is None, the MemLSTM will be removed.
    norm_type: gln, cln.
        cln is for causal implementation.
    seg_model: class
        The model that processes the within segment elements
    mem_model: class
        The memory model that ensures continuity between the segments

    Example
    ---------
    >>> x = torch.randn(10, 100, 64)
    >>> seg_mdl = SBTransformerBlock_wnormandskip(1, 64, 8)
    >>> mem_mdl = SBTransformerBlock_wnormandskip(1, 64, 8)
    >>> resepf_pipeline = ResourceEfficientSeparationPipeline(64, 64, 128, seg_model=seg_mdl, mem_model=mem_mdl)
    >>> out = resepf_pipeline.forward(x)
    >>> out.shape
    torch.Size([10, 100, 128])
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        dropout=0.0,
        num_blocks=2,
        segment_size=20,
        bidirectional=True,
        mem_type="av",
        norm_type="gln",
        seg_model=None,
        mem_model=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.segment_size = segment_size
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.mem_type = mem_type
        self.norm_type = norm_type

        assert mem_type in [
            "hc",
            "h",
            "c",
            "id",
            "av",
            None,
        ], f"only support 'hc', 'h', 'c', 'id', 'av' and None, current type: {mem_type}"

        self.seg_model = nn.ModuleList([])
        for i in range(num_blocks):
            self.seg_model.append(copy.deepcopy(seg_model))

        if self.mem_type is not None:
            self.mem_model = nn.ModuleList([])
            for i in range(num_blocks - 1):
                self.mem_model.append(copy.deepcopy(mem_model))

        self.output_fc = nn.Sequential(
            nn.PReLU(), nn.Conv1d(input_size, output_size, 1)
        )

    def forward(self, input):
        """The forward function of the ResourceEfficientSeparatioPipeline

        This takes in a tensor of size [B, (S*K), D]

        Arguments
        ---------
        input : torch.Tensor
                Tensor shape [B, (S*K), D],
                where, B = Batchsize,
                       S = Number of chunks
                       K = Chunksize
                       D = number of features
        """
        B, T, D = input.shape

        input, rest = self._padfeature(input=input)
        input = input.view(B, -1, self.segment_size, D)  # B, S, K, D
        B, S, K, D = input.shape

        assert K == self.segment_size

        output = input.reshape(B * S, K, D)  # BS, K, D

        if self.mem_type == "av":
            hc = torch.zeros(
                output.shape[0], 1, output.shape[-1], device=output.device
            )
        else:
            hc = None

        for i in range(self.num_blocks):
            seg_model_type = type(self.seg_model[0]).__name__
            if seg_model_type == "SBTransformerBlock_wnormandskip":
                output = self.seg_model[i](output + hc)  # BS, K, D
            elif seg_model_type == "SegLSTM":
                output, hc = self.seg_model[i](output, hc)  # BS, K, D
            else:
                raise ValueError("Unsupported segment model class")

            if i < (self.num_blocks - 1):
                if self.mem_type == "av":
                    hc = output.mean(1).unsqueeze(0)
                    hc = self.mem_model[i](hc).permute(1, 0, 2)
                else:
                    hc = self.mem_model[i](hc, S)

        output = output.reshape(B, S * K, D)[:, :T, :]  # B, T, D
        output = self.output_fc(output.transpose(1, 2)).transpose(1, 2)

        return output

    def _padfeature(self, input):
        """
        Argument:
        ----------
        input : torch.Tensor of size [B, T, D]
                    where B is Batchsize
                          T is the chunk length
                          D is the feature dimensionality
        """
        B, T, D = input.shape
        rest = self.segment_size - T % self.segment_size

        if rest > 0:
            input = torch.nn.functional.pad(input, (0, 0, 0, rest))
        return input, rest


class ResourceEfficientSeparator(nn.Module):
    """Resource Efficient Source Separator
    This is the class that implements RE-SepFormer

    Arguments:
    ----------
    input_dim: int,
        Input feature dimension
    causal: bool,
        Whether the system is causal.
    num_spk: int,
        Number of target speakers.
    nonlinear: class
        the nonlinear function for mask estimation,
        select from 'relu', 'tanh', 'sigmoid'
    layer: int,
        number of blocks. Default is 2 for RE-SepFormer.
    unit: int,
        Dimensionality of the hidden state.
    segment_size: int,
        Chunk size for splitting long features
    dropout: float,
        dropout ratio. Default is 0.
    mem_type: 'hc', 'h', 'c', 'id', 'av'  or None.
        This controls whether a memory representation will be used to ensure continuity between segments.
        In 'av' mode, the summary state is is calculated by simply averaging over the time dimension of each segment
        In 'id' mode, both the hidden and cell states
        will be identically returned.
        When mem_type is None, the memory model will be removed.
    seg_model: class,
        The model that processes the within segment elements
    mem_model: class,
        The memory model that ensures continuity between the segments

    Example
    ---------
    >>> x = torch.randn(10, 64, 100)
    >>> seg_mdl = SBTransformerBlock_wnormandskip(1, 64, 8)
    >>> mem_mdl = SBTransformerBlock_wnormandskip(1, 64, 8)
    >>> resepformer = ResourceEfficientSeparator(64, num_spk=3, mem_type='av', seg_model=seg_mdl, mem_model=mem_mdl)
    >>> out = resepformer.forward(x)
    >>> out.shape
    torch.Size([3, 10, 64, 100])
    """

    def __init__(
        self,
        input_dim: int,
        causal: bool = True,
        num_spk: int = 2,
        nonlinear: str = "relu",
        layer: int = 3,
        unit: int = 512,
        segment_size: int = 20,
        dropout: float = 0.0,
        mem_type: str = "hc",
        seg_model=None,
        mem_model=None,
    ):

        super().__init__()

        self.num_spk = num_spk

        self.segment_size = segment_size

        if mem_type not in ("hc", "h", "c", "id", "av", None):
            raise ValueError("Not supporting mem_type={}".format(mem_type))

        self.model = ResourceEfficientSeparationPipeline(
            input_size=input_dim,
            hidden_size=unit,
            output_size=input_dim * num_spk,
            dropout=dropout,
            num_blocks=layer,
            bidirectional=(not causal),
            norm_type="cln" if causal else "gln",
            segment_size=segment_size,
            mem_type=mem_type,
            seg_model=seg_model,
            mem_model=mem_model,
        )

        if nonlinear not in ("sigmoid", "relu", "tanh"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))

        self.nonlinear = {
            "sigmoid": torch.nn.Sigmoid(),
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
        }[nonlinear]

    def forward(self, inpt: torch.Tensor):
        """Forward.
        Arguments:
        ----------
            inpt (torch.Tensor):
                Encoded feature [B, T, N]
        """

        inpt = inpt.permute(0, 2, 1)

        B, T, N = inpt.shape
        processed = self.model(inpt)  # B,T, N

        processed = processed.reshape(B, T, N, self.num_spk)
        masks = self.nonlinear(processed).unbind(dim=3)

        mask_tensor = torch.stack([m.permute(0, 2, 1) for m in masks])

        return mask_tensor
