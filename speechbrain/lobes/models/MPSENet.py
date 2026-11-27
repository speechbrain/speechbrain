"""MP-SENet: parallel magnitude and phase spectra denoising.

The network is a dense-convolution encoder, a stack of time-frequency
blocks, and two parallel decoders (magnitude mask and phase). The
sequence model inside each time-frequency block is swappable: any module
mapping ``(N, L, C)`` to ``(N, L, C)`` works.

Reference: Lu, Ai, Ling, "MP-SENet: A Speech Enhancement Model with
Parallel Denoising of Magnitude and Phase Spectra", Interspeech 2023.
Adapted from https://github.com/yxlu-0102/MP-SENet (MIT license).

Differences from the original implementation:

* Spectra are ``[B, T, F]`` (original: ``[B, F, T]``). Phase losses are
  remapped so group delay and instantaneous frequency keep their meaning.
  The discriminator sees ``[B, 2, T, F]``; no discriminator weights were
  released, so this only matters when training.
* The original code puts ``(N, L, C)`` into ``nn.MultiheadAttention`` and
  ``nn.GRU`` without ``batch_first=True``, so they run along ``N``
  (batch x other axis) instead of ``L``. ``GRUTransformerBlock`` fixes
  both by default; ``batch_first=False`` reproduces the released
  checkpoints exactly.
* The released checkpoints use ``GRUTransformerBlock``, not the Conformer
  described in the paper. ``ConformerBlock`` reuses SpeechBrain's
  ``ConformerEncoderLayer`` (regular attention, no positional encoding,
  like the original), so it is not numerically identical to the
  original's unused ``conformer.py``.
* Released checkpoints need a key rename to load (``TSTransformer`` ->
  ``ts_blocks``, ``time/freq_transformer`` -> ``time/freq_block``,
  ``ffn.`` dropped, ``lsigmoid.slope`` squeezed to ``[F]``).
* The dropout inside the GRU feed-forward is dropped (all released
  configs use dropout 0).

Authors
* Manil Bendali 2026
"""

import torch
from torch import nn

from speechbrain.lobes.models.transformer.Conformer import ConformerEncoderLayer


def mag_pha_stft(wav, n_fft=400, hop_length=100, win_length=400, compress=0.3):
    """Computes the compressed magnitude, phase and complex spectra.

    Arguments
    ---------
    wav : torch.Tensor
        Waveforms, shape ``[B, samples]``.
    n_fft : int
        FFT size.
    hop_length : int
        Hop size in samples.
    win_length : int
        Hann window size in samples.
    compress : float
        Power-law compression factor applied to the magnitude.

    Returns
    -------
    mag : torch.Tensor
        Compressed magnitude, shape ``[B, T, F]``.
    pha : torch.Tensor
        Phase, shape ``[B, T, F]``.
    com : torch.Tensor
        Compressed complex spectrum, shape ``[B, T, F, 2]``.

    Example
    -------
    >>> mag, pha, com = mag_pha_stft(torch.randn(2, 16000))
    >>> mag.shape, com.shape
    (torch.Size([2, 161, 201]), torch.Size([2, 161, 201, 2]))
    """
    window = torch.hann_window(win_length, device=wav.device)
    spec = torch.stft(
        wav,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    spec = torch.view_as_real(spec).transpose(1, 2)  # [B, T, F, 2]
    # Epsilons match the original implementation (the released checkpoints
    # were trained with them).
    mag = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    pha = torch.atan2(spec[..., 1] + 1e-10, spec[..., 0] + 1e-5)
    mag = mag.pow(compress)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
    return mag, pha, com


def mag_pha_istft(
    mag,
    pha,
    n_fft=400,
    hop_length=100,
    win_length=400,
    compress=0.3,
    length=None,
):
    """Inverts compressed magnitude and phase back to a waveform.

    Arguments
    ---------
    mag : torch.Tensor
        Compressed magnitude, shape ``[B, T, F]``.
    pha : torch.Tensor
        Phase, shape ``[B, T, F]``.
    n_fft : int
        FFT size.
    hop_length : int
        Hop size in samples.
    win_length : int
        Hann window size in samples.
    compress : float
        Compression factor used in ``mag_pha_stft``.
    length : int, optional
        Output length in samples (default: ``(T - 1) * hop_length``).

    Returns
    -------
    torch.Tensor
        Waveforms, shape ``[B, samples]``.

    Example
    -------
    >>> wav = torch.randn(2, 16050)
    >>> mag, pha, _ = mag_pha_stft(wav)
    >>> rec = mag_pha_istft(mag, pha, length=16050)
    >>> torch.allclose(rec, wav, atol=1e-3)
    True
    """
    mag = mag.pow(1.0 / compress)
    spec = torch.polar(mag, pha).transpose(1, 2)  # [B, F, T]
    window = torch.hann_window(win_length, device=mag.device)
    return torch.istft(
        spec,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        length=length,
    )


class LearnableSigmoid(nn.Module):
    """Sigmoid with a learnable slope per feature, scaled by ``beta``.

    Arguments
    ---------
    in_features : int
        Size of the last dimension.
    beta : float
        Output upper bound.

    Example
    -------
    >>> act = LearnableSigmoid(201, beta=2.0)
    >>> act(torch.zeros(1, 10, 201)).max().item()
    1.0
    """

    def __init__(self, in_features, beta=1.0):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))

    def forward(self, x):
        """Applies the activation on the last dimension."""
        return self.beta * torch.sigmoid(self.slope * x)


class SPConvTranspose2d(nn.Module):
    """Sub-pixel convolution upsampling along the frequency axis.

    Arguments
    ---------
    in_channels : int
        Input channels.
    out_channels : int
        Output channels.
    kernel_size : tuple
        Convolution kernel size.
    r : int
        Upsampling factor.

    Example
    -------
    >>> conv = SPConvTranspose2d(8, 8, (1, 3), r=2)
    >>> conv(torch.randn(1, 8, 10, 17)).shape
    torch.Size([1, 8, 10, 34])
    """

    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super().__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size)
        self.r = r

    def forward(self, x):
        """Upsamples ``[B, C, T, F]`` to ``[B, C, T, F * r]``."""
        out = self.conv(self.pad1(x))
        b, c, t, f = out.shape
        out = out.view(b, self.r, c // self.r, t, f).permute(0, 2, 3, 4, 1)
        return out.reshape(b, c // self.r, t, f * self.r)


class DenseBlock(nn.Module):
    """Densely connected stack of time-dilated 2D convolutions.

    Arguments
    ---------
    channels : int
        Channels of each convolution.
    kernel_size : tuple
        Kernel size (time, frequency).
    depth : int
        Number of convolutions; dilation doubles at each one.

    Example
    -------
    >>> block = DenseBlock(8)
    >>> block(torch.randn(1, 8, 10, 17)).shape
    torch.Size([1, 8, 10, 17])
    """

    def __init__(self, channels, kernel_size=(2, 3), depth=4):
        super().__init__()
        self.dense_block = nn.ModuleList(
            nn.Sequential(
                nn.ConstantPad2d((1, 1, 2**i, 0), value=0.0),
                nn.Conv2d(
                    channels * (i + 1),
                    channels,
                    kernel_size,
                    dilation=(2**i, 1),
                ),
                nn.InstanceNorm2d(channels, affine=True),
                nn.PReLU(channels),
            )
            for i in range(depth)
        )

    def forward(self, x):
        """Processes ``[B, C, T, F]``."""
        skip = x
        for conv in self.dense_block:
            x = conv(skip)
            skip = torch.cat([x, skip], dim=1)
        return x


def _conv_norm_act(in_channels, channels, *conv_args, **conv_kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, channels, *conv_args, **conv_kwargs),
        nn.InstanceNorm2d(channels, affine=True),
        nn.PReLU(channels),
    )


class DenseEncoder(nn.Module):
    """Encodes stacked magnitude and phase, halving the frequency axis.

    Arguments
    ---------
    in_channels : int
        Input channels (2: magnitude and phase).
    channels : int
        Hidden channels.

    Example
    -------
    >>> enc = DenseEncoder(2, 8)
    >>> enc(torch.randn(1, 2, 10, 33)).shape
    torch.Size([1, 8, 10, 17])
    """

    def __init__(self, in_channels, channels):
        super().__init__()
        self.dense_conv_1 = _conv_norm_act(in_channels, channels, (1, 1))
        self.dense_block = DenseBlock(channels, depth=4)
        self.dense_conv_2 = _conv_norm_act(
            channels, channels, (1, 3), (1, 2), padding=(0, 1)
        )

    def forward(self, x):
        """Maps ``[B, 2, T, F]`` to ``[B, C, T, F // 2 + 1]``."""
        return self.dense_conv_2(self.dense_block(self.dense_conv_1(x)))


class MaskDecoder(nn.Module):
    """Predicts a bounded magnitude mask.

    Arguments
    ---------
    channels : int
        Hidden channels.
    n_freqs : int
        Number of frequency bins of the output (``n_fft // 2 + 1``).
    beta : float
        Upper bound of the mask.

    Example
    -------
    >>> dec = MaskDecoder(8, 33)
    >>> dec(torch.randn(1, 8, 10, 17)).shape
    torch.Size([1, 10, 33])
    """

    def __init__(self, channels, n_freqs, beta=2.0):
        super().__init__()
        self.dense_block = DenseBlock(channels, depth=4)
        self.mask_conv = nn.Sequential(
            SPConvTranspose2d(channels, channels, (1, 3), 2),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
            nn.Conv2d(channels, 1, (1, 2)),
        )
        self.lsigmoid = LearnableSigmoid(n_freqs, beta=beta)

    def forward(self, x):
        """Maps ``[B, C, T, F']`` to a mask ``[B, T, F]``."""
        x = self.mask_conv(self.dense_block(x)).squeeze(1)
        return self.lsigmoid(x)


class PhaseDecoder(nn.Module):
    """Predicts the phase from pseudo real and imaginary parts.

    Arguments
    ---------
    channels : int
        Hidden channels.

    Example
    -------
    >>> dec = PhaseDecoder(8)
    >>> dec(torch.randn(1, 8, 10, 17)).shape
    torch.Size([1, 10, 33])
    """

    def __init__(self, channels):
        super().__init__()
        self.dense_block = DenseBlock(channels, depth=4)
        self.phase_conv = nn.Sequential(
            SPConvTranspose2d(channels, channels, (1, 3), 2),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.phase_conv_r = nn.Conv2d(channels, 1, (1, 2))
        self.phase_conv_i = nn.Conv2d(channels, 1, (1, 2))

    def forward(self, x):
        """Maps ``[B, C, T, F']`` to a phase ``[B, T, F]``."""
        x = self.phase_conv(self.dense_block(x))
        pha = torch.atan2(self.phase_conv_i(x), self.phase_conv_r(x))
        return pha.squeeze(1)


class ConformerBlock(nn.Module):
    """Conformer sequence block (MP-SENet as described in the paper).

    Uses regular multi-head attention without positional encoding, like
    the original implementation.

    Arguments
    ---------
    d_model : int
        Feature size.
    nhead : int
        Number of attention heads.
    kernel_size : int
        Depthwise convolution kernel size.
    ffn_mult : int
        Feed-forward expansion factor.
    dropout : float
        Dropout rate.

    Example
    -------
    >>> block = ConformerBlock(16)
    >>> block(torch.randn(3, 20, 16)).shape
    torch.Size([3, 20, 16])
    """

    def __init__(
        self, d_model, nhead=4, kernel_size=31, ffn_mult=4, dropout=0.0
    ):
        super().__init__()
        self.layer = ConformerEncoderLayer(
            d_model=d_model,
            d_ffn=d_model * ffn_mult,
            nhead=nhead,
            kernel_size=kernel_size,
            dropout=dropout,
            attention_type="regularMHA",
        )

    def forward(self, x):
        """Processes ``(N, L, C)``."""
        return self.layer(x)[0]


class GRUTransformerBlock(nn.Module):
    """Attention + bidirectional GRU block used by the released checkpoints.

    Arguments
    ---------
    d_model : int
        Feature size.
    nhead : int
        Number of attention heads.
    batch_first : bool
        ``False`` reproduces the original code, where attention and GRU run
        along the ``N`` axis instead of ``L`` (needed to match the released
        checkpoints). Use ``True`` when training from scratch.
    dropout : float
        Dropout rate.

    Example
    -------
    >>> block = GRUTransformerBlock(16)
    >>> block(torch.randn(3, 20, 16)).shape
    torch.Size([3, 20, 16])
    """

    def __init__(self, d_model, nhead=4, batch_first=True, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.gru = nn.GRU(
            d_model, d_model * 2, batch_first=batch_first, bidirectional=True
        )
        self.linear = nn.Linear(d_model * 4, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x):
        """Processes ``(N, L, C)``."""
        xt = self.norm1(x)
        x = x + self.dropout1(self.attention(xt, xt, xt)[0])
        xt = self.gru(self.norm2(x))[0]
        x = x + self.dropout2(self.linear(nn.functional.leaky_relu(xt)))
        return self.norm3(x)


class TSBlock(nn.Module):
    """Runs one sequence block along time, then one along frequency.

    Arguments
    ---------
    seq_block : callable
        Builds a module mapping ``(N, L, C)`` to ``(N, L, C)`` when called
        with ``d_model``. Called twice (separate time and frequency weights).
    d_model : int
        Feature size.

    Example
    -------
    >>> block = TSBlock(GRUTransformerBlock, 8)
    >>> block(torch.randn(2, 8, 10, 17)).shape
    torch.Size([2, 8, 10, 17])
    """

    def __init__(self, seq_block, d_model):
        super().__init__()
        self.time_block = seq_block(d_model)
        self.freq_block = seq_block(d_model)

    def forward(self, x):
        """Processes ``[B, C, T, F]``."""
        b, c, t, f = x.shape
        x = x.permute(0, 3, 2, 1).reshape(b * f, t, c)
        x = self.time_block(x) + x
        x = x.reshape(b, f, t, c).transpose(1, 2).reshape(b * t, f, c)
        x = self.freq_block(x) + x
        return x.reshape(b, t, f, c).permute(0, 3, 1, 2)


class MPSENet(nn.Module):
    """MP-SENet generator with a swappable time-frequency sequence block.

    Arguments
    ---------
    seq_block : callable
        Builds the sequence model of each time-frequency block, e.g.
        ``ConformerBlock`` (MP-SENet) or
        ``functools.partial(GRUTransformerBlock, batch_first=False)`` to load
        the released checkpoints.
    num_blocks : int
        Number of time-frequency blocks.
    channels : int
        Hidden channels.
    n_fft : int
        FFT size of the input spectra.
    beta : float
        Upper bound of the magnitude mask.

    Example
    -------
    >>> model = MPSENet(ConformerBlock, num_blocks=1, channels=16, n_fft=64)
    >>> mag, pha, _ = mag_pha_stft(
    ...     torch.randn(2, 800), n_fft=64, hop_length=16, win_length=64
    ... )
    >>> mag_g, pha_g, com_g = model(mag, pha)
    >>> mag_g.shape, com_g.shape
    (torch.Size([2, 51, 33]), torch.Size([2, 51, 33, 2]))
    """

    def __init__(
        self, seq_block, num_blocks=4, channels=64, n_fft=400, beta=2.0
    ):
        super().__init__()
        self.dense_encoder = DenseEncoder(2, channels)
        self.ts_blocks = nn.ModuleList(
            TSBlock(seq_block, channels) for _ in range(num_blocks)
        )
        self.mask_decoder = MaskDecoder(channels, n_fft // 2 + 1, beta)
        self.phase_decoder = PhaseDecoder(channels)

    def forward(self, noisy_mag, noisy_pha):
        """Denoises compressed magnitude and phase.

        Arguments
        ---------
        noisy_mag : torch.Tensor
            Compressed magnitude, shape ``[B, T, F]``.
        noisy_pha : torch.Tensor
            Phase, shape ``[B, T, F]``.

        Returns
        -------
        mag : torch.Tensor
            Denoised compressed magnitude, ``[B, T, F]``.
        pha : torch.Tensor
            Denoised phase, ``[B, T, F]``.
        com : torch.Tensor
            Denoised compressed complex spectrum, ``[B, T, F, 2]``.
        """
        x = self.dense_encoder(torch.stack((noisy_mag, noisy_pha), dim=1))
        for block in self.ts_blocks:
            x = block(x)
        mag = noisy_mag * self.mask_decoder(x)
        pha = self.phase_decoder(x)
        com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
        return mag, pha, com


class MetricDiscriminator(nn.Module):
    """Predicts the normalized PESQ of a degraded magnitude vs the clean one.

    Arguments
    ---------
    channels : int
        Channels of the first convolution; doubled at each of the 4 layers.

    Example
    -------
    >>> disc = MetricDiscriminator()
    >>> clean, enhanced = torch.rand(2, 2, 50, 201)
    >>> disc(clean, enhanced).shape
    torch.Size([2, 1])
    """

    def __init__(self, channels=16):
        super().__init__()
        sn = nn.utils.spectral_norm
        layers, in_ch = [], 2
        for out_ch in (channels, channels * 2, channels * 4, channels * 8):
            layers += [
                sn(
                    nn.Conv2d(in_ch, out_ch, (4, 4), (2, 2), (1, 1), bias=False)
                ),
                nn.InstanceNorm2d(out_ch, affine=True),
                nn.PReLU(out_ch),
            ]
            in_ch = out_ch
        layers += [
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            sn(nn.Linear(channels * 8, channels * 4)),
            nn.Dropout(0.3),
            nn.PReLU(channels * 4),
            sn(nn.Linear(channels * 4, 1)),
            LearnableSigmoid(1),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, clean_mag, mag):
        """Scores ``mag`` against ``clean_mag``, both ``[B, T, F]``.

        Returns
        -------
        torch.Tensor
            Scores in ``[0, 1]``, shape ``[B, 1]``.
        """
        return self.layers(torch.stack((clean_mag, mag), dim=1))


def anti_wrapping(x):
    """Distance to the nearest multiple of 2*pi, so phase errors don't wrap.

    Arguments
    ---------
    x : torch.Tensor
        Phase differences.

    Returns
    -------
    torch.Tensor
        ``|x - 2*pi*round(x / (2*pi))|``, in ``[0, pi]``.

    Example
    -------
    >>> anti_wrapping(torch.tensor([2 * torch.pi + 0.1, -0.1]))
    tensor([0.1000, 0.1000])
    """
    two_pi = 2 * torch.pi
    return torch.abs(x - torch.round(x / two_pi) * two_pi)


def phase_losses(clean_pha, pha):
    """Anti-wrapping phase losses from the MP-SENet paper.

    Arguments
    ---------
    clean_pha : torch.Tensor
        Clean phase, shape ``[B, T, F]``.
    pha : torch.Tensor
        Estimated phase, shape ``[B, T, F]``.

    Returns
    -------
    ip_loss : torch.Tensor
        Instantaneous phase loss.
    gd_loss : torch.Tensor
        Group delay loss (phase differences along frequency).
    iaf_loss : torch.Tensor
        Instantaneous angular frequency loss (differences along time).

    Example
    -------
    >>> pha = torch.rand(2, 50, 201) * 6 - 3
    >>> [loss.item() for loss in phase_losses(pha, pha)]
    [0.0, 0.0, 0.0]
    """
    ip_loss = anti_wrapping(clean_pha - pha).mean()
    gd_loss = anti_wrapping(
        torch.diff(clean_pha, dim=2) - torch.diff(pha, dim=2)
    ).mean()
    iaf_loss = anti_wrapping(
        torch.diff(clean_pha, dim=1) - torch.diff(pha, dim=1)
    ).mean()
    return ip_loss, gd_loss, iaf_loss
