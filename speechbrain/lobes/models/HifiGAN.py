"""
Neural network modules for the HiFi-GAN: Generative Adversarial Networks for
Efficient and High Fidelity Speech Synthesis

For more details: https://arxiv.org/pdf/2010.05646.pdf, https://arxiv.org/abs/2406.10735

Authors
 * Jarod Duret 2021
 * Yingzhi WANG 2022
"""

# Adapted from https://github.com/jik876/hifi-gan/ and https://github.com/coqui-ai/TTS/
# MIT License

# Copyright (c) 2020 Jungil Kong

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import transforms

import speechbrain as sb
from speechbrain.nnet.CNN import Conv1d, Conv2d, ConvTranspose1d

LRELU_SLOPE = 0.1


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """Dynamique range compression for audio signals"""
    return torch.log(torch.clamp(x, min=clip_val) * C)


def mel_spectogram(
    sample_rate,
    hop_length,
    win_length,
    n_fft,
    n_mels,
    f_min,
    f_max,
    power,
    normalized,
    norm,
    mel_scale,
    compression,
    audio,
):
    """calculates MelSpectrogram for a raw audio signal

    Arguments
    ---------
    sample_rate : int
        Sample rate of audio signal.
    hop_length : int
        Length of hop between STFT windows.
    win_length : int
        Window size.
    n_fft : int
        Size of FFT.
    n_mels : int
        Number of mel filterbanks.
    f_min : float
        Minimum frequency.
    f_max : float
        Maximum frequency.
    power : float
        Exponent for the magnitude spectrogram.
    normalized : bool
        Whether to normalize by magnitude after stft.
    norm : str or None
        If "slaney", divide the triangular mel weights by the width of the mel band
    mel_scale : str
        Scale to use: "htk" or "slaney".
    compression : bool
        whether to do dynamic range compression
    audio : torch.tensor
        input audio signal

    Returns
    -------
    Mel spectrogram
    """

    audio_to_mel = transforms.MelSpectrogram(
        sample_rate=sample_rate,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=power,
        normalized=normalized,
        norm=norm,
        mel_scale=mel_scale,
    ).to(audio.device)

    mel = audio_to_mel(audio)

    if compression:
        mel = dynamic_range_compression(mel)

    return mel


def process_duration(code, code_feat):
    """
    Process a given batch of code to extract consecutive unique elements and their associated features.

    Arguments
    ---------
    code : torch.Tensor (batch, time)
        Tensor of code indices.
    code_feat : torch.Tensor (batch, time, channel)
        Tensor of code features.

    Returns
    -------
    uniq_code_feat_filtered : torch.Tensor (batch, time)
        Features of consecutive unique codes.
    mask : torch.Tensor (batch, time)
        Padding mask for the unique codes.
    uniq_code_count : torch.Tensor (n)
        Count of unique codes.

    Example
    -------
    >>> code = torch.IntTensor([[40, 18, 18, 10]])
    >>> code_feat = torch.rand([1, 4, 128])
    >>> out_tensor, mask, uniq_code = process_duration(code, code_feat)
    >>> out_tensor.shape
    torch.Size([1, 1, 128])
    >>> mask.shape
    torch.Size([1, 1])
    >>> uniq_code.shape
    torch.Size([1])
    """
    uniq_code_count = []
    uniq_code_feat = []
    for i in range(code.size(0)):
        _, count = torch.unique_consecutive(code[i, :], return_counts=True)
        if len(count) > 2:
            # remove first and last code as segment sampling may cause incomplete segment length
            uniq_code_count.append(count[1:-1])
            uniq_code_idx = count.cumsum(dim=0)[:-2]
        else:
            uniq_code_count.append(count)
            uniq_code_idx = count.cumsum(dim=0) - 1
        uniq_code_feat.append(
            code_feat[i, uniq_code_idx, :].view(-1, code_feat.size(2))
        )
    uniq_code_count = torch.cat(uniq_code_count)

    # collate
    max_len = max(feat.size(0) for feat in uniq_code_feat)
    uniq_code_feat_filtered = uniq_code_feat[0].new_zeros(
        (len(uniq_code_feat), max_len, uniq_code_feat[0].size(1))
    )
    mask = torch.arange(max_len).repeat(len(uniq_code_feat), 1)
    for i, v in enumerate(uniq_code_feat):
        uniq_code_feat_filtered[i, : v.size(0)] = v
        mask[i, :] = mask[i, :] < v.size(0)

    return uniq_code_feat_filtered, mask.bool(), uniq_code_count.float()


##################################
# Generator
##################################


class ResBlock1(torch.nn.Module):
    """
    Residual Block Type 1, which has 3 convolutional layers in each convolution block.

    Arguments
    ---------
    channels : int
        number of hidden channels for the convolutional layers.
    kernel_size : int
        size of the convolution filter in each layer.
    dilation : list
        list of dilation value for each conv layer in a block.
    """

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation[0],
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation[1],
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation[2],
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=1,
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=1,
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=1,
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
            ]
        )

    def forward(self, x):
        """Returns the output of ResBlock1

        Arguments
        ---------
        x : torch.Tensor (batch, channel, time)
            input tensor.

        Returns
        -------
        The ResBlock outputs
        """

        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        """This functions removes weight normalization during inference."""
        for layer in self.convs1:
            layer.remove_weight_norm()
        for layer in self.convs2:
            layer.remove_weight_norm()


class ResBlock2(torch.nn.Module):
    """
    Residual Block Type 2, which has 2 convolutional layers in each convolution block.

    Arguments
    ---------
    channels : int
        number of hidden channels for the convolutional layers.
    kernel_size : int
        size of the convolution filter in each layer.
    dilation : list
        list of dilation value for each conv layer in a block.
    """

    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation[0],
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation[1],
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
            ]
        )

    def forward(self, x):
        """Returns the output of ResBlock1

        Arguments
        ---------
        x : torch.Tensor (batch, channel, time)
            input tensor.

        Returns
        -------
        The ResBlock outputs
        """

        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        """This functions removes weight normalization during inference."""
        for layer in self.convs:
            layer.remove_weight_norm()


class HifiganGenerator(torch.nn.Module):
    """HiFiGAN Generator with Multi-Receptive Field Fusion (MRF)

    Arguments
    ---------
    in_channels : int
        number of input tensor channels.
    out_channels : int
        number of output tensor channels.
    resblock_type : str
        type of the `ResBlock`. '1' or '2'.
    resblock_dilation_sizes : List[List[int]]
        list of dilation values in each layer of a `ResBlock`.
    resblock_kernel_sizes : List[int]
        list of kernel sizes for each `ResBlock`.
    upsample_kernel_sizes : List[int]
        list of kernel sizes for each transposed convolution.
    upsample_initial_channel : int
        number of channels for the first upsampling layer. This is divided by 2
        for each consecutive upsampling layer.
    upsample_factors : List[int]
        upsampling factors (stride) for each upsampling layer.
    inference_padding : int
       constant padding applied to the input at inference time. Defaults to 5.
    cond_channels : int
        If provided, adds a conv layer to the beginning of the forward.
    conv_post_bias : bool
        Whether to add a bias term to the final conv.

    Example
    -------
    >>> inp_tensor = torch.rand([4, 80, 33])
    >>> hifigan_generator= HifiganGenerator(
    ...    in_channels = 80,
    ...    out_channels = 1,
    ...    resblock_type = "1",
    ...    resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    ...    resblock_kernel_sizes = [3, 7, 11],
    ...    upsample_kernel_sizes = [16, 16, 4, 4],
    ...    upsample_initial_channel = 512,
    ...    upsample_factors = [8, 8, 2, 2],
    ... )
    >>> out_tensor = hifigan_generator(inp_tensor)
    >>> out_tensor.shape
    torch.Size([4, 1, 8448])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        resblock_type,
        resblock_dilation_sizes,
        resblock_kernel_sizes,
        upsample_kernel_sizes,
        upsample_initial_channel,
        upsample_factors,
        inference_padding=5,
        cond_channels=0,
        conv_post_bias=True,
    ):
        super().__init__()
        self.inference_padding = inference_padding
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)
        # initial upsampling layers
        self.conv_pre = Conv1d(
            in_channels=in_channels,
            out_channels=upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding="same",
            skip_transpose=True,
            weight_norm=True,
        )
        resblock = ResBlock1 if resblock_type == "1" else ResBlock2
        # upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_factors, upsample_kernel_sizes)):
            self.ups.append(
                ConvTranspose1d(
                    in_channels=upsample_initial_channel // (2**i),
                    out_channels=upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=k,
                    stride=u,
                    padding=(k - u) // 2,
                    skip_transpose=True,
                    weight_norm=True,
                )
            )
        # MRF blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for _, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))
        # post convolution layer
        self.conv_post = Conv1d(
            in_channels=ch,
            out_channels=1,
            kernel_size=7,
            stride=1,
            padding="same",
            skip_transpose=True,
            bias=conv_post_bias,
            weight_norm=True,
        )
        if cond_channels > 0:
            self.cond_layer = Conv1d(
                in_channels=cond_channels,
                out_channels=upsample_initial_channel,
                kernel_size=1,
            )

    def forward(self, x, g=None):
        """
        Arguments
        ---------
        x : torch.Tensor (batch, channel, time)
            feature input tensor.
        g : torch.Tensor (batch, 1, time)
            global conditioning input tensor.

        Returns
        -------
        The generator outputs
        """

        o = self.conv_pre(x)
        if hasattr(self, "cond_layer"):
            o = o + self.cond_layer(g)
        for i in range(self.num_upsamples):
            o = F.leaky_relu(o, LRELU_SLOPE)
            o = self.ups[i](o)
            z_sum = None
            for j in range(self.num_kernels):
                if z_sum is None:
                    z_sum = self.resblocks[i * self.num_kernels + j](o)
                else:
                    z_sum += self.resblocks[i * self.num_kernels + j](o)
            o = z_sum / self.num_kernels
        o = F.leaky_relu(o)
        o = self.conv_post(o)
        o = torch.tanh(o)
        return o

    def remove_weight_norm(self):
        """This functions removes weight normalization during inference."""

        for layer in self.ups:
            layer.remove_weight_norm()
        for layer in self.resblocks:
            layer.remove_weight_norm()
        self.conv_pre.remove_weight_norm()
        self.conv_post.remove_weight_norm()

    @torch.no_grad()
    def inference(self, c, padding=True):
        """The inference function performs a padding and runs the forward method.

        Arguments
        ---------
        c : torch.Tensor (batch, channel, time)
            feature input tensor.
        padding : bool
            Whether to pad tensor before forward.

        Returns
        -------
        The generator outputs
        """
        if padding:
            c = torch.nn.functional.pad(
                c, (self.inference_padding, self.inference_padding), "replicate"
            )
        return self.forward(c)


class VariancePredictor(nn.Module):
    """Variance predictor inspired from FastSpeech2

    Arguments
    ---------
    encoder_embed_dim : int
        number of input tensor channels.
    var_pred_hidden_dim : int
        size of hidden channels for the convolutional layers.
    var_pred_kernel_size : int
        size of the convolution filter in each layer.
    var_pred_dropout : float
        dropout probability of each layer.

    Example
    -------
    >>> inp_tensor = torch.rand([4, 80, 128])
    >>> duration_predictor = VariancePredictor(
    ...    encoder_embed_dim = 128,
    ...    var_pred_hidden_dim = 128,
    ...    var_pred_kernel_size = 3,
    ...    var_pred_dropout = 0.5,
    ... )
    >>> out_tensor = duration_predictor (inp_tensor)
    >>> out_tensor.shape
    torch.Size([4, 80])
    """

    def __init__(
        self,
        encoder_embed_dim,
        var_pred_hidden_dim,
        var_pred_kernel_size,
        var_pred_dropout,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            Conv1d(
                in_channels=encoder_embed_dim,
                out_channels=var_pred_hidden_dim,
                kernel_size=var_pred_kernel_size,
                padding="same",
                skip_transpose=True,
                weight_norm=True,
            ),
            nn.ReLU(),
        )
        self.dropout = var_pred_dropout
        self.conv2 = nn.Sequential(
            Conv1d(
                in_channels=var_pred_hidden_dim,
                out_channels=var_pred_hidden_dim,
                kernel_size=var_pred_kernel_size,
                padding="same",
                skip_transpose=True,
                weight_norm=True,
            ),
            nn.ReLU(),
        )
        self.proj = nn.Linear(var_pred_hidden_dim, 1)

    def forward(self, x):
        """
        Arguments
        ---------
        x : torch.Tensor (batch, channel, time)
            feature input tensor.

        Returns
        -------
        Variance predictor output
        """
        x = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.proj(x).squeeze(dim=2)


class UnitHifiganGenerator(HifiganGenerator):
    """The UnitHiFiGAN generator takes discrete speech tokens as input.
    The generator is adapted to support bitrate scalability training.
    For more details, refer to: https://arxiv.org/abs/2406.10735.

    Arguments
    ---------
    in_channels : int
        number of input tensor channels.
    out_channels : int
        number of output tensor channels.
    resblock_type : str
        type of the `ResBlock`. '1' or '2'.
    resblock_dilation_sizes : List[List[int]]
        list of dilation values in each layer of a `ResBlock`.
    resblock_kernel_sizes : List[int]
        list of kernel sizes for each `ResBlock`.
    upsample_kernel_sizes : List[int]
        list of kernel sizes for each transposed convolution.
    upsample_initial_channel : int
        number of channels for the first upsampling layer. This is divided by 2
        for each consecutive upsampling layer.
    upsample_factors : List[int]
        upsampling factors (stride) for each upsampling layer.
    inference_padding : int
        constant padding applied to the input at inference time. Defaults to 5.
    cond_channels : int
        Whether to add a conv to the front
    conv_post_bias : bool
        Whether to add a bias to the last conv
    vocab_size : int
        size of the dictionary of embeddings.
    embedding_dim : int
        size of each embedding vector.
    attn_dim : int
        size of attention dimension.
    duration_predictor : bool
        enable duration predictor module.
    var_pred_hidden_dim : int
        size of hidden channels for the convolutional layers of the duration predictor.
    var_pred_kernel_size : int
        size of the convolution filter in each layer of the duration predictor.
    var_pred_dropout : float
        dropout probability of each layer in the duration predictor.
    multi_speaker : bool
        enable multi speaker training.
    normalize_speaker_embeddings: bool
        enable normalization of speaker embeddings.
    skip_token_embedding: bool
        Whether to skip the embedding layer in the case of continuous input.
    pooling_type: str, optional
        The type of pooling to use. Must be one of ["attention", "sum", "none"].
        Defaults to "attention" for scalable vocoder.

    Example
    -------
    >>> inp_tensor = torch.randint(0, 100, (4, 10, 1))
    >>> unit_hifigan_generator= UnitHifiganGenerator(
    ...    in_channels = 128,
    ...    out_channels = 1,
    ...    resblock_type = "1",
    ...    resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    ...    resblock_kernel_sizes = [3, 7, 11],
    ...    upsample_kernel_sizes = [11, 8, 8, 4, 4],
    ...    upsample_initial_channel = 512,
    ...    upsample_factors = [5, 4, 4, 2, 2],
    ...    vocab_size = 100,
    ...    embedding_dim = 128,
    ...    duration_predictor = True,
    ...    var_pred_hidden_dim = 128,
    ...    var_pred_kernel_size = 3,
    ...    var_pred_dropout = 0.5,
    ... )
    >>> out_tensor, _ = unit_hifigan_generator(inp_tensor)
    >>> out_tensor.shape
    torch.Size([4, 1, 3200])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        resblock_type,
        resblock_dilation_sizes,
        resblock_kernel_sizes,
        upsample_kernel_sizes,
        upsample_initial_channel,
        upsample_factors,
        inference_padding=5,
        cond_channels=0,
        conv_post_bias=True,
        vocab_size=100,
        embedding_dim=128,
        attn_dim=128,
        duration_predictor=False,
        var_pred_hidden_dim=128,
        var_pred_kernel_size=3,
        var_pred_dropout=0.5,
        multi_speaker=False,
        normalize_speaker_embeddings=False,
        skip_token_embedding=False,
        pooling_type="attention",
    ):
        super().__init__(
            in_channels,
            out_channels,
            resblock_type,
            resblock_dilation_sizes,
            resblock_kernel_sizes,
            upsample_kernel_sizes,
            upsample_initial_channel,
            upsample_factors,
            inference_padding,
            cond_channels,
            conv_post_bias,
        )
        self.unit_embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.pooling_type = pooling_type
        if pooling_type == "attention":
            self.attn_pooling = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, attn_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(attn_dim, 1, bias=False),
            )

        self.duration_predictor = duration_predictor
        if duration_predictor:
            self.var_predictor = VariancePredictor(
                embedding_dim,
                var_pred_hidden_dim,
                var_pred_kernel_size,
                var_pred_dropout,
            )
        self.multi_speaker = multi_speaker
        self.normalize_speaker_embeddings = normalize_speaker_embeddings
        self.skip_token_embedding = skip_token_embedding

    @staticmethod
    def _upsample(x, max_frames):
        """
        Upsamples the input tensor to match the specified max_frames.
        """
        batch, hidden_dim, cond_length = x.size()
        x = x.unsqueeze(3).repeat(1, 1, 1, max_frames // cond_length)
        x = x.view(batch, hidden_dim, max_frames)
        return x

    def forward(self, x, g=None, spk=None):
        """
        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            feature input tensor.
        g : torch.Tensor (batch, 1, time)
            global conditioning input tensor.
        spk : torch.Tensor
            Speaker embeddings

        Returns
        -------
        Generator output
        """
        if self.skip_token_embedding:
            u = x
        else:
            u = self.unit_embedding(x)

        batch_size, time, channel, emb_size = u.shape
        u_ = u.view(batch_size * time, channel, emb_size)

        if self.pooling_type == "attention":
            attn_scores = self.attn_pooling(u_)
            attn_weights = F.softmax(attn_scores, dim=1)
            u_weighted = u_ * attn_weights
            u_pooled = torch.sum(u_weighted, dim=1)
        elif self.pooling_type == "sum":
            u_pooled = torch.sum(u_, dim=1)
        elif self.pooling_type == "none":
            u_pooled = u_

        u = u_pooled.view(batch_size, time, emb_size)
        u = u.transpose(1, 2)

        log_dur = None
        log_dur_pred = None

        if self.duration_predictor:
            uniq_code_feat, uniq_code_mask, dur = process_duration(x, u.transpose(1, 2))
            log_dur_pred = self.var_predictor(uniq_code_feat)
            log_dur_pred = log_dur_pred[uniq_code_mask]
            log_dur = torch.log(dur + 1)

        if self.multi_speaker:
            if self.normalize_speaker_embeddings:
                spk = torch.nn.functional.normalize(spk)
            spk = spk.unsqueeze(-1)
            spk = self._upsample(spk, u.shape[-1])
            u = torch.cat([u, spk], dim=1)

        return super().forward(u), (log_dur_pred, log_dur)

    @torch.no_grad()
    def inference(self, x, spk=None):
        """The inference function performs duration prediction and runs the forward method.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            feature input tensor.
        spk : torch.Tensor
            Speaker embeddings

        Returns
        -------
        Generator output
        """
        if not self.skip_token_embedding:
            x = self.unit_embedding(x)

        batch_size, time, channel, emb_size = x.shape
        x_ = x.view(batch_size * time, channel, emb_size)

        if self.pooling_type == "attention":
            attn_scores = self.attn_pooling(x_)
            attn_weights = F.softmax(attn_scores, dim=1)
            x_weighted = x_ * attn_weights
            x_pooled = torch.sum(x_weighted, dim=1)
        elif self.pooling_type == "sum":
            x_pooled = torch.sum(x_, dim=1)
        elif self.pooling_type == "none":
            x_pooled = x_

        x = x_pooled.view(batch_size, time, emb_size)
        x = x.transpose(1, 2)

        if self.duration_predictor:
            assert x.size(0) == 1, "only support single sample batch in inference"
            log_dur_pred = self.var_predictor(x.transpose(1, 2))
            dur_out = torch.clamp(
                torch.round((torch.exp(log_dur_pred) - 1)).long(), min=1
            )
            # B x C x T
            x = torch.repeat_interleave(x, dur_out.view(-1), dim=2)

        if self.multi_speaker:
            if self.normalize_speaker_embeddings:
                spk = torch.nn.functional.normalize(spk)
            spk = spk.unsqueeze(-1)
            spk = self._upsample(spk, x.shape[-1])
            x = torch.cat([x, spk], dim=1)

        return super().forward(x)


##################################
# DISCRIMINATOR
##################################


class DiscriminatorP(torch.nn.Module):
    """HiFiGAN Periodic Discriminator
    Takes every Pth value from the input waveform and applies a stack of convolutions.
    Note:
        if period is 2
        waveform = [1, 2, 3, 4, 5, 6 ...] --> [1, 3, 5 ... ] --> convs -> score, feat

    Arguments
    ---------
    period : int
       Take every a new value every `period`
    kernel_size : int
        Size of 1-d kernel for conv stack
    stride : int
        Stride of conv stack
    """

    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period

        self.convs = nn.ModuleList(
            [
                Conv2d(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
                Conv2d(
                    in_channels=32,
                    out_channels=128,
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
                Conv2d(
                    in_channels=128,
                    out_channels=512,
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
                Conv2d(
                    in_channels=512,
                    out_channels=1024,
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
                Conv2d(
                    in_channels=1024,
                    out_channels=1024,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
            ]
        )
        self.conv_post = Conv2d(
            in_channels=1024,
            out_channels=1,
            kernel_size=(3, 1),
            stride=1,
            padding="same",
            skip_transpose=True,
            weight_norm=True,
        )

    def forward(self, x):
        """
        Arguments
        ---------
        x : torch.Tensor (batch, 1, time)
            input waveform.

        Returns
        -------
        Scores and features
        """

        feat = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            feat.append(x)
        x = self.conv_post(x)
        feat.append(x)
        x = torch.flatten(x, 1, -1)

        return x, feat


class MultiPeriodDiscriminator(torch.nn.Module):
    """HiFiGAN Multi-Period Discriminator (MPD)
    Wrapper for the `PeriodDiscriminator` to apply it in different periods.
    Periods are suggested to be prime numbers to reduce the overlap between each discriminator.
    """

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(2),
                DiscriminatorP(3),
                DiscriminatorP(5),
                DiscriminatorP(7),
                DiscriminatorP(11),
            ]
        )

    def forward(self, x):
        """Returns Multi-Period Discriminator scores and features

        Arguments
        ---------
        x : torch.Tensor (batch, 1, time)
            input waveform.

        Returns
        -------
        Scores and features
        """

        scores = []
        feats = []
        for _, d in enumerate(self.discriminators):
            score, feat = d(x)
            scores.append(score)
            feats.append(feat)
        return scores, feats


class DiscriminatorS(torch.nn.Module):
    """HiFiGAN Scale Discriminator.
    It is similar to `MelganDiscriminator` but with a specific architecture explained in the paper.
    SpeechBrain CNN wrappers are not used here because spectral_norm is not often used

    Arguments
    ---------
    use_spectral_norm : bool
        if `True` switch to spectral norm instead of weight norm.
    """

    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        """
        Arguments
        ---------
        x : torch.Tensor (batch, 1, time)
            input waveform.

        Returns
        -------
        Scores and features
        """

        feat = []
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            feat.append(x)
        x = self.conv_post(x)
        feat.append(x)
        x = torch.flatten(x, 1, -1)
        return x, feat


class MultiScaleDiscriminator(torch.nn.Module):
    """HiFiGAN Multi-Scale Discriminator.
    Similar to MultiScaleMelganDiscriminator but specially tailored for HiFiGAN as in the paper.
    """

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorS(use_spectral_norm=True),
                DiscriminatorS(),
                DiscriminatorS(),
            ]
        )
        self.meanpools = nn.ModuleList(
            [nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)]
        )

    def forward(self, x):
        """
        Arguments
        ---------
        x : torch.Tensor (batch, 1, time)
            input waveform.

        Returns
        -------
        Scores and features
        """

        scores = []
        feats = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                x = self.meanpools[i - 1](x)
            score, feat = d(x)
            scores.append(score)
            feats.append(feat)
        return scores, feats


class HifiganDiscriminator(nn.Module):
    """HiFiGAN discriminator wrapping MPD and MSD.

    Example
    -------
    >>> inp_tensor = torch.rand([4, 1, 8192])
    >>> hifigan_discriminator= HifiganDiscriminator()
    >>> scores, feats = hifigan_discriminator(inp_tensor)
    >>> len(scores)
    8
    >>> len(feats)
    8

    """

    def __init__(self):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

    def forward(self, x):
        """Returns list of list of features from each layer of each discriminator.

        Arguments
        ---------
        x : torch.Tensor
            input waveform.

        Returns
        -------
        Features from each discriminator layer
        """

        scores, feats = self.mpd(x)
        scores_, feats_ = self.msd(x)
        return scores + scores_, feats + feats_


#################################
# GENERATOR LOSSES
#################################


def stft(x, n_fft, hop_length, win_length, window_fn="hann_window"):
    """computes the Fourier transform of short overlapping windows of the input"""
    o = torch.stft(
        x.squeeze(1),
        n_fft,
        hop_length,
        win_length,
    )
    M = o[:, :, :, 0]
    P = o[:, :, :, 1]
    S = torch.sqrt(torch.clamp(M**2 + P**2, min=1e-8))
    return S


class STFTLoss(nn.Module):
    """STFT loss. Input generate and real waveforms are converted
    to spectrograms compared with L1 and Spectral convergence losses.
    It is from ParallelWaveGAN paper https://arxiv.org/pdf/1910.11480.pdf

    Arguments
    ---------
    n_fft : int
        size of Fourier transform.
    hop_length : int
        the distance between neighboring sliding window frames.
    win_length : int
        the size of window frame and STFT filter.
    """

    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, y_hat, y):
        """Returns magnitude loss and spectral convergence loss

        Arguments
        ---------
        y_hat : torch.tensor
            generated waveform tensor
        y : torch.tensor
            real waveform tensor

        Returns
        -------
        Magnitude loss and spectral convergence loss
        """

        y_hat_M = stft(y_hat, self.n_fft, self.hop_length, self.win_length)
        y_M = stft(y, self.n_fft, self.hop_length, self.win_length)
        # magnitude loss
        loss_mag = F.l1_loss(torch.log(y_M), torch.log(y_hat_M))
        # spectral convergence loss
        loss_sc = torch.norm(y_M - y_hat_M, p="fro") / torch.norm(y_M, p="fro")
        return loss_mag, loss_sc


class MultiScaleSTFTLoss(torch.nn.Module):
    """Multi-scale STFT loss. Input generate and real waveforms are converted
    to spectrograms compared with L1 and Spectral convergence losses.
    It is from ParallelWaveGAN paper https://arxiv.org/pdf/1910.11480.pdf"""

    def __init__(
        self,
        n_ffts=(1024, 2048, 512),
        hop_lengths=(120, 240, 50),
        win_lengths=(600, 1200, 240),
    ):
        super().__init__()
        self.loss_funcs = torch.nn.ModuleList()
        for n_fft, hop_length, win_length in zip(n_ffts, hop_lengths, win_lengths):
            self.loss_funcs.append(STFTLoss(n_fft, hop_length, win_length))

    def forward(self, y_hat, y):
        """Returns multi-scale magnitude loss and spectral convergence loss

        Arguments
        ---------
        y_hat : torch.tensor
            generated waveform tensor
        y : torch.tensor
            real waveform tensor

        Returns
        -------
        Magnitude loss and spectral convergence loss
        """

        N = len(self.loss_funcs)
        loss_sc = 0
        loss_mag = 0
        for f in self.loss_funcs:
            lm, lsc = f(y_hat, y)
            loss_mag += lm
            loss_sc += lsc
        loss_sc /= N
        loss_mag /= N
        return loss_mag, loss_sc


class L1SpecLoss(nn.Module):
    """L1 Loss over Spectrograms as described in HiFiGAN paper https://arxiv.org/pdf/2010.05646.pdf
    Note : L1 loss helps leaning details compared with L2 loss

    Arguments
    ---------
    sample_rate : int
        Sample rate of audio signal.
    hop_length : int
        Length of hop between STFT windows.
    win_length : int
        Window size.
    n_mel_channels : int
        Number of mel filterbanks.
    n_fft : int
        Size of FFT.
    n_stft : int
        Size of STFT.
    mel_fmin : float
        Minimum frequency.
    mel_fmax : float
        Maximum frequency.
    mel_normalized : bool
        Whether to normalize by magnitude after stft.
    power : float
        Exponent for the magnitude spectrogram.
    norm : str or None
        If "slaney", divide the triangular mel weights by the width of the mel band
    mel_scale : str
        Scale to use: "htk" or "slaney".
    dynamic_range_compression : bool
        whether to do dynamic range compression
    """

    def __init__(
        self,
        sample_rate=22050,
        hop_length=256,
        win_length=24,
        n_mel_channels=80,
        n_fft=1024,
        n_stft=1024 // 2 + 1,
        mel_fmin=0.0,
        mel_fmax=8000.0,
        mel_normalized=False,
        power=1.0,
        norm="slaney",
        mel_scale="slaney",
        dynamic_range_compression=True,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.n_fft = n_fft
        self.n_stft = n_fft // 2 + 1
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.mel_normalized = mel_normalized
        self.power = power
        self.norm = norm
        self.mel_scale = mel_scale
        self.dynamic_range_compression = dynamic_range_compression

    def forward(self, y_hat, y):
        """Returns L1 Loss over Spectrograms

        Arguments
        ---------
        y_hat : torch.tensor
            generated waveform tensor
        y : torch.tensor
            real waveform tensor

        Returns
        -------
        L1 loss
        """
        y_hat_M = mel_spectogram(
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.n_fft,
            self.n_mel_channels,
            self.mel_fmin,
            self.mel_fmax,
            self.power,
            self.mel_normalized,
            self.norm,
            self.mel_scale,
            self.dynamic_range_compression,
            y_hat,
        )
        # y_M = mel_spectogram(self.mel_params, y)
        y_M = mel_spectogram(
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.n_fft,
            self.n_mel_channels,
            self.mel_fmin,
            self.mel_fmax,
            self.power,
            self.mel_normalized,
            self.norm,
            self.mel_scale,
            self.dynamic_range_compression,
            y,
        )

        # magnitude loss
        # loss_mag = F.l1_loss(torch.log(y_M), torch.log(y_hat_M))
        loss_mag = F.l1_loss(y_M, y_hat_M)
        return loss_mag


class MSEGLoss(nn.Module):
    """Mean Squared Generator Loss
    The generator is trained to fake the discriminator by updating the sample quality
    to be classified to a value almost equal to 1.
    """

    def forward(self, score_fake):
        """Returns Generator GAN loss

        Arguments
        ---------
        score_fake : list
            discriminator scores of generated waveforms D(G(s))

        Returns
        -------
        Generator loss
        """

        loss_fake = F.mse_loss(score_fake, score_fake.new_ones(score_fake.shape))
        return loss_fake


class MelganFeatureLoss(nn.Module):
    """Calculates the feature matching loss, which is a learned similarity metric measured by
    the difference in features of the discriminator between a ground truth sample and a generated
    sample (Larsen et al., 2016, Kumar et al., 2019).
    """

    def __init__(
        self,
    ):
        super().__init__()
        self.loss_func = nn.L1Loss()

    # pylint: disable=no-self-use
    def forward(self, fake_feats, real_feats):
        """Returns feature matching loss

        Arguments
        ---------
        fake_feats : list
            discriminator features of generated waveforms
        real_feats : list
            discriminator features of groundtruth waveforms

        Returns
        -------
        Feature matching loss
        """

        loss_feats = 0
        num_feats = 0
        for idx, _ in enumerate(fake_feats):
            for fake_feat, real_feat in zip(fake_feats[idx], real_feats[idx]):
                loss_feats += self.loss_func(fake_feat, real_feat)
                num_feats += 1
        loss_feats = loss_feats / num_feats
        return loss_feats


##################################
# DISCRIMINATOR LOSSES
##################################


class MSEDLoss(nn.Module):
    """Mean Squared Discriminator Loss
    The discriminator is trained to classify ground truth samples to 1,
    and the samples synthesized from the generator to 0.
    """

    def __init__(
        self,
    ):
        super().__init__()
        self.loss_func = nn.MSELoss()

    def forward(self, score_fake, score_real):
        """Returns Discriminator GAN losses

        Arguments
        ---------
        score_fake : list
            discriminator scores of generated waveforms
        score_real : list
            discriminator scores of groundtruth waveforms

        Returns
        -------
        Discriminator losses
        """

        loss_real = self.loss_func(score_real, score_real.new_ones(score_real.shape))
        loss_fake = self.loss_func(score_fake, score_fake.new_zeros(score_fake.shape))
        loss_d = loss_real + loss_fake
        return loss_d, loss_real, loss_fake


#####################################
# LOSS WRAPPERS
#####################################


def _apply_G_adv_loss(scores_fake, loss_func):
    """Compute Generator adversarial loss function
    and normalize values

    Arguments
    ---------
    scores_fake : list
        discriminator scores of generated waveforms
    loss_func : object
        object of target generator loss

    Returns
    -------
    Generator loss
    """

    adv_loss = 0
    if isinstance(scores_fake, list):
        for score_fake in scores_fake:
            fake_loss = loss_func(score_fake)
            adv_loss += fake_loss
        # adv_loss /= len(scores_fake)
    else:
        fake_loss = loss_func(scores_fake)
        adv_loss = fake_loss
    return adv_loss


def _apply_D_loss(scores_fake, scores_real, loss_func):
    """Compute Discriminator losses and normalize loss values

    Arguments
    ---------
    scores_fake : list
        discriminator scores of generated waveforms
    scores_real : list
        discriminator scores of groundtruth waveforms
    loss_func : object
        object of target discriminator loss

    Returns
    -------
    Discriminator losses
    """

    loss = 0
    real_loss = 0
    fake_loss = 0
    if isinstance(scores_fake, list):
        # multi-scale loss
        for score_fake, score_real in zip(scores_fake, scores_real):
            total_loss, real_loss, fake_loss = loss_func(
                score_fake=score_fake, score_real=score_real
            )
            loss += total_loss
            real_loss += real_loss
            fake_loss += fake_loss
        # normalize loss values with number of scales (discriminators)
        # loss /= len(scores_fake)
        # real_loss /= len(scores_real)
        # fake_loss /= len(scores_fake)
    else:
        # single scale loss
        total_loss, real_loss, fake_loss = loss_func(scores_fake, scores_real)
        loss = total_loss
    return loss, real_loss, fake_loss


##################################
# MODEL LOSSES
##################################


class GeneratorLoss(nn.Module):
    """Creates a summary of generator losses
    and applies weights for different losses

    Arguments
    ---------
    stft_loss : object
        object of stft loss
    stft_loss_weight : float
        weight of STFT loss
    mseg_loss : object
        object of mseg loss
    mseg_loss_weight : float
        weight of mseg loss
    feat_match_loss : object
        object of feature match loss
    feat_match_loss_weight : float
        weight of feature match loss
    l1_spec_loss : object
        object of L1 spectrogram loss
    l1_spec_loss_weight : float
        weight of L1 spectrogram loss
    mseg_dur_loss : object
        object of mseg duration loss
    mseg_dur_loss_weight : float
        weight of mseg duration loss
    """

    def __init__(
        self,
        stft_loss=None,
        stft_loss_weight=0,
        mseg_loss=None,
        mseg_loss_weight=0,
        feat_match_loss=None,
        feat_match_loss_weight=0,
        l1_spec_loss=None,
        l1_spec_loss_weight=0,
        mseg_dur_loss=None,
        mseg_dur_loss_weight=0,
    ):
        super().__init__()
        self.stft_loss = stft_loss
        self.stft_loss_weight = stft_loss_weight
        self.mseg_loss = mseg_loss
        self.mseg_loss_weight = mseg_loss_weight
        self.feat_match_loss = feat_match_loss
        self.feat_match_loss_weight = feat_match_loss_weight
        self.l1_spec_loss = l1_spec_loss
        self.l1_spec_loss_weight = l1_spec_loss_weight
        self.mseg_dur_loss = mseg_dur_loss
        self.mseg_dur_loss_weight = mseg_dur_loss_weight

    def forward(
        self,
        stage,
        y_hat=None,
        y=None,
        scores_fake=None,
        feats_fake=None,
        feats_real=None,
        log_dur_pred=None,
        log_dur=None,
    ):
        """Returns a dictionary of generator losses and applies weights

        Arguments
        ---------
        stage : speechbrain.Stage
            training, validation or testing
        y_hat : torch.tensor
            generated waveform tensor
        y : torch.tensor
            real waveform tensor
        scores_fake : list
            discriminator scores of generated waveforms
        feats_fake : list
            discriminator features of generated waveforms
        feats_real : list
            discriminator features of groundtruth waveforms
        log_dur_pred : torch.Tensor
            Predicted duration for duration loss
        log_dur : torch.Tensor
            Real duration for duration loss

        Returns
        -------
        Dictionary of generator losses
        """

        gen_loss = 0
        adv_loss = 0
        dur_loss = 0
        loss = {}

        # STFT Loss
        if self.stft_loss:
            stft_loss_mg, stft_loss_sc = self.stft_loss(
                y_hat[:, :, : y.size(2)].squeeze(1), y.squeeze(1)
            )
            loss["G_stft_loss_mg"] = stft_loss_mg
            loss["G_stft_loss_sc"] = stft_loss_sc
            gen_loss = gen_loss + self.stft_loss_weight * (stft_loss_mg + stft_loss_sc)

        # L1 Spec loss
        if self.l1_spec_loss:
            l1_spec_loss = self.l1_spec_loss(y_hat, y)
            loss["G_l1_spec_loss"] = l1_spec_loss
            gen_loss = gen_loss + self.l1_spec_loss_weight * l1_spec_loss

        # multiscale MSE adversarial loss
        if self.mseg_loss and scores_fake is not None:
            mse_fake_loss = _apply_G_adv_loss(scores_fake, self.mseg_loss)
            loss["G_mse_fake_loss"] = mse_fake_loss
            adv_loss = adv_loss + self.mseg_loss_weight * mse_fake_loss

        # Feature Matching Loss
        if self.feat_match_loss and feats_fake is not None:
            feat_match_loss = self.feat_match_loss(feats_fake, feats_real)
            loss["G_feat_match_loss"] = feat_match_loss
            adv_loss = adv_loss + self.feat_match_loss_weight * feat_match_loss

        # Duration loss
        if self.mseg_dur_loss and stage == sb.Stage.TRAIN:
            dur_loss = F.mse_loss(log_dur_pred, log_dur, reduction="mean")
            loss["G_dur_loss"] = dur_loss
            dur_loss *= self.mseg_dur_loss_weight

        loss["G_loss"] = gen_loss + adv_loss + dur_loss
        loss["G_gen_loss"] = gen_loss
        loss["G_adv_loss"] = adv_loss

        return loss


class DiscriminatorLoss(nn.Module):
    """Creates a summary of discriminator losses

    Arguments
    ---------
    msed_loss : object
        object of MSE discriminator loss
    """

    def __init__(self, msed_loss=None):
        super().__init__()
        self.msed_loss = msed_loss

    def forward(self, scores_fake, scores_real):
        """Returns a dictionary of discriminator losses

        Arguments
        ---------
        scores_fake : list
            discriminator scores of generated waveforms
        scores_real : list
            discriminator scores of groundtruth waveforms

        Returns
        -------
        Dictionary of discriminator losses
        """

        disc_loss = 0
        loss = {}

        if self.msed_loss:
            mse_D_loss, mse_D_real_loss, mse_D_fake_loss = _apply_D_loss(
                scores_fake=scores_fake,
                scores_real=scores_real,
                loss_func=self.msed_loss,
            )
            loss["D_mse_gan_loss"] = mse_D_loss
            loss["D_mse_gan_real_loss"] = mse_D_real_loss
            loss["D_mse_gan_fake_loss"] = mse_D_fake_loss
            disc_loss += mse_D_loss

        loss["D_loss"] = disc_loss
        return loss
