"""
Neural network modules for DIFFWAVE:
A VERSATILE DIFFUSION MODEL FOR AUDIO SYNTHESIS

For more details: https://arxiv.org/pdf/2009.09761.pdf

Authors
 * Yingzhi WANG 2022
"""

# This code uses a significant portion of the LMNT implementation, even though it
# has been modified and enhanced

# https://github.com/lmnt-com/diffwave/blob/master/src/diffwave/model.py
# *****************************************************************************
# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet import linear
from speechbrain.nnet.diffusion import DenoisingDiffusion
from math import sqrt
from torchaudio import transforms


Linear = linear.Linear
ConvTranspose2d = nn.ConvTranspose2d


@torch.jit.script
def silu(x):
    """sigmoid linear unit activation function
    """
    return x * torch.sigmoid(x)


def diffwave_mel_spectogram(
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
    audio,
):
    """calculates MelSpectrogram for a raw audio signal
    and preprocesses it for diffwave training

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
    audio : torch.tensor
        input audio signal
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

    mel = audio_to_mel(torch.clamp(audio, -1.0, 1.0))
    mel = 20 * torch.log10(torch.clamp(mel, min=1e-5)) - 20
    mel = torch.clamp((mel + 100) / 100, 0.0, 1.0)
    return mel


class DiffusionEmbedding(nn.Module):
    """Embeds the diffusion step into an input vector of DiffWave

    Arguments
    ---------
    max_steps: int
        total difussion steps

    Example
    -------
    >>> from speechbrain.lobes.models.DiffWave import DiffusionEmbedding
    >>> diffusion_embedding = DiffusionEmbedding(max_steps=50)
    >>> time_step = torch.randint(50, (1,))
    >>> step_embedding = diffusion_embedding(time_step)
    >>> step_embedding.shape
    torch.Size([1, 512])
    """

    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(max_steps), persistent=False
        )
        self.projection1 = Linear(input_size=128, n_neurons=512)
        self.projection2 = Linear(input_size=512, n_neurons=512)

    def forward(self, diffusion_step):
        """forward function of diffusion step embedding

        Arguments
        ---------
        diffusion_step: torch.Tensor
            which step of diffusion to execute
        Returns
        -------
        diffusion step embedding: tensor [bs, 512]
        """
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        """Deals with the cases where diffusion_step is not int

        Arguments
        ---------
        t: torch.Tensor
            which step of diffusion to execute
        """
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        """Build embeddings in a designed way

        Arguments
        ---------
        max_steps: int
            total diffusion steps
        """
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)  # [1,64]
        table = steps * 10.0 ** (dims * 4.0 / 63.0)  # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class SpectrogramUpsampler(nn.Module):
    """Upsampler for spectrograms with Transposed Conv
    Only the upsamling is done here, the layer-specific Conv can be found
    in residual bloack to map the mel bands into 2× residual channels

    Example
    -------
    >>> from speechbrain.lobes.models.DiffWave import SpectrogramUpsampler
    >>> spec_upsampler = SpectrogramUpsampler()
    >>> mel_input = torch.rand(3, 80, 100)
    >>> upsampled_mel = spec_upsampler(mel_input)
    >>> upsampled_mel.shape
    torch.Size([3, 80, 25600])
    """

    def __init__(self):
        super().__init__()
        self.conv1 = ConvTranspose2d(
            1, 1, [3, 32], stride=[1, 16], padding=[1, 8]
        )
        self.conv2 = ConvTranspose2d(
            1, 1, [3, 32], stride=[1, 16], padding=[1, 8]
        )

    def forward(self, x):
        """Upsamples spectrograms 256 times to match the length of audios
        Hop length should be 256 when extracting mel spectrograms

        Arguments
        ---------
        x: torch.Tensor
            input mel spectrogram [bs, 80, mel_len]

        Returns
        -------
        upsampled spectrogram [bs, 80, mel_len*256]
        """
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        x = torch.squeeze(x, 1)
        return x


class ResidualBlock(nn.Module):
    """
    Residual Block with dilated convolution

    Arguments
    ---------
    n_mels:
        input mel channels of conv1x1 for conditional vocoding task
    residual_channels:
        channels of audio convolution
    dilation:
        dilation cycles of audio convolution
    uncond:
        conditional/unconditional generation

    Example
    -------
    >>> from speechbrain.lobes.models.DiffWave import ResidualBlock
    >>> res_block = ResidualBlock(n_mels=80, residual_channels=64, dilation=3)
    >>> noisy_audio = torch.randn(1, 1, 22050)
    >>> timestep_embedding = torch.rand(1, 512)
    >>> upsampled_mel = torch.rand(1, 80, 22050)
    >>> output = res_block(noisy_audio, timestep_embedding, upsampled_mel)
    >>> output[0].shape
    torch.Size([1, 64, 22050])
    """

    def __init__(self, n_mels, residual_channels, dilation, uncond=False):
        super().__init__()
        self.dilated_conv = Conv1d(
            in_channels=residual_channels,
            out_channels=2 * residual_channels,
            kernel_size=3,
            dilation=dilation,
            skip_transpose=True,
            padding="same",
            conv_init="kaiming",
        )
        self.diffusion_projection = Linear(
            input_size=512, n_neurons=residual_channels
        )

        # conditional model
        if not uncond:
            self.conditioner_projection = Conv1d(
                in_channels=n_mels,
                out_channels=2 * residual_channels,
                kernel_size=1,
                skip_transpose=True,
                padding="same",
                conv_init="kaiming",
            )
        # unconditional model
        else:
            self.conditioner_projection = None

        self.output_projection = Conv1d(
            in_channels=residual_channels,
            out_channels=2 * residual_channels,
            kernel_size=1,
            skip_transpose=True,
            padding="same",
            conv_init="kaiming",
        )

    def forward(self, x, diffusion_step, conditioner=None):
        """
        forward function of Residual Block

        Arguments
        ---------
        x: torch.Tensor
            input sample [bs, 1, time]
        diffusion_step: torch.Tensor
            the embedding of which step of diffusion to execute
        conditioner: torch.Tensor
            the condition used for conditional generation
        Returns
        -------
        residual output [bs, residual_channels, time]
        a skip of residual branch [bs, residual_channels, time]
        """
        assert (
            conditioner is None and self.conditioner_projection is None
        ) or (
            conditioner is not None and self.conditioner_projection is not None
        )

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step
        if self.conditioner_projection is None:  # using a unconditional model
            y = self.dilated_conv(y)
        else:
            conditioner = self.conditioner_projection(conditioner)
            # for inference make sure that they have the same length
            # conditioner = conditioner[:, :, y.shape[-1]]
            y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class DiffWave(nn.Module):
    """
    DiffWave Model with dilated residual blocks

    Arguments
    ---------
    input_channels:
        input mel channels of conv1x1 for conditional vocoding task
    residual_layers:
        number of residual blocks
    residual_channels:
        channels of audio convolution
    dilation_cycle_length:
        dilation cycles of audio convolution
    total_steps:
        total steps of diffusion
    unconditional:
        conditional/unconditional generation

    Example
    -------
    >>> from speechbrain.lobes.models.DiffWave import DiffWave
    >>> diffwave = DiffWave(
    ...     input_channels=80,
    ...     residual_layers=30,
    ...     residual_channels=64,
    ...     dilation_cycle_length=10,
    ...     total_steps=50,
    ... )
    >>> noisy_audio = torch.randn(1, 1, 25600)
    >>> timestep = torch.randint(50, (1,))
    >>> input_mel = torch.rand(1, 80, 100)
    >>> predicted_noise = diffwave(noisy_audio, timestep, input_mel)
    >>> predicted_noise.shape
    torch.Size([1, 1, 25600])
    """

    def __init__(
        self,
        input_channels,
        residual_layers,
        residual_channels,
        dilation_cycle_length,
        total_steps,
        unconditional=False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.residual_layers = residual_layers
        self.residual_channels = residual_channels
        self.dilation_cycle_length = dilation_cycle_length
        self.unconditional = unconditional
        self.total_steps = total_steps
        self.input_projection = Conv1d(
            in_channels=1,
            out_channels=self.residual_channels,
            kernel_size=1,
            skip_transpose=True,
            padding="same",
            conv_init="kaiming",
        )
        self.diffusion_embedding = DiffusionEmbedding(self.total_steps)

        if self.unconditional:  # use unconditional model
            self.spectrogram_upsampler = None
        else:
            self.spectrogram_upsampler = SpectrogramUpsampler()

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    self.input_channels,
                    self.residual_channels,
                    2 ** (i % self.dilation_cycle_length),
                    uncond=self.unconditional,
                )
                for i in range(self.residual_layers)
            ]
        )
        self.skip_projection = Conv1d(
            in_channels=self.residual_channels,
            out_channels=self.residual_channels,
            kernel_size=1,
            skip_transpose=True,
            padding="same",
            conv_init="kaiming",
        )
        self.output_projection = Conv1d(
            in_channels=self.residual_channels,
            out_channels=1,
            kernel_size=1,
            skip_transpose=True,
            padding="same",
            conv_init="zero",
        )

    def forward(self, audio, diffusion_step, spectrogram=None, length=None):
        """
        DiffWave forward function

        Arguments
        ---------
        audio: torch.Tensor
            input gaussian sample [bs, 1, time]
        diffusion_steps: torch.Tensor
            which timestep of diffusion to execute [bs, 1]
        spectrogram: torch.Tensor
            spectrogram data [bs, 80, mel_len]
        length: torch.Tensor
            sample lengths - not used - provided for compatibility only
        Returns
        -------
        predicted noise [bs, 1, time]
        """
        assert (spectrogram is None and self.spectrogram_upsampler is None) or (
            spectrogram is not None and self.spectrogram_upsampler is not None
        )

        x = self.input_projection(audio)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        if self.spectrogram_upsampler:  # use conditional model
            spectrogram = self.spectrogram_upsampler(spectrogram)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, spectrogram)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x


class DiffWaveDiffusion(DenoisingDiffusion):
    """An enhanced diffusion implementation with DiffWave-specific inference

    Arguments
    ---------
    model: nn.Module
        the underlying model
    timesteps: int
        the total number of timesteps
    noise: str|nn.Module
        the type of noise being used
        "gaussian" will produce standard Gaussian noise
    beta_start: float
        the value of the "beta" parameter at the beginning of the process
        (see DiffWave paper)
    beta_end: float
        the value of the "beta" parameter at the end of the process
    show_progress: bool
        whether to show progress during inference

    Example
    -------
    >>> from speechbrain.lobes.models.DiffWave import DiffWave
    >>> diffwave = DiffWave(
    ...     input_channels=80,
    ...     residual_layers=30,
    ...     residual_channels=64,
    ...     dilation_cycle_length=10,
    ...     total_steps=50,
    ... )
    >>> from speechbrain.lobes.models.DiffWave import DiffWaveDiffusion
    >>> from speechbrain.nnet.diffusion import GaussianNoise
    >>> diffusion = DiffWaveDiffusion(
    ...     model=diffwave,
    ...     beta_start=0.0001,
    ...     beta_end=0.05,
    ...     timesteps=50,
    ...     noise=GaussianNoise,
    ... )
    >>> input_mel = torch.rand(1, 80, 100)
    >>> output = diffusion.inference(
    ...     unconditional=False,
    ...     scale=256,
    ...     condition=input_mel,
    ...     fast_sampling=True,
    ...     fast_sampling_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],
    ... )
    >>> output.shape
    torch.Size([1, 25600])
    """

    def __init__(
        self,
        model,
        timesteps=None,
        noise=None,
        beta_start=None,
        beta_end=None,
        sample_min=None,
        sample_max=None,
        show_progress=False,
    ):
        super().__init__(
            model,
            timesteps,
            noise,
            beta_start,
            beta_end,
            sample_min,
            sample_max,
            show_progress,
        )

    @torch.no_grad()
    def inference(
        self,
        unconditional,
        scale,
        condition=None,
        fast_sampling=False,
        fast_sampling_noise_schedule=None,
        device=None,
    ):
        """Processes the inference for diffwave
        One inference function for all the locally/globally conditional
        generation and unconditional generation tasks
        Arguments
        ---------
        unconditional: bool
            do unconditional generation if True, else do conditional generation
        scale: int
            scale to get the final output wave length
            for conditional genration, the output wave length is scale * condition.shape[-1]
            for example, if the condition is spectrogram (bs, n_mel, time), scale should be hop length
            for unconditional generation, scale should be the desired audio length
        condition: torch.Tensor
            input spectrogram for vocoding or other conditions for other
            conditional generation, should be None for unconditional generation
        fast_sampling: bool
            whether to do fast sampling
        fast_sampling_noise_schedule: list
            the noise schedules used for fast sampling
        device: str|torch.device
            inference device
        Returns
        ---------
        predicted_sample: torch.Tensor
            the predicted audio (bs, 1, t)
        """
        if device is None:
            device = torch.device("cuda")
        # either condition or uncondition
        if unconditional:
            assert condition is None
        else:
            assert condition is not None
            device = condition.device

        # must define fast_sampling_noise_schedule during fast sampling
        if fast_sampling:
            assert fast_sampling_noise_schedule is not None

        if fast_sampling and fast_sampling_noise_schedule is not None:
            inference_noise_schedule = fast_sampling_noise_schedule
            inference_alphas = 1 - torch.tensor(inference_noise_schedule)
            inference_alpha_cum = inference_alphas.cumprod(dim=0)
        else:
            inference_noise_schedule = self.betas
            inference_alphas = self.alphas
            inference_alpha_cum = self.alphas_cumprod

        inference_steps = []
        for s in range(len(inference_noise_schedule)):
            for t in range(self.timesteps - 1):
                if (
                    self.alphas_cumprod[t + 1]
                    <= inference_alpha_cum[s]
                    <= self.alphas_cumprod[t]
                ):
                    twiddle = (
                        self.alphas_cumprod[t] ** 0.5
                        - inference_alpha_cum[s] ** 0.5
                    ) / (
                        self.alphas_cumprod[t] ** 0.5
                        - self.alphas_cumprod[t + 1] ** 0.5
                    )
                    inference_steps.append(t + twiddle)
                    break

        if not unconditional:
            if (
                len(condition.shape) == 2
            ):  # Expand rank 2 tensors by adding a batch dimension.
                condition = condition.unsqueeze(0)
            audio = torch.randn(
                condition.shape[0], scale * condition.shape[-1], device=device,
            )
        else:
            audio = torch.randn(1, scale, device=device)
        # noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)

        for n in range(len(inference_alphas) - 1, -1, -1):
            c1 = 1 / inference_alphas[n] ** 0.5
            c2 = (
                inference_noise_schedule[n]
                / (1 - inference_alpha_cum[n]) ** 0.5
            )
            # predict noise
            noise_pred = self.model(
                audio,
                torch.tensor([inference_steps[n]], device=device),
                condition,
            ).squeeze(1)
            # mean
            audio = c1 * (audio - c2 * noise_pred)
            # add variance
            if n > 0:
                noise = torch.randn_like(audio)
                sigma = (
                    (1.0 - inference_alpha_cum[n - 1])
                    / (1.0 - inference_alpha_cum[n])
                    * inference_noise_schedule[n]
                ) ** 0.5
                audio += sigma * noise
            audio = torch.clamp(audio, -1.0, 1.0)
        return audio
