"""
Neural network modules for DIFFWAVE: A VERSATILE DIFFUSION MODEL FOR
AUDIO SYNTHESIS

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

from math import sqrt


Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    """sigmoid linear unit activation function
    """
    return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
    """Embeds the diffusion step into an input vector of DiffWave
    
    Arguments
    ---------
    max_steps: int
        total difussion steps
    """
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        self.projection1 = Linear(128, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, diffusion_step):
        """forward function of diffusion step embedding

        Arguments
        ---------
        diffusion_step:
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
        t: which step of diffusion to execute
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
            total difussion steps
        """
        steps = torch.arange(max_steps).unsqueeze(1) # [T,1]
        dims = torch.arange(64).unsqueeze(0) # [1,64]
        table = steps * 10.0**(dims * 4.0 / 63.0) # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class SpectrogramUpsampler(nn.Module):
    """Upsampler for spectrograms with Transposed Conv
    Only the upsamling is done here, the layer-specific Conv can be found
    in residual bloack to map the mel bands into 2× residual channels
    """
    def __init__(self):
        super().__init__()
        self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
        self.conv2 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])

    def forward(self, x):
        """Upsamples spectrograms 256 times to match the length of audios
        Hop length should be 256 when extracting mel spectrograms

        Arguments
        ---------
        x: torch.tensor
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
    """
    def __init__(
        self,
        n_mels,
        residual_channels,
        dilation,
        uncond=False
    ):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(512, residual_channels)
        if not uncond: # conditional model
            self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)
        else: # unconditional model
            self.conditioner_projection = None

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, conditioner=None):
        """
        forward function of Residual Block

        Arguments
        ---------
        x: torch.tensor
            input sample [bs, 1, time]
        diffusion_step: int
            which step of diffusion to execute
        conditioner: torch.tensor
            the condition used for conditional generation
        Returns
        -------
        residual output [bs, residual_channels, time]
        a skip of residual branch [bs, residual_channels, time]
        """
        assert (conditioner is None and self.conditioner_projection is None) or \
                     (conditioner is not None and self.conditioner_projection is not None)

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step
        if self.conditioner_projection is None: # using a unconditional model
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
        self.input_projection = Conv1d(1, self.residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(self.total_steps)

        if self.unconditional: # use unconditional model
            self.spectrogram_upsampler = None
        else:
            self.spectrogram_upsampler = SpectrogramUpsampler()

        self.residual_layers = nn.ModuleList([
                ResidualBlock(self.input_channels, self.residual_channels, 2**(i % self.dilation_cycle_length), uncond=self.unconditional)
                for i in range(self.residual_layers)
        ])
        self.skip_projection = Conv1d(self.residual_channels, self.residual_channels, 1)
        self.output_projection = Conv1d(self.residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, audio, diffusion_step, spectrogram=None):
        """
        DiffWave forward function

        Arguments
        ---------
        audio:
            input gaussian sample
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
        Returns
        -------
        predicted noise [bs, 1, time]
        """
        assert (spectrogram is None and self.spectrogram_upsampler is None) or \
                     (spectrogram is not None and self.spectrogram_upsampler is not None)

        x = self.input_projection(audio)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        if self.spectrogram_upsampler: # use conditional model
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