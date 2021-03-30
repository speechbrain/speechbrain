"""The SpeechBrain implementation of WaveNet by
https://arxiv.org/pdf/1609.03499.pdf
Inspired by:
https://github.com/r9y9/wavenet_vocoder 
"""

from __future__ import with_statement, print_function, absolute_import

import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from modules import Embedding

from modules import Conv1d1x1, ResidualConv1dGLU, ConvTranspose2d
from mixture import sample_from_discretized_mix_logistic
from mixture import sample_from_mix_gaussian
from wavenet_vocoder import upsample


def _expand_global_features(B, T, g, bct=True):
    """Expand global conditioning features to all time steps

    Args:
        B (int): Batch size.
        T (int): Time length.
        g (Tensor): Global features, (B x C) or (B x C x 1).
        bct (bool) : returns (B x C x T) if True, otherwise (B x T x C)

    Returns:
        Tensor: B x C x T or B x T x C or None
    """
    if g is None:
        return None
    g = g.unsqueeze(-1) if g.dim() == 2 else g
    if bct:
        g_bct = g.expand(B, -1, T)
        return g_bct.contiguous()
    else:
        g_btc = g.expand(B, -1, T).transpose(1, 2)
        return g_btc.contiguous()


def receptive_field_size(total_layers, num_cycles, kernel_size,
                         dilation=lambda x: 2**x):
    """Compute receptive field size

    Args:
        total_layers (int): total layers
        num_cycles (int): cycles
        kernel_size (int): kernel size
        dilation (lambda): lambda to compute dilation factor. ``lambda x : 1``
          to disable dilated convolution.

    Returns:
        int: receptive field size in sample

    """
    assert total_layers % num_cycles == 0
    layers_per_cycle = total_layers // num_cycles
    dilations = [dilation(i % layers_per_cycle) for i in range(total_layers)]
    return (kernel_size - 1) * sum(dilations) + 1


class WaveNet(nn.Module):
    """The WaveNet model that supports local and global conditioning.

    Args:
        out_channels (int): Output channels. If input_type is mu-law quantized
          one-hot vecror. this must equal to the quantize channels. Other wise
          num_mixtures x 3 (pi, mu, log_scale).
        layers (int): Number of total layers
        stacks (int): Number of dilation cycles
        residual_channels (int): Residual input / output channels
        gate_channels (int): Gated activation channels.
        skip_out_channels (int): Skip connection channels.
        kernel_size (int): Kernel size of convolution layers.
        dropout (float): Dropout probability.
        cin_channels (int): Local conditioning channels. If negative value is
          set, local conditioning is disabled.
        gin_channels (int): Global conditioning channels. If negative value is
          set, global conditioning is disabled.
        n_speakers (int): Number of speakers. Used only if global conditioning
          is enabled.
        upsample_conditional_features (bool): Whether upsampling local
          conditioning features by transposed convolution layers or not.
        upsample_scales (list): List of upsample scale.
          ``np.prod(upsample_scales)`` must equal to hop size. Used only if
          upsample_conditional_features is enabled.
        freq_axis_kernel_size (int): Freq-axis kernel_size for transposed
          convolution layers for upsampling. If you only care about time-axis
          upsampling, set this to 1.
        scalar_input (Bool): If True, scalar input ([-1, 1]) is expected, otherwise
          quantized one-hot vector is expected.
        use_speaker_embedding (Bool): Use speaker embedding or Not. Set to False
          if you want to disable embedding layer and use external features
          directly.
    """

    def __init__(self, out_channels=256, layers=20, stacks=2,
                 residual_channels=512,
                 gate_channels=512,
                 skip_out_channels=512,
                 kernel_size=3, dropout=1 - 0.95,
                 cin_channels=-1, gin_channels=-1, n_speakers=None,
                 upsample_conditional_features=False,
                 upsample_net="ConvInUpsampleNetwork",
                 upsample_params={"upsample_scales": [4, 4, 4, 4]},
                 scalar_input=False,
                 use_speaker_embedding=False,
                 output_distribution="Logistic",
                 cin_pad=0,
                 ):
        super(WaveNet, self).__init__()
        self.scalar_input = scalar_input
        self.out_channels = out_channels
        self.cin_channels = cin_channels
        self.output_distribution = output_distribution
        assert layers % stacks == 0
        layers_per_stack = layers // stacks
        if scalar_input:
            self.first_conv = Conv1d1x1(1, residual_channels)
        else:
            self.first_conv = Conv1d1x1(out_channels, residual_channels)

        self.conv_layers = nn.ModuleList()
        for layer in range(layers):
            dilation = 2**(layer % layers_per_stack)
            conv = ResidualConv1dGLU(
                residual_channels, gate_channels,
                kernel_size=kernel_size,
                skip_out_channels=skip_out_channels,
                bias=True,  # magenda uses bias, but musyoku doesn't
                dilation=dilation, dropout=dropout,
                cin_channels=cin_channels,
                gin_channels=gin_channels)
            self.conv_layers.append(conv)
        self.last_conv_layers = nn.ModuleList([
            nn.ReLU(inplace=True),
            Conv1d1x1(skip_out_channels, skip_out_channels),
            nn.ReLU(inplace=True),
            Conv1d1x1(skip_out_channels, out_channels),
        ])

        if gin_channels > 0 and use_speaker_embedding:
            assert n_speakers is not None
            self.embed_speakers = Embedding(
                n_speakers, gin_channels, padding_idx=None, std=0.1)
        else:
            self.embed_speakers = None

        # Upsample conv net
        if upsample_conditional_features:
            self.upsample_net = getattr(upsample, upsample_net)(**upsample_params)
        else:
            self.upsample_net = None

        self.receptive_field = receptive_field_size(layers, stacks, kernel_size)

    def has_speaker_embedding(self):
        return self.embed_speakers is not None

    def local_conditioning_enabled(self):
        return self.cin_channels > 0

    def forward(self, x, c=None, g=None, softmax=False):
        """Forward step

        Args:
            x (Tensor): One-hot encoded audio signal, shape (B x C x T)
            c (Tensor): Local conditioning features,
              shape (B x cin_channels x T)
            g (Tensor): Global conditioning features,
              shape (B x gin_channels x 1) or speaker Ids of shape (B x 1).
              Note that ``self.use_speaker_embedding`` must be False when you
              want to disable embedding layer and use external features
              directly (e.g., one-hot vector).
              Also type of input tensor must be FloatTensor, not LongTensor
              in case of ``self.use_speaker_embedding`` equals False.
            softmax (bool): Whether applies softmax or not.

        Returns:
            Tensor: output, shape B x out_channels x T
        """
        B, _, T = x.size()

        if g is not None:
            if self.embed_speakers is not None:
                # (B x 1) -> (B x 1 x gin_channels)
                g = self.embed_speakers(g.view(B, -1))
                # (B x gin_channels x 1)
                g = g.transpose(1, 2)
                assert g.dim() == 3
        # Expand global conditioning features to all time steps
        g_bct = _expand_global_features(B, T, g, bct=True)

        if c is not None and self.upsample_net is not None:
            c = self.upsample_net(c)
            assert c.size(-1) == x.size(-1)

        # Feed data to network
        x = self.first_conv(x)
        skips = 0
        for f in self.conv_layers:
            x, h = f(x, c, g_bct)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_layers))

        x = skips
        for f in self.last_conv_layers:
            x = f(x)

        x = F.softmax(x, dim=1) if softmax else x

        return x

    def incremental_forward(self, initial_input=None, c=None, g=None,
                            T=100, test_inputs=None,
                            tqdm=lambda x: x, softmax=True, quantize=True,
                            log_scale_min=-50.0):
        """Incremental forward step

        Due to linearized convolutions, inputs of shape (B x C x T) are reshaped
        to (B x T x C) internally and fed to the network for each time step.
        Input of each time step will be of shape (B x 1 x C).

        Args:
            initial_input (Tensor): Initial decoder input, (B x C x 1)
            c (Tensor): Local conditioning features, shape (B x C' x T)
            g (Tensor): Global conditioning features, shape (B x C'' or B x C''x 1)
            T (int): Number of time steps to generate.
            test_inputs (Tensor): Teacher forcing inputs (for debugging)
            tqdm (lamda) : tqdm
            softmax (bool) : Whether applies softmax or not
            quantize (bool): Whether quantize softmax output before feeding the
              network output to input for the next time step. TODO: rename
            log_scale_min (float):  Log scale minimum value.

        Returns:
            Tensor: Generated one-hot encoded samples. B x C x T　
              or scaler vector B x 1 x T
        """
        self.clear_buffer()
        B = 1

        # Note: shape should be **(B x T x C)**, not (B x C x T) opposed to
        # batch forward due to linealized convolution
        if test_inputs is not None:
            if self.scalar_input:
                if test_inputs.size(1) == 1:
                    test_inputs = test_inputs.transpose(1, 2).contiguous()
            else:
                if test_inputs.size(1) == self.out_channels:
                    test_inputs = test_inputs.transpose(1, 2).contiguous()

            B = test_inputs.size(0)
            if T is None:
                T = test_inputs.size(1)
            else:
                T = max(T, test_inputs.size(1))
        # cast to int in case of numpy.int64...
        T = int(T)

        # Global conditioning
        if g is not None:
            if self.embed_speakers is not None:
                g = self.embed_speakers(g.view(B, -1))
                # (B x gin_channels, 1)
                g = g.transpose(1, 2)
                assert g.dim() == 3
        g_btc = _expand_global_features(B, T, g, bct=False)

        # Local conditioning
        if c is not None:
            B = c.shape[0]
            if self.upsample_net is not None:
                c = self.upsample_net(c)
                assert c.size(-1) == T
            if c.size(-1) == T:
                c = c.transpose(1, 2).contiguous()

        outputs = []
        if initial_input is None:
            if self.scalar_input:
                initial_input = torch.zeros(B, 1, 1)
            else:
                initial_input = torch.zeros(B, 1, self.out_channels)
                initial_input[:, :, 127] = 1  # TODO: is this ok?
            # https://github.com/pytorch/pytorch/issues/584#issuecomment-275169567
            if next(self.parameters()).is_cuda:
                initial_input = initial_input.cuda()
        else:
            if initial_input.size(1) == self.out_channels:
                initial_input = initial_input.transpose(1, 2).contiguous()

        current_input = initial_input

        for t in tqdm(range(T)):
            if test_inputs is not None and t < test_inputs.size(1):
                current_input = test_inputs[:, t, :].unsqueeze(1)
            else:
                if t > 0:
                    current_input = outputs[-1]

            # Conditioning features for single time step
            ct = None if c is None else c[:, t, :].unsqueeze(1)
            gt = None if g is None else g_btc[:, t, :].unsqueeze(1)

            x = current_input
            x = self.first_conv.incremental_forward(x)
            skips = 0
            for f in self.conv_layers:
                x, h = f.incremental_forward(x, ct, gt)
                skips += h
            skips *= math.sqrt(1.0 / len(self.conv_layers))
            x = skips
            for f in self.last_conv_layers:
                try:
                    x = f.incremental_forward(x)
                except AttributeError:
                    x = f(x)

            # Generate next input by sampling
            if self.scalar_input:
                if self.output_distribution == "Logistic":
                    x = sample_from_discretized_mix_logistic(
                        x.view(B, -1, 1), log_scale_min=log_scale_min)
                elif self.output_distribution == "Normal":
                    x = sample_from_mix_gaussian(
                        x.view(B, -1, 1), log_scale_min=log_scale_min)
                else:
                    assert False
            else:
                x = F.softmax(x.view(B, -1), dim=1) if softmax else x.view(B, -1)
                if quantize:
                    dist = torch.distributions.OneHotCategorical(x)
                    x = dist.sample()
            outputs += [x.data]
        # T x B x C
        outputs = torch.stack(outputs)
        # B x C x T
        outputs = outputs.transpose(0, 1).transpose(1, 2).contiguous()

        self.clear_buffer()
        return outputs

    def clear_buffer(self):
        self.first_conv.clear_buffer()
        for f in self.conv_layers:
            f.clear_buffer()
        for f in self.last_conv_layers:
            try:
                f.clear_buffer()
            except AttributeError:
                pass

    def make_generation_fast_(self):
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return
        self.apply(remove_weight_norm)
