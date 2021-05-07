"""The SpeechBrain implementation of WaveNet
https://arxiv.org/pdf/1609.03499.pdf
Inspired by:
https://github.com/r9y9/wavenet_vocoder 

Written by:
Aleksandar Rachkov, 2021
"""

import math
import numpy as np
import speechbrain.nnet.CNN as CNN
import torch
from torch import nn
from torch.nn import functional as F

class Stretch2d(nn.Module):
    """
    Performs a stretch in the time domain of the input conditioning features

    Arguments
    ----------
    x_scale: int
        Scale of stretching the time axis
    y_scale: int
        Scale of stretching the frequency/mel axis (default=1)
    mode: str
        Mode used for upsampling by torch.nn.functional.interpolate

    """
    def __init__(self, x_scale, y_scale, mode="nearest"):
        super(Stretch2d, self).__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.mode = mode

    def forward(self, x):
        return F.interpolate(
            x, scale_factor=(self.y_scale, self.x_scale), mode=self.mode)

class UpsampleNetwork(nn.Module):
    """
    The learned upsampling network

    Arguments
    ----------
    upsample_scales: list
        List of upsampling scales. np.prod(upsample_scales) must be equal to hop_length
    mode: str
        Mode used for upsampling by torch.nn.functional.interpolate
    freq_axis_kernel_size: int
        Size of the kernel's frequency axis
    cin_channels: int
        Number of local conditioning channels. Set to -1 to disable local conditions
    cin_pad: int
        Padding of the conditional features to capture a wider context during upsampling
    """
    def __init__(self, upsample_scales, mode="nearest",
                 freq_axis_kernel_size=1, cin_pad=0, cin_channels=80):
        super(UpsampleNetwork, self).__init__()
        self.up_layers = nn.ModuleList()
        total_scale = np.prod(upsample_scales)
        self.indent = cin_pad * total_scale
        for scale in upsample_scales:
            freq_axis_padding = (freq_axis_kernel_size - 1) // 2
            k_size = (freq_axis_kernel_size, scale * 2 + 1)
            padding = (freq_axis_padding, scale)
            stretch = Stretch2d(scale, 1, mode)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
            conv.weight.data.fill_(1. / np.prod(k_size))
            conv = nn.utils.weight_norm(conv)
            self.up_layers.append(stretch)
            self.up_layers.append(conv)

    def forward(self, c):
        """
        Arguments:
        ----------
        c : torch.tensor, (B x T x C)
            Input local conditioning features
        
        Returns:
        ----------
        c : torch.tensor, (B x T x C)
            Upsampled features
        """
        # input is in B x T x C, convert to B x C x T for upsampling
        c = c.transpose(1,2).contiguous()
        # B x 1 x C x T
        c = c.unsqueeze(1)

        for f in self.up_layers:
            c = f(c)

        # B x C x T
        c = c.squeeze(1)

        if self.indent > 0:
            c = c[:, :, self.indent:-self.indent]
        
        # convert back to BxTxC
        c = c.transpose(1,2).contiguous()

        return c

class ConvInUpsampleNetwork(nn.Module):
    """
    Performs a convolution to capture wider-context information on conditional features (meaningless if cin_pad=0)
    and then upsamples the local conditioning features to match with input audio' time resolution

    Arguments
    ----------
    upsample_scales: list
        List of upsampling scales. np.prod(upsample_scales) must be equal to hop_length
    mode: str
        Mode used for upsampling by torch.nn.functional.interpolate
    freq_axis_kernel_size: int
        Size of the kernel's frequency axis
    cin_channels: int
        Number of local conditioning channels. Set to -1 to disable local conditions
    cin_pad: int
        Padding of the conditional features to capture a wider context during upsampling
    """
    def __init__(self, upsample_scales, mode="nearest",
                 freq_axis_kernel_size=1, cin_pad=0,
                 cin_channels=80):
        super(ConvInUpsampleNetwork, self).__init__()

        ks = 2 * cin_pad + 1
        self.conv_in = CNN.Conv1d(in_channels = cin_channels, out_channels = cin_channels, kernel_size=ks, bias=False, padding="valid")
        self.upsample = UpsampleNetwork(
            upsample_scales, mode, freq_axis_kernel_size, cin_pad=0, cin_channels=cin_channels)

    def forward(self, c):
        """
        Calculates the upsampled local conditioning features

        Arguments
        ----------
        c: torch.Tensor, (B x T x C)
            Local conditioning features. In the case of TTS pipeline, these are the mel spectrograms
        
        Returns
        ----------
        c_up: torch.Tensor, (B x T x C)
            Upsampled features
        """
        c_up = self.upsample(self.conv_in(c))
        return c_up

class IncrementalConv1d(CNN.Conv1d):
    """
    An extension of the standard SpeechBrain Conv1d that
    supports "Incremental Forward" mode. Used for synthesizing new signal from input features
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.clear_buffer()
        self._linearized_weight = None
        self.register_backward_hook(self._clear_linearized_weight)

    def incremental_forward(self, input):
        """
        Performs one incremental forward timestep. 
        It is a bit messy, but the gist of it is that the convolutional weights are linearized and passed through a linear layer.

        Arguments
        ----------
        input: torch.Tensor, (B x 1 x C)
            Input for one timestep
        
        Returns
        output: torch.Tensor, (B x 1 x C)
            Output for next timestep
    
        """
        if self.training:
            raise RuntimeError('incremental_forward only supports eval mode')
        # run forward pre hooks (e.g., weight norm)
        for hook in self._forward_pre_hooks.values():
            hook(self, input)
        # reshape weight
        weight = self._get_linearized_weight()
        kw = self.conv.kernel_size[0]
        dilation = self.conv.dilation[0]
        bsz = input.size(0)  # input: bsz x len x dim
        if kw > 1:
            input = input.data
            if self.input_buffer is None:
                self.input_buffer = input.new(bsz, kw + (kw - 1) * (dilation - 1), input.size(2))
                self.input_buffer.zero_()
            else:
                # shift buffer
                self.input_buffer[:, :-1, :] = self.input_buffer[:, 1:, :].clone()
            # append next input
            self.input_buffer[:, -1, :] = input[:, -1, :]
            input = self.input_buffer
            if dilation > 1:
                input = input[:, 0::dilation, :].contiguous()
        output = F.linear(input.view(bsz, -1), weight, self.conv.bias)
        output = output.unsqueeze(-1)
        return output.view(bsz, 1, -1)
    def clear_buffer(self):
        self.input_buffer = None
    def _get_linearized_weight(self):
        if self._linearized_weight is None:
            kw = self.conv.kernel_size[0]
            weight = self.conv.weight.transpose(1, 2).contiguous()
            assert weight.size() == (self.conv.out_channels, kw, self.conv.in_channels)
            self._linearized_weight = weight.view(self.conv.out_channels, -1)
        return self._linearized_weight
    def _clear_linearized_weight(self, *args):
        self._linearized_weight = None        


def Conv1dkxk(in_channels, out_channels, kernel_size, dropout=0, padding="causal", **kwargs):
    """
    Performs Conv1D using the IncrementalConv1d class for incremental forward option.
    Weights have kaiming normalization applied to them.
    """
    m = IncrementalConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, **kwargs)
    nn.init.kaiming_normal_(m.conv.weight, nonlinearity="relu")
    if m.conv.bias is not None:
        nn.init.constant_(m.conv.bias, 0)
    nn.utils.weight_norm(m.conv)
    return m

def Embedding(num_embeddings, embedding_dim, padding_idx, std=0.01):
    """
    Used by the model to learn the speaker embeddings
    """
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, std)
    return m

def Conv1d1x1(in_channels, out_channels, padding="causal", bias=True):
    """
    1-by-1 convolution layer
    """
    return IncrementalConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=padding,
                  dilation=1, bias=bias)


def _conv1x1_forward(conv, x, is_incremental):
    """
    Forward for a 1-by-1 convolution, with an option for incremental forward
    """
    if is_incremental:
        x = conv.incremental_forward(x)
    else:
        x = conv(x)
    return x


class ResidualBlockLayer(nn.Module):
    """
    Residual dilated conv1d + Gated linear unit

    Arguments
    ----------
    residual_channels: int 
        Number of residual input / output channels
    gate_channels: int
        Number of gate activation unit channels        
    skip_out_channels: int
        Number of skip connection channels
    kernel_size: int
        Kernel size of convolution layers.
    dropout: float
        Dropout probability 
    cin_channels: int
        Number of local conditioning channels. Set to -1 to disable local conditions
    gin_channels: int
        Number of global conditioning channels. Set to -1 to disable global conditions
    dropout: float
        Dropout probability.
    dilation: int
        Dilation factor
    """

    def __init__(self, residual_channels, gate_channels, kernel_size,
                 skip_out_channels=None,
                 cin_channels=-1, gin_channels=-1,
                 dropout=1 - 0.95, padding="causal", dilation=1,
                 bias=True, *args, **kwargs):
        super(ResidualBlockLayer, self).__init__()
        self.dropout = dropout
        if skip_out_channels is None:
            skip_out_channels = residual_channels

        self.conv = Conv1dkxk(residual_channels, gate_channels, kernel_size,
                           padding=padding, dilation=dilation,
                           bias=bias, *args, **kwargs)

        # local conditioning
        if cin_channels > 0:
            self.conv1x1c = Conv1d1x1(cin_channels, gate_channels, bias=False)
        else:
            self.conv1x1c = None

        # global conditioning
        if gin_channels > 0:
            self.conv1x1g = Conv1d1x1(gin_channels, gate_channels, bias=False)
        else:
            self.conv1x1g = None

        # conv output is split into two groups
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels, bias=bias)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_out_channels, bias=bias)

    def forward(self, x, c=None, g=None):
        return self._forward(x, c, g, False)

    def incremental_forward(self, x, c=None, g=None):
        return self._forward(x, c, g, True)

    def _forward(self, x, c, g, is_incremental):
        """
        Forward step

        Arguments
        ----------
            x: torch.tensor, (B x T x C)
                signal
            c: torch.tensor, (B x T x C)
                Local conditioning features
            g: torch.tensor, (B x T x C) 
                Expanded global conditioning features
            is_incremental: bool
                Whether incremental mode or not

        Returns:
            x: torch.tensor, (B x T x C) 
                output from Residual Block Layer
            s: torch.tensor, (B x T x C)
                skip-connection
        """
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)

        if is_incremental:
            x = self.conv.incremental_forward(x)
        else:
            x = self.conv(x)

        splitdim = -1

        a, b = x.split(x.size(splitdim) // 2, dim=splitdim)

        # local conditioning
        if c is not None:
            assert self.conv1x1c is not None
            c = _conv1x1_forward(self.conv1x1c, c, is_incremental)
            ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
            a, b = a + ca, b + cb

        # global conditioning
        if g is not None:
            assert self.conv1x1g is not None
            g = _conv1x1_forward(self.conv1x1g, g, is_incremental)
            ga, gb = g.split(g.size(splitdim) // 2, dim=splitdim)
            a, b = a + ga, b + gb

        x = torch.tanh(a) * torch.sigmoid(b)

        # For skip connection
        s = _conv1x1_forward(self.conv1x1_skip, x, is_incremental)

        # For residual connection
        x = _conv1x1_forward(self.conv1x1_out, x, is_incremental)

        x = (x + residual) * math.sqrt(0.5)
        return x, s

    def clear_buffer(self):
        for c in [self.conv, self.conv1x1_out, self.conv1x1_skip,
                  self.conv1x1c, self.conv1x1g]:
            if c is not None:
                c.clear_buffer()

def _expand_global_features(B, T, g):
    """
    Expand global conditioning features to all time steps

    Args:
        B: int
            Batch size.
        T: int
            Time length.
        g: torch.tensor, (B x C)
            Global features 

    Returns:
        g_btc: torch.tensor (B x T x C)
            Global features expanded
    """
    if g is None:
        return None

    g = g.unsqueeze(1) if g.dim() == 2 else g
    g_btc = g.expand(B,T,-1)
    return g_btc.contiguous()

class WaveNet(nn.Module):
    """
    The WaveNet model that supports local and global conditioning.

    Reference paper: https://arxiv.org/pdf/1609.03499.pdf

    Arguments
    ----------
    out_channels : int
        Number of output channels = number of quantized channels (mu=256)
    layers: int
        Number of total layers
    stacks: int
        Number of dilation cycles. Number of layers must be a multiple of number of stacks
    residual_channels: int 
        Number of residual input / output channels
    gate_channels: int
        Number of gate activation unit channels
    skip_out_channels: int
        Number of skip connection channels
    kernel_size: int
        Kernel size of convolution layers.
    dropout: float
        Dropout probability 
    cin_channels: int
        Number of local conditioning channels. Set to -1 to disable local conditions
    cin_pad: int
        Padding of the conditional features to capture a wider context during upsampling
    upsample_params: dict
        Contains parameters needed for upsamplng:
        - cin_channels
        - cin_pad
        - upsample scales: list
            List of upsample scales. np.prod(upsample_scales) must be equal to hop_length
    gin_channels: int
        Number of global conditioning channels. Set to -1 to disable global conditions
    n_speakers: int
        Number of speakers
    """

    def __init__(self, out_channels=256, layers=20, stacks=2,
                 residual_channels=512,
                 gate_channels=512,
                 skip_out_channels=512,
                 kernel_size=3, dropout=1 - 0.95,
                 cin_channels=-1, gin_channels=-1, n_speakers=None,
                 upsample_params={},
                 cin_pad=0,
                 ):
        super(WaveNet, self).__init__()

        self.out_channels = out_channels
        self.cin_channels = cin_channels
        self.gin_channels = gin_channels
        assert layers % stacks == 0
        layers_per_stack = layers // stacks

        self.first_conv = Conv1d1x1(in_channels=out_channels, out_channels=residual_channels)
 
        self.conv_layers = nn.ModuleList()
        for layer in range(layers):
            dilation = 2**(layer % layers_per_stack)
            conv = ResidualBlockLayer(
                residual_channels, gate_channels,
                kernel_size=kernel_size,
                skip_out_channels=skip_out_channels,
                bias=True, 
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

        if self.gin_channels > 0:
            assert n_speakers is not None
            self.embed_speakers = Embedding(
                n_speakers, gin_channels, padding_idx=None, std=0.1)
        else:
            self.embed_speakers = None

        # Upsample conv net
        if self.cin_channels>0:
            self.upsample_net = ConvInUpsampleNetwork(**upsample_params)
        else:
            self.upsample_net = None

    def forward(self, x, c=None, g=None):
        """
        Forward step

        Arguments
        ----------
            x: torch.tensor, (B x T x C)
                signal
            c: torch.tensor, (B x T x C)
                Local conditioning features
            g: torch.tensor, (B x T x C) 
                Expanded global conditioning features

        Returns:
            x: torch.tensor, (B x T x C)
                Class scores for each timestep 
        """
        B, T, _ = x.size()

        if g is not None:
            if self.embed_speakers is not None:
                # (B x 1) -> (B x 1 x gin_channels)
                g = self.embed_speakers(g.view(B, -1))
                assert g.dim() == 3

        # Expand global conditioning features to all time steps
        g_bct = _expand_global_features(B, T, g)

        if c is not None and self.upsample_net is not None:
            c = self.upsample_net(c)
            assert c.size(1) == x.size(1)

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

        return x

    def incremental_forward(self, initial_input=None, c=None, g=None,
                            T=100, test_inputs=None,
                            tqdm=lambda x: x):
        """
        Incremental forward step
        Input of each time step is of shape (B x 1 x C).

        Arguments
        ----------
            initial_input: torch.tensor, (B x 1 x C)
                Initial decoder input
            c: torch.tensor, (B x T x C)
                Local conditioning features
            g: torch.tensor, (B x C)
                Global conditioning features
            T: int
                Number of time steps to generate.
            tqdm (lamda) : tqdm

        Returns:
            outputs: torch.tensor, (B x T x C)
                Generated one-hot encoded samples　
        """
        self.clear_buffer()
        B = 1

        # cast to int in case of numpy.int64...
        T = int(T)

        # Global conditioning
        if g is not None:
            if self.embed_speakers is not None:
                g = self.embed_speakers(g.view(B, -1))
                assert g.dim() == 3
        g_btc = _expand_global_features(B, T, g)

        # Local conditioning
        if c is not None:
            B = c.shape[0]
            if self.upsample_net is not None:
                c = self.upsample_net(c)
                assert c.size(1) == T

        outputs = []
        if initial_input is None:
            initial_input = torch.zeros(B, 1, self.out_channels)
            initial_input[:, :, 127] = 1 
            # https://github.com/pytorch/pytorch/issues/584#issuecomment-275169567
            if next(self.parameters()).is_cuda:
                initial_input = initial_input.cuda()
        else:
            assert initial_input.size(-1) == self.out_channels

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

            # Generate next input 
            x = F.softmax(x.view(B, -1), dim=1)

            dist = torch.distributions.OneHotCategorical(x)
            x = dist.sample()
            outputs += [x.data]

        # T x B x C
        outputs = torch.stack(outputs)
        # B x T x C
        outputs = outputs.transpose(0, 1).contiguous()

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
        """
        Removes weight norm for faster generation
        """
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return
        self.apply(remove_weight_norm)

class Loss(nn.Module):
    """
    Masked Cross-Entropy Loss. 
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predictions, target, lengths):
        """
        Calculates the masked cross-entropy loss

        Arguments
        ---------
        predictions: torch.Tensor, (B,T,C,1)
            batch of predicted output
        target: torch.Tensor, (B,T,1)
            batch of targets
        lengths: torch.Tensor, (B)
            target lengths

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.     
        """
        # (B,T) -> (B,1,T)
        mask = self.sequence_mask(lengths).unsqueeze(1)        
        # (B,1,T) -> (B,T,1)
        mask = mask.transpose(1,2).contiguous()
        mask = mask[:,1:,:]
        mask_ = mask.expand_as(target)

        # cross entropy loss expects LongTensor target
        # cross entropy loss requires input to be of (N,Classes,...)
        # classes in our case are the quantized channels
        # B,T,C,1 -> B,C,T,1
        predictions = predictions.transpose(1,2).contiguous()
        losses = self.criterion(predictions, target.long())

        return ((losses * mask_).sum()) / mask_.sum()
    
    def sequence_mask(self, sequence_length):
        """
        Function that computes sequence mask based on a series of sequence lengths
        
        Arguments
        ---------
        sequence_length: torch.Tensor, (L)
            Path to the file name

        Returns
        -------
        mask : torch.Tensor, (L x L)
            A tensor containing the mask

        Ex:
        >>> sequence_length = torch.tensor([1,2,3,4,5,6])
        >>> print(sequence_mask(sequence_length))
        tensor([[1., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0.],
        [1., 1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1.]])
        """
        max_len = sequence_length.data.max()
        batch_size = sequence_length.size(0)
        seq_range = torch.arange(0, max_len).long()

        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)

        if sequence_length.is_cuda:
            seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = sequence_length.unsqueeze(1) \
            .expand_as(seq_range_expand)
        return (seq_range_expand < seq_length_expand).float()
