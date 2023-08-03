"""
Neural network modules for the Tacotron2 end-to-end neural
Text-to-Speech (TTS) model

Authors
* Georges Abous-Rjeili 2021
* Artem Ploujnikov 2021
"""

# This code uses a significant portion of the NVidia implementation, even though it
# has been modified and enhanced

# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/tacotron2/model.py
# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

from math import sqrt
from speechbrain.nnet.loss.guidedattn_loss import GuidedAttentionLoss
import torch
from torch import nn
from torch.nn import functional as F
from collections import namedtuple


class LinearNorm(torch.nn.Module):
    """A linear layer with Xavier initialization

    Arguments
    ---------
    in_dim: int
        the input dimension
    out_dim: int
        the output dimension
    bias: bool
        whether or not to use a bias
    w_init_gain: linear
        the weight initialization gain type (see torch.nn.init.calculate_gain)

    Example
    -------
    >>> import torch
    >>> from speechbrain.lobes.models.Tacotron2 import LinearNorm
    >>> layer = LinearNorm(in_dim=5, out_dim=3)
    >>> x = torch.randn(3, 5)
    >>> y = layer(x)
    >>> y.shape
    torch.Size([3, 3])
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain),
        )

    def forward(self, x):
        """Computes the forward pass

        Arguments
        ---------
        x: torch.Tensor
            a (batch, features) input tensor


        Returns
        -------
        output: torch.Tensor
            the linear layer output

        """
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    """A 1D convolution layer with Xavier initialization

    Arguments
    ---------
    in_channels: int
        the number of input channels
    out_channels: int
        the number of output channels
    kernel_size: int
        the kernel size
    stride: int
        the convolutional stride
    padding: int
        the amount of padding to include. If not provided, it will be calculated
        as dilation * (kernel_size - 1) / 2
    dilation: int
        the dilation of the convolution
    bias: bool
        whether or not to use a bias
    w_init_gain: linear
        the weight initialization gain type (see torch.nn.init.calculate_gain)

    Example
    -------
    >>> import torch
    >>> from speechbrain.lobes.models.Tacotron2 import ConvNorm
    >>> layer = ConvNorm(in_channels=10, out_channels=5, kernel_size=3)
    >>> x = torch.randn(3, 10, 5)
    >>> y = layer(x)
    >>> y.shape
    torch.Size([3, 5, 5])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super().__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, signal):
        """Computes the forward pass

        Arguments
        ---------
        signal: torch.Tensor
            the input to the convolutional layer

        Returns
        -------
        output: torch.Tensor
            the output
        """
        return self.conv(signal)


class LocationLayer(nn.Module):
    """A location-based attention layer consisting of a Xavier-initialized
    convolutional layer followed by a dense layer

    Arguments
    ---------
    attention_n_filters: int
        the number of filters used in attention

    attention_kernel_size: int
        the kernel size of the attention layer

    attention_dim: int
        the dimension of linear attention layers


    Example
    -------
    >>> import torch
    >>> from speechbrain.lobes.models.Tacotron2 import LocationLayer
    >>> layer = LocationLayer()
    >>> attention_weights_cat = torch.randn(3, 2, 64)
    >>> processed_attention = layer(attention_weights_cat)
    >>> processed_attention.shape
    torch.Size([3, 64, 128])

    """

    def __init__(
        self,
        attention_n_filters=32,
        attention_kernel_size=31,
        attention_dim=128,
    ):
        super().__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(
            2,
            attention_n_filters,
            kernel_size=attention_kernel_size,
            padding=padding,
            bias=False,
            stride=1,
            dilation=1,
        )
        self.location_dense = LinearNorm(
            attention_n_filters, attention_dim, bias=False, w_init_gain="tanh"
        )

    def forward(self, attention_weights_cat):
        """Performs the forward pass for the attention layer

        Arguments
        ---------
        attention_weights_cat: torch.Tensor
            the concatenating attention weights

        Results
        -------
        processed_attention: torch.Tensor
            the attention layer output

        """
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    """The Tacotron attention layer. Location-based attention is used.

    Arguments
    ---------
    attention_rnn_dim: int
        the dimension of the RNN to which the attention layer
        is applied
    embedding_dim: int
        the embedding dimension
    attention_dim: int
        the dimension of the memory cell
    attenion_location_n_filters: int
        the number of location filters
    attention_location_kernel_size: int
        the kernel size of the location layer

    Example
    -------
    >>> import torch
    >>> from speechbrain.lobes.models.Tacotron2 import (
    ...     Attention, get_mask_from_lengths)
    >>> layer = Attention()
    >>> attention_hidden_state = torch.randn(2, 1024)
    >>> memory = torch.randn(2, 173, 512)
    >>> processed_memory = torch.randn(2, 173, 128)
    >>> attention_weights_cat = torch.randn(2, 2, 173)
    >>> memory_lengths = torch.tensor([173, 91])
    >>> mask = get_mask_from_lengths(memory_lengths)
    >>> attention_context, attention_weights = layer(
    ...    attention_hidden_state,
    ...    memory,
    ...    processed_memory,
    ...    attention_weights_cat,
    ...    mask
    ... )
    >>> attention_context.shape, attention_weights.shape
    (torch.Size([2, 512]), torch.Size([2, 173]))
    """

    def __init__(
        self,
        attention_rnn_dim=1024,
        embedding_dim=512,
        attention_dim=128,
        attention_location_n_filters=32,
        attention_location_kernel_size=31,
    ):
        super().__init__()
        self.query_layer = LinearNorm(
            attention_rnn_dim, attention_dim, bias=False, w_init_gain="tanh"
        )
        self.memory_layer = LinearNorm(
            embedding_dim, attention_dim, bias=False, w_init_gain="tanh"
        )
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(
            attention_location_n_filters,
            attention_location_kernel_size,
            attention_dim,
        )
        self.score_mask_value = -float("inf")

    def get_alignment_energies(
        self, query, processed_memory, attention_weights_cat
    ):
        """Computes the alignment energies

        Arguments
        ---------
        query: torch.Tensor
            decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: torch.Tensor
            processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: torch.Tensor
            cumulative and prev. att weights (B, 2, max_time)

        Returns
        -------
        alignment : torch.Tensor
            (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(
            torch.tanh(
                processed_query + processed_attention_weights + processed_memory
            )
        )

        energies = energies.squeeze(2)
        return energies

    def forward(
        self,
        attention_hidden_state,
        memory,
        processed_memory,
        attention_weights_cat,
        mask,
    ):
        """Computes the forward pass

        Arguments
        ---------
        attention_hidden_state: torch.Tensor
            attention rnn last output
        memory: torch.Tensor
            encoder outputs
        processed_memory: torch.Tensor
            processed encoder outputs
        attention_weights_cat: torch.Tensor
            previous and cummulative attention weights
        mask: torch.Tensor
            binary mask for padded data

        Returns
        -------
        result: tuple
            a (attention_context, attention_weights) tuple
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat
        )

        alignment = alignment.masked_fill(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    """The Tacotron pre-net module consisting of a specified number of
    normalized (Xavier-initialized) linear layers

    Arguments
    ---------
    in_dim: int
        the input dimensions
    sizes: int
        the dimension of the hidden layers/outout
    dropout: float
        the dropout probability

    Example
    -------
    >>> import torch
    >>> from speechbrain.lobes.models.Tacotron2 import Prenet
    >>> layer = Prenet()
    >>> x = torch.randn(862, 2, 80)
    >>> output = layer(x)
    >>> output.shape
    torch.Size([862, 2, 256])
    """

    def __init__(self, in_dim=80, sizes=[256, 256], dropout=0.5):
        super().__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [
                LinearNorm(in_size, out_size, bias=False)
                for (in_size, out_size) in zip(in_sizes, sizes)
            ]
        )
        self.dropout = dropout

    def forward(self, x):
        """Computes the forward pass for the prenet

        Arguments
        ---------
        x: torch.Tensor
            the prenet inputs

        Returns
        -------
        output: torch.Tensor
            the output
        """
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=self.dropout, training=True)
        return x


class Postnet(nn.Module):
    """The Tacotron postnet consists of a number of 1-d convolutional layers
    with Xavier initialization and a tanh activation, with batch normalization.
    Depending on configuration, the postnet may either refine the MEL spectrogram
    or upsample it to a linear spectrogram

    Arguments
    ---------
    n_mel_channels: int
        the number of MEL spectrogram channels
    postnet_embedding_dim: int
        the postnet embedding dimension
    postnet_kernel_size: int
        the kernel size of the convolutions within the decoders
    postnet_n_convolutions: int
        the number of convolutions in the postnet

    Example
    -------
    >>> import torch
    >>> from speechbrain.lobes.models.Tacotron2 import Postnet
    >>> layer = Postnet()
    >>> x = torch.randn(2, 80, 861)
    >>> output = layer(x)
    >>> output.shape
    torch.Size([2, 80, 861])
    """

    def __init__(
        self,
        n_mel_channels=80,
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
    ):
        super().__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    n_mel_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="tanh",
                ),
                nn.BatchNorm1d(postnet_embedding_dim),
            )
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=int((postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="linear",
                ),
                nn.BatchNorm1d(n_mel_channels),
            )
        )
        self.n_convs = len(self.convolutions)

    def forward(self, x):
        """Computes the forward pass of the postnet

        Arguments
        ---------
        x: torch.Tensor
            the postnet input (usually a MEL spectrogram)

        Returns
        -------
        output: torch.Tensor
            the postnet output (a refined MEL spectrogram or a
            linear spectrogram depending on how the model is
            configured)
        """
        i = 0
        for conv in self.convolutions:
            if i < self.n_convs - 1:
                x = F.dropout(torch.tanh(conv(x)), 0.5, training=self.training)
            else:
                x = F.dropout(conv(x), 0.5, training=self.training)
            i += 1

        return x


class Encoder(nn.Module):
    """The Tacotron2 encoder module, consisting of a sequence of  1-d convolution banks (3 by default)
    and a bidirectional LSTM

    Arguments
    ---------
    encoder_n_convolutions: int
        the number of encoder convolutions
    encoder_embedding_dim: int
        the dimension of the encoder embedding
    encoder_kernel_size: int
        the kernel size of the 1-D convolutional layers within
        the encoder

    Example
    -------
    >>> import torch
    >>> from speechbrain.lobes.models.Tacotron2 import Encoder
    >>> layer = Encoder()
    >>> x = torch.randn(2, 512, 128)
    >>> input_lengths = torch.tensor([128, 83])
    >>> outputs = layer(x, input_lengths)
    >>> outputs.shape
    torch.Size([2, 128, 512])

    """

    def __init__(
        self,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,
        encoder_kernel_size=5,
    ):
        super().__init__()

        convolutions = []
        for _ in range(encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(
                    encoder_embedding_dim,
                    encoder_embedding_dim,
                    kernel_size=encoder_kernel_size,
                    stride=1,
                    padding=int((encoder_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(encoder_embedding_dim),
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(
            encoder_embedding_dim,
            int(encoder_embedding_dim / 2),
            1,
            batch_first=True,
            bidirectional=True,
        )

    @torch.jit.ignore
    def forward(self, x, input_lengths):
        """Computes the encoder forward pass

        Arguments
        ---------
        x: torch.Tensor
            a batch of inputs (sequence embeddings)

        input_lengths: torch.Tensor
            a tensor of input lengths

        Returns
        -------
        outputs: torch.Tensor
            the encoder output
        """
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True
        )

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs

    @torch.jit.export
    def infer(self, x, input_lengths):
        """Performs a forward stap in the inference context

        Arguments
        ---------
        x: torch.Tensor
            a batch of inputs (sequence embeddings)

        input_lengths: torch.Tensor
            a tensor of input lengths

        Returns
        -------
        outputs: torch.Tensor
            the encoder output
        """
        device = x.device
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x.to(device))), 0.5, self.training)

        x = x.transpose(1, 2)

        input_lengths = input_lengths.cpu()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True
        )
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs


class Decoder(nn.Module):
    """The Tacotron decoder

    Arguments
    ---------
    n_mel_channels: int
        the number of channels in the MEL sepctrogram
    n_frames_per_step:
        the number of frames in the spectrogram for each
        time step of the decoder
    encoder_embedding_dim: int
        the dimension of the encoder embedding
    attention_location_n_filters: int
        the number of filters in location-based attention
    attention_location_kernel_size: int
        the kernel size of location-based attention
    attention_rnn_dim: int
        RNN dimension for the attention layer
    decoder_rnn_dim: int
        the encoder RNN dimension
    prenet_dim: int
        the dimension of the prenet (inner and output layers)
    max_decoder_steps: int
        the maximum number of decoder steps for the longest utterance
        expected for the model
    gate_threshold: float
        the fixed threshold to which the outputs of the decoders will be compared
    p_attention_dropout: float
        dropout probability for attention layers

    Example
    -------
    >>> import torch
    >>> from speechbrain.lobes.models.Tacotron2 import Decoder
    >>> layer = Decoder()
    >>> memory = torch.randn(2, 173, 512)
    >>> decoder_inputs = torch.randn(2, 80, 173)
    >>> memory_lengths = torch.tensor([173, 91])
    >>> mel_outputs, gate_outputs, alignments = layer(
    ...     memory, decoder_inputs, memory_lengths)
    >>> mel_outputs.shape, gate_outputs.shape, alignments.shape
    (torch.Size([2, 80, 173]), torch.Size([2, 173]), torch.Size([2, 173, 173]))
    """

    def __init__(
        self,
        n_mel_channels=80,
        n_frames_per_step=1,
        encoder_embedding_dim=512,
        attention_dim=128,
        attention_location_n_filters=32,
        attention_location_kernel_size=31,
        attention_rnn_dim=1024,
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,
        early_stopping=True,
    ):
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step
        self.encoder_embedding_dim = encoder_embedding_dim
        self.attention_rnn_dim = attention_rnn_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dim = prenet_dim
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        self.p_attention_dropout = p_attention_dropout
        self.p_decoder_dropout = p_decoder_dropout
        self.early_stopping = early_stopping

        self.prenet = Prenet(
            n_mel_channels * n_frames_per_step, [prenet_dim, prenet_dim]
        )

        self.attention_rnn = nn.LSTMCell(
            prenet_dim + encoder_embedding_dim, attention_rnn_dim
        )

        self.attention_layer = Attention(
            attention_rnn_dim,
            encoder_embedding_dim,
            attention_dim,
            attention_location_n_filters,
            attention_location_kernel_size,
        )

        self.decoder_rnn = nn.LSTMCell(
            attention_rnn_dim + encoder_embedding_dim, decoder_rnn_dim, 1
        )

        self.linear_projection = LinearNorm(
            decoder_rnn_dim + encoder_embedding_dim,
            n_mel_channels * n_frames_per_step,
        )

        self.gate_layer = LinearNorm(
            decoder_rnn_dim + encoder_embedding_dim,
            1,
            bias=True,
            w_init_gain="sigmoid",
        )

    def get_go_frame(self, memory):
        """Gets all zeros frames to use as first decoder input

        Arguments
        ---------
        memory: torch.Tensor
            decoder outputs

        Returns
        -------
        decoder_input: torch.Tensor
            all zeros frames
        """
        B = memory.size(0)
        dtype = memory.dtype
        device = memory.device
        decoder_input = torch.zeros(
            B,
            self.n_mel_channels * self.n_frames_per_step,
            dtype=dtype,
            device=device,
        )
        return decoder_input

    def initialize_decoder_states(self, memory):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory

        Arguments
        ---------
        memory: torch.Tensor
            Encoder outputs
        mask: torch.Tensor
            Mask for padded data if training, expects None for inference

        Returns
        -------
        result: tuple
            A tuple of tensors
            (
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
                processed_memory,
            )
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)
        dtype = memory.dtype
        device = memory.device

        attention_hidden = torch.zeros(
            B, self.attention_rnn_dim, dtype=dtype, device=device
        )
        attention_cell = torch.zeros(
            B, self.attention_rnn_dim, dtype=dtype, device=device
        )

        decoder_hidden = torch.zeros(
            B, self.decoder_rnn_dim, dtype=dtype, device=device
        )
        decoder_cell = torch.zeros(
            B, self.decoder_rnn_dim, dtype=dtype, device=device
        )

        attention_weights = torch.zeros(B, MAX_TIME, dtype=dtype, device=device)
        attention_weights_cum = torch.zeros(
            B, MAX_TIME, dtype=dtype, device=device
        )
        attention_context = torch.zeros(
            B, self.encoder_embedding_dim, dtype=dtype, device=device
        )

        processed_memory = self.attention_layer.memory_layer(memory)

        return (
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
            processed_memory,
        )

    def parse_decoder_inputs(self, decoder_inputs):
        """Prepares decoder inputs, i.e. mel outputs
        Arguments
        ----------
        decoder_inputs: torch.Tensor
            inputs used for teacher-forced training, i.e. mel-specs

        Returns
        -------
        decoder_inputs: torch.Tensor
            processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1) / self.n_frames_per_step),
            -1,
        )
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """Prepares decoder outputs for output

        Arguments
        ---------
        mel_outputs: torch.Tensor
            MEL-scale spectrogram outputs
        gate_outputs: torch.Tensor
            gate output energies
        alignments: torch.Tensor
            the alignment tensor

        Returns
        -------
        mel_outputs: torch.Tensor
            MEL-scale spectrogram outputs
        gate_outputs: torch.Tensor
            gate output energies
        alignments: torch.Tensor
            the alignment tensor
        """
        # (T_out, B) -> (B, T_out)
        alignments = alignments.transpose(0, 1).contiguous()
        # (T_out, B) -> (B, T_out)
        if gate_outputs.dim() == 1:
            gate_outputs = gate_outputs.unsqueeze(0)
        else:
            gate_outputs = gate_outputs.transpose(0, 1).contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = mel_outputs.transpose(0, 1).contiguous()
        # decouple frames per step
        shape = (mel_outputs.shape[0], -1, self.n_mel_channels)
        mel_outputs = mel_outputs.view(*shape)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(
        self,
        decoder_input,
        attention_hidden,
        attention_cell,
        decoder_hidden,
        decoder_cell,
        attention_weights,
        attention_weights_cum,
        attention_context,
        memory,
        processed_memory,
        mask,
    ):
        """Decoder step using stored states, attention and memory
        Arguments
        ---------
        decoder_input: torch.Tensor
            previous mel output
        attention_hidden: torch.Tensor
            the hidden state of the attention module
        attention_cell: torch.Tensor
            the attention cell state
        decoder_hidden: torch.Tensor
            the decoder hidden state
        decoder_cell: torch.Tensor
            the decoder cell state
        attention_weights: torch.Tensor
            the attention weights
        attention_weights_cum: torch.Tensor
            cumulative attention weights
        attention_context: torch.Tensor
            the attention context tensor
        memory: torch.Tensor
            the memory tensor
        processed_memory: torch.Tensor
            the processed memory tensor
        mask: torch.Tensor



        Returns
        -------
        mel_output: torch.Tensor
            the MEL-scale outputs
        gate_output: torch.Tensor
            gate output energies
        attention_weights: torch.Tensor
            attention weights
        """
        cell_input = torch.cat((decoder_input, attention_context), -1)

        attention_hidden, attention_cell = self.attention_rnn(
            cell_input, (attention_hidden, attention_cell)
        )
        attention_hidden = F.dropout(
            attention_hidden, self.p_attention_dropout, self.training
        )

        attention_weights_cat = torch.cat(
            (
                attention_weights.unsqueeze(1),
                attention_weights_cum.unsqueeze(1),
            ),
            dim=1,
        )
        attention_context, attention_weights = self.attention_layer(
            attention_hidden,
            memory,
            processed_memory,
            attention_weights_cat,
            mask,
        )

        attention_weights_cum += attention_weights
        decoder_input = torch.cat((attention_hidden, attention_context), -1)

        decoder_hidden, decoder_cell = self.decoder_rnn(
            decoder_input, (decoder_hidden, decoder_cell)
        )
        decoder_hidden = F.dropout(
            decoder_hidden, self.p_decoder_dropout, self.training
        )

        decoder_hidden_attention_context = torch.cat(
            (decoder_hidden, attention_context), dim=1
        )
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context
        )

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        return (
            decoder_output,
            gate_prediction,
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
        )

    @torch.jit.ignore
    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training

        Arguments
        ----------
        memory: torch.Tensor
            Encoder outputs
        decoder_inputs: torch.Tensor
            Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: torch.Tensor
            Encoder output lengths for attention masking.

        Returns
        -------
        mel_outputs: torch.Tensor
            mel outputs from the decoder
        gate_outputs: torch.Tensor
            gate outputs from the decoder
        alignments: torch.Tensor
            sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        mask = get_mask_from_lengths(memory_lengths)
        (
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
            processed_memory,
        ) = self.initialize_decoder_states(memory)

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            (
                mel_output,
                gate_output,
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
            ) = self.decode(
                decoder_input,
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
                memory,
                processed_memory,
                mask,
            )

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            torch.stack(mel_outputs),
            torch.stack(gate_outputs),
            torch.stack(alignments),
        )

        return mel_outputs, gate_outputs, alignments

    @torch.jit.export
    def infer(self, memory, memory_lengths):
        """ Decoder inference

        Arguments
        ---------
        memory: torch.Tensor
            Encoder outputs

        Returns
        -------
        mel_outputs: torch.Tensor
            mel outputs from the decoder
        gate_outputs: torch.Tensor
            gate outputs from the decoder
        alignments: torch.Tensor
            sequence of attention weights from the decoder
        mel_lengths: torch.Tensor
            the length of MEL spectrograms
        """
        decoder_input = self.get_go_frame(memory)

        mask = get_mask_from_lengths(memory_lengths)
        (
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
            processed_memory,
        ) = self.initialize_decoder_states(memory)

        mel_lengths = torch.zeros(
            [memory.size(0)], dtype=torch.int32, device=memory.device
        )
        not_finished = torch.ones(
            [memory.size(0)], dtype=torch.int32, device=memory.device
        )

        mel_outputs, gate_outputs, alignments = (
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
        )
        first_iter = True
        while True:
            decoder_input = self.prenet(decoder_input)
            (
                mel_output,
                gate_output,
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
            ) = self.decode(
                decoder_input,
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
                memory,
                processed_memory,
                mask,
            )

            if first_iter:
                mel_outputs = mel_output.unsqueeze(0)
                gate_outputs = gate_output
                alignments = attention_weights
                first_iter = False
            else:
                mel_outputs = torch.cat(
                    (mel_outputs, mel_output.unsqueeze(0)), dim=0
                )
                gate_outputs = torch.cat((gate_outputs, gate_output), dim=0)
                alignments = torch.cat((alignments, attention_weights), dim=0)

            dec = (
                torch.le(torch.sigmoid(gate_output), self.gate_threshold)
                .to(torch.int32)
                .squeeze(1)
            )

            not_finished = not_finished * dec
            mel_lengths += not_finished
            if self.early_stopping and torch.sum(not_finished) == 0:
                break
            if len(mel_outputs) == self.max_decoder_steps:
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments
        )

        return mel_outputs, gate_outputs, alignments, mel_lengths


class Tacotron2(nn.Module):
    """The Tactron2 text-to-speech model, based on the NVIDIA implementation.

    This class is the main entry point for the model, which is responsible
    for instantiating all submodules, which, in turn, manage the individual
    neural network layers

    Simplified STRUCTURE: input->word embedding ->encoder ->attention \
    ->decoder(+prenet) -> postnet ->output

    prenet(input is decoder previous time step) output is input to decoder
    concatenanted with the attention output

    Arguments
    ---------
    mask_padding: bool
        whether or not to mask pad-outputs of tacotron

    #mel generation parameter in data io
    n_mel_channels: int
        number of mel channels for constructing spectrogram

    #symbols
    n_symbols:  int=128
        number of accepted char symbols defined in textToSequence
    symbols_embedding_dim: int
        number of embeding dimension for symbols fed to nn.Embedding

    # Encoder parameters
    encoder_kernel_size: int
        size of kernel processing the embeddings
    encoder_n_convolutions: int
        number of convolution layers in encoder
    encoder_embedding_dim: int
        number of kernels in encoder, this is also the dimension
        of the bidirectional LSTM in the encoder

    # Attention parameters
    attention_rnn_dim: int
        input dimension
    attention_dim: int
        number of hidden represetation in attention
    # Location Layer parameters
    attention_location_n_filters: int
        number of 1-D convulation filters in attention
    attention_location_kernel_size: int
        length of the 1-D convolution filters

    # Decoder parameters
    n_frames_per_step: int=1
        only 1 generated mel-frame per step is supported for the decoder as of now.
    decoder_rnn_dim: int
        number of 2 unidirectionnal stacked LSTM units
    prenet_dim: int
        dimension of linear prenet layers
    max_decoder_steps: int
        maximum number of steps/frames the decoder generates before stopping
    p_attention_dropout: float
        attention drop out probability
    p_decoder_dropout: float
        decoder drop  out probability

    gate_threshold: int
        cut off level any output probabilty above that is considered
        complete and stops genration so we have variable length outputs
    decoder_no_early_stopping: bool
        determines early stopping of decoder
        along with gate_threshold . The logical inverse of this is fed to the decoder


    #Mel-post processing network parameters
    postnet_embedding_dim: int
        number os postnet dfilters
    postnet_kernel_size: int
        1d size of posnet kernel
    postnet_n_convolutions: int
        number of convolution layers in postnet

    Example
    -------
    >>> import torch
    >>> _ = torch.manual_seed(213312)
    >>> from speechbrain.lobes.models.Tacotron2 import Tacotron2
    >>> model = Tacotron2(
    ...    mask_padding=True,
    ...    n_mel_channels=80,
    ...    n_symbols=148,
    ...    symbols_embedding_dim=512,
    ...    encoder_kernel_size=5,
    ...    encoder_n_convolutions=3,
    ...    encoder_embedding_dim=512,
    ...    attention_rnn_dim=1024,
    ...    attention_dim=128,
    ...    attention_location_n_filters=32,
    ...    attention_location_kernel_size=31,
    ...    n_frames_per_step=1,
    ...    decoder_rnn_dim=1024,
    ...    prenet_dim=256,
    ...    max_decoder_steps=32,
    ...    gate_threshold=0.5,
    ...    p_attention_dropout=0.1,
    ...    p_decoder_dropout=0.1,
    ...    postnet_embedding_dim=512,
    ...    postnet_kernel_size=5,
    ...    postnet_n_convolutions=5,
    ...    decoder_no_early_stopping=False
    ... )
    >>> _ = model.eval()
    >>> inputs = torch.tensor([
    ...     [13, 12, 31, 14, 19],
    ...     [31, 16, 30, 31, 0],
    ... ])
    >>> input_lengths = torch.tensor([5, 4])
    >>> outputs, output_lengths, alignments = model.infer(inputs, input_lengths)
    >>> outputs.shape, output_lengths.shape, alignments.shape
    (torch.Size([2, 80, 1]), torch.Size([2]), torch.Size([2, 1, 5]))
    """

    def __init__(
        self,
        mask_padding=True,
        n_mel_channels=80,
        n_symbols=148,
        symbols_embedding_dim=512,
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,
        attention_rnn_dim=1024,
        attention_dim=128,
        attention_location_n_filters=32,
        attention_location_kernel_size=31,
        n_frames_per_step=1,
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
        decoder_no_early_stopping=False,
    ):
        super().__init__()
        self.mask_padding = mask_padding
        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step
        self.embedding = nn.Embedding(n_symbols, symbols_embedding_dim)
        std = sqrt(2.0 / (n_symbols + symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(
            encoder_n_convolutions, encoder_embedding_dim, encoder_kernel_size
        )
        self.decoder = Decoder(
            n_mel_channels,
            n_frames_per_step,
            encoder_embedding_dim,
            attention_dim,
            attention_location_n_filters,
            attention_location_kernel_size,
            attention_rnn_dim,
            decoder_rnn_dim,
            prenet_dim,
            max_decoder_steps,
            gate_threshold,
            p_attention_dropout,
            p_decoder_dropout,
            not decoder_no_early_stopping,
        )
        self.postnet = Postnet(
            n_mel_channels,
            postnet_embedding_dim,
            postnet_kernel_size,
            postnet_n_convolutions,
        )

    def parse_output(self, outputs, output_lengths, alignments_dim=None):
        """
        Masks the padded part of output

        Arguments
        ---------
        outputs: list
            a list of tensors - raw outputs
        outputs_lengths: torch.Tensor
            a tensor representing the lengths of all outputs
        alignments_dim: int
            the desired dimension of the alignments along the last axis
            Optional but needed for data-parallel training


        Returns
        -------
        result: tuple
            a (mel_outputs, mel_outputs_postnet, gate_outputs, alignments) tuple with
            the original outputs - with the mask applied
        """
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = outputs
        if self.mask_padding and output_lengths is not None:
            mask = get_mask_from_lengths(
                output_lengths, max_len=mel_outputs.size(-1)
            )
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            mel_outputs.clone().masked_fill_(mask, 0.0)
            mel_outputs_postnet.masked_fill_(mask, 0.0)
            gate_outputs.masked_fill_(mask[:, 0, :], 1e3)  # gate energies
        if alignments_dim is not None:
            alignments = F.pad(
                alignments, (0, alignments_dim - alignments.size(-1))
            )

        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments

    def forward(self, inputs, alignments_dim=None):
        """Decoder forward pass for training

        Arguments
        ---------
        inputs: tuple
            batch object
        alignments_dim: int
            the desired dimension of the alignments along the last axis
            Optional but needed for data-parallel training

        Returns
        ---------
        mel_outputs: torch.Tensor
            mel outputs from the decoder
        mel_outputs_postnet: torch.Tensor
            mel outputs from postnet
        gate_outputs: torch.Tensor
            gate outputs from the decoder
        alignments: torch.Tensor
            sequence of attention weights from the decoder
        output_legnths: torch.Tensor
            length of the output without padding
        """

        inputs, input_lengths, targets, max_len, output_lengths = inputs
        input_lengths, output_lengths = input_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, input_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, targets, memory_lengths=input_lengths
        )

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths,
            alignments_dim,
        )

    def infer(self, inputs, input_lengths):
        """Produces outputs


        Arguments
        ---------
        inputs: torch.tensor
            text or phonemes converted

        input_lengths: torch.tensor
            the lengths of input parameters

        Returns
        -------
        mel_outputs_postnet: torch.Tensor
            final mel output of tacotron 2
        mel_lengths: torch.Tensor
            length of mels
        alignments: torch.Tensor
            sequence of attention weights
        """

        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.infer(embedded_inputs, input_lengths)
        mel_outputs, gate_outputs, alignments, mel_lengths = self.decoder.infer(
            encoder_outputs, input_lengths
        )

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        BS = mel_outputs_postnet.size(0)
        alignments = alignments.unfold(1, BS, BS).transpose(0, 2)

        return mel_outputs_postnet, mel_lengths, alignments


def get_mask_from_lengths(lengths, max_len=None):
    """Creates a mask from a tensor of lengths

    Arguments
    ---------
    lengths: torch.Tensor
        a tensor of sequence lengths

    Returns
    -------
    mask: torch.Tensor
        the mask
    max_len: int
        The maximum length, i.e. the last dimension of
        the mask tensor. If not provided, it will be
        calculated automatically
    """
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = (ids < lengths.unsqueeze(1)).byte()
    mask = torch.le(mask, 0)
    return mask


def infer(model, text_sequences, input_lengths):
    """
    An inference hook for pretrained synthesizers

    Arguments
    ---------
    model: Tacotron2
        the tacotron model
    text_sequences: torch.Tensor
        encoded text sequences
    input_lengths: torch.Tensor
        input lengths

    Returns
    -------
    result: tuple
        (mel_outputs_postnet, mel_lengths, alignments) - the exact
        model output
    """
    return model.infer(text_sequences, input_lengths)


LossStats = namedtuple(
    "TacotronLoss", "loss mel_loss gate_loss attn_loss attn_weight"
)


class Loss(nn.Module):
    """The Tacotron loss implementation

    The loss consists of an MSE loss on the spectrogram, a BCE gate loss
    and a guided attention loss (if enabled) that attempts to make the
    attention matrix diagonal

    The output of the moduel is a LossStats tuple, which includes both the
    total loss

    Arguments
    ---------
    guided_attention_sigma: float
        The guided attention sigma factor, controling the "width" of
        the mask
    gate_loss_weight: float
        The constant by which the hate loss will be multiplied
    guided_attention_weight: float
        The weight for the guided attention
    guided_attention_scheduler: callable
        The scheduler class for the guided attention loss
    guided_attention_hard_stop: int
        The number of epochs after which guided attention will be compeltely
        turned off

    Example:
    >>> import torch
    >>> _ = torch.manual_seed(42)
    >>> from speechbrain.lobes.models.Tacotron2 import Loss
    >>> loss = Loss(guided_attention_sigma=0.2)
    >>> mel_target = torch.randn(2, 80, 861)
    >>> gate_target = torch.randn(1722, 1)
    >>> mel_out = torch.randn(2, 80, 861)
    >>> mel_out_postnet = torch.randn(2, 80, 861)
    >>> gate_out = torch.randn(2, 861)
    >>> alignments = torch.randn(2, 861, 173)
    >>> targets = mel_target, gate_target
    >>> model_outputs = mel_out, mel_out_postnet, gate_out, alignments
    >>> input_lengths = torch.tensor([173,  91])
    >>> target_lengths = torch.tensor([861, 438])
    >>> loss(model_outputs, targets, input_lengths, target_lengths, 1)
    TacotronLoss(loss=tensor(4.8566), mel_loss=tensor(4.0097), gate_loss=tensor(0.8460), attn_loss=tensor(0.0010), attn_weight=tensor(1.))
    """

    def __init__(
        self,
        guided_attention_sigma=None,
        gate_loss_weight=1.0,
        guided_attention_weight=1.0,
        guided_attention_scheduler=None,
        guided_attention_hard_stop=None,
    ):
        super().__init__()
        if guided_attention_weight == 0:
            guided_attention_weight = None
        self.guided_attention_weight = guided_attention_weight
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.guided_attention_loss = GuidedAttentionLoss(
            sigma=guided_attention_sigma
        )
        self.gate_loss_weight = gate_loss_weight
        self.guided_attention_weight = guided_attention_weight
        self.guided_attention_scheduler = guided_attention_scheduler
        self.guided_attention_hard_stop = guided_attention_hard_stop

    def forward(
        self, model_output, targets, input_lengths, target_lengths, epoch
    ):
        """Computes the loss

        Arguments
        ---------
        model_output: tuple
            the output of the model's forward():
            (mel_outputs, mel_outputs_postnet, gate_outputs, alignments)
        targets: tuple
            the targets
        input_lengths: torch.Tensor
            a (batch, length) tensor of input lengths
        target_lengths: torch.Tensor
            a (batch, length) tensor of target (spectrogram) lengths
        epoch: int
            the current epoch number (used for the scheduling of the guided attention
            loss) A StepScheduler is typically used

        Returns
        -------
        result: LossStats
            the total loss - and individual losses (mel and gate)

        """
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, alignments = model_output

        gate_out = gate_out.view(-1, 1)
        mel_loss = self.mse_loss(mel_out, mel_target) + self.mse_loss(
            mel_out_postnet, mel_target
        )
        gate_loss = self.gate_loss_weight * self.bce_loss(gate_out, gate_target)
        attn_loss, attn_weight = self.get_attention_loss(
            alignments, input_lengths, target_lengths, epoch
        )
        total_loss = mel_loss + gate_loss + attn_loss
        return LossStats(
            total_loss, mel_loss, gate_loss, attn_loss, attn_weight
        )

    def get_attention_loss(
        self, alignments, input_lengths, target_lengths, epoch
    ):
        """Computes the attention loss

        Arguments
        ---------
        alignments: torch.Tensor
            the aligment matrix from the model
        input_lengths: torch.Tensor
            a (batch, length) tensor of input lengths
        target_lengths: torch.Tensor
            a (batch, length) tensor of target (spectrogram) lengths
        epoch: int
            the current epoch number (used for the scheduling of the guided attention
            loss) A StepScheduler is typically used

        Returns
        -------
        attn_loss: torch.Tensor
            the attention loss value
        """
        zero_tensor = torch.tensor(0.0, device=alignments.device)
        if (
            self.guided_attention_weight is None
            or self.guided_attention_weight == 0
        ):
            attn_weight, attn_loss = zero_tensor, zero_tensor
        else:
            hard_stop_reached = (
                self.guided_attention_hard_stop is not None
                and epoch > self.guided_attention_hard_stop
            )
            if hard_stop_reached:
                attn_weight, attn_loss = zero_tensor, zero_tensor
            else:
                attn_weight = self.guided_attention_weight
                if self.guided_attention_scheduler is not None:
                    _, attn_weight = self.guided_attention_scheduler(epoch)
            attn_weight = torch.tensor(attn_weight, device=alignments.device)
            attn_loss = attn_weight * self.guided_attention_loss(
                alignments, input_lengths, target_lengths
            )
        return attn_loss, attn_weight


class TextMelCollate:
    """ Zero-pads model inputs and targets based on number of frames per step

    Arguments
    ---------
    n_frames_per_step: int
        the number of output frames per step

    Returns
    -------
    result: tuple
        a tuple of tensors to be used as inputs/targets
        (
            text_padded,
            input_lengths,
            mel_padded,
            gate_padded,
            output_lengths,
            len_x
        )
    """

    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    # TODO: Make this more intuitive, use the pipeline
    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        Arguments
        ---------
        batch: list
            [text_normalized, mel_normalized]
        """

        # TODO: Remove for loops and this dirty hack
        raw_batch = list(batch)
        for i in range(
            len(batch)
        ):  # the pipline return a dictionary wiht one elemnent
            batch[i] = batch[i]["mel_text_pair"]

        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True
        )
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, : text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += (
                self.n_frames_per_step - max_target_len % self.n_frames_per_step
            )
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        labels, wavs = [], []
        for i in range(len(ids_sorted_decreasing)):
            idx = ids_sorted_decreasing[i]
            mel = batch[idx][1]
            mel_padded[i, :, : mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1 :] = 1
            output_lengths[i] = mel.size(1)
            labels.append(raw_batch[idx]["label"])
            wavs.append(raw_batch[idx]["wav"])

        # count number of items - characters in text
        len_x = [x[2] for x in batch]
        len_x = torch.Tensor(len_x)
        return (
            text_padded,
            input_lengths,
            mel_padded,
            gate_padded,
            output_lengths,
            len_x,
            labels,
            wavs,
        )


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """Dynamic range compression for audio signals
    """
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
    """
    from torchaudio import transforms

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
