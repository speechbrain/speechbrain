# coding: utf-8

#TODO: Reference implementation - remove this.
#It was added for comparison purposes - minimal modifications
#from the r9y9 original

import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np



class Conv1dModule(nn.Conv1d):
    """Extended nn.Conv1d for incremental dilated convolutions
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clear_buffer()
        self._linearized_weight = None
        self.register_backward_hook(self._clear_linearized_weight)

    def incremental_forward(self, input):
        # input: (B, T, C)
        if self.training:
            raise RuntimeError('incremental_forward only supports eval mode')

        # run forward pre hooks (e.g., weight norm)
        for hook in self._forward_pre_hooks.values():
            hook(self, input)

        # reshape weight
        weight = self._get_linearized_weight()
        kw = self.kernel_size[0]
        dilation = self.dilation[0]

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
        output = F.linear(input.view(bsz, -1), weight, self.bias)
        return output.view(bsz, 1, -1)

    def clear_buffer(self):
        self.input_buffer = None

    def _get_linearized_weight(self):
        if self._linearized_weight is None:
            kw = self.kernel_size[0]
            # nn.Conv1d
            if self.weight.size() == (self.out_channels, self.in_channels, kw):
                weight = self.weight.transpose(1, 2).contiguous()
            else:
                # fairseq.modules.conv_tbc.ConvTBC
                weight = self.weight.transpose(2, 1).transpose(1, 0).contiguous()
            assert weight.size() == (self.out_channels, kw, self.in_channels)
            self._linearized_weight = weight.view(self.out_channels, -1)
        return self._linearized_weight

    def _clear_linearized_weight(self, *args):
        self._linearized_weight = None



def position_encoding_init(n_position, d_pos_vec, position_rate=1.0,
                           sinusoidal=True):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [position_rate * pos / np.power(10000, 2 * (i // 2) / d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc = torch.from_numpy(position_enc).float()
    if sinusoidal:
        position_enc[1:, 0::2] = torch.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = torch.cos(position_enc[1:, 1::2])  # dim 2i+1

    return position_enc


def sinusoidal_encode(x, w):
    y = w * x
    y[1:, 0::2] = torch.sin(y[1:, 0::2].clone())
    y[1:, 1::2] = torch.cos(y[1:, 1::2].clone())
    return y


class SinusoidalEncoding(nn.Embedding):

    def __init__(self, num_embeddings, embedding_dim,
                 *args, **kwargs):
        super(SinusoidalEncoding, self).__init__(num_embeddings, embedding_dim,
                                                 padding_idx=0,
                                                 *args, **kwargs)
        self.weight.data = position_encoding_init(num_embeddings, embedding_dim,
                                                  position_rate=1.0,
                                                  sinusoidal=False)

    def forward(self, x, w=1.0):
        isscaler = np.isscalar(w)
        assert self.padding_idx is not None

        if isscaler or w.size(0) == 1:
            weight = sinusoidal_encode(self.weight, w)
            return F.embedding(
                x, weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)
        else:
            # TODO: cannot simply apply for batch
            # better to implement efficient function
            pe = []
            for batch_idx, we in enumerate(w):
                weight = sinusoidal_encode(self.weight, we)
                pe.append(F.embedding(
                    x[batch_idx], weight, self.padding_idx, self.max_norm,
                    self.norm_type, self.scale_grad_by_freq, self.sparse))
            pe = torch.stack(pe)
            return pe


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        ctx.mark_shared_storage((x, res))
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def Linear(in_features, out_features, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def Embedding(num_embeddings, embedding_dim, padding_idx, std=0.01):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, std)
    return m


def Conv1d(in_channels, out_channels, kernel_size, dropout=0, std_mul=4.0, **kwargs):
    m = Conv1dModule(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((std_mul * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def ConvTranspose1d(in_channels, out_channels, kernel_size, dropout=0,
                    std_mul=1.0, **kwargs):
    m = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((std_mul * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


class Conv1dGLU(nn.Module):
    """(Dilated) Conv1d + Gated linear unit + (optionally) speaker embedding
    """

    def __init__(self, n_speakers, speaker_embed_dim,
                 in_channels, out_channels, kernel_size,
                 dropout, padding=None, dilation=1, causal=False, residual=False,
                 *args, **kwargs):
        super(Conv1dGLU, self).__init__()
        self.dropout = dropout
        self.residual = residual
        if padding is None:
            # no future time stamps available
            if causal:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
        self.causal = causal

        self.conv = Conv1d(in_channels, 2 * out_channels, kernel_size,
                           dropout=dropout, padding=padding, dilation=dilation,
                           *args, **kwargs)
        if n_speakers > 1:
            self.speaker_proj = Linear(speaker_embed_dim, out_channels)
        else:
            self.speaker_proj = None

    def forward(self, x, speaker_embed=None):
        return self._forward(x, speaker_embed, False)

    def incremental_forward(self, x, speaker_embed=None):
        return self._forward(x, speaker_embed, True)

    def _forward(self, x, speaker_embed, is_incremental):
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        if is_incremental:
            splitdim = -1
            x = self.conv.incremental_forward(x)
        else:
            splitdim = 1
            x = self.conv(x)
            # remove future time steps
            x = x[:, :, :residual.size(-1)] if self.causal else x

        a, b = x.split(x.size(splitdim) // 2, dim=splitdim)
        if self.speaker_proj is not None:
            softsign = F.softsign(self.speaker_proj(speaker_embed))
            # Since conv layer assumes BCT, we need to transpose
            softsign = softsign if is_incremental else softsign.transpose(1, 2)
            a = a + softsign
        x = a * torch.sigmoid(b)
        return (x + residual) * math.sqrt(0.5) if self.residual else x

    def clear_buffer(self):
        self.conv.clear_buffer()


class HighwayConv1d(nn.Module):
    """Weight normzlized Conv1d + Highway network (support incremental forward)
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, padding=None,
                 dilation=1, causal=False, dropout=0, std_mul=None, glu=False):
        super(HighwayConv1d, self).__init__()
        if std_mul is None:
            std_mul = 4.0 if glu else 1.0
        if padding is None:
            # no future time stamps available
            if causal:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
        self.causal = causal
        self.dropout = dropout
        self.glu = glu

        self.conv = Conv1d(in_channels, 2 * out_channels,
                           kernel_size=kernel_size, padding=padding,
                           dilation=dilation, dropout=dropout,
                           std_mul=std_mul)

    def forward(self, x):
        return self._forward(x, False)

    def incremental_forward(self, x):
        return self._forward(x, True)

    def _forward(self, x, is_incremental):
        """Forward

        Args:
            x: (B, in_channels, T)
        returns:
            (B, out_channels, T)
        """

        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        if is_incremental:
            splitdim = -1
            x = self.conv.incremental_forward(x)
        else:
            splitdim = 1
            x = self.conv(x)
            # remove future time steps
            x = x[:, :, :residual.size(-1)] if self.causal else x

        if self.glu:
            x = F.glu(x, dim=splitdim)
            return (x + residual) * math.sqrt(0.5)
        else:
            a, b = x.split(x.size(splitdim) // 2, dim=splitdim)
            T = torch.sigmoid(b)
            return (T * a + (1 - T) * residual)

    def clear_buffer(self):
        self.conv.clear_buffer()


def get_mask_from_lengths(memory, memory_lengths):
    """Get mask tensor from list of length
    Args:
        memory: (batch, max_time, dim)
        memory_lengths: array like
    """
    max_len = max(memory_lengths)
    mask = torch.arange(max_len).expand(memory.size(0), max_len) < torch.tensor(memory_lengths).unsqueeze(-1)
    mask = mask.to(memory.device)
    return ~mask

def expand_speaker_embed(inputs_btc, speaker_embed=None, tdim=1):
    if speaker_embed is None:
        return None
    # expand speaker embedding for all time steps
    # (B, N) -> (B, T, N)
    ss = speaker_embed.size()
    speaker_embed_btc = speaker_embed.unsqueeze(1).expand(
        ss[0], inputs_btc.size(tdim), ss[-1])
    return speaker_embed_btc


class Encoder(nn.Module):
    def __init__(self, n_vocab, embed_dim, n_speakers, speaker_embed_dim,
                 padding_idx=None, embedding_weight_std=0.1,
                 convolutions=((64, 5, .1),) * 7,
                 max_positions=512, dropout=0.1, apply_grad_scaling=False):
        super(Encoder, self).__init__()
        self.dropout = dropout
        self.num_attention_layers = None
        self.apply_grad_scaling = apply_grad_scaling

        # Text input embeddings
        self.embed_tokens = Embedding(
            n_vocab, embed_dim, padding_idx, embedding_weight_std)

        # Speaker embedding
        if n_speakers > 1:
            self.speaker_fc1 = Linear(speaker_embed_dim, embed_dim, dropout=dropout)
            self.speaker_fc2 = Linear(speaker_embed_dim, embed_dim, dropout=dropout)
        self.n_speakers = n_speakers

        # Non causual convolution blocks
        in_channels = embed_dim
        self.convolutions = nn.ModuleList()
        std_mul = 1.0
        for (out_channels, kernel_size, dilation) in convolutions:
            if in_channels != out_channels:
                # Conv1d + ReLU
                self.convolutions.append(
                    Conv1d(in_channels, out_channels, kernel_size=1, padding=0,
                           dilation=1, std_mul=std_mul))
                self.convolutions.append(nn.ReLU(inplace=True))
                in_channels = out_channels
                std_mul = 2.0
            self.convolutions.append(
                Conv1dGLU(n_speakers, speaker_embed_dim,
                          in_channels, out_channels, kernel_size, causal=False,
                          dilation=dilation, dropout=dropout, std_mul=std_mul,
                          residual=True))
            in_channels = out_channels
            std_mul = 4.0
        # Last 1x1 convolution
        self.convolutions.append(Conv1d(in_channels, embed_dim, kernel_size=1,
                                        padding=0, dilation=1, std_mul=std_mul,
                                        dropout=dropout))

    def forward(self, text_sequences, text_positions=None, lengths=None,
                speaker_embed=None):
        assert self.n_speakers == 1 or speaker_embed is not None

        # embed text_sequences
        x = self.embed_tokens(text_sequences.long())
        x = F.dropout(x, p=self.dropout, training=self.training)

        # expand speaker embedding for all time steps
        speaker_embed_btc = expand_speaker_embed(x, speaker_embed)
        if speaker_embed_btc is not None:
            speaker_embed_btc = F.dropout(speaker_embed_btc, p=self.dropout, training=self.training)
            x = x + F.softsign(self.speaker_fc1(speaker_embed_btc))

        input_embedding = x

        # B x T x C -> B x C x T
        x = x.transpose(1, 2)

        # １D conv blocks
        for f in self.convolutions:
            x = f(x, speaker_embed_btc) if isinstance(f, Conv1dGLU) else f(x)

        # Back to B x T x C
        keys = x.transpose(1, 2)

        if speaker_embed_btc is not None:
            keys = keys + F.softsign(self.speaker_fc2(speaker_embed_btc))

        # scale gradients (this only affects backward, not forward)
        if self.apply_grad_scaling and self.num_attention_layers is not None:
            keys = GradMultiply.apply(keys, 1.0 / (2.0 * self.num_attention_layers))

        # add output to input embedding for attention
        values = (keys + input_embedding) * math.sqrt(0.5)

        return keys, values


class AttentionLayer(nn.Module):
    def __init__(self, conv_channels, embed_dim, dropout=0.1,
                 window_ahead=3, window_backward=1,
                 key_projection=True, value_projection=True):
        super(AttentionLayer, self).__init__()
        self.query_projection = Linear(conv_channels, embed_dim)
        if key_projection:
            self.key_projection = Linear(embed_dim, embed_dim)
            # According to the DeepVoice3 paper, intiailize weights to same values
            # TODO: Does this really work well? not sure..
            if conv_channels == embed_dim:
                self.key_projection.weight.data = self.query_projection.weight.data.clone()
        else:
            self.key_projection = None
        if value_projection:
            self.value_projection = Linear(embed_dim, embed_dim)
        else:
            self.value_projection = None

        self.out_projection = Linear(embed_dim, conv_channels)
        self.dropout = dropout
        self.window_ahead = window_ahead
        self.window_backward = window_backward

    def forward(self, query, encoder_out, mask=None, last_attended=None):
        keys, values = encoder_out
        residual = query
        if self.value_projection is not None:
            values = self.value_projection(values)
        # TODO: yes, this is inefficient
        if self.key_projection is not None:
            keys = self.key_projection(keys.transpose(1, 2)).transpose(1, 2)

        # attention
        x = self.query_projection(query)
        x = torch.bmm(x, keys)

        mask_value = -float("inf")
        if mask is not None:
            mask = mask.view(query.size(0), 1, -1)
            x.data.masked_fill_(mask, mask_value)

        if last_attended is not None:
            backward = last_attended - self.window_backward
            if backward > 0:
                x[:, :, :backward] = mask_value
            ahead = last_attended + self.window_ahead
            if ahead < x.size(-1):
                x[:, :, ahead:] = mask_value

        # softmax over last dim
        # (B, tgt_len, src_len)
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=1)
        x = x.view(sz)
        attn_scores = x

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = torch.bmm(x, values)

        # scale attention output
        s = values.size(1)
        x = x * (s * math.sqrt(1.0 / s))

        # project back
        x = self.out_projection(x)
        x = (x + residual) * math.sqrt(0.5)
        return x, attn_scores


class Decoder(nn.Module):
    def __init__(self, embed_dim, n_speakers, speaker_embed_dim,
                 in_dim=80, r=5,
                 max_positions=512, padding_idx=None,
                 preattention=((128, 5, 1),) * 4,
                 convolutions=((128, 5, 1),) * 4,
                 attention=True, dropout=0.1,
                 use_memory_mask=False,
                 force_monotonic_attention=False,
                 query_position_rate=1.0,
                 key_position_rate=1.29,
                 window_ahead=3,
                 window_backward=1,
                 key_projection=True,
                 value_projection=True,
                 ):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.r = r
        self.query_position_rate = query_position_rate
        self.key_position_rate = key_position_rate

        in_channels = in_dim * r
        if isinstance(attention, bool):
            # expand True into [True, True, ...] and do the same with False
            attention = [attention] * len(convolutions)

        # Position encodings for query (decoder states) and keys (encoder states)
        self.embed_query_positions = SinusoidalEncoding(
            max_positions, convolutions[0][0])
        self.embed_keys_positions = SinusoidalEncoding(
            max_positions, embed_dim)
        # Used for compute multiplier for positional encodings
        if n_speakers > 1:
            self.speaker_proj1 = Linear(speaker_embed_dim, 1, dropout=dropout)
            self.speaker_proj2 = Linear(speaker_embed_dim, 1, dropout=dropout)
        else:
            self.speaker_proj1, self.speaker_proj2 = None, None

        # Prenet: causal convolution blocks
        self.preattention = nn.ModuleList()
        in_channels = in_dim * r
        std_mul = 1.0
        for out_channels, kernel_size, dilation in preattention:
            if in_channels != out_channels:
                # Conv1d + ReLU
                self.preattention.append(
                    Conv1d(in_channels, out_channels, kernel_size=1, padding=0,
                           dilation=1, std_mul=std_mul))
                self.preattention.append(nn.ReLU(inplace=True))
                in_channels = out_channels
                std_mul = 2.0
            self.preattention.append(
                Conv1dGLU(n_speakers, speaker_embed_dim,
                          in_channels, out_channels, kernel_size, causal=True,
                          dilation=dilation, dropout=dropout, std_mul=std_mul,
                          residual=True))
            in_channels = out_channels
            std_mul = 4.0

        # Causal convolution blocks + attention layers
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()

        for i, (out_channels, kernel_size, dilation) in enumerate(convolutions):
            assert in_channels == out_channels
            self.convolutions.append(
                Conv1dGLU(n_speakers, speaker_embed_dim,
                          in_channels, out_channels, kernel_size, causal=True,
                          dilation=dilation, dropout=dropout, std_mul=std_mul,
                          residual=False))
            self.attention.append(
                AttentionLayer(out_channels, embed_dim,
                               dropout=dropout,
                               window_ahead=window_ahead,
                               window_backward=window_backward,
                               key_projection=key_projection,
                               value_projection=value_projection)
                if attention[i] else None)
            in_channels = out_channels
            std_mul = 4.0
        # Last 1x1 convolution
        self.last_conv = Conv1d(in_channels, in_dim * r, kernel_size=1,
                                padding=0, dilation=1, std_mul=std_mul,
                                dropout=dropout)

        # Mel-spectrogram (before sigmoid) -> Done binary flag
        self.fc = Linear(in_dim * r, 1)

        self.max_decoder_steps = 200
        self.min_decoder_steps = 10
        self.use_memory_mask = use_memory_mask
        if isinstance(force_monotonic_attention, bool):
            self.force_monotonic_attention = [force_monotonic_attention] * len(convolutions)
        else:
            self.force_monotonic_attention = force_monotonic_attention

    def forward(self, encoder_out, inputs=None,
                text_positions=None, frame_positions=None,
                speaker_embed=None, lengths=None):
        if inputs is None:
            assert text_positions is not None
            self.start_fresh_sequence()
            outputs = self.incremental_forward(encoder_out, text_positions, speaker_embed)
            return outputs

        # Grouping multiple frames if necessary
        if inputs.size(-1) == self.in_dim:
            inputs = inputs.view(inputs.size(0), inputs.size(1) // self.r, -1)
        assert inputs.size(-1) == self.in_dim * self.r

        # expand speaker embedding for all time steps
        speaker_embed_btc = expand_speaker_embed(inputs, speaker_embed)
        if speaker_embed_btc is not None:
            speaker_embed_btc = F.dropout(speaker_embed_btc, p=self.dropout, training=self.training)

        keys, values = encoder_out

        if self.use_memory_mask and lengths is not None:
            mask = get_mask_from_lengths(keys, lengths)
        else:
            mask = None

        # position encodings
        if text_positions is not None:
            w = self.key_position_rate
            # TODO: may be useful to have projection per attention layer
            if self.speaker_proj1 is not None:
                w = w * torch.sigmoid(self.speaker_proj1(speaker_embed)).view(-1)
            text_pos_embed = self.embed_keys_positions(text_positions, w)
            keys = keys + text_pos_embed
        if frame_positions is not None:
            w = self.query_position_rate
            if self.speaker_proj2 is not None:
                w = w * torch.sigmoid(self.speaker_proj2(speaker_embed)).view(-1)
            frame_pos_embed = self.embed_query_positions(frame_positions, w)

        # transpose only once to speed up attention layers
        keys = keys.transpose(1, 2).contiguous()

        x = inputs
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Generic case: B x T x C -> B x C x T
        x = x.transpose(1, 2)

        # Prenet
        for f in self.preattention:
            x = f(x, speaker_embed_btc) if isinstance(f, Conv1dGLU) else f(x)

        # Casual convolutions + Multi-hop attentions
        alignments = []
        for f, attention in zip(self.convolutions, self.attention):
            residual = x

            x = f(x, speaker_embed_btc) if isinstance(f, Conv1dGLU) else f(x)

            # Feed conv output to attention layer as query
            if attention is not None:
                assert isinstance(f, Conv1dGLU)
                # (B x T x C)
                x = x.transpose(1, 2)
                x = x if frame_positions is None else x + frame_pos_embed
                x, alignment = attention(x, (keys, values), mask=mask)
                # (T x B x C)
                x = x.transpose(1, 2)
                alignments += [alignment]

            if isinstance(f, Conv1dGLU):
                x = (x + residual) * math.sqrt(0.5)

        # decoder state (B x T x C):
        # internal representation before compressed to output dimention
        decoder_states = x.transpose(1, 2).contiguous()
        x = self.last_conv(x)

        # Back to B x T x C
        x = x.transpose(1, 2)

        # project to mel-spectorgram
        outputs = torch.sigmoid(x)

        # Done flag
        done = torch.sigmoid(self.fc(x))

        return outputs, torch.stack(alignments), done, decoder_states

    def incremental_forward(self, encoder_out, text_positions, speaker_embed=None,
                            initial_input=None, test_inputs=None):
        keys, values = encoder_out
        B = keys.size(0)

        # position encodings
        w = self.key_position_rate
        # TODO: may be useful to have projection per attention layer
        if self.speaker_proj1 is not None:
            w = w * torch.sigmoid(self.speaker_proj1(speaker_embed)).view(-1)
        text_pos_embed = self.embed_keys_positions(text_positions, w)
        keys = keys + text_pos_embed

        # transpose only once to speed up attention layers
        keys = keys.transpose(1, 2).contiguous()

        decoder_states = []
        outputs = []
        alignments = []
        dones = []
        # intially set to zeros
        last_attended = [None] * len(self.attention)
        for idx, v in enumerate(self.force_monotonic_attention):
            last_attended[idx] = 0 if v else None

        num_attention_layers = sum([layer is not None for layer in self.attention])
        t = 0
        if initial_input is None:
            initial_input = keys.data.new(B, 1, self.in_dim * self.r).zero_()
        current_input = initial_input
        while True:
            # frame pos start with 1.
            frame_pos = keys.data.new(B, 1).fill_(t + 1).long()
            w = self.query_position_rate
            if self.speaker_proj2 is not None:
                w = w * torch.sigmoid(self.speaker_proj2(speaker_embed)).view(-1)
            frame_pos_embed = self.embed_query_positions(frame_pos, w)

            if test_inputs is not None:
                if t >= test_inputs.size(1):
                    break
                current_input = test_inputs[:, t, :].unsqueeze(1)
            else:
                if t > 0:
                    current_input = outputs[-1]
            x = current_input
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Prenet
            for f in self.preattention:
                if isinstance(f, Conv1dGLU):
                    x = f.incremental_forward(x, speaker_embed)
                else:
                    try:
                        x = f.incremental_forward(x)
                    except AttributeError as e:
                        x = f(x)

            # Casual convolutions + Multi-hop attentions
            ave_alignment = None
            for idx, (f, attention) in enumerate(zip(self.convolutions,
                                                     self.attention)):
                residual = x
                if isinstance(f, Conv1dGLU):
                    x = f.incremental_forward(x, speaker_embed)
                else:
                    try:
                        x = f.incremental_forward(x)
                    except AttributeError as e:
                        x = f(x)

                # attention
                if attention is not None:
                    assert isinstance(f, Conv1dGLU)
                    x = x + frame_pos_embed
                    x, alignment = attention(x, (keys, values),
                                             last_attended=last_attended[idx])
                    if self.force_monotonic_attention[idx]:
                        last_attended[idx] = alignment.max(-1)[1].view(-1).data[0]
                    if ave_alignment is None:
                        ave_alignment = alignment
                    else:
                        ave_alignment = ave_alignment + ave_alignment

                # residual
                if isinstance(f, Conv1dGLU):
                    x = (x + residual) * math.sqrt(0.5)

            decoder_state = x
            x = self.last_conv.incremental_forward(x)
            ave_alignment = ave_alignment.div_(num_attention_layers)

            # Ooutput & done flag predictions
            output = torch.sigmoid(x)
            done = torch.sigmoid(self.fc(x))

            decoder_states += [decoder_state]
            outputs += [output]
            alignments += [ave_alignment]
            dones += [done]

            t += 1
            if test_inputs is None:
                if (done > 0.5).all() and t > self.min_decoder_steps:
                    break
                elif t > self.max_decoder_steps:
                    break

        # Remove 1-element time axis
        alignments = list(map(lambda x: x.squeeze(1), alignments))
        decoder_states = list(map(lambda x: x.squeeze(1), decoder_states))
        outputs = list(map(lambda x: x.squeeze(1), outputs))

        # Combine outputs for all time steps
        alignments = torch.stack(alignments).transpose(0, 1)
        decoder_states = torch.stack(decoder_states).transpose(0, 1).contiguous()
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()
        dones = torch.cat(dones, dim=1)

        return outputs, alignments, dones, decoder_states

    def start_fresh_sequence(self):
        _clear_modules(self.preattention)
        _clear_modules(self.convolutions)
        self.last_conv.clear_buffer()


def _clear_modules(modules):
    for m in modules:
        try:
            m.clear_buffer()
        except AttributeError as e:
            pass


class Converter(nn.Module):
    def __init__(self, n_speakers, speaker_embed_dim,
                 in_dim, out_dim, convolutions=((256, 5, 1),) * 4,
                 time_upsampling=1,
                 dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_speakers = n_speakers

        # Non causual convolution blocks
        in_channels = convolutions[0][0]
        # Idea from nyanko
        if time_upsampling == 4:
            self.convolutions = nn.ModuleList([
                Conv1d(in_dim, in_channels, kernel_size=1, padding=0, dilation=1,
                       std_mul=1.0),
                ConvTranspose1d(in_channels, in_channels, kernel_size=2,
                                padding=0, stride=2, std_mul=1.0),
                Conv1dGLU(n_speakers, speaker_embed_dim,
                          in_channels, in_channels, kernel_size=3, causal=False,
                          dilation=1, dropout=dropout, std_mul=1.0, residual=True),
                Conv1dGLU(n_speakers, speaker_embed_dim,
                          in_channels, in_channels, kernel_size=3, causal=False,
                          dilation=3, dropout=dropout, std_mul=4.0, residual=True),
                ConvTranspose1d(in_channels, in_channels, kernel_size=2,
                                padding=0, stride=2, std_mul=4.0),
                Conv1dGLU(n_speakers, speaker_embed_dim,
                          in_channels, in_channels, kernel_size=3, causal=False,
                          dilation=1, dropout=dropout, std_mul=1.0, residual=True),
                Conv1dGLU(n_speakers, speaker_embed_dim,
                          in_channels, in_channels, kernel_size=3, causal=False,
                          dilation=3, dropout=dropout, std_mul=4.0, residual=True),
            ])
        elif time_upsampling == 2:
            self.convolutions = nn.ModuleList([
                Conv1d(in_dim, in_channels, kernel_size=1, padding=0, dilation=1,
                       std_mul=1.0),
                ConvTranspose1d(in_channels, in_channels, kernel_size=2,
                                padding=0, stride=2, std_mul=1.0),
                Conv1dGLU(n_speakers, speaker_embed_dim,
                          in_channels, in_channels, kernel_size=3, causal=False,
                          dilation=1, dropout=dropout, std_mul=1.0, residual=True),
                Conv1dGLU(n_speakers, speaker_embed_dim,
                          in_channels, in_channels, kernel_size=3, causal=False,
                          dilation=3, dropout=dropout, std_mul=4.0, residual=True),
            ])
        elif time_upsampling == 1:
            self.convolutions = nn.ModuleList([
                # 1x1 convolution first
                Conv1d(in_dim, in_channels, kernel_size=1, padding=0, dilation=1,
                       std_mul=1.0),
                Conv1dGLU(n_speakers, speaker_embed_dim,
                          in_channels, in_channels, kernel_size=3, causal=False,
                          dilation=3, dropout=dropout, std_mul=4.0, residual=True),
            ])
        else:
            raise ValueError("Not supported")

        std_mul = 4.0
        for (out_channels, kernel_size, dilation) in convolutions:
            if in_channels != out_channels:
                self.convolutions.append(
                    Conv1d(in_channels, out_channels, kernel_size=1, padding=0,
                           dilation=1, std_mul=std_mul))
                self.convolutions.append(nn.ReLU(inplace=True))
                in_channels = out_channels
                std_mul = 2.0
            self.convolutions.append(
                Conv1dGLU(n_speakers, speaker_embed_dim,
                          in_channels, out_channels, kernel_size, causal=False,
                          dilation=dilation, dropout=dropout, std_mul=std_mul,
                          residual=True))
            in_channels = out_channels
            std_mul = 4.0
        # Last 1x1 convolution
        self.convolutions.append(Conv1d(in_channels, out_dim, kernel_size=1,
                                        padding=0, dilation=1, std_mul=std_mul,
                                        dropout=dropout))

    def forward(self, x, speaker_embed=None):
        assert self.n_speakers == 1 or speaker_embed is not None

        # expand speaker embedding for all time steps
        speaker_embed_btc = expand_speaker_embed(x, speaker_embed)
        if speaker_embed_btc is not None:
            speaker_embed_btc = F.dropout(speaker_embed_btc, p=self.dropout, training=self.training)

        # Generic case: B x T x C -> B x C x T
        x = x.transpose(1, 2)

        for f in self.convolutions:
            # Case for upsampling
            if speaker_embed_btc is not None and speaker_embed_btc.size(1) != x.size(-1):
                speaker_embed_btc = expand_speaker_embed(x, speaker_embed, tdim=-1)
                speaker_embed_btc = F.dropout(
                    speaker_embed_btc, p=self.dropout, training=self.training)
            x = f(x, speaker_embed_btc) if isinstance(f, Conv1dGLU) else f(x)

        # Back to B x T x C
        x = x.transpose(1, 2)

        return torch.sigmoid(x)


def guided_attention(N, max_N, T, max_T, g):
    W = np.zeros((max_N, max_T), dtype=np.float32)
    # TODO: Vectorize
    for n in range(N):
        for t in range(T):
            W[n, t] = 1 - np.exp(-(n / N - t / T)**2 / (2 * g * g))
    return W


def guided_attentions(input_lengths, target_lengths, max_target_len, g=0.2):
    B = len(input_lengths)
    max_input_len = input_lengths.max()
    W = np.zeros((B, max_target_len, max_input_len), dtype=np.float32)
    for b in range(B):
        W[b] = guided_attention(input_lengths[b], max_input_len,
                                target_lengths[b], max_target_len, g).T
    return W

def deepvoice3(n_vocab, embed_dim=256, mel_dim=80, linear_dim=513, r=4,
               downsample_step=1,
               n_speakers=1, speaker_embed_dim=16, padding_idx=0,
               dropout=(1 - 0.95), kernel_size=5,
               encoder_channels=128,
               decoder_channels=256,
               converter_channels=256,
               query_position_rate=1.0,
               key_position_rate=1.29,
               use_memory_mask=False,
               trainable_positional_encodings=False,
               force_monotonic_attention=True,
               use_decoder_state_for_postnet_input=True,
               max_positions=512,
               embedding_weight_std=0.1,
               speaker_embedding_weight_std=0.01,
               freeze_embedding=False,
               window_ahead=3,
               window_backward=1,
               key_projection=False,
               value_projection=False,
               **kwargs
               ):
    """Build deepvoice3
    """
    time_upsampling = max(downsample_step // r, 1)

    # Seq2seq
    h = encoder_channels  # hidden dim (channels)
    k = kernel_size   # kernel size
    encoder = Encoder(
        n_vocab, embed_dim, padding_idx=padding_idx,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        dropout=dropout, max_positions=max_positions,
        embedding_weight_std=embedding_weight_std,
        # (channels, kernel_size, dilation)
        convolutions=[(h, k, 1), (h, k, 3), (h, k, 9), (h, k, 27),
                      (h, k, 1), (h, k, 3), (h, k, 9), (h, k, 27),
                      (h, k, 1), (h, k, 3)],
    )

    h = decoder_channels
    decoder = Decoder(
        embed_dim, in_dim=mel_dim, r=r, padding_idx=padding_idx,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        dropout=dropout, max_positions=max_positions,
        preattention=[(h, k, 1), (h, k, 3)],
        convolutions=[(h, k, 1), (h, k, 3), (h, k, 9), (h, k, 27),
                      (h, k, 1)],
        attention=[True, False, False, False, True],
        force_monotonic_attention=force_monotonic_attention,
        query_position_rate=query_position_rate,
        key_position_rate=key_position_rate,
        use_memory_mask=use_memory_mask,
        window_ahead=window_ahead,
        window_backward=window_backward,
        key_projection=key_projection,
        value_projection=value_projection,
    )

    seq2seq = AttentionSeq2Seq(encoder, decoder)

    # Post net
    if use_decoder_state_for_postnet_input:
        in_dim = h // r
    else:
        in_dim = mel_dim
    h = converter_channels
    converter = Converter(
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        in_dim=in_dim, out_dim=linear_dim, dropout=dropout,
        time_upsampling=time_upsampling,
        convolutions=[(h, k, 1), (h, k, 3), (2 * h, k, 1), (2 * h, k, 3)],
    )

    # Seq2seq + post net
    model = MultiSpeakerTTSModel(
        seq2seq, converter, padding_idx=padding_idx,
        mel_dim=mel_dim, linear_dim=linear_dim,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        trainable_positional_encodings=trainable_positional_encodings,
        use_decoder_state_for_postnet_input=use_decoder_state_for_postnet_input,
        speaker_embedding_weight_std=speaker_embedding_weight_std,
        freeze_embedding=freeze_embedding)

    return model

class MultiSpeakerTTSModel(nn.Module):
    """Attention seq2seq model + post processing network

    """

    def __init__(self, seq2seq, postnet,
                 mel_dim=80, linear_dim=513,
                 n_speakers=1, speaker_embed_dim=16, padding_idx=None,
                 trainable_positional_encodings=False,
                 use_decoder_state_for_postnet_input=False,
                 speaker_embedding_weight_std=0.01,
                 freeze_embedding=False):
        super(MultiSpeakerTTSModel, self).__init__()
        self.seq2seq = seq2seq
        self.postnet = postnet  # referred as "Converter" in DeepVoice3
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim
        self.trainable_positional_encodings = trainable_positional_encodings
        self.use_decoder_state_for_postnet_input = use_decoder_state_for_postnet_input
        self.freeze_embedding = freeze_embedding

        # Speaker embedding
        if n_speakers > 1:
            self.embed_speakers = Embedding(
                n_speakers, speaker_embed_dim, padding_idx=None,
                std=speaker_embedding_weight_std)
        self.n_speakers = n_speakers
        self.speaker_embed_dim = speaker_embed_dim

    def make_generation_fast_(self):

        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return
        self.apply(remove_weight_norm)

    def get_trainable_parameters(self):
        freezed_param_ids = set()

        encoder, decoder = self.seq2seq.encoder, self.seq2seq.decoder

        # Avoid updating the position encoding
        if not self.trainable_positional_encodings:
            pe_query_param_ids = set(map(id, decoder.embed_query_positions.parameters()))
            pe_keys_param_ids = set(map(id, decoder.embed_keys_positions.parameters()))
            freezed_param_ids |= (pe_query_param_ids | pe_keys_param_ids)
        # Avoid updating the text embedding
        if self.freeze_embedding:
            embed_param_ids = set(map(id, encoder.embed_tokens.parameters()))
            freezed_param_ids |= embed_param_ids

        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def forward(self, text_sequences, mel_targets=None, speaker_ids=None,
                text_positions=None, frame_positions=None, input_lengths=None):
        B = text_sequences.size(0)

        if speaker_ids is not None:
            assert self.n_speakers > 1
            speaker_embed = self.embed_speakers(speaker_ids)
        else:
            speaker_embed = None

        # Apply seq2seq
        # (B, T//r, mel_dim*r)
        mel_outputs, alignments, done, decoder_states = self.seq2seq(
            text_sequences, mel_targets, speaker_embed,
            text_positions, frame_positions, input_lengths)

        # Reshape
        # (B, T, mel_dim)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        # Prepare postnet inputs
        if self.use_decoder_state_for_postnet_input:
            postnet_inputs = decoder_states.view(B, mel_outputs.size(1), -1)
        else:
            postnet_inputs = mel_outputs

        # (B, T, linear_dim)
        # Convert coarse mel-spectrogram (or decoder hidden states) to
        # high resolution spectrogram
        linear_outputs = self.postnet(postnet_inputs, speaker_embed)
        assert linear_outputs.size(-1) == self.linear_dim

        return mel_outputs, linear_outputs, alignments, done


class AttentionSeq2Seq(nn.Module):
    """Encoder + Decoder with attention
    """

    def __init__(self, encoder, decoder):
        super(AttentionSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        if isinstance(self.decoder.attention, nn.ModuleList):
            self.encoder.num_attention_layers = sum(
                [layer is not None for layer in decoder.attention])

    def forward(self, text_sequences, mel_targets=None, speaker_embed=None,
                text_positions=None, frame_positions=None, input_lengths=None):
        # (B, T, text_embed_dim)
        encoder_outputs = self.encoder(
            text_sequences, lengths=input_lengths, speaker_embed=speaker_embed)

        # Mel: (B, T//r, mel_dim*r)
        # Alignments: (N, B, T_target, T_input)
        # Done: (B, T//r, 1)
        mel_outputs, alignments, done, decoder_states = self.decoder(
            encoder_outputs, mel_targets,
            text_positions=text_positions, frame_positions=frame_positions,
            speaker_embed=speaker_embed, lengths=input_lengths)

        return mel_outputs, alignments, done, decoder_states


class Loss(nn.Module):
    """
    The loss for the DeepVoice3 model
    """

    def __init__(
        self,
        linear_dim: int,
        downsample_step: int,
        outputs_per_step: int,
        masked_loss_weight: float,
        binary_divergence_weight: float,
        priority_freq_weight: float,
        priority_freq: float,
        sample_rate: float,
        guided_attention_sigma: float):
        """
        Class constructor

        Arguments
        ----------
        linear_dim : int
            The dimension of the linear layer
        downsample_step : int
            The number of steps of signal downsampling
        outputs_per_step: int
            The number of output steps for each decoder input step
        masked_loss_weight: float
            The relative weight of the masked loss
        binary_divergence_weight: float
            The relative weight of the binary divergence criterion (comparing linear outputs)
        

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient
        """

        super().__init__()
        self.linear_dim = linear_dim
        self.downsample_step = downsample_step
        self.outputs_per_step = outputs_per_step
        self.masked_loss_weight = masked_loss_weight
        self.binary_divergence_weight = binary_divergence_weight
        self.priority_freq_weight = priority_freq_weight
        self.priority_freq = priority_freq
        self.sample_rate = sample_rate
        self.guided_attention_sigma = guided_attention_sigma
        self.binary_criterion = nn.BCELoss()
        self.masked_l1 = MaskedL1Loss()
        self.l1 = nn.L1Loss()


    def forward(self, input, target, input_lengths):
        input_mel, input_linear, attention, input_done, _ = input
        target_mel, target_linear, target_done, target_lengths = target
        r = self.outputs_per_step
                
        decoder_target_mask = sequence_mask(
            target_lengths // (self.outputs_per_step * self.downsample_step),
            max_len=target_mel.size(1),
            device=input_mel.device).unsqueeze(-1)
        
        if self.downsample_step > 1:
            # spectrogram-domain mask
            target_mask = sequence_mask(
                target_lengths, max_len=target_linear.size(1),
                device=target_lengths.device).unsqueeze(-1)
        else:
            target_mask = decoder_target_mask
        
        decoder_target_mask = decoder_target_mask[:, r:, :]
        target_mask = target_mask[:, r:, :]

        mel_l1_loss, mel_binary_div = self.spec_loss(
            input_mel[:, :-self.outputs_per_step, :], target_mel[:, self.outputs_per_step:, :], decoder_target_mask,
            masked_loss_weight=self.masked_loss_weight,
            binary_divergence_weight=self.binary_divergence_weight)
        mel_loss = (1 - self.masked_loss_weight) * mel_l1_loss + self.masked_loss_weight * mel_binary_div
        done_loss = self.binary_criterion(input_done, target_done)

        n_priority_freq = int(self.priority_freq / (self.sample_rate * 0.5) * self.linear_dim)


        linear_l1_loss, linear_binary_div = self.spec_loss(
            input_linear[:, :-self.outputs_per_step, :], target_linear[:, self.outputs_per_step:, :], target_mask,
            priority_bin=n_priority_freq,
            priority_w=self.priority_freq_weight,
            masked_loss_weight=self.masked_loss_weight,
            binary_divergence_weight=self.binary_divergence_weight)
        linear_loss = (1 - self.masked_loss_weight) * linear_l1_loss + self.masked_loss_weight * linear_binary_div  
        
        decoder_lengths = target_lengths.cpu().long().numpy() // r // self.downsample_step
        soft_mask = guided_attentions(
            input_lengths, decoder_lengths,
            attention.size(-2), self.guided_attention_sigma)
        soft_mask = torch.from_numpy(soft_mask).to(input_mel.device)
        attn_loss = (attention * soft_mask).mean()

        loss = mel_loss + linear_loss + done_loss + attn_loss
        return loss

    def spec_loss(self, y_hat, y, mask, priority_bin=None, priority_w=0, masked_loss_weight=0., binary_divergence_weight=0.):

        w = masked_loss_weight
        # L1 loss
        if w > 0:
            assert mask is not None
            l1_loss = w * self.masked_l1(y_hat, y, mask=mask) + (1 - w) * self.l1(y_hat, y)
        else:
            assert mask is None
            l1_loss = self.l1(y_hat, y)

        # Priority L1 loss
        if priority_bin is not None and priority_w > 0:
            if w > 0:
                priority_loss = w * self.masked_l1(
                    y_hat[:, :, :priority_bin], y[:, :, :priority_bin], mask=mask) \
                    + (1 - w) * self.l1(y_hat[:, :, :priority_bin], y[:, :, :priority_bin])
            else:
                priority_loss = self.l1(y_hat[:, :, :priority_bin], y[:, :, :priority_bin])
            l1_loss = (1 - priority_w) * l1_loss + priority_w * priority_loss

        # Binary divergence loss
        if binary_divergence_weight <= 0:
            binary_div = y.data.new(1).zero_()
        else:
            y_hat_logits = logit(y_hat)
            z = -y * y_hat_logits + torch.log1p(torch.exp(y_hat_logits))
            if w > 0:
                binary_div = w * masked_mean(z, mask) + (1 - w) * z.mean()
            else:
                binary_div = z.mean()

        return l1_loss, binary_div



def sequence_mask(sequence_length, max_len=None, device=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len, device=device).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = sequence_length.unsqueeze(1) \
        .expand_as(seq_range_expand)
    result = (seq_range_expand < seq_length_expand.to(seq_range_expand.device)).float()
    if device is not None:
        result.to(device)
    return result




def logit(x, eps=1e-8):
    return torch.log(x + eps) - torch.log(1 - x + eps)


def masked_mean(y, mask):
    # (B, T, D)
    mask_ = mask.expand_as(y)
    return (y * mask_).sum() / mask_.sum()


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.criterion = nn.L1Loss(reduction="sum")

    def forward(self, input, target, lengths=None, mask=None, max_len=None):
        if lengths is None and mask is None:
            raise RuntimeError("Should provide either lengths or mask")

        # (B, T, 1)
        if mask is None:
            mask = sequence_mask(lengths, max_len).unsqueeze(-1)

        # (B, T, D)
        mask_ = mask.expand_as(input)
        loss = self.criterion(input * mask_, target * mask_)
        return loss / mask_.sum()

