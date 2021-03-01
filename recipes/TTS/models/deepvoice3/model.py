"""
DeepVoice 3 - Building Blocks

Elements of the architecture are inspired by
https://github.com/r9y9/deepvoice3_pytorch
"""

import math
import torch
import numpy as np
from torch.nn import functional as F
from torch import nn
from speechbrain.nnet import CNN
from typing import List


class WeightNorm(nn.Module):
    """
    A weight normalization wrapper for convolutional layers
    """
    def __init__(self, inner: CNN.Conv1d, dropout: float=0.1, std_mul: float=4.0):
        """
        Class constructor

        :param inner: A convolutional layer
        :param dropout: The drop-out rate (0.0 to 1.0)
        :param std_mul: The standard deviation multiplier
        """
        super().__init__()
        self.inner = inner
        self.dropout = dropout
        self.std_mul = std_mul
        self._apply_weight_norm()

    def _apply_weight_norm(self):
        std = math.sqrt(
            (self.std_mul * (1.0 - self.dropout)) / (self.inner.conv.kernel_size[0] * self.inner.conv.in_channels))
        self.inner.conv.weight.data.normal_(mean=0, std=std)
        self.inner.conv.bias.data.zero_()
    
    def forward(self, *args, **kwargs):
        return self.inner.forward(*args, **kwargs)



class ConvBlock(nn.Module):
    """
    A wrapper for the standard SpeechBrain convolution applying the weight normalization
    described in the paper
    """
    def __init__(
        self,
        padding: int=None,
        kernel_size: int=None,
        dilation: int=None,
        dropout: float=0.,
        std_mul: float=4.0,
        causal: bool=False,
        residual: bool=False,
        *args,
        **kwargs):
        """
        Class constructor. Any arguments not explicitly specified
        will be passed through to the Conv1d instance

        :param dropout: the dropout rate
        :param std_mul: the standard deviation multiplier
        """
        super().__init__()
        self.dropout = dropout
        self.std_mul = std_mul
        self.causal = causal
        self.residual = residual
        if padding is None:
            padding = 'causal' if self.causal else 'same'
        self.conv = WeightNorm(
            inner=CNN.Conv1d(
                *args, skip_transpose=True,
                kernel_size=kernel_size, padding=padding, dilation=dilation,
                **kwargs),
            dropout=dropout,
            std_mul=std_mul)
        self.glu = nn.GLU(dim=-2)
        self.multiplier = math.sqrt(0.5)

    def _apply_weight_norm(self):
        std = math.sqrt(
            (self.std_mul * (1.0 - self.dropout)) / (self.conv.kernel_size[0] * self.conv.in_channels))
        self.conv.weight.data.normal_(mean=0, std=std)
        self.conv.bias.data.zero_()
    
    def forward(self, x, *args, **kwargs):
        residual = x
        x = self.conv(x)
        x = self.glu(x)
        if self.causal:
            x = x[:, :, :residual.size(-1)]
        return (x + residual) * self.multiplier if self.residual else x


class Encoder(nn.Module):
    def __init__(self, 
                 n_vocab: int,
                 embed_dim: int,
                 convolutions: List[nn.Module]=[],
                 embedding_weight_std: float=0.1,
                 dropout: float=0.1):
        super().__init__()
        self.n_vocab = n_vocab
        self.embed_dim = embed_dim
        self.embed_tokens = nn.Embedding(
            n_vocab, embed_dim, max_norm=embedding_weight_std)
        self.convolutions = nn.Sequential(*convolutions)
        self.dropout = dropout

    def forward(self, text_sequences, text_positions=None, lengths=None,
                speaker_embed=None):
        # embed text_sequences
        x = self.embed_tokens(text_sequences.long())
        x = F.dropout(x, p=self.dropout, training=self.training)

        input_embedding = x

        # B x T x C -> B x C x T
        x = x.transpose(1, 2)

        # １D conv blocks
        x = self.convolutions(x)

        # Back to B x T x C
        keys = x.transpose(1, 2)

        # add output to input embedding for attention
        values = (keys + input_embedding) * math.sqrt(0.5)

        return keys, values


class AttentionLayer(nn.Module):
    """
    The attention layer module
    """
    def __init__(
            self, 
            conv_channels: int, 
            embed_dim: int, 
            query_projection: nn.Module,
            key_projection: nn.Module,
            value_projection: nn.Module,
            out_projection: nn.Module,
            dropout: float=0.1,
            window_ahead: int=3,
            window_backward: int=1):
        super().__init__()
        self.query_projection = query_projection
        self.key_projection = key_projection
        if conv_channels == embed_dim:
            self.key_projection.weight.data = self.query_projection.weight.data.clone()
        self.value_projection = value_projection
        self.out_projection = out_projection
        self.dropout = dropout
        self.window_ahead = window_ahead
        self.window_backward = window_backward

    def forward(self, query, encoder_out, mask=None, last_attended=None):
        keys, values = encoder_out
        residual = query
        values = self.value_projection(values)
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
    def __init__(
        self,
        embed_dim: int,
        preattention: List[nn.Module],
        convolutions: List[nn.Module],
        attention: List[nn.Module],
        output: nn.Module,
        max_positions: int=512,
        in_dim: int=80,
        in_channels: int=128,
        r: int=5,
        dropout: float=0.1,
        query_position_rate: float=1.0,
        key_position_rate: float=1.29):

        super().__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.in_channels = in_channels
        self.r = r
        self.query_position_rate = query_position_rate
        self.key_position_rate = key_position_rate

        in_channels = in_dim * r
        
        # Position encodings for query (decoder states) and keys (encoder states)
        self.embed_query_positions = SinusoidalEncoding(
            max_positions, self.in_channels)
        self.embed_keys_positions = SinusoidalEncoding(
            max_positions, embed_dim)
        # Used for compute multiplier for positional encodings

        # Prenet: causal convolution blocks
        in_channels = in_dim * r
        std_mul = 1.0
        self.preattention = nn.Sequential(*preattention)
        self.convolutions = nn.ModuleList(convolutions)
        self.attention = nn.ModuleList(attention)

        self.output = output

        # Mel-spectrogram (before sigmoid) -> Done binary flag
        self.fc = nn.Linear(in_dim * r, 1)

        self.max_decoder_steps = 200
        self.min_decoder_steps = 10

    def forward(self, encoder_out, inputs=None,
                text_positions=None, frame_positions=None, lengths=None):

        # Grouping multiple frames if necessary
        if inputs.size(-1) == self.in_dim:
            inputs = inputs.view(inputs.size(0), inputs.size(1) // self.r, -1)

        assert inputs.size(-1) == self.in_dim * self.r

        keys, values = encoder_out

        # position encodings
        if text_positions is not None:
            w = self.key_position_rate
            # TODO: may be useful to have projection per attention layer
            text_pos_embed = self.embed_keys_positions(text_positions, w)
            keys = keys + text_pos_embed
        if frame_positions is not None:
            w = self.query_position_rate
            frame_pos_embed = self.embed_query_positions(frame_positions, w)

        # transpose only once to speed up attention layers
        keys = keys.transpose(1, 2).contiguous()

        x = inputs
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Generic case: B x T x C -> B x C x T
        x = x.transpose(1, 2)

        # Prenet
        x = self.preattention(x)

        # Casual convolutions + Multi-hop attentions
        alignments = []
        for f, attention in zip(self.convolutions, self.attention):
            residual = x

            x = f(x)

            # Feed conv output to attention layer as query
            if attention is not None:
                # (B x T x C)
                x = x.transpose(1, 2)
                x = x if frame_positions is None else x + frame_pos_embed
                x, alignment = attention(x, (keys, values))
                # (T x B x C)
                x = x.transpose(1, 2)
                alignments += [alignment]
                x = (x + residual) * math.sqrt(0.5)

        # decoder state (B x T x C):
        # internal representation before compressed to output dimention
        decoder_states = x.transpose(1, 2).contiguous()
        x = self.output(x)

        # Back to B x T x C
        x = x.transpose(1, 2)

        # project to mel-spectorgram
        outputs = torch.sigmoid(x)

        # Done flag
        done = torch.sigmoid(self.fc(x))

        return outputs, torch.stack(alignments), done, decoder_states

    
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

class SinusoidalEncoding(nn.Embedding):
    """
    A sinusoidal encoding implementation
    """
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

class Converter(nn.Module):
    def __init__(self, in_dim, out_dim, convolutions):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.convolutions = nn.Sequential(*convolutions)


    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.convolutions(x)
        x = x.transpose(1, 2)
        return torch.sigmoid(x)


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

    def forward(self, text_sequences, mel_targets=None, 
                text_positions=None, frame_positions=None, input_lengths=None):
        # (B, T, text_embed_dim)
        encoder_outputs = self.encoder(
            text_sequences, lengths=input_lengths)

        # Mel: (B, T//r, mel_dim*r)
        # Alignments: (N, B, T_target, T_input)
        # Done: (B, T//r, 1)
        mel_outputs, alignments, done, decoder_states = self.decoder(
            encoder_outputs, mel_targets,
            text_positions=text_positions, frame_positions=frame_positions,
            lengths=input_lengths)

        return mel_outputs, alignments, done, decoder_states


class TTSModel(nn.Module):
    """Attention seq2seq model + post processing network
    """

    def __init__(self, 
                 seq2seq: nn.Module, 
                 postnet: nn.Module,
                 mel_dim: int=80,
                 linear_dim: int=513,
                 trainable_positional_encodings=False,
                 freeze_embedding=False):
        super().__init__()
        self.seq2seq = seq2seq
        self.postnet = postnet  # referred as "Converter" in DeepVoice3
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim
        self.trainable_positional_encodings = trainable_positional_encodings
        self.freeze_embedding = freeze_embedding


    def make_generation_fast_(self):

        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return
        self.apply(remove_weight_norm)

    def get_trainable_parameters(self):
        frozen_param_ids = set()

        encoder, decoder = self.seq2seq.encoder, self.seq2seq.decoder

        # Avoid updating the position encoding
        if not self.trainable_positional_encodings:
            pe_query_param_ids = set(map(id, decoder.embed_query_positions.parameters()))
            pe_keys_param_ids = set(map(id, decoder.embed_keys_positions.parameters()))
            frozen_param_ids |= (pe_query_param_ids | pe_keys_param_ids)
        # Avoid updating the text embedding
        if self.freeze_embedding:
            embed_param_ids = set(map(id, encoder.embed_tokens.parameters()))
            frozen_param_ids |= embed_param_ids

        return (p for p in self.parameters() if id(p) not in frozen_param_ids)

    def forward(self, text_sequences, mel_targets=None, 
                text_positions=None, frame_positions=None, input_lengths=None):
        B = text_sequences.size(0)

        # Apply seq2seq
        # (B, T//r, mel_dim*r)
        mel_outputs, alignments, done, decoder_states = self.seq2seq(
            text_sequences, mel_targets,
            text_positions, frame_positions, input_lengths)

        # Reshape
        # (B, T, mel_dim)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        # Prepare postnet inputs
        postnet_inputs = decoder_states.view(B, mel_outputs.size(1), -1)
        
        # (B, T, linear_dim)
        # Convert coarse mel-spectrogram (or decoder hidden states) to
        # high resolution spectrogram
        linear_outputs = self.postnet(postnet_inputs)
        assert linear_outputs.size(-1) == self.linear_dim

        return mel_outputs, linear_outputs, alignments, done
