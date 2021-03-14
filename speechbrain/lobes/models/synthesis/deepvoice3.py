"""
DeepVoice 3 - Building Blocks

Elements of the architecture are inspired by
https://github.com/r9y9/deepvoice3_pytorch
"""

import math
import torch
import numpy as np
from torch.nn import functional as F
from torch import nn, Tensor
from speechbrain.nnet import CNN
from typing import List, Tuple


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


class EdgeConvBlock(nn.Module):
    """
    A convolution block found at the "edge" of multi-layer
    stacks within DeepVoice3, typically, the first or last
    layer consisting of a "regular" convolutional layer

    """
    def __init__(self, dropout: float, std_mul: float, *args, **kwargs):
        super().__init__()
        self.conv = WeightNorm(
            inner=IncrementalConv1d(*args, **kwargs),
            std_mul=std_mul,
            dropout=dropout)

    def forward(self, *args, **kwargs):
        return self.conv(*args, **kwargs)

    def incremental_forward(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class IncrementalConv1d(CNN.Conv1d):
    """
    An extension of the standard SpeechBrain Conv1d that
    supports "Incremental Forward" mode.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clear_buffer()
        self._linearized_weight = None
        self.register_backward_hook(self._clear_linearized_weight)

    def incremental_forward(self, input):
        input = input.transpose(1, 2).contiguous()
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
        return output

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


class ConvBlock(nn.Module):
    """
    A wrapper for the standard SpeechBrain convolution applying the weight normalization
    described in the paper
    """
    def __init__(
        self,
        padding: str=None,
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


        Arguments
        ----------
        padding
            the type of padding used (e.g. "same", "valid)
        kernel_size
            the convolution kernel size (i.e. the area covered by a single "step" in the convolution)
        dilation
            the convolution dilation
        dropout
            the amount of dropout used
        causal
            whether or not this is a causal convolution
        residual
            whether or not to use a residual connection
        """
        super().__init__()
        self.dropout = dropout
        self.std_mul = std_mul
        self.causal = causal
        self.residual = residual
        if padding is None:
            padding = 'causal' if self.causal else 'same'
        self.conv = WeightNorm(
            inner=IncrementalConv1d(
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
    #TODO: Remove this
    # def forward(self, x: Tensor, *args, **kwargs):
    #     residual = x
    #     x = self.conv(x)
    #     x = self.glu(x)
    #     if self.causal:
    #         x = x[:, :, :residual.size(-1)]
    #     return (x + residual) * self.multiplier if self.residual else x

    def forward(self, x, speaker_embed=None):
        return self._forward(x, speaker_embed, False)    

    def incremental_forward(self, x, speaker_embed=None):
        return self._forward(x, speaker_embed, True)

    def _forward(self, x, speaker_embed, is_incremental):
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        if is_incremental:
            x = self.conv.inner.incremental_forward(x)
        else:
            x = self.conv(x)
            # remove future time steps
            x = x[:, :, :residual.size(-1)] if self.causal else x

        # TODO: There is too much transposing going on, 
        a, b = x.split(x.size(1) // 2, dim=1)
        x = a * torch.sigmoid(b)
        x = (x + residual) * math.sqrt(0.5) if self.residual else x
        return x
        


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
                speaker_embed=None) -> Tuple[Tensor, Tensor]:
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
        clone_projection_weights = (
            conv_channels == embed_dim 
            and hasattr(self.key_projection, 'weight')
            and hasattr(self.query_projection, 'weight')
        )
        if clone_projection_weights:
            self.key_projection.weight.data = self.query_projection.weight.data.clone()
        self.value_projection = value_projection
        self.out_projection = out_projection
        self.dropout = dropout
        self.window_ahead = window_ahead
        self.window_backward = window_backward

    def forward(self, query: Tensor, encoder_out: Tensor, mask=None, last_attended: Tensor=None) -> Tuple[Tensor, Tensor]:
        keys, values = encoder_out
        residual = query
        values = self.value_projection(values)
        keys = self.key_projection(keys.transpose(1, 2))

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


DECODER_OUTPUT_TYPE = Tuple[Tensor, Tensor, Tensor, Tensor]

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
        outputs_per_step: int=5,
        dropout: float=0.1,
        query_position_rate: float=1.0,
        key_position_rate: float=1.29):

        super().__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.in_channels = in_channels
        self.outputs_per_step = outputs_per_step
        self.query_position_rate = query_position_rate
        self.key_position_rate = key_position_rate

        # Position encodings for query (decoder states) and keys (encoder states)
        self.embed_query_positions = SinusoidalEncoding(
            max_positions + 1, self.in_channels)
        self.embed_keys_positions = SinusoidalEncoding(
            max_positions + 1, embed_dim)
        # Used for compute multiplier for positional encodings

        # Prenet: causal convolution blocks
        self.preattention = nn.Sequential(*preattention)
        self.convolutions = nn.ModuleList(convolutions)
        self.attention = nn.ModuleList(attention)

        self.output = output

        # Mel-spectrogram (before sigmoid) -> Done binary flag
        self.fc = nn.Linear(in_dim * outputs_per_step, 1)

        self.max_decoder_steps = 200
        self.min_decoder_steps = 10

    def forward(self, encoder_out: Tensor, inputs: Tensor=None,
                text_positions: Tensor=None, frame_positions: Tensor=None,
                lengths: Tensor=None) -> DECODER_OUTPUT_TYPE:

        if inputs is None:
            return self.incremental_forward(encoder_out, text_positions)
        # Grouping multiple frames if necessary
        if inputs.size(-1) == self.in_dim:
            inputs = inputs.reshape(inputs.size(0), inputs.size(1) // self.outputs_per_step, -1)

        assert inputs.size(-1) == self.in_dim * self.outputs_per_step

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

    # TODO: Improve this method
    def incremental_forward(self, encoder_out, text_positions, speaker_embed=None,
                            initial_input=None, test_inputs=None):
        keys, values = encoder_out
        B = keys.size(0)

        # position encodings
        w = self.key_position_rate
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

        num_attention_layers = sum([layer is not None for layer in self.attention])
        t = 0
        if initial_input is None:
            initial_input = keys.data.new(B, self.in_dim * self.outputs_per_step, 1).zero_()
        current_input = initial_input
        n = 0
        while True:
            n += 1
            if n > 10:
                break
            # frame pos start with 1.
            frame_pos = keys.data.new(B, 1).fill_(t + 1).long()
            w = self.query_position_rate
            frame_pos_embed = self.embed_query_positions(frame_pos, w).transpose(1, 2)

            if test_inputs is not None:
                if t >= test_inputs.size(1):
                    break
                current_input = test_inputs[:, t, :].unsqueeze(1)
            else:
                if t > 0:
                    current_input = outputs[-1].transpose(1, 2)
            x = current_input
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Prenet
            for f in self.preattention:
                x = f(x)

            # Casual convolutions + Multi-hop attentions
            ave_alignment = None
            for idx, (f, attention) in enumerate(zip(self.convolutions,
                                                     self.attention)):
                residual = x
                x = f.incremental_forward(x)
                # attention
                if attention is not None:
                    x = x + frame_pos_embed
                    x, alignment = attention(x.transpose(1, 2), (keys, values),
                                             last_attended=last_attended[idx])
                    x = x.transpose(1, 2)
                    if ave_alignment is None:
                        ave_alignment = alignment
                    else:
                        ave_alignment = ave_alignment + ave_alignment

                # TODO: REVIEW THIS
                # if isinstance(f, Conv1dGLU):
                #    x = (x + residual) * math.sqrt(0.5)
                x = (x + residual) * math.sqrt(0.5)

            decoder_state = x
            x = self.output(x)
            ave_alignment = ave_alignment.div_(num_attention_layers)

            x = x.transpose(1, 2)
            # Output & done flag predictions
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


        # Combine outputs for all time steps
        alignments = torch.stack(alignments).transpose(0, 1).squeeze(2)
        decoder_states = torch.cat(decoder_states, dim=2).transpose(1, 2).squeeze(2).contiguous()
        outputs = torch.stack(outputs).transpose(0, 1).squeeze(2)
        dones = torch.cat(dones, dim=1).transpose(1, 2)
        return outputs, alignments, dones, decoder_states


    
    def start_fresh_sequence(self):
        _clear_modules(self.preattention)
        _clear_modules(self.convolutions)
        _clear_modules(self.output)

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
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 *args, **kwargs):
        super(SinusoidalEncoding, self).__init__(num_embeddings, embedding_dim,
                                                 padding_idx=0,
                                                 *args, **kwargs)
        self.weight.data = position_encoding_init(num_embeddings, embedding_dim,
                                                  position_rate=1.0,
                                                  sinusoidal=False)

    def forward(self, x: Tensor, w: float=1.0) -> Tensor:
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

def position_encoding_init(n_position: int, d_pos_vec: int, position_rate: float=1.0,
                           sinusoidal: bool=True):
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


def sinusoidal_encode(x: Tensor, w: Tensor):
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


    def forward(self, x: Tensor) -> Tensor:
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

    def forward(self, text_sequences: Tensor, mel_targets: Tensor=None, 
                text_positions: Tensor=None, frame_positions: Tensor=None,
                input_lengths: Tensor=None):
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


MODEL_OUTPUT_TYPE = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

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
                text_positions=None, frame_positions=None, input_lengths=None, target_lengths=None) -> MODEL_OUTPUT_TYPE:        
        B = text_sequences.size(0)

        if mel_targets is not None:
            mel_targets = mel_targets.transpose(1, 2)
        # Apply seq2seq
        # (B, T//r, mel_dim*r)
        mel_outputs, alignments, done, decoder_states = self.seq2seq(
            text_sequences, mel_targets,
            text_positions, frame_positions, input_lengths)

        # Reshape
        # (B, T, mel_dim)
        
        #mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        # Prepare postnet inputs
        postnet_inputs = decoder_states.view(B, mel_outputs.size(1), -1)
        
        # (B, T, linear_dim)
        # Convert coarse mel-spectrogram (or decoder hidden states) to
        # high resolution spectrogram
        linear_outputs = self.postnet(postnet_inputs)
        assert linear_outputs.size(-1) == self.linear_dim
        linear_outputs = linear_outputs.transpose(1, 2)
        
        mel_outputs = mel_outputs.transpose(1, 2)
        return mel_outputs, linear_outputs, alignments, done, target_lengths


def logit(x, eps=1e-8):
    return torch.log(x + eps) - torch.log(1 - x + eps)


def masked_mean(y, mask):
    # (B, T, D)
    mask_ = mask.expand_as(y)
    return (y * mask_).sum() / mask_.sum()


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = sequence_length.unsqueeze(1) \
        .expand_as(seq_range_expand)
    return (seq_range_expand < seq_length_expand).float()


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


def spec_loss(y_hat, y, mask, priority_bin=None, priority_w=0, masked_loss_weight=0., binary_divergence_weight=0.):
    masked_l1 = MaskedL1Loss()
    l1 = nn.L1Loss()

    # L1 loss
    if masked_loss_weight > 0:
        l1_loss = masked_loss_weight * masked_l1(y_hat, y, mask=mask) + (1 - masked_loss_weight) * l1(y_hat, y)
    else:
        l1_loss = l1(y_hat, y)

    # Priority L1 loss
    if priority_bin is not None and priority_w > 0:
        if masked_loss_weight > 0:
            priority_loss = masked_loss_weight * masked_l1(
                y_hat[:, :, :priority_bin], y[:, :, :priority_bin], mask=mask) \
                + (1 - masked_loss_weight) * l1(y_hat[:, :, :priority_bin], y[:, :, :priority_bin])
        else:
            priority_loss = l1(y_hat[:, :, :priority_bin], y[:, :, :priority_bin])
        l1_loss = (1 - priority_w) * l1_loss + priority_w * priority_loss

    # Binary divergence loss
    if binary_divergence_weight <= 0:
        binary_div = y.data.new(1).zero_()
    else:
        y_hat_logits = logit(y_hat)
        z = -y * y_hat_logits + torch.log1p(torch.exp(y_hat_logits))
        if masked_loss_weight > 0:
            binary_div = masked_loss_weight * masked_mean(z, mask) + (1 - masked_loss_weight) * z.mean()
        else:
            binary_div = z.mean()

    return l1_loss, binary_div


LOSS_INPUT_TYPE = Tuple[Tensor, Tensor, Tensor, Tensor]


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
        sample_rate: float):
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
        self.binary_criterion = nn.BCELoss()

    def forward(self, input: LOSS_INPUT_TYPE, target: LOSS_INPUT_TYPE) -> Tensor:
        input_mel, input_linear, input_done, _ = input
        target_mel, target_linear, target_done, target_lengths = target
        decoder_target_mask = sequence_mask(
            target_lengths // (self.outputs_per_step * self.downsample_step),
            max_len=target_mel.size(1)).unsqueeze(-1)
        mel_l1_loss, mel_binary_div = spec_loss(
            input_mel[:, :-self.outputs_per_step, :], target_mel[:, self.outputs_per_step:, :], decoder_target_mask)
        mel_loss = (1 - self.masked_loss_weight) * mel_l1_loss + self.masked_loss_weight * mel_binary_div
        done_loss = self.binary_criterion(target_done.squeeze(), input_done)

        target_mask = sequence_mask(
            target_lengths, max_len=target_linear.size(1)).unsqueeze(-1)

        n_priority_freq = int(self.priority_freq / (self.sample_rate * 0.5) * self.linear_dim)
        linear_l1_loss, linear_binary_div = spec_loss(
            input_linear[:, :-self.outputs_per_step, :], target_linear[:, self.outputs_per_step:, :], target_mask,
            priority_bin=n_priority_freq,
            priority_w=self.priority_freq_weight)
        linear_loss = (1 - self.masked_loss_weight) * linear_l1_loss + self.masked_loss_weight * linear_binary_div

        # Combine losses
        loss = mel_loss + linear_loss + done_loss
        return loss
