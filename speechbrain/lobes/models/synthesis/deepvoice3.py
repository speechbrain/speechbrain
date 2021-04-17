# coding: utf-8

# Adopted and modified from https://github.com/r9y9/deepvoice3_pytorch

import dataclasses
import torch
import math
import numpy as np
from torch import nn
from torch.nn import functional as F
from speechbrain.nnet import CNN
from dataclasses import dataclass


class WeightNorm(nn.Module):
    """
    A weight normalization wrapper for convolutional layers
    """
    def __init__(self, inner: CNN.Conv1d, dropout: float=0.1, std_mul: float=4.0):
        """
        Class constructor

        Arguments
        ---------
        inner
            A convolutional layer
        dropout 
            The drop-out rate (0.0 to 1.0)
        std_mul 
            The standard deviation multiplier
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
        """
        Computes the forward pass. The arguments are the same
        as those to CNN.Conv1d
        """
        return self.inner.forward(*args, **kwargs)        

    def incremental_forward(self, *args, **kwargs):
        """
        Computes the forward pass as part of an incremental
        pass
        """
        return self.inner.incremental_forward(*args, **kwargs)        


class IncrementalConv1d(CNN.Conv1d):
    """
    An extension of the standard SpeechBrain Conv1d that
    supports "Incremental Forward" mode.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, padding_mode='constant', **kwargs)
        self.clear_buffer()
        self._linearized_weight = None
        self.register_backward_hook(self._clear_linearized_weight)

    def incremental_forward(self, input):
        """
        Performs the incremental forward step
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
        """
        Clears the input buffers, which are used during
        incremental passes
        """
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



class ReLU(nn.ReLU):
    """
    A ReLU equivalent with a pass-through incremental_forward
    implementation
    """
    def incremental_forward(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class EdgeConvBlock(nn.Module):
    """
    A convolution block found at the "edge" of multi-layer
    stacks within DeepVoice3, typically, the first or last
    layer consisting of a "regular" convolutional layer

    """
    def __init__(
            self,
            dropout: float=0.,
            std_mul: float=1., *args, **kwargs):
        super().__init__()
        self.conv = WeightNorm(
            inner=IncrementalConv1d(
                skip_transpose=True, *args, **kwargs),
            std_mul=std_mul,
            dropout=dropout)

    def forward(self, *args, **kwargs):
        """
        Computes the forward pass. The arguments are the same
        as those to CNN.Conv1d
        """
        return self.conv(*args, **kwargs)

    def incremental_forward(self, x, *args, **kwargs):
        """
        Computes the forward pass. The arguments are the same
        as those to CNN.Conv1d
        """
        x = x.transpose(1, 2)
        x = self.forward(x, *args, **kwargs)
        x = x.transpose(1, 2)
        return x

    def clear_buffer(self):
        self.conv.inner.clear_buffer()


class TransposeConvBlock(nn.Module):
    """
    A transposed convolution block
    """
    def __init__(
            self, 
            dropout: float=0.,
            std_mul: float=1., *args, **kwargs):
        """
        Class constructor
        
        Arguments
        ---------
        dropout
            the dropout rate
        std_mul
            the standard deviation multiplier
        """
        super().__init__()
        self.conv = WeightNorm(
            inner=CNN.TransposeConv1d(
                skip_transpose=True, *args, **kwargs),
            std_mul=std_mul,
            dropout=dropout)        

    def forward(self, *args, **kwargs):
        """
        Computes a forward pass.
        """
        return self.conv(*args, **kwargs)

    def incremental_forward(self, *args, **kwargs):
        """
        Computes a forward pass incrementally (used when generating a sample
        from a train model)
        """
        return self.forward(*args, **kwargs)


def get_padding(causal, kernel_size, dilation):
    """
    Computes the amount of padding to be used

    Arguments
    ---------
    causal
        whether the convolution block is causal
    kernel_size
        the kernel size of the convolution block
    dilation
        the dilation of the convolution block
    """
    padding = (kernel_size - 1)
    if not causal:
        padding //= 2
    padding *= dilation
    return padding


class ConvBlock(nn.Module):
    """
    A wrapper for the standard SpeechBrain convolution applying the weight normalization
    described in the paper
    """
    def __init__(
        self,
        n_speakers: int=1,
        speaker_embed_dim: int=16,
        in_channels: int=256,
        out_channels: int=256,
        kernel_size: int=5,
        padding: str=None,
        dilation: int=1,
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
        in_channels
            the number of input channels
        out_channels
            the number of output channels
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
        padding_raw = None
        if padding is None:
            padding_raw = get_padding(causal, kernel_size, dilation)

        self.conv = WeightNorm(            
            inner=IncrementalConv1d(
                skip_transpose=True,
                in_channels=in_channels,
                out_channels=out_channels * 2,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                padding_raw=padding_raw,
                **kwargs),
            dropout=dropout,
            std_mul=std_mul)
        self.multiplier = math.sqrt(0.5)
        if n_speakers > 1:
            self.speaker_proj = Linear(speaker_embed_dim, out_channels)
        else:
            self.speaker_proj = None


    def _apply_weight_norm(self):
        std = math.sqrt(
            (self.std_mul * (1.0 - self.dropout)) / (self.conv.kernel_size[0] * self.conv.in_channels))
        self.conv.weight.data.normal_(mean=0, std=std)
        self.conv.bias.data.zero_()

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
    """
    A sinusoidal encoding implementation for DeepVoice3

    Arguments
    ---------
    num_embeddings: int
        the number of embeddings to be used
    embedding_dim: int
        the embedding dimension (i.e. the dimension of
        each embedding vector)
    """
    def __init__(self, num_embeddings, embedding_dim,
                 *args, **kwargs):
        super().__init__(
            num_embeddings, embedding_dim, padding_idx=0,
            *args, **kwargs)
        self.weight.data = position_encoding_init(
            num_embeddings, embedding_dim, position_rate=1.0,
            sinusoidal=False)

    def forward(self, x, w=1.0):
        """
        Computes the forward pass
        
        Arguments
        ---------
        x: torch.Tensor
            the inputs
        w: float
            the weight/multiplier
        """
        isscalar = np.isscalar(w)
        assert self.padding_idx is not None

        if isscalar or w.size(0) == 1:
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
    """
    The gradient multiplier implementation
    """
    @staticmethod
    def forward(ctx, x, scale):
        """
        Computes the forward pass

        Arguments
        ---------
        x: torch.Tensor
            the input
        scale: float
            the scaling factor
        """
        ctx.scale = scale
        res = x.new(x)
        ctx.mark_shared_storage((x, res))
        return res

    @staticmethod    
    def backward(ctx, grad):
        """
        Computes the forward pass

        Arguments
        ---------
        grad: torch.Tensor
            the gradient
        """        
        return grad * ctx.scale, None


def Linear(in_features, out_features, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def Embedding(num_embeddings, embedding_dim, padding_idx, std=0.01):
    """
    Creates a new embedding

    Arguments
    ---------
    num_embeddings: int
        the number of embeddings
    embedding_dim: int
        the embedding dimension
    padding_idx: int
        the padding index
    """
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, std)
    return m


def get_mask_from_lengths(memory, memory_lengths):
    """Get mask tensor from list of length
    
    Arguments
    ---------
    memory: tuple
        (batch, max_time, dim)
    
    memory_lengths: array-like
        the length of the memory cell
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
    """
    The encoder block

    Arguments
    ---------

    n_vocab: int
        the number of possible values of input sequences
    embed_dim: int
        the embedding dimension
    n_speakers: int
        the number of speakers
    speaker_embed_dim: int
        the dimension of speaker embeddings to be used
    padding_idx: int
        the padding index
    embedding_weight_std: float
        the standard deviation of embedding weights
    convolutions: list
        convolutional layers (a list of nn.Module instances)
    dropout: float
        the dropout rate
    apply_grad_scaling: bool
        whether to apply gradient scaling

    """
    def __init__(self, n_vocab, embed_dim, n_speakers, speaker_embed_dim,
                 padding_idx=None, embedding_weight_std=0.1,
                 convolutions=[],
                 dropout=0.1, apply_grad_scaling=False):
        super().__init__()
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
        self.convolutions = nn.ModuleList(convolutions)


    def forward(self, text_sequences, speaker_embed=None):
        """
        Computes the forward pass

        Arguments
        ---------
        text_sequences: torch.Tensor
            the text sequences
        speaker_embed: torch.Tensor
            speaker embeddings
        """
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
        i = 1
        for f in self.convolutions:
            x = f(x, speaker_embed_btc) if isinstance(f, ConvBlock) else f(x)

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
    """
    Implements a DeepVoice3 attention layer

    Arguments
    ---------
    conv_channels
        the number of channels in the convolutional layers
    embed_dim
        the embedding imension
    dropout
        the dropout rate
    window_ahead
        the size of the window to look ahead in the sequence
    window_backward
        the size of the window to look behind
    key_projection
        whether to use key projections
    value_projection
        whether to use value projections
    """
    def __init__(self, conv_channels, embed_dim, dropout=0.1,
                 window_ahead=3, window_backward=1,
                 key_projection=True, value_projection=True):

        super().__init__()
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
        """
        Performs a forward step
        
        query
            the attention queries
        encoder_out
            the output of the encoder layer
        mask
            the mask tensor to be used
        last_attended
            the last attended steps
        """
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
    """
    The decoder block
    
    embed_dim
        the embedding dimension
    n_speaker
        the number of speakers
    speaker_embed_dim
        the dimension of the speaker embedding
    in_channels
        the number of channels in the first input convolution
    in_dim
        the input dimension
    max_positions
        the the maximum number of positions in the decoded sequence
    preattention
        the pre-attention layers (a list of Torch modules)
    convolutions
        the convolution layers (a list of Torch modules)
    output
        the output layer
    attention
        whether or not to use attention
    dropout
        the dropout rate
    use_memory_mask
        whether or not to use a memory mask
    force_monotonic_attention
        whether or not to force monotonic attention
    query_position_rate
        the query position rate
    key_position_rate
        the key position rate
    """
    def __init__(self, embed_dim, n_speakers, speaker_embed_dim,
                 in_channels=256,
                 in_dim=80, r=5,
                 max_positions=512,
                 preattention=[],
                 convolutions=[],
                 output=None,
                 attention=True,
                 dropout=0.1,
                 use_memory_mask=False,
                 force_monotonic_attention=False,
                 query_position_rate=1.0,
                 key_position_rate=1.29,
                 ):
        super().__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.r = r
        self.query_position_rate = query_position_rate
        self.key_position_rate = key_position_rate

        if isinstance(attention, bool):
            # expand True into [True, True, ...] and do the same with False
            attention = [attention] * len(convolutions)

        # Position encodings for query (decoder states) and keys (encoder states)
        self.embed_query_positions = SinusoidalEncoding(
            max_positions, in_channels)
        self.embed_keys_positions = SinusoidalEncoding(
            max_positions, embed_dim)
        # Used for compute multiplier for positional encodings
        if n_speakers > 1:
            self.speaker_proj1 = Linear(speaker_embed_dim, 1, dropout=dropout)
            self.speaker_proj2 = Linear(speaker_embed_dim, 1, dropout=dropout)
        else:
            self.speaker_proj1, self.speaker_proj2 = None, None

        # Prenet: causal convolution blocks
        self.preattention = nn.ModuleList(preattention)
        self.convolutions = nn.ModuleList(convolutions)
        self.attention = nn.ModuleList(attention)
        #self.output = output
        self.output = output


        # Mel-spectrogram (before sigmoid) -> Done binary flag
        self.fc = Linear(in_dim * r, 1)

        self.max_decoder_steps = 200
        self.min_decoder_steps = 20
        self.use_memory_mask = use_memory_mask
        
        if isinstance(force_monotonic_attention, bool):
            self.force_monotonic_attention = [force_monotonic_attention] * len(convolutions)
        else:
            self.force_monotonic_attention = force_monotonic_attention

    def forward(self, encoder_out, inputs=None,
                text_positions=None, frame_positions=None,
                speaker_embed=None, lengths=None):
        """
        Computes the forward pass

        Arguments
        ---------
        encoder_out
            the output of the encoder layer
        inputs
            the inputs (sequences of characters or phonemes)
        text_positions
            a tensor of positions (1 for the first character, 2 for the second, and so on)
        frame_positions
            a tensor of positions for the decoded steps (1 for the first, 2 for the second, etc)
        speaker_embed
            speaker embeddings (if available)
        lengths
            a tensor of lengths in the original inputs. The tensor passed in the
            `inputs` parameter may be padded - the value in lengths corresponds to the length
            of the original (pre-padding) sequence
        """
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
            x = f(x, speaker_embed_btc) if isinstance(f, ConvBlock) else f(x)
        # Casual convolutions + Multi-hop attentions
        alignments = []
        for f, attention in zip(self.convolutions, self.attention):
            residual = x

            x = f(x, speaker_embed_btc) if isinstance(f, ConvBlock) else f(x)

            # Feed conv output to attention layer as query
            if attention is not None:
                assert isinstance(f, ConvBlock)
                # (B x T x C)
                x = x.transpose(1, 2)
                x = x if frame_positions is None else x + frame_pos_embed
                x, alignment = attention(x, (keys, values), mask=mask)
                # (T x B x C)
                x = x.transpose(1, 2)                
                alignments += [alignment]

            if isinstance(f, ConvBlock):
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

    def incremental_forward(self, encoder_out, text_positions, speaker_embed=None,
                            initial_input=None, test_inputs=None):

        """
        Computes the forward pass incrementally (needed when producing an output without a target)

        Arguments
        ---------
        encoder_out
            the output of the encoder layer
        inputs
            the inputs (sequences of characters or phonemes)
        text_positions
            a tensor of positions (1 for the first character, 2 for the second, and so on)
        speaker_embed
            speaker embeddings (if available)
        initial_input
            the initial input
        test_inputs
            test inputs (optional, the last output will be used if not provided)
        """                            
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
                if isinstance(f, ConvBlock):
                    x = f.incremental_forward(x, speaker_embed)
                else:
                    x = f.incremental_forward(x)

            # Casual convolutions + Multi-hop attentions
            ave_alignment = None
            for idx, (f, attention) in enumerate(zip(self.convolutions,
                                                     self.attention)):
                residual = x
                if isinstance(f, ConvBlock):
                    x = f.incremental_forward(x, speaker_embed)
                else:
                    x = f.incremental_forward(x)

                # attention
                if attention is not None:
                    assert isinstance(f, ConvBlock)
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
                if isinstance(f, ConvBlock):
                    x = (x + residual) * math.sqrt(0.5)

            decoder_state = x
            x = self.output.incremental_forward(x)
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
        """
        Starts a fresh sequence - used when 
        """
        _clear_modules(self.preattention)
        _clear_modules(self.convolutions)
        self.output.clear_buffer()




def _clear_modules(modules):
    """
    Clears the specified modules' buffers
    """
    for m in modules:
        try:
            m.clear_buffer()
        except AttributeError as e:
            pass


class Converter(nn.Module):
    """
    The deepvoice3 converter module (also referred to as the "postnet")

    Arguments
    ---------
    n_speakers
        the number of speakers
    in_dim
        the number of output dimensions
    out_dim
        the number of output dimensions
    convolutions
        a list of convolution layers
    dropout
        the dropout rate (0.0 - 1.0)
    """
    def __init__(self, n_speakers, in_dim, out_dim, convolutions,
                 dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_speakers = n_speakers

        # Non causual convolution blocks
        self.convolutions = nn.ModuleList(convolutions)

    def forward(self, x, speaker_embed=None):
        """
        Computes the forward pass
        
        Arguments
        ---------
        x
            the converter inputs
        speaker_embed
            the speaker embedding (if applicable)
        """
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
            x = f(x, speaker_embed_btc) if isinstance(f, ConvBlock) else f(x)

        # Back to B x T x C
        x = x.transpose(1, 2)

        return torch.sigmoid(x)




class TTSModel(nn.Module):
    """
    The complete Deepvoice3 text-to-speech model

    Arguments
    ---------
    seq2seq
        the sequence-to-sequence model
    mel_dim
        the MEL spectrogram dimension
    linear-dim
        the linear spectrogram dimension
    n_speakers
        the number of speakers
    speaker_embed_dim
        the dimension of speaker embeddings
    trainable_positional_encodings
        whether positional encodings should be trainable
    speaker_embedding_weight_std
        the standard deviations
    freeze_embedding
        whether speaker embeddings should be frozen
    """

    def __init__(self, seq2seq, postnet,
                 mel_dim=80, linear_dim=513,
                 n_speakers=1, speaker_embed_dim=16,
                 trainable_positional_encodings=False,
                 use_decoder_state_for_postnet_input=False,
                 speaker_embedding_weight_std=0.01,
                 freeze_embedding=False):
        super().__init__()
        self.seq2seq = seq2seq
        self.postnet = postnet  
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
        """
        Returns an array of trainable parameters (ones that should not be frozen)
        """
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

    def forward(self, text_sequences, mel_targets=None, speaker_ids=None,
                text_positions=None, frame_positions=None, input_lengths=None):
        """
        Computes the forward pass

        Arguments
        ---------
        text_sequences
            the original sequences of characters or phonemes (with each symbol encoded as an integer)
        mel_targets
            the target MEL-scale diagrams
        speaker_ids
            speaker identifiers
        text_positions
            a tensor encoding the position of each character
            (shape = same as text_sequences)
        frame_positions
            a tensor indicating the position of each frame in the spectrogram
            (shape = Batch x MEL dimension)
        input_lengths
            the lengths of the input sequences
        """

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
    
    Arguments
    ---------
    encoder: Encoder
        the encoder model
    decoder: Decoder
        the decoder model
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        if isinstance(self.decoder.attention, nn.ModuleList):
            self.encoder.num_attention_layers = sum(
                [layer is not None for layer in decoder.attention])

    def forward(self, text_sequences, mel_targets=None, speaker_embed=None,
                text_positions=None, frame_positions=None, input_lengths=None):

        """
        Computes the forward pass

        Arguments
        ---------
        text_sequences: torch.Tensor
            the original text sequences with each character, word or phoneme
            encoded as an integer
        
        mel_targets: torch.Tensor
            the MEL spectrogram targets - optional (provided during training only)
        
        speaker_embed: torch.Tensor
            the speaker embeddings (optional)

        text_positions: torch.Tensor
            a (batch size X max text length) tensor in which every element value
            corresponds to the position of that particular element within the
            sequence.

        frame_positions: torch.Tensor
            similar to text_positions but for spectrogram frames
        
        input_lengths: torch.Tensor
            a single-dimensional tensor in which each position indicates
            the length of the sample

        """
        # (B, T, text_embed_dim)
        encoder_outputs = self.encoder(
            text_sequences, speaker_embed=speaker_embed)

        # Mel: (B, T//r, mel_dim*r)
        # Alignments: (N, B, T_target, T_input)
        # Done: (B, T//r, 1)
        mel_outputs, alignments, done, decoder_states = self.decoder(
            encoder_outputs, mel_targets,
            text_positions=text_positions, frame_positions=frame_positions,
            speaker_embed=speaker_embed, lengths=input_lengths)

        return mel_outputs, alignments, done, decoder_states


@dataclass
class LossStats:
    """
    Represents the different losses available in DeepVoice3,
    including the total loss
    """
    loss: torch.Tensor
    mel_loss: torch.Tensor
    mel_l1_loss: torch.Tensor
    mel_binary_div: torch.Tensor
    linear_loss: torch.Tensor
    linear_l1_loss: torch.Tensor
    linear_binary_div: torch.Tensor
    done_loss: torch.Tensor
    attn_loss: torch.Tensor = None
    
    def as_scalar(self):
        fields = dataclasses.fields(self)
        return {field.name: getattr(self, field.name).item()
                for field in fields}



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


    #TODO: Make this more friendly
    def forward(self, input, target):
        """
        Computes the losses
        
        inputs : tuple
            a tuple of (MEL input, linear input, attention, done, target lengths)
            The *input* to the loss is the *output* of the model
        targets: tuple
            a tuple of (MEL targets, linear targets, done, target lengths)

        Returns
        -------
        loss : LossStats
            the diferent losses from DeepVoice3, collected into
            a single object
        """
        input_mel, input_linear, attention, input_done, input_lengths = input
        target_mel, target_linear, target_done, target_lengths = target

        decoder_target_mask, target_mask = self.compute_masks(
            target_linear, target_mel, target_lengths)               

        mel_loss, mel_l1_loss, mel_binary_div = self.mel_loss(
            input_mel, target_mel, decoder_target_mask)
        done_loss = self.binary_criterion(input_done, target_done)

        linear_loss, linear_l1_loss, linear_binary_div = self.linear_loss(
            input_linear, target_linear, target_mask)
        
        decoder_lengths = (
            target_lengths.long()
            // self.outputs_per_step
            // self.downsample_step)
        soft_mask = self.guided_attentions(
            input_lengths, decoder_lengths,
            attention.size(-2), self.guided_attention_sigma)
        attn_loss = (attention * soft_mask).mean()

        loss = mel_loss + linear_loss + done_loss + attn_loss
        loss_stats = LossStats(
            loss=loss,
            mel_loss=mel_loss,
            mel_l1_loss=mel_l1_loss,
            mel_binary_div=mel_binary_div,
            linear_loss=linear_loss,
            linear_l1_loss=linear_l1_loss,
            linear_binary_div=linear_binary_div,
            done_loss=done_loss,
            attn_loss=attn_loss   
        )
        return loss_stats

    def compute_masks(self, target_linear, target_mel, target_lengths):
        """
        Computes the masks to be used for the calculation of the losses

        Arguments:
        ----------
        target_linear: torch.Tensor
            the linear spectrogram targets
        target_mel: torch.Tensor
            the MEL spectrogram targets
        target_lengths
            the target lengths

        Returns
        -------
        masks: tuple
            a tuple of (decoder_target_mask, target_mask)
        """
        decoder_target_mask = self.sequence_mask(
            target_lengths // (self.outputs_per_step * self.downsample_step),
            max_len=target_mel.size(1),
            device=target_mel.device).unsqueeze(-1)
        
        if self.downsample_step > 1:
            # spectrogram-domain mask
            target_mask = self.sequence_mask(
                target_lengths, max_len=target_linear.size(1),
                device=target_lengths.device).unsqueeze(-1)
        else:
            target_mask = decoder_target_mask
        
        decoder_target_mask = decoder_target_mask[:, self.outputs_per_step:, :]
        target_mask = target_mask[:, self.outputs_per_step:, :]
        return decoder_target_mask, target_mask


    def mel_loss(self, input_mel, target_mel, decoder_target_mask):
        """
        Computes the MEL scale spectrogram loss

        Arguments
        ---------
        input_linear: torch.Tensor
            the linear inputs (i.e. the output of the model)
        target_linear: torch.Tensor
            the linear target (the linear spectrogram of the original audio)
        target_mask: torch.Tensor
            the target mask to be applied

        Returns
        -------
        losses: tuple
            a tuple of (mel_loss, mel_l1_loss, mel_binary_div) - weighted MEL loss,
            MEL L1 loss, MEL binary divergence
        """

        mel_l1_loss, mel_binary_div = self.spec_loss(
            input_mel[:, :-self.outputs_per_step, :], target_mel[:, self.outputs_per_step:, :], decoder_target_mask,
            masked_loss_weight=self.masked_loss_weight,
            binary_divergence_weight=self.binary_divergence_weight)
        mel_loss = (1 - self.binary_divergence_weight) * mel_l1_loss + self.binary_divergence_weight * mel_binary_div
        return mel_loss, mel_l1_loss, mel_binary_div

    def linear_loss(self, input_linear, target_linear, target_mask):
        """
        Computes the linear spectrogram loss

        Arguments
        ---------
        input_linear: torch.Tensor
            the linear inputs (i.e. the output of the model)
        target_linear: torch.Tensor
            the linear target (the linear spectrogram of the original audio)
        target_mask: torch.Tensor
            the target mask to be applied

        Returns
        -------
        losses: tuple
            a tuple of (linear_loss, linear_l1_loss,
            linear_binary_div) - linear MEL loss, linear L1 loss, 
            linear binary divergence

        """
        n_priority_freq = int(self.priority_freq / (self.sample_rate * 0.5) 
                              * self.linear_dim)

        linear_l1_loss, linear_binary_div = self.spec_loss(
            input_linear[:, :-self.outputs_per_step, :], 
            target_linear[:, self.outputs_per_step:, :], 
            target_mask,
            priority_bin=n_priority_freq,
            priority_w=self.priority_freq_weight,
            masked_loss_weight=self.masked_loss_weight,
            binary_divergence_weight=self.binary_divergence_weight)
        linear_loss = (
            (1 - self.binary_divergence_weight) * linear_l1_loss 
             + self.binary_divergence_weight * linear_binary_div)
        return linear_loss, linear_l1_loss, linear_binary_div
        

    def spec_loss(self, y_hat, y, mask, priority_bin=None, priority_w=0, masked_loss_weight=0., binary_divergence_weight=0.):
        """
        Computes the DeepVoice3 mask spectrogram loss
        
        y_hat: torch.Tensor
            the predicted spectrogram
        y: torch.Tensor
            the target spectrogram
        mask: torch.Tensor
            the masked to be applied
        priority_w: torch.Tensor
            the weight of the priority loss
        """
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

    def sequence_mask(self, sequence_length, max_len=None, device=None):
        """
        Computes a sequence mask

        Arguments
        ---------
        sequence_length: int
            the length of the sequence
        max_len: int
            the maximum length of the sequence
        """
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

    def guided_attention(self, N, max_N, T, max_T, g):
        """
        Computes a single guided attention matrix for a sample
        
        Arguments
        ---------
        N
            the input length
        max_N
            the maximum input length
        T
            the target length
        max_T
            the maximum target length
        g
            the guided attention weight
        """
        W = torch.zeros((max_N, max_T)).float()
        n, t = torch.meshgrid(
            torch.arange(N).to(N.device), 
            torch.arange(T).to(N.device))
        value = 1. - torch.exp(-(n / N - t / T) ** 2 / (2 * g * g))
        return value


    def guided_attentions(self, input_lengths, target_lengths, max_target_len, g=0.2):
        """
        Computes guided attention matrices

        Arguments
        ---------
        input_lengths
            a tensor of input lengths
        target_lengths
            a tensor of target (spectrogram) length
        max_target_len
            the maximum target length
        g
            the attention weight
        """
        B = len(input_lengths)
        max_input_len = input_lengths.max()
        W = (torch.zeros((B, max_target_len, max_input_len.item()))
            .float()
            .to(input_lengths.device))
        #TODO: Attempt to vectorize here as well
        for b in range(B):
            attention = self.guided_attention(
                input_lengths[b],
                max_input_len,
                target_lengths[b],
                max_target_len,
                g).T
            W[b, :attention.size(0), :attention.size(1)] = attention
        return W


def logit(x, eps=1e-8):
    """
    Computes the logit value: log(x + eps) - log (1 - x + eps)

    Arguments
    ---------
    x: torch.Tensor
        the input
    eps: float
        the epsilon value
    """
    return torch.log(x + eps) - torch.log(1 - x + eps)


def masked_mean(y, mask):
    """
    Multiplies the provided tensor by the provided mask and
    then computes the mean

    y : torch.Tensor
        the tensor whose mean is being computed
    mask: torch.Tensor
        the mask to be applied
    """
    # (B, T, D)
    mask_ = mask.expand_as(y)
    return (y * mask_).sum() / mask_.sum()


class MaskedL1Loss(nn.Module):
    """
    A masked L1 loss implementation
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss(reduction="sum")

    def forward(self, input, target, lengths=None, mask=None, max_len=None):
        """
        Computes the loss value

        Arguments
        ---------
        inputs
            the input tensor
        target
            the target tensor
        lengths
            the actual lengths (inputs and targets may be padded)
        mask
            the mask to be applied (optional - a sequence mask based on
            the lengths will be computed if omitted)
        max_len
            the maximum length        
        """
        if lengths is None and mask is None:
            raise RuntimeError("Should provide either lengths or mask")

        # (B, T, 1)
        if mask is None:
            mask = self.sequence_mask(lengths, max_len).unsqueeze(-1)

        # (B, T, D)
        mask_ = mask.expand_as(input)
        loss = self.criterion(input * mask_, target * mask_)
        return loss / mask_.sum()

