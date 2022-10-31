"""InterleaveFormer implementaion in the SpeechBrain style.
Authors
* Yiqi Wang, 2022
* Aditya Rathod, 2022
"""
import math
import torch
import torch.nn as nn
import speechbrain as sb
from typing import Optional
import numpy as np


from .Conformer import ConformerEncoder
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.attention import RelPosEncXL


class InterleaveFormerInterface(nn.Module):
    """This is an interface for Interleaveformer model.
    the forward function is modifed such that there's only autoregressive prediction
    rather than tranditional encoder-decoder workflow since InterleaveFormer is a big "decoder"
    arxiv PLACE HOLDER
    Arguments
    ----------
    d_model: int
        The number of expected features in the model inputs (default=512).
    nhead: int
        The number of heads in the multi-head attention models (default=8).
    num_encoder_layers: int, optional
        The number of layers in the causal encoder.
    num_decoder_layers: int, optional
        The number of decoder layers in the decoder.
        This part is kept to follow the TransformerInterface but not used.
    dim_ffn: int, optional
        The dimension of the feedforward network model hidden layer.
    dropout: int, optional
        The dropout value.
    activation: torch.nn.Module, optional
        The activation function for Feed-Forward Netowrk layer,
        e.g., relu or gelu or swish.
    custom_src_module: torch.nn.Module, optional
        Module that processes the src features to expected feature dim.
    custom_tgt_module: torch.nn.Module, optional
        Module that processes the tgt features to expected feature dim.
    positional_encoding: str, optional
        Type of positional encoding used. e.g. 'fixed_abs_sine' for fixed absolute positional encodings.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    kernel_size: int, optional
        Kernel size in convolutional layers when Conformer is used.
        Not used in InterleaveFormer
    bias: bool, optional
        Whether to use bias in Conformer convolutional layers.
        Not used in InterleaveFormer
    encoder_module: str, optional
        Choose between Conformer and Transformer for the encoder. The decoder is fixed to be a Transformer.
    conformer_activation: torch.nn.Module, optional
        Activation module used after Conformer convolutional layers. E.g. Swish, ReLU etc. it has to be a torch Module.
    attention_type: str, optional
        Type of attention layer used in all Transformer or Conformer layers.
        e.g. regularMHA or RelPosMHA.
    max_length: int, optional
        Max length for the target and source sequence in input.
        Used for positional encodings.
    causal: bool, optional
        InterleaveFormer's encoder is always causal
    encoder_kdim: int, optional
        Dimension of the key for the encoder.
    encoder_vdim: int, optional
        Dimension of the value for the encoder.
    decoder_kdim: int, optional
        Dimension of the key for the decoder.
    decoder_vdim: int, optional
        Dimension of the value for the decoder.
    """

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=0,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        custom_src_module=None,
        custom_tgt_module=None,
        positional_encoding="fixed_abs_sine",
        normalize_before=True,
        kernel_size: Optional[int] = 31,
        bias: Optional[bool] = True,
        encoder_module: Optional[str] = "InterleaveFormer",
        conformer_activation: Optional[nn.Module] = Swish,
        attention_type: Optional[str] = "regularMHA",
        max_length: Optional[int] = 2500,
        causal: Optional[bool] = True,
        encoder_kdim: Optional[int] = None,
        encoder_vdim: Optional[int] = None,
        decoder_kdim: Optional[int] = None,
        decoder_vdim: Optional[int] = None,
    ):
        super().__init__()
        self.causal = causal
        self.attention_type = attention_type
        self.positional_encoding_type = positional_encoding
        self.encoder_kdim = encoder_kdim
        self.encoder_vdim = encoder_vdim
        self.decoder_kdim = decoder_kdim
        self.decoder_vdim = decoder_vdim

        assert attention_type in ["regularMHA", "RelPosMHAXL"]
        assert positional_encoding in ["fixed_abs_sine", None]

        assert (
            num_encoder_layers + num_decoder_layers > 0
        ), "number of encoder layers and number of decoder layers cannot both be 0!"

        if positional_encoding == "fixed_abs_sine":
            self.positional_encoding = PositionalEncoding(d_model, max_length)
        elif positional_encoding is None:
            pass
            # no positional encodings

        # overrides any other pos_embedding
        if attention_type == "RelPosMHAXL":
            self.positional_encoding = RelPosEncXL(d_model)
            self.positional_encoding_decoder = PositionalEncoding(
                d_model, max_length
            )

        # initialize the encoder
        if num_encoder_layers > 0:
            if custom_src_module is not None:
                self.custom_src_module = custom_src_module(d_model)
            if encoder_module == "InterleaveFormer":
                self.encoder = InterleaveFormer(
                    nhead=nhead,
                    num_layers=num_encoder_layers,
                    d_ffn=d_ffn,
                    d_model=d_model,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    causal=self.causal,
                    attention_type=self.attention_type,
                    kdim=self.encoder_kdim,
                    vdim=self.encoder_vdim,
                )
            else:
                assert False,f"Unsupported model type. Needs a InterleaveFormer, but have {encoder_module}"

    def forward(self, **kwags):
        """Check InterleaveFormerASR.py InterleaveFormerASR's forward()"""
        raise NotImplementedError


class PositionalEncoding(nn.Module):
    """This class implements the absolute sinusoidal positional encoding function.
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    Arguments
    ---------
    input_size: int
        Embedding dimension.
    max_len : int, optional
        Max length of the input sequences (default 2500).
    Example
    -------
    >>> a = torch.rand((8, 120, 512))
    >>> enc = PositionalEncoding(input_size=a.shape[-1])
    >>> b = enc(a)
    >>> b.shape
    torch.Size([1, 120, 512])
    """

    def __init__(self, input_size, max_len=2500):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(self.max_len, input_size, requires_grad=False)
        positions = torch.arange(0, self.max_len).unsqueeze(1).float()
        denominator = torch.exp(
            torch.arange(0, input_size, 2).float()
            * -(math.log(10000.0) / input_size)
        )

        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments
        ---------
        x : tensor
            Input feature shape (batch, time, fea)
        """
        return self.pe[:, : x.size(1)].clone().detach()


class InterleaveFormerLayer(nn.Module):
    """This is an implementation of causal self-attention multiway encoder layer
    for the InterleaveFormer model. 2 modality expert FFN are added
    Arguments
    ----------
    d_ffn: int, optional
        The dimension of the feedforward network model hidden layer.
    nhead: int
        The number of heads in the multi-head attention models (default=8).
    d_model: int
        The number of expected features in the encoder/decoder inputs (default=512).
    kdim: int, optional
        Dimension of the key.
    vdim: int, optional
        Dimension of the value.
    dropout: int, optional
        The dropout value.
    activation: torch.nn.Module, optional
        The activation function for Feed-Forward Netowrk layer,
        e.g., relu or gelu or swish.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    attention_type: str, optional
        Type of attention layer used in all Transformer or Conformer layers.
        e.g. regularMHA or RelPosMHA.
    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = InterleaveFormerLayer(512, 8, d_model=512)
    >>> output = net(x)
    >>> output[0].shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_ffn,
        nhead,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        attention_type="regularMHA",
        causal=True,
    ):
        super().__init__()

        if attention_type == "regularMHA":
            self.self_att = sb.nnet.attention.MultiheadAttention(
                nhead=nhead,
                d_model=d_model,
                dropout=dropout,
                kdim=kdim,
                vdim=vdim,
            )

        elif attention_type == "RelPosMHAXL":
            self.self_att = sb.nnet.attention.RelPosMHAXL(
                d_model, nhead, dropout, mask_pos_future=causal
            )

        # instantiated our 2 modality experts here????????
        # rename the pos_ffn with something COOLER???
        self.pos_ffn = sb.nnet.attention.PositionalwiseFeedForward(
            d_ffn=d_ffn,
            input_size=d_model,
            dropout=dropout,
            activation=activation,
        )

        self.norm1 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.norm2 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def forward(
        self,
        src,
        seg_stats,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder layer.
        seg_stats:  dictionary
            Bi-modality indicator { key_i: array for i in batch} used by modality expert.
            Key_i: key to index sample.
            Value_of_key_i: Even element is audio segment true length. Odd element is text tokens true length.
        src_mask : torch.Tensor
            The mask for the src query for each example in the batch.
            now it actually a TGT_MASK that is passed in, which is a causal mask!
        src_key_padding_mask : torch.Tensor, optional
            The mask for the src keys for each example in the batch.
        """
        assert type(seg_stats) == type(dict()), f"Need seg_stats to be a valid dictionary for modality expert!"

        if self.normalize_before:
            src1 = self.norm1(src)
        else:
            src1 = src

        output, self_attn = self.self_att(
            src1,
            src1,
            src1,
            attn_mask=src_mask, # remember it has to be a causal mask now!
            key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs,
        )

        # add & norm
        src = src + self.dropout1(output)
        if not self.normalize_before:
            src = self.norm1(src)

        if self.normalize_before:
            src1 = self.norm2(src)
        else:
            src1 = src
        # apply 2 modality expert on their own modality using seg_stats
        output = self.pos_ffn(src1)

        # add & norm
        output = src + self.dropout2(output)
        if not self.normalize_before:
            output = self.norm2(output)
        return output, self_attn


class InterleaveFormer(nn.Module):
    """This class implements the InterleaveFormer causal encoder.
    Arguments
    ---------
    num_layers : int
        Number of transformer layers to include.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    d_model : int
        The dimension of the input embedding.
    kdim : int
        Dimension for key (Optional).
    vdim : int
        Dimension for value (Optional).
    dropout : float
        Dropout for the encoder (Optional).
    input_module: torch class
        The module to process the source input feature to expected
        feature dimension (Optional).
    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = InterleaveFormer(1, 8, 512, d_model=512)
    >>> output, _ = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        input_shape=None,
        d_model=None,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        causal=False,
        layerdrop_prob=0.0,
        attention_type="regularMHA",
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [
                InterleaveFormerLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    causal=causal,
                    attention_type=attention_type,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.layerdrop_prob = layerdrop_prob
        self.rng = np.random.default_rng()

    def forward(
        self,
        src,
        seg_stats,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ----------
        src : tensor
            The sequence to the encoder layer (required).
        seg_stats:  dictionary
            Bi-modality indicator { key_i: array for i in batch} used by modality expert.
            Key_i: key to index sample.
            Value_of_key_i: Even element is audio segment true length. Odd element is text tokens true length.
        src_mask : tensor
            The mask for the src sequence (optional).
            now it actually a TGT_MASK that is passed in, which is a causal mask!
        src_key_padding_mask : tensor
            The mask for the src keys per batch (optional).
        """
        output = src
        if self.layerdrop_prob > 0.0:
            keep_probs = self.rng.random(len(self.layers))
        else:
            keep_probs = None
        attention_lst = []
        for i, enc_layer in enumerate(self.layers):
            if (
                not self.training
                or self.layerdrop_prob == 0.0
                or keep_probs[i] > self.layerdrop_prob
            ):
                output, attention = enc_layer(
                    output,
                    seg_stats,
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    pos_embs=pos_embs,
                )

                attention_lst.append(attention)
        output = self.norm(output)
        return output, attention_lst



class NormalizedEmbedding(nn.Module):
    """This class implements the normalized embedding layer for the transformer.
    Since the dot product of the self-attention is always normalized by sqrt(d_model)
    and the final linear projection for prediction shares weight with the embedding layer,
    we multiply the output of the embedding by sqrt(d_model).
    Arguments
    ---------
    d_model: int
        The number of expected features in the encoder/decoder inputs (default=512).
    vocab: int
        The vocab size.
    Example
    -------
    >>> emb = NormalizedEmbedding(512, 1000)
    >>> trg = torch.randint(0, 999, (8, 50))
    >>> emb_fea = emb(trg)
    """

    def __init__(self, d_model, vocab):
        super().__init__()
        self.emb = sb.nnet.embedding.Embedding(
            num_embeddings=vocab, embedding_dim=d_model, blank_id=0
        )
        self.d_model = d_model

    def forward(self, x):
        """ Processes the input tensor x and returns an output tensor."""
        return self.emb(x) * math.sqrt(self.d_model)


def get_key_padding_mask(padded_input, pad_idx):
    """Creates a binary mask to prevent attention to padded locations.
    Arguments
    ----------
    padded_input: int
        Padded input.
    pad_idx:
        idx for padding element.
    Example
    -------
    >>> a = torch.LongTensor([[1,1,0], [2,3,0], [4,5,0]])
    >>> get_key_padding_mask(a, pad_idx=0)
    tensor([[False, False,  True],
            [False, False,  True],
            [False, False,  True]])
    """
    if len(padded_input.shape) == 4:
        bz, time, ch1, ch2 = padded_input.shape
        padded_input = padded_input.reshape(bz, time, ch1 * ch2)

    key_padded_mask = padded_input.eq(pad_idx).to(padded_input.device)

    # if the input is more than 2d, mask the locations where they are silence
    # across all channels
    if len(padded_input.shape) > 2:
        key_padded_mask = key_padded_mask.float().prod(dim=-1).bool()
        return key_padded_mask.detach()

    return key_padded_mask.detach()


def get_lookahead_mask(padded_input):
    """Creates a binary mask for each sequence which maskes future frames.
    Arguments
    ---------
    padded_input: torch.Tensor
        Padded input tensor.
    Example
    -------
    >>> a = torch.LongTensor([[1,1,0], [2,3,0], [4,5,0]])
    >>> get_lookahead_mask(a)
    tensor([[0., -inf, -inf],
            [0., 0., -inf],
            [0., 0., 0.]])
    """
    seq_len = padded_input.shape[1]
    mask = (
        torch.triu(torch.ones((seq_len, seq_len), device=padded_input.device))
        == 1
    ).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask.detach().to(padded_input.device)

def get_lookahead_hopping_mask(padded_input, seg_stats):
    """Creates a binary mask for each sequence which maskes future frames in a hopping style
    since audio segments are not predicted autoregressively but text does.
    Arguments
    ---------
    padded_input: torch.Tensor
        Padded input tensor.
    seg_stats:
        array of [max_audio_len, max_text_len]
        # dictionary
        # Bi-modality indicator { key_i: array for i in batch} used by modality expert.
        # Key_i: key to index sample.
        # Value_of_key_i: Even element is audio segment true length. Odd element is text tokens true length.
    Example
    -------
    >>> a = torch.rand(1,10) # a random Tensor represent a sample
    >>> seg_stats = [3,2] # first 3 features belongs to audio where next 2 features are text
    >>> get_lookahead_mask(a) # notice that althought seq len is 10 but the shape of mask is 6x10 instead of 10x10
    tensor([[0., 0., 0., -inf, -inf],
            [0., 0., 0., -inf, -inf],
            [0., 0., 0., -inf, -inf],
            [0., 0., 0., 0.,   -inf],
            [0., 0., 0., 0.,   0.  ],]
            )
    """
    # NOT implemented YET!
    # pleaes follow get_lookahead_mask(padded_input) style
    assert padded_input.shape[1] == seg_stats[1]
    max_audio, max_text = seg_stats
    inf = float( 'inf')
    total_len = sum(seg_stats)
    tgt_mask = torch.tensor( - inf).repeat(total_len,total_len)
    tgt_mask = torch.triu(tgt_mask,1)
    tgt_mask[:max_audio,:max_audio] = 0

    return tgt_mask.detach().to(padded_input.device)
