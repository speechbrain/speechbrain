"""Transformer for ASR in the SpeechBrain sytle.

Authors
* Jianyuan Zhong 2020
"""

import torch  # noqa 42
from torch import nn
from typing import Optional

from speechbrain.nnet.linear import Linear
from speechbrain.nnet.containers import ModuleList
from speechbrain.lobes.models.transformer.TransformerUtils import (
    get_lookahead_mask,
    get_key_padding_mask,
    NormalizedEmbedding,
    PositionalEncoding,
)
from speechbrain.nnet.attention import RelPosEncXL

from speechbrain.dataio.dataio import length_to_mask


class TransformerASR(nn.Module):
    """This is an implementation of transformer model for ASR.

    The architecture is based on the paper "Attention Is All You Need":
    https://arxiv.org/pdf/1706.03762.pdf

    Arguments
    ----------
    tgt_vocab: int
        Size of vocabulary.
    input_size: int
        Input feature size.
    d_model : int, optional
        Embedding dimension size.
        (default=512).
    encoder: encoder module
        A module to be used as the encoder part in the Transformer
    decoder: decoder module
        A module to be used as the decoder part in the Transformer
    dropout : int, optional
        The dropout value (default=0.1).
    positional_encoding_encoder: Module
        Module to be used for the Positional Encoding Encoder part
    positional_encoding_decoder: Module
        Module to be used for the Positional Encoding Encoder part

    Example
    -------
    >>> import torch
    >>> from speechbrain.nnet.attention import MultiheadAttention
    >>> from speechbrain.lobes.models.transformer.TransformerUtils import TransformerEncoder, TransformerDecoder
    >>> x = torch.rand((8, 60, 512))
    >>> inputs = torch.rand([8, 60, 512])
    >>> mha1 = MultiheadAttention(nhead=8, d_model=inputs.shape[-1])
    >>> mha2 = MultiheadAttention(nhead=8, d_model=inputs.shape[-1])
    >>> mha3 = MultiheadAttention(nhead=8, d_model=inputs.shape[-1])
    >>> enc = TransformerEncoder(1, 512, mha1, d_model=512)
    >>> dec = TransformerDecoder(8, mha2, mha3, d_model=512, d_ffn=512)
    >>> pos_enc = PositionalEncoding(input_size=512)
    >>> pos_dec = PositionalEncoding(input_size=512)
    >>> src = torch.rand([8, 120, 512])
    >>> tgt = torch.randint(0, 720, [8, 120])
    >>> net = TransformerASR(tgt_vocab=720, input_size=512, d_model=512, encoder=enc, decoder=dec,
    ... positional_encoding_encoder=pos_enc, positional_encoding_decoder=pos_dec, dropout=0.1)
    >>> enc_out, dec_out = net.forward(src, tgt)
    >>> enc_out.shape
    torch.Size([8, 120, 512])
    >>> dec_out.shape
    torch.Size([8, 120, 512])
    """

    def __init__(
        self,
        tgt_vocab,
        input_size,
        d_model,
        dropout,
        encoder,
        decoder,
        positional_encoding_encoder: Optional[object] = None,
        positional_encoding_decoder: Optional[object] = None,
    ):
        super().__init__()
        self.positional_encoding = positional_encoding_encoder
        self.positional_encoding_decoder = positional_encoding_decoder
        self.encoder = encoder
        self.decoder = decoder if decoder is not None else None
        self.custom_src_module = ModuleList(
            Linear(
                input_size=input_size,
                n_neurons=d_model,
                bias=True,
                combine_dims=False,
            ),
            torch.nn.Dropout(dropout),
        )
        self.custom_tgt_module = ModuleList(
            NormalizedEmbedding(d_model, tgt_vocab)
        )
        # reset parameters using xavier_normal_
        self._init_params()

    def forward(
        self, src, tgt, wav_len=None, pad_idx=0,
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder.
        tgt : torch.Tensor
            The sequence to the decoder.
        wav_len: torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        pad_idx : int, optional
            The index for <pad> token (default=0).
        """

        # reshpae the src vector to [Batch, Time, Fea] is a 4d vector is given
        if src.dim() == 4:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        (
            src_key_padding_mask,
            tgt_key_padding_mask,
            src_mask,
            tgt_mask,
        ) = self.make_masks(src, tgt, wav_len, pad_idx=pad_idx)

        src = self.custom_src_module(src)
        # add pos encoding to queries if are sinusoidal ones else
        if isinstance(self.positional_encoding, RelPosEncXL):
            pos_embs_encoder = self.positional_encoding(src)
        elif isinstance(self.positional_encoding, PositionalEncoding):
            src = src + self.positional_encoding(src)  # add the encodings here
            pos_embs_encoder = None

        encoder_out, _ = self.encoder(
            src=src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_encoder,
        )

        tgt = self.custom_tgt_module(tgt)

        if isinstance(self.positional_encoding, RelPosEncXL):
            # use standard sinusoidal pos encoding in decoder
            tgt = tgt + self.positional_encoding_decoder(tgt)
            src = src + self.positional_encoding_decoder(src)
            pos_embs_encoder = None  # self.positional_encoding(src)
            pos_embs_target = None
        elif isinstance(self.positional_encoding, PositionalEncoding):
            tgt = tgt + self.positional_encoding(tgt)
            pos_embs_target = None
            pos_embs_encoder = None

        decoder_out, _, _ = self.decoder(
            tgt=tgt,
            memory=encoder_out,
            memory_mask=src_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            pos_embs_tgt=pos_embs_target,
            pos_embs_src=pos_embs_encoder,
        )

        return encoder_out, decoder_out

    def make_masks(self, src, tgt, wav_len=None, pad_idx=0):
        """This method generates the masks for training the transformer model.

        Arguments
        ---------
        src : tensor
            The sequence to the encoder (required).
        tgt : tensor
            The sequence to the decoder (required).
        pad_idx : int
            The index for <pad> token (default=0).
        """
        src_key_padding_mask = None
        if wav_len is not None and self.training:
            abs_len = torch.round(wav_len * src.shape[1])
            src_key_padding_mask = (1 - length_to_mask(abs_len)).bool()
        tgt_key_padding_mask = get_key_padding_mask(tgt, pad_idx=pad_idx)

        src_mask = None
        tgt_mask = get_lookahead_mask(tgt)
        return src_key_padding_mask, tgt_key_padding_mask, src_mask, tgt_mask

    def decode(self, tgt, encoder_out):
        """This method implements a decoding step for the transformer model.

        Arguments
        ---------
        tgt : torch.Tensor
            The sequence to the decoder.
        encoder_out : torch.Tensor
            Hidden output of the encoder.
        """
        tgt_mask = get_lookahead_mask(tgt)
        tgt = self.custom_tgt_module(tgt)
        if isinstance(self.positional_encoding, RelPosEncXL):
            # we use fixed positional encodings in the decoder
            tgt = tgt + self.positional_encoding_decoder(tgt)
            encoder_out = encoder_out + self.positional_encoding_decoder(
                encoder_out
            )
            # pos_embs_target = self.positional_encoding(tgt)
            pos_embs_encoder = None  # self.positional_encoding(src)
            pos_embs_target = None
        elif isinstance(self.positional_encoding, PositionalEncoding):
            tgt = tgt + self.positional_encoding(tgt)  # add the encodings here
            pos_embs_target = None
            pos_embs_encoder = None

        prediction, self_attns, multihead_attns = self.decoder(
            tgt,
            encoder_out,
            tgt_mask=tgt_mask,
            pos_embs_tgt=pos_embs_target,
            pos_embs_src=pos_embs_encoder,
        )
        return prediction, multihead_attns[-1]

    def encode(
        self, src, wav_len=None,
    ):
        """
        Encoder forward pass

        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder.
        wav_len: torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        """
        # reshape the src vector to [Batch, Time, Fea] if a 4d vector is given
        if src.dim() == 4:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        src_key_padding_mask = None
        if wav_len is not None and self.training:
            abs_len = torch.round(wav_len * src.shape[1])
            src_key_padding_mask = (1 - length_to_mask(abs_len)).bool()

        src = self.custom_src_module(src)
        if isinstance(self.positional_encoding, RelPosEncXL):
            pos_embs_source = self.positional_encoding(src)

        elif isinstance(self.positional_encoding, PositionalEncoding):
            src = src + self.positional_encoding(src)
            pos_embs_source = None

        encoder_out, _ = self.encoder(
            src=src,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_source,
        )
        return encoder_out

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)


class EncoderWrapper(nn.Module):
    """This is a wrapper of any ASR transformer encoder. By default, the
    TransformerASR .forward() function encodes and decodes. With this wrapper
    the .forward() function becomes .encode() only.

    Important: The TransformerASR class must contain a .encode() function.

    Arguments
    ----------
    transformer : sb.lobes.models.TransformerInterface
        A Transformer instance that contains a .encode() function.

    Example
    -------
    >>> import torch
    >>> from speechbrain.nnet.attention import MultiheadAttention
    >>> from speechbrain.lobes.models.transformer.TransformerUtils import TransformerEncoder, TransformerDecoder
    >>> x = torch.rand((8, 60, 512))
    >>> inputs = torch.rand([8, 60, 512])
    >>> mha1 = MultiheadAttention(nhead=8, d_model=inputs.shape[-1])
    >>> mha2 = MultiheadAttention(nhead=8, d_model=inputs.shape[-1])
    >>> mha3 = MultiheadAttention(nhead=8, d_model=inputs.shape[-1])
    >>> enc = TransformerEncoder(1, 512, mha1, d_model=512)
    >>> dec = TransformerDecoder(8, mha2, mha3, d_model=512, d_ffn=512)
    >>> pos_enc = PositionalEncoding(input_size=512)
    >>> pos_dec = PositionalEncoding(input_size=5122)
    >>> src = torch.rand([8, 120, 512])
    >>> tgt = torch.randint(0, 720, [8, 120])
    >>> net = TransformerASR(tgt_vocab=720, input_size=512, d_model=512, encoder=enc, decoder=dec,
    ... positional_encoding_encoder=pos_enc, positional_encoding_decoder=pos_dec, dropout=0.1)
    >>> enc_out, dec_out = net.forward(src, tgt)
    >>> enc_out.shape
    torch.Size([8, 120, 512])
    """

    def __init__(self, transformer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer = transformer

    def forward(self, x, wav_lens=None):
        x = self.transformer.encode(x, wav_lens)
        return x
