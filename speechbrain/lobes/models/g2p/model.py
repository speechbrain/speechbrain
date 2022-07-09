"""The Attentional RNN model for Grapheme-to-Phoneme

Authors
 * Mirco Ravinelli 2021
 * Artem Ploujnikov 2021
"""

from speechbrain.lobes.models.transformer.Transformer import (
    TransformerInterface,
    get_lookahead_mask,
    get_key_padding_mask,
)

import torch
from torch import nn
from speechbrain.nnet.linear import Linear
from speechbrain.nnet import normalization


class AttentionSeq2Seq(nn.Module):
    """
    The Attentional RNN encoder-decoder model

    Arguments
    ---------
    enc: torch.nn.Module
        the encoder module
    encoder_emb: torch.nn.Module
        the encoder_embedding_module
    emb: torch.nn.Module
        the embedding module
    dec: torch.nn.Module
        the decoder module
    lin: torch.nn.Module
        the linear module
    out: torch.nn.Module
        the output layer (typically log_softmax)
    use_word_emb: bool
        whether or not to use word embedding
    bos_token: int
        the index of teh Beginning-of-Sentence token
    word_emb_enc: nn.Module
        a module to encode word embeddings


    Returns
    -------
    result: tuple
        a (p_seq, char_lens) tuple

    """

    def __init__(
        self,
        enc,
        encoder_emb,
        emb,
        dec,
        lin,
        out,
        bos_token=0,
        use_word_emb=False,
        word_emb_enc=None,
    ):
        super().__init__()
        self.enc = enc
        self.encoder_emb = encoder_emb
        self.emb = emb
        self.dec = dec
        self.lin = lin
        self.out = out
        self.bos_token = bos_token
        self.use_word_emb = use_word_emb
        self.word_emb_enc = word_emb_enc if use_word_emb else None

    def forward(
        self, grapheme_encoded, phn_encoded=None, word_emb=None, **kwargs
    ):
        """Computes the forward pass

        Arguments
        ---------
        grapheme_encoded: torch.Tensor
            graphemes encoded as a Torch tensor

        phn_encoded: torch.Tensor
            the encoded phonemes

        word_emb: torch.Tensor
            word embeddings (optional)

        Returns
        -------
        p_seq: torch.Tensor
            a (batch x position x token) tensor of token probabilities in each
            position
        char_lens: torch.Tensor
            a tensor of character sequence lengths
        encoder_out:
            the raw output of the encoder
        """

        chars, char_lens = grapheme_encoded
        if phn_encoded is None:
            phn_bos = get_dummy_phonemes(chars.size(0), chars.device)
        else:
            phn_bos, _ = phn_encoded

        emb_char = self.encoder_emb(chars)
        if self.use_word_emb:
            emb_char = _apply_word_emb(self.word_emb_enc, emb_char, word_emb)

        encoder_out, _ = self.enc(emb_char)
        e_in = self.emb(phn_bos)
        h, w = self.dec(e_in, encoder_out, char_lens)
        logits = self.lin(h)
        p_seq = self.out(logits)

        return p_seq, char_lens, encoder_out, w

    def _apply_word_emb(self, emb_char, word_emb):
        """Concatenate character embeddings with word embeddeings,
        possibly encoding the word embeddings if an encoder
        is provided

        Arguments
        ---------
        emb_char: torch.Tensor
            the character embedding tensor
        word_emb: torch.Tensor
            the word embedding tensor

        Returns
        -------
        result: torch.Tensor
            the concatenation of the tensor"""
        word_emb_enc = (
            self.word_emb_enc(word_emb)
            if self.word_emb_enc is not None
            else word_emb
        )
        return torch.cat([emb_char, word_emb_enc], dim=-1)


class WordEmbeddingEncoder(nn.Module):
    """A small encoder module that reduces the dimensionality
    and normalizes word embeddings

    Arguments
    ---------
    word_emb_dim: int
        the dimension of the original word embeddings
    word_emb_enc_dim: int
        the dimension of the encoded word embeddings
    norm: torch.nn.Module
        the normalization to be used (
            e.g. speechbrain.nnet.normalization.LayerNorm)
    norm_type: str
        the type of normalization to be used
    """

    def __init__(
        self, word_emb_dim, word_emb_enc_dim, norm=None, norm_type=None
    ):
        super().__init__()
        self.word_emb_dim = word_emb_dim
        self.word_emb_enc_dim = word_emb_enc_dim
        if norm_type:
            self.norm = self._get_norm(norm_type, word_emb_dim)
        else:
            self.norm = norm
        self.lin = Linear(n_neurons=word_emb_enc_dim, input_size=word_emb_dim)
        self.activation = nn.Tanh()

    def _get_norm(self, norm, dim):
        """Determines the type of normalizer

        Arguments
        ---------
        norm: str
            the normalization type: "batch", "layer" or "instance
        dim: int
            the dimensionality of the inputs
        """
        norm_cls = self.NORMS.get(norm)
        if not norm_cls:
            raise ValueError(f"Invalid norm: {norm}")
        return norm_cls(input_size=dim)

    def forward(self, emb):
        """Computes the forward pass of the embedding

        Arguments
        ---------
        emb: torch.Tensor
            the original word embeddings

        Returns
        -------
        emb_enc: torch.Tensor
            encoded word embeddings
        """
        if self.norm is not None:
            x = self.norm(emb)
        x = self.lin(x)
        x = self.activation(x)
        return x

    NORMS = {
        "batch": normalization.BatchNorm1d,
        "layer": normalization.LayerNorm,
        "instance": normalization.InstanceNorm1d,
    }


class TransformerG2P(TransformerInterface):
    """
    A Transformer-based Grapheme-to-Phoneme model

    Arguments
    ----------
    emb: torch.nn.Module
        the embedding module
    encoder_emb: torch.nn.Module
        the encoder embedding module
    char_lin: torch.nn.Module
        a linear module connecting the inputs
        to the transformer
    phn_lin: torch.nn.Module
        a linear module connecting the outputs to
        the transformer
    out: torch.nn.Module
        the decoder module (usually Softmax)
    lin: torch.nn.Module
        the linear module for outputs
    d_model: int
        The number of expected features in the encoder/decoder inputs (default=512).
    nhead: int
        The number of heads in the multi-head attention models (default=8).
    num_encoder_layers: int, optional
        The number of encoder layers in1ì the encoder.
    num_decoder_layers: int, optional
        The number of decoder layers in the decoder.
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
        Module that processes the src features to expected feature dim.
    positional_encoding: str, optional
        Type of positional encoding used. e.g. 'fixed_abs_sine' for fixed absolute positional encodings.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    kernel_size: int, optional
        Kernel size in convolutional layers when Conformer is used.
    bias: bool, optional
        Whether to use bias in Conformer convolutional layers.
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
        Whether the encoder should be causal or not (the decoder is always causal).
        If causal the Conformer convolutional layer is causal.
    pad_idx: int
        the padding index (for masks)
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
        emb,
        encoder_emb,
        char_lin,
        phn_lin,
        lin,
        out,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        custom_src_module=None,
        custom_tgt_module=None,
        positional_encoding="fixed_abs_sine",
        normalize_before=True,
        kernel_size=15,
        bias=True,
        encoder_module="transformer",
        attention_type="regularMHA",
        max_length=2500,
        causal=False,
        pad_idx=0,
        encoder_kdim=None,
        encoder_vdim=None,
        decoder_kdim=None,
        decoder_vdim=None,
        use_word_emb=False,
        word_emb_enc=None,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            custom_src_module=custom_src_module,
            custom_tgt_module=custom_tgt_module,
            positional_encoding=positional_encoding,
            normalize_before=normalize_before,
            kernel_size=kernel_size,
            bias=bias,
            encoder_module=encoder_module,
            attention_type=attention_type,
            max_length=max_length,
            causal=causal,
            encoder_kdim=encoder_kdim,
            encoder_vdim=encoder_vdim,
            decoder_kdim=decoder_kdim,
            decoder_vdim=decoder_vdim,
        )
        self.emb = emb
        self.encoder_emb = encoder_emb
        self.char_lin = char_lin
        self.phn_lin = phn_lin
        self.lin = lin

        self.out = out
        self.pad_idx = pad_idx
        self.use_word_emb = use_word_emb
        self.word_emb_enc = word_emb_enc
        self._reset_params()

    def forward(
        self, grapheme_encoded, phn_encoded=None, word_emb=None, **kwargs
    ):
        """Computes the forward pass

        Arguments
        ---------
        grapheme_encoded: torch.Tensor
            graphemes encoded as a Torch tensor

        phn_encoded: torch.Tensor
            the encoded phonemes

        word_emb: torch.Tensor
            word embeddings (if applicable)

        Returns
        -------
        p_seq: torch.Tensor
            the log-probabilities of individual tokens i a sequence
        char_lens: torch.Tensor
            the character length syntax
        encoder_out: torch.Tensor
            the encoder state
        attention: torch.Tensor
            the attention state
        """

        chars, char_lens = grapheme_encoded

        if phn_encoded is None:
            phn = get_dummy_phonemes(chars.size(0), chars.device)
        else:
            phn, _ = phn_encoded

        emb_char = self.encoder_emb(chars)
        if self.use_word_emb:
            emb_char = _apply_word_emb(self.word_emb_enc, emb_char, word_emb)

        src = self.char_lin(emb_char)
        tgt = self.emb(phn)
        tgt = self.phn_lin(tgt)

        (
            src_key_padding_mask,
            tgt_key_padding_mask,
            src_mask,
            tgt_mask,
        ) = self.make_masks(src, tgt, char_lens, pad_idx=self.pad_idx)

        pos_embs_encoder = None
        if self.attention_type == "RelPosMHAXL":
            pos_embs_encoder = self.positional_encoding(src)
        elif self.positional_encoding_type == "fixed_abs_sine":
            src = src + self.positional_encoding(src)  # add the encodings here
            pos_embs_encoder = None

        encoder_out, _ = self.encoder(
            src=src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_encoder,
        )

        if self.attention_type == "RelPosMHAXL":
            # use standard sinusoidal pos encoding in decoder
            tgt = tgt + self.positional_encoding_decoder(tgt)
            src = src + self.positional_encoding_decoder(src)
            pos_embs_encoder = None
            pos_embs_target = None
        elif self.positional_encoding_type == "fixed_abs_sine":
            tgt = tgt + self.positional_encoding(tgt)
            pos_embs_target = None
            pos_embs_encoder = None

        decoder_out, _, attention = self.decoder(
            tgt=tgt,
            memory=encoder_out,
            memory_mask=src_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            pos_embs_tgt=pos_embs_target,
            pos_embs_src=pos_embs_encoder,
        )
        logits = self.lin(decoder_out)
        p_seq = self.out(logits)
        return p_seq, char_lens, encoder_out, attention

    def _reset_params(self):
        """Resets the parameters of the model"""
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)

    def make_masks(self, src, tgt, src_len=None, pad_idx=0):
        """This method generates the masks for training the transformer model.

        Arguments
        ---------
        src : tensor
            The sequence to the encoder (required).
        tgt : tensor
            The sequence to the decoder (required).
        pad_idx : int
            The index for <pad> token (default=0).


        Returns
        -------
        src_key_padding_mask: torch.Tensor
            the source key padding mask
        tgt_key_padding_mask: torch.Tensor
            the target key padding masks
        src_mask: torch.Tensor
            the source mask
        tgt_mask: torch.Tensor
            the target mask
        """
        if src_len is not None:
            abs_len = torch.round(src_len * src.shape[1])
            src_key_padding_mask = (
                torch.arange(src.shape[1])[None, :].to(abs_len)
                > abs_len[:, None]
            )

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

        Returns
        -------
        prediction: torch.Tensor
            the predicted sequence
        attention: torch.Tensor
            the attention matrix corresponding to the last attention head
            (useful for plotting attention)
        """
        tgt_mask = get_lookahead_mask(tgt)
        tgt = self.emb(tgt)
        tgt = self.phn_lin(tgt)
        if self.attention_type == "RelPosMHAXL":
            # we use fixed positional encodings in the decoder
            tgt = tgt + self.positional_encoding_decoder(tgt)
            encoder_out = encoder_out + self.positional_encoding_decoder(
                encoder_out
            )
        elif self.positional_encoding_type == "fixed_abs_sine":
            tgt = tgt + self.positional_encoding(tgt)  # add the encodings here
        prediction, self_attns, multihead_attns = self.decoder(
            tgt,
            encoder_out,
            tgt_mask=tgt_mask,
            pos_embs_tgt=None,
            pos_embs_src=None,
        )
        attention = multihead_attns[-1]
        return prediction, attention


def input_dim(use_word_emb, embedding_dim, word_emb_enc_dim):
    """Computes the input dimension (intended for hparam files)

    Arguments
    ---------
    use_word_emb: bool
        whether to use word embeddings

    embedding_dim: int
        the embedding dimension

    word_emb_enc_dim: int
        the dimension of encoded word embeddings

    Returns
    -------
    input_dim: int
        the input dimension
    """
    return embedding_dim + use_word_emb * word_emb_enc_dim


def _apply_word_emb(word_emb_enc, emb_char, word_emb):
    """
    Concatenates character and word embeddings together, possibly
    applying a custom encoding/transformation

    Arguments
    ---------
    word_emb_enc: callable
        an encoder to apply (typically, speechbrain.lobes.models.g2p.model.WordEmbeddingEncoder)
    emb_char: torch.Tensor
        character embeddings
    word_emb: char
        word embeddings

    Returns
    -------
    result: torch.Tensor
        the resulting (concatenated) tensor
    """
    word_emb_enc = (
        word_emb_enc(word_emb.data)
        if word_emb_enc is not None
        else word_emb.data
    )
    return torch.cat([emb_char, word_emb_enc], dim=-1)


def get_dummy_phonemes(batch_size, device):
    """
    Creates a dummy phoneme sequence

    Arguments
    ---------
    batch_size: int
        the batch size
    device: str
        the target device

    Returns
    -------
    result: torch.Tensor
    """
    return torch.tensor([0], device=device).expand(batch_size, 1)
