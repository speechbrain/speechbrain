"""The Attentional RNN model for Grapheme-to-Phoneme

Authors
 * Mirco Ravinelli 2021
 * Artem Ploujnikov 2021 (slight refactoring only - for pretrainer
   compatibility)
"""

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
    max_len: int
        the maximum length
    use_word_emb: bool
        whether or not to use word embedding
    word_emb_enc: nn.Module
        a module to encode word embeddings


    Returns
    -------
    result: tuple
        a (p_seq, char_lens) tuple

    """

    def __init__(
        self, enc, encoder_emb, emb, dec, lin, out, bos_token=0, max_len=50, use_word_emb=False, word_emb_enc=None
    ):
        super().__init__()
        self.enc = enc
        self.encoder_emb = encoder_emb
        self.emb = emb
        self.dec = dec
        self.lin = lin
        self.out = out
        self.bos_token = bos_token
        self.max_len = max_len
        self.use_word_emb = use_word_emb
        self.word_emb_enc = word_emb_enc if use_word_emb else None

    def forward(
        self,
        grapheme_encoded,
        phn_encoded=None,
        word_emb=None,
        **kwargs
    ):
        """
        Computes the forward pass

        Arguments
        ---------
        grapheme_encoded: torch.Tensor
            graphemes encoded as a Torch tensor

        phn_encoded: torch.Tensor
            the encoded phonemes

        Returns
        -------
        result: tuple
            a tuple of (p_seq, char_lens, encoder_out) - sequence
            probabilities, character lengths and
        """

        chars, char_lens = grapheme_encoded
        if phn_encoded is None:
            phn_bos = self._get_dummy_phonemes(chars.size(0), chars.device)
        else:
            phn_bos, _ = phn_encoded

        emb_char = self.encoder_emb(chars)
        if self.use_word_emb:
            emb_char = self._apply_word_emb(emb_char, word_emb)

        encoder_out, _ = self.enc(emb_char)
        e_in = self.emb(phn_bos)
        h, w = self.dec(e_in, encoder_out, char_lens)
        logits = self.lin(h)
        p_seq = self.out(logits)

        return p_seq, char_lens, encoder_out, w

    def _apply_word_emb(self, emb_char, word_emb):
        word_emb_enc = (
            self.word_emb_enc(word_emb)
            if self.word_emb_enc is not None
            else word_emb)
        return torch.cat([emb_char, word_emb_enc], dim=-1)

    def _get_dummy_phonemes(self, batch_size, device):
        return torch.tensor([0], device=device).expand(batch_size, 1)


class WordEmbeddingEncoder(nn.Module):
    NORMS = {
        "batch": normalization.BatchNorm1d,
        "layer": normalization.LayerNorm,
        "instance": normalization.InstanceNorm1d
    }
    """A small encoder module that reduces the dimensionality
    and normalizes word embeddings

    Arguments
    ---------
    word_emb_dim: int
        the dimension of the original word embeddings
    word_emb_enc_dim: int
        the dimension of the encoded word embeddings
    norm_type: str
        the type of normalization to be used
    norm: torch.nn.Module
        the normalization to be used (
            e.g. speechbrain.nnet.normalization.LayerNorm)
    """
    def __init__(self, word_emb_dim, word_emb_enc_dim, norm=None, norm_type=None):
        super().__init__()
        self.word_emb_dim = word_emb_dim
        self.word_emb_enc_dim = word_emb_enc_dim
        if norm_type:
            self.norm = self._get_norm(norm_type, word_emb_dim)
        else:
            self.norm = norm
        self.lin = Linear(
            n_neurons=word_emb_enc_dim, input_size=word_emb_dim)
        self.activation = nn.Tanh()

    def _get_norm(self, norm, dim):
        """Determines the type of normalizer"""
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