"""The Attentional RNN model for Grapheme-to-Phoneme

Authors
 * Mirco Ravinelli 2021
 * Artem Ploujnikov 2021 (slight refactoring only - for pretrainer
   compatibility)
"""

import torch
from torch import nn


class Model(nn.Module):
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
    max_len
        the maximum length


    Returns
    -------
    result: tuple
        a (p_seq, char_lens) tuple

    """

    def __init__(
        self, enc, encoder_emb, emb, dec, lin, out, bos_token=0, max_len=50
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

    def forward(self, grapheme_encoded, phn_encoded=None, **kwargs):
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
        encoder_out, _ = self.enc(emb_char)

        e_in = self.emb(phn_bos)
        h, w = self.dec(e_in, encoder_out, char_lens)
        logits = self.lin(h)
        p_seq = self.out(logits)

        return p_seq, char_lens, encoder_out

    def _get_dummy_phonemes(self, batch_size, device):
        return torch.tensor([0], device=device).expand(batch_size, 1)
