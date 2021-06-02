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
        self,
        enc,
        encoder_emb,
        emb,
        dec,
        lin,
        out,
        bos_token=0,
        max_len=50
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

    def forward(self, grapheme_encoded, phn_encoded=None):
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
        if phn_encoded is None:
            return self.infer(grapheme_encoded)

        chars, char_lens = grapheme_encoded
        phn_bos, _ = phn_encoded

        emb_char = self.encoder_emb(chars)
        encoder_out, _ = self.enc(emb_char)

        e_in = self.emb(phn_bos)
        h, w = self.dec(e_in, encoder_out, char_lens)
        logits = self.lin(h)
        p_seq = self.out(logits)

        return p_seq, char_lens, encoder_out

    def infer(self, grapheme_encoded):
        done = False
        batch_size = len(grapheme_encoded)
        decoded_sequence = self._get_initial_sequence(batch_size)
        idx = 1
        while not done:
            step_grapheme_encoded = grapheme_encoded[:, :idx]
            step_phn_encoded = torch.tensor(decoded_sequence)
            p_seq, char_lens, _ = self.forward(
                grapheme_encoded=step_grapheme_encoded,
                phn_encoded=step_phn_encoded
            )
            #TODO: Beam search
            predictions = p_seq.argmax(dim=-1)
            decoded_sequence.append(predictions)
            idx += 1
            done = idx < self.max_len
        return p_seq, char_lens

    def _get_initial_sequence(self, batch_size):
        return torch.ones(batch_size) * self.bos_token
