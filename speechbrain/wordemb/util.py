"""
Utilities for word embeddings

Authors
* Artem Ploujnikov 2021
"""
import torch


def expand_to_chars(emb, seq, seq_len, word_separator):
    """Expands word embeddings to a sequence of character
    embeddings, assigning each character the word embedding
    of the word to which it belongs
    """
    word_boundaries = seq == word_separator
    words = word_boundaries.cumsum(dim=-1)

    # TODO: Find a way to vectorize over the batch axis
    char_word_emb = torch.zeros(
        emb.size(0), seq.size(-1), emb.size(-1)).to(emb.device)
    seq_len_idx = (seq_len * seq.size(-1)).int()
    for idx, (item, item_length) in enumerate(zip(words, seq_len_idx)):
        char_word_emb[idx] = emb[idx, item]
        char_word_emb[idx, item_length:, :] = 0
        char_word_emb[idx, word_boundaries[idx], :] = 0

    return char_word_emb

