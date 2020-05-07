"""
Description:
    This module implements N-gram counting, discounting, interpolation,
    and backoff
Author:
    Aku Rouhe
"""


# The following functions are essentially copying the NLTK ngram counting
# pipeline with minor differences. Written from scratch, but with enough
# inspiration that I feel I want to mention the inspiration source:
# NLTK is licenced under the Apache 2.0 Licence, same as SpeechBrain
# See https://github.com/nltk/nltk
# The NLTK implementation is highly focused on getting lazy evaluation.
def pad_ends(sequence, n, left_pad_symbol="<s>", right_pad_symbol="</s>"):
    """
    Description:
        In speech recognition it is important to predict the end of sentence
        and use the start of sentence to condition predictions. Typically this
        is done by adding special tokens (usually <s> and </s>) at the ends of
        each sentence. The <s> token should not be predicted, so some special
        care needs to be taken for unigrams.
    Author:
        Aku Rouhe
    """
    if n == 1:
        return tuple(sequence) + (right_pad_symbol,)
    else:
        return (left_pad_symbol,) + tuple(sequence) + (right_pad_symbol,)


def ngrams(sequence, n, pad=True):
    """
    Description:
        Produce all ngrams from the sequence. This will generally be used in an
        ngram counting pipeline.
    Returns:

    Author:
        Aku Rouhe
    """
    if pad:
        sequence = pad_ends(sequence)
    ngrams = []
    del ngrams  # TODO
