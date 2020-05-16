"""
N-gram counting, discounting, interpolation, and backoff

Author
------
Aku Rouhe 2020
"""
import itertools
import collections


# The following functions are essentially copying the NLTK ngram counting
# pipeline with minor differences. Written from scratch, but with enough
# inspiration that I feel I want to mention the inspiration source:
# NLTK is licenced under the Apache 2.0 Licence, same as SpeechBrain
# See https://github.com/nltk/nltk
# The NLTK implementation is highly focused on getting lazy evaluation.
def pad_ends(sequence, n, left_pad_symbol="<s>", right_pad_symbol="</s>"):
    """
    Pad sentence ends with start- and end-of-sentence tokens

    In speech recognition it is important to predict the end of sentence
    and use the start of sentence to condition predictions. Typically this
    is done by adding special tokens (usually <s> and </s>) at the ends of
    each sentence. The <s> token should not be predicted, so some special
    care needs to be taken for unigrams.
    """
    if n == 1:
        return itertools.chain(tuple(sequence), (right_pad_symbol,))
    else:
        return itertools.chain(
            (left_pad_symbol,), tuple(sequence), (right_pad_symbol,)
        )


def ngrams(sequence, n):
    """
    Produce all Nth order ngrams from the sequence.

    This will generally be used in an
    ngram counting pipeline.
    """
    if n <= 0:
        raise ValueError("N must be >=1")
    # Handle the unigram case specially:
    if n == 1:
        for token in sequence:
            yield (token,)
        return
    iterator = iter(sequence)
    history = []
    for hist_length, token in enumerate(iterator, start=1):
        history.append(token)
        if hist_length == n - 1:
            break
    else:  # For-else is obscure but fits here perfectly
        return
    for token in iterator:
        yield tuple(history) + (token,)
        history.append(token)
        del history[0]
    return


def ngrams_for_evaluation(sequence, max_n):
    if max_n <= 0:
        raise ValueError("Max N must be >=1")
    # Handle the unigram case specially:
    if max_n == 1:
        for token in sequence:
            yield (token,), tuple()
        return
    iterator = iter(sequence)
    history = []
    for token in iterator:
        yield token, tuple(history)
        history.append(token)
        if len(history) == max_n:
            del history[0]
    return


def ngram_evaluation_details(data, LM):
    logprob = 0.0
    num_tokens = 0
    orders_hit = collections.Counter()
    for sentence in data:
        for token, context in ngrams_for_evaluation(
            sentence, max_n=LM.top_order
        ):
            num_tokens += 1
            order_hit, lp = LM.logprob(token, context)
            orders_hit[order_hit] += 1
            logprob += lp
    return {
        "logprob": logprob,
        "num_tokens": num_tokens,
        "orders_hit": orders_hit,
    }


def ngram_perplexity(eval_details, logbase=10):
    exponent = -eval_details["logprob"] / eval_details["num_tokens"]
    perplexity = logbase ** exponent
    return perplexity
