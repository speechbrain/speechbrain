#!/usr/bin/env python3

# Copyright 2020 Xiaomi Corporation (Author: Junbo Zhang, Haowen Qiu)
# Apache 2.0
"""
arpa2fst.py:
Similar to Kaldi's `lmbin/arpa2fst.cc`, rewritten in Python.

Generally, we do the following. Suppose we are adding an n-gram "A B C". Then find the node for "A B", add a new node
for "A B C", and connect them with the arc accepting "C" with the specified weight. Also, add a backoff arc from the
new "A B C" node to its backoff state "B C".

Two notable exceptions are the highest order n-grams, and final n-grams.

When adding a highest order n-gram (e. g., our "A B C" is in a 3-gram LM), the following optimization is performed.
There is no point adding a node for "A B C" with a "C" arc from "A B", since there will be no other arcs ingoing to
this node, and an epsilon backoff arc into the backoff model "B C", with the weight of \bar{1}. To save a node,
create an arc accepting "C" directly from "A B" to "B C". This saves as many nodes as there are the highest order
n-grams, which is typically about half the size of a large 3-gram model.

Indeed, this does not apply to n-grams ending in EOS, since they do not back off. These are special, as they do not
have a back-off state, and the node for "(..anything..) </s>" is always final. These are handled in one of the two
possible ways, If symbols <s> and </s> are being replaced by epsilons, neither node nor arc is created,
and the logprob of the n-gram is applied to its source node as final weight. If <s> and </s> are preserved,
then a special final node for </s> is allocated and used as the destination of the "</s>" acceptor arc. """

import re
import argparse
from typing import List, NamedTuple


def get_args():
    parser = argparse.ArgumentParser(
        description="""This script is similar to Kaldi's `lmbin/arpa2fst.cc`,
                but was rewritten in Python. The output goes to the stdout.""")
    parser.add_argument('arpa', type=str, help="""The arpa file.""")
    parser.add_argument('--bos',
                        type=str,
                        default='<s>',
                        help="""The begin symbol.""")
    parser.add_argument('--eos',
                        type=str,
                        default='</s>',
                        help="""The ending symbol.""")
    parser.add_argument('--disambig_symbol',
                        type=str,
                        default='#0',
                        help="""The disambig symbol.""")
    parser.add_argument('--to-natural-log',
                        type=bool,
                        default=True,
                        help="""Convert to natural log.""")
    args = parser.parse_args()
    return args


class Ngram(NamedTuple):
    logprob: float
    prev_words: List
    cur_word: str
    backoff: float


def parse_arpa_file(filename, to_natural_log=True):
    ngrams = []
    with open(filename) as f:
        stage = 0  # stage 0 for header, 1 for unigram, 2 for bigram, ...
        for line in f:
            line = re.sub(r'\s+$', '', line)
            if re.match(r'^\s*$', line) or re.match(
                    r'\\(data|end)\\', line) or re.match(r'ngram \d+=', line):
                continue
            elif re.match(r'\\\d+-grams:', line):
                stage = int(re.match(r'\\(\d+)-grams:', line).group(1))
                ngrams.append([])
                assert len(ngrams) == stage
            else:
                items = re.split(r'\s+', line)
                assert stage + 1 <= len(
                    items) <= stage + 2, f'Invalid arpa line: {line}'

                logprob = float(items[0]) if not to_natural_log else float(
                    items[0]) * 2.30258509299404568402
                words = items[1:stage + 1]
                backoff = float(items[stage +
                                      1]) if len(items) == stage + 2 else 0.0
                if to_natural_log:
                    backoff = backoff * 2.30258509299404568402

                if stage == 1:
                    prev_words = []
                    assert len(words) == 1
                    cur_word = words[0]
                else:
                    prev_words = words[0:-1]
                    cur_word = words[-1]

                ngrams[stage - 1].append(
                    Ngram(logprob=logprob,
                          prev_words=prev_words,
                          cur_word=cur_word,
                          backoff=backoff))

    return ngrams


def create_backoff(key, state, state_id, weight, sub_eps):
    while key not in state_id:
        key = ' '.join((key.split(' '))[1:])
    dest = state_id[key]
    print(f'{state} {dest} {sub_eps} {weight}')


# TODO(haowen): refactor the code (and support case where sub_eps is empty?)
def print_fst_from_ngrams(ngram_lm, bos='<s>', eos='</s>', sub_eps='#0'):
    # for now this version only support case that disambig sybmol is not zero
    assert sub_eps != ''
    highest_order = len(ngram_lm)
    state_id = {bos: 0, '': 1, eos: 2}
    state_count = len(state_id)
    for order, ngrams in enumerate(ngram_lm, start=1):
        for logprob, prev_words, cur_word, backoff in ngrams:
            assert len(prev_words) + 1 == order
            prev_words_str = ' '.join(prev_words)
            whole_words_str = ' '.join(prev_words + [cur_word])
            if prev_words_str not in state_id:
                continue  # no parent (n-1) gram
            source = state_id[prev_words_str]
            weight = -logprob
            assert cur_word != sub_eps
            if cur_word == eos:
                if sub_eps == '':
                    dest = state_id[eos]
                else:
                    # treat </s> as if it was epsilon; mark source final
                    print(f'{source} {weight}')
                    continue
            else:
                key = whole_words_str if order != highest_order else ' '.join(
                    (prev_words + [cur_word])[1:])
                if key not in state_id:
                    state_id[key] = state_count
                    dest = state_count
                    state_count += 1
                    tails = ' '.join((key.split(' '))[1:])
                    create_backoff(tails, dest, state_id, -backoff, sub_eps)
                else:
                    dest = state_id[key]

            if cur_word == bos:
                weight = 0
                if sub_eps != '':
                    continue

            print(f'{source} {dest} {cur_word} {weight}')

    if sub_eps == '':
        print(f'{state_id[eos]} 0')


def main():
    args = get_args()
    print_fst_from_ngrams(parse_arpa_file(args.arpa, args.to_natural_log),
                          args.bos, args.eos, args.disambig_symbol)


if __name__ == '__main__':
    main()
