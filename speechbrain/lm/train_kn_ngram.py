#!/usr/bin/env python3
"""
This is an implementation of computing Kneser-Ney smoothed language model
in the same way as srilm. This is a back-off, unmodified version of
Kneser-Ney smoothing, which produces the same results as the following
command (as an example) of srilm:

$ ngram-count -order 4 -kn-modify-counts-at-end -ukndiscount -gt1min 0 -gt2min 0 \
    -gt3min 0 -gt4min 0 -text corpus.txt -lm lm.arpa

The data structure is based on: kaldi/egs/wsj/s5/utils/lang/make_phone_lm.py
The smoothing algorithm is based on: http://www.speech.sri.com/projects/srilm/manpages/ngram-discount.7.html

The code below is adjusted from icefall to work with SpeechBrain.

Modified by:
    * Georgios Karakasidis 2023
"""

import io
import math
import os
from pathlib import Path
import re
import sys
import logging
from collections import Counter, defaultdict
from typing import Iterable, List


logger = logging.getLogger(__name__)
# For encoding-agnostic scripts, we assume byte stream as input.
# Need to be very careful about the use of strip() and split()
# in this case, because there is a latin-1 whitespace character
# (nbsp) which is part of the unicode encoding range.
# Ref: kaldi/egs/wsj/s5/utils/lang/bpe/prepend_words.py @ 69cd717
DEFAULT_ENCODING = "latin-1"


class CountsForHistory:
    """This class (which is more like a struct) stores the counts seen in a
    particular history-state.  It is used inside class NgramCounts. It really
    does the job of a dict from int to float, but it also keeps track of the
    total count.
    """
    def __init__(self):
        # The 'lambda: defaultdict(float)' is an anonymous function taking no
        # arguments that returns a new defaultdict(float).
        self.word_to_count = defaultdict(int)
        # using a set to count the number of unique contexts
        self.word_to_context = defaultdict(set)
        self.word_to_f = dict()  # discounted probability
        self.word_to_bow = dict()  # back-off weight
        self.total_count = 0

    def words(self):
        return self.word_to_count.keys()

    def __str__(self):
        # e.g. returns ' total=12: 3->4, 4->6, -1->2'
        return " total={0}: {1}".format(
            str(self.total_count),
            ", ".join(
                [
                    "{0} -> {1}".format(word, count)
                    for word, count in self.word_to_count.items()
                ]
            ),
        )

    def add_count(self, predicted_word, context_word, count):
        assert count >= 0

        self.total_count += count
        self.word_to_count[predicted_word] += count
        if context_word is not None:
            self.word_to_context[predicted_word].add(context_word)


class NgramCounts:
    """A note on data-structure.  Firstly, all words are represented as integers.
    We store n-gram counts as an array, indexed by (history-length == n-gram order
    minus one) (note: python calls arrays "lists") of dicts from histories to counts,
    where histories are arrays of integers and "counts" are dicts from integer to float.
    For instance, when accumulating the 4-gram count for the '8' in the sequence
    '5 6 7 8', we'd do as follows: self.counts[3][[5,6,7]][8] += 1.0 where the [3]
    indexes an array, the [[5,6,7]] indexes a dict, and the [8] indexes a dict.

    Arguments
    ---------
    ngram_order : int
        The order of n-gram.
    bos_symbol : str
        The symbol for beginning of sentence.
    eos_symbol : str
        The symbol for end of sentence.

    Example
    -------
    >>> import sys
    >>> from speechbrain.k2_integration.train_ngram import NgramCounts
    >>> ngram_counts = NgramCounts(ngram_order=4)
    >>> text = ["utterance1", "utterance2"]
    >>> ngram_counts.add_raw_counts_from_list_of_strs(text)
    >>> ngram_counts.cal_discounting_constants()
    >>> ngram_counts.cal_f()
    >>> ngram_counts.cal_bow()
    >>> ngram_counts.print_as_arpa(sys.stdout)
    """
    def __init__(self, ngram_order, bos_symbol="<s>", eos_symbol="</s>"):
        assert ngram_order >= 2

        self.ngram_order = ngram_order
        self.bos_symbol = bos_symbol
        self.eos_symbol = eos_symbol

        self.strip_chars = " \t\r\n"
        self.whitespace = re.compile("[ \t]+")

        self.counts = []
        for n in range(ngram_order):
            self.counts.append(defaultdict(lambda: CountsForHistory()))

        self.d = []  # list of discounting factor for each order of ngram

    def add_count(self, history, predicted_word, context_word, count):
        """Add a raw count (called while processing input data).
        Suppose we see the sequence '6 7 8 9' and ngram_order=4, 'history'
        would be (6,7,8) and 'predicted_word' would be 9; 'count' would be
        1.

        Arguments
        ---------
        history : tuple
            The history of n-gram.
        predicted_word : str
            The predicted word.
        context_word : str
            The context word.
        count : int
            The count of the n-gram.
        """
        self.counts[len(history)][history].add_count(
            predicted_word, context_word, count
        )

    def add_raw_counts_from_line(self, line):
        """Add the un-smoothed counts from this line of text.
        'line' is a string containing a sequence of integer word-ids.
        This function adds the un-smoothed counts from this line of text.

        Arguments
        ---------
        line : str
            The line of text as a string containing a sequence of integer word-ids.
        """
        if line == "":
            words = [self.bos_symbol, self.eos_symbol]
        else:
            words = [self.bos_symbol] + self.whitespace.split(line) + [self.eos_symbol]
        for i in range(len(words)):
            for n in range(1, self.ngram_order + 1):
                if i + n > len(words):
                    break
                ngram = words[i : i + n]
                predicted_word = ngram[-1]
                history = tuple(ngram[:-1])
                if i == 0 or n == self.ngram_order:
                    context_word = None
                else:
                    context_word = words[i - 1]

                self.add_count(history, predicted_word, context_word, 1)

    def add_raw_counts_from_standard_input(self):
        """Add the un-smoothed counts from standard input."""
        lines_processed = 0
        # byte stream as input
        infile = io.TextIOWrapper(sys.stdin.buffer, encoding=DEFAULT_ENCODING)
        for line in infile:
            line = line.strip(self.strip_chars)
            self.add_raw_counts_from_line(line)
            lines_processed += 1
        logger.debug(
            "processed {0} lines of input".format(
                lines_processed
            )
        )

    def add_raw_counts_from_file(self, filename):
        """Read a file containing text sentences in each line and add the un-smoothed
        counts from its content.

        Arguments
        ---------
        filename : str
            The file containing text sentences in each line.
        """
        lines_processed = 0
        with open(filename, encoding=DEFAULT_ENCODING) as fp:
            for line in fp:
                line = line.strip(self.strip_chars)
                self.add_raw_counts_from_line(line)
                lines_processed += 1
        logger.info(
            "processed {0} lines of input".format(
                lines_processed
            )
        )

    def add_raw_counts_from_list_of_strs(self, list_of_tokens: Iterable[str]):
        """Add the un-smoothed counts from a list of strings.
        
        Arguments
        ---------
        list_of_tokens : Iterable[str]
            The list of strings, where each string contains a sequence of integer word-ids.
        """
        lines_processed = 0
        for line in list_of_tokens:
            line = line.strip(self.strip_chars)
            # print("Using line: ", line)
            self.add_raw_counts_from_line(line)
            lines_processed += 1
            # if lines_processed == 10:
            #     raise
        logger.debug(
            "processed {0} lines of input".format(
                lines_processed
            )
        )

    def cal_discounting_constants(self):
        """This function calculates the discounting constants D_N for each order N of N-grams.
        For each order N of N-grams, we calculate discounting constant D_N = n1_N / (n1_N + 2 * n2_N),
        where n1_N is the number of unique N-grams with count = 1 (counts-of-counts).
        This constant is used similarly to absolute discounting.

        For the lowest order, i.e., 1-gram, we do not need to discount, thus the constant is 0
        This is a special case: as we currently assumed having seen all vocabularies in the dictionary,
        but perhaps this is not the case for some other scenarios.

        Return
        ------
        d : list
            A list of floats, where d[N+1] = D_N
        """
        self.d = [0]
        for n in range(1, self.ngram_order):
            this_order_counts = self.counts[n]
            n1 = 0
            n2 = 0
            for hist, counts_for_hist in this_order_counts.items():
                stat = Counter(counts_for_hist.word_to_count.values())
                n1 += stat[1]
                n2 += stat[2]
            assert n1 + 2 * n2 > 0

            # We are doing this max(0.001, xxx) to avoid zero discounting constant D due to n1=0,
            # which could happen if the number of symbols is small.
            # Otherwise, zero discounting constant can cause division by zero in computing BOW.
            self.d.append(max(0.1, n1 * 1.0) / (n1 + 2 * n2))
        logger.info("`cal_discounting_constants` done.")

    def cal_f(self):
        """This function calculates the discounted probability of each N-gram.
        f(a_z) is a probability distribution of word sequence a_z.
        Typically f(a_z) is discounted to be less than the ML estimate so we have
        some leftover probability for the z words unseen in the context (a_).
        
        f(a_z) = (c(a_z) - D0) / c(a_)    ;; for highest order N-grams
        f(_z)  = (n(*_z) - D1) / n(*_*)	;; for lower order N-grams
        
        where c(a_z) is the count of N-gram a_z, c(a_) is the count of N-1-gram a_,
        n(*_z) is the number of unique N-grams with count c(a_z) (modified count),
        n(*_*) is the number of unique N-grams with count c(a_) (modified count),
        D0 is the discounting constant for highest order N-grams,
        D1 is the discounting constant for lower order N-grams.
        """

        # highest order N-grams
        n = self.ngram_order - 1
        this_order_counts = self.counts[n]
        for hist, counts_for_hist in this_order_counts.items():
            for w, c in counts_for_hist.word_to_count.items():
                counts_for_hist.word_to_f[w] = (
                    max((c - self.d[n]), 0) * 1.0 / counts_for_hist.total_count
                )

        # lower order N-grams
        for n in range(0, self.ngram_order - 1):
            this_order_counts = self.counts[n]
            for hist, counts_for_hist in this_order_counts.items():

                n_star_star = 0
                for w in counts_for_hist.word_to_count.keys():
                    n_star_star += len(counts_for_hist.word_to_context[w])

                if n_star_star != 0:
                    for w in counts_for_hist.word_to_count.keys():
                        n_star_z = len(counts_for_hist.word_to_context[w])
                        counts_for_hist.word_to_f[w] = (
                            max((n_star_z - self.d[n]), 0) * 1.0 / n_star_star
                        )
                else:  # patterns begin with <s>, they do not have "modified count", so use raw count instead
                    for w in counts_for_hist.word_to_count.keys():
                        n_star_z = counts_for_hist.word_to_count[w]
                        counts_for_hist.word_to_f[w] = (
                            max((n_star_z - self.d[n]), 0)
                            * 1.0
                            / counts_for_hist.total_count
                        )
        logger.info("`cal_f` done.")

    def cal_bow(self):
        """Backoff weights (BOWs) are only necessary for ngrams which form a prefix 
        of a longer ngram. Thus, two sorts of ngrams do not have a bow:
            1) highest order ngrams
            2) ngrams ending in </s>
        
        bow(a_) = (1 - Sum_Z1 f(a_z)) / (1 - Sum_Z1 f(_z))
        where Z1 is the set of all words with c(a_z) > 0,
        Sum_Z1 f(a_z) is the sum of f(a_z) for all words in Z1,
        Sum_Z1 f(_z) is the sum of f(_z) for all words in Z1.
        """

        # highest order N-grams
        n = self.ngram_order - 1
        this_order_counts = self.counts[n]
        for hist, counts_for_hist in this_order_counts.items():
            for w in counts_for_hist.word_to_count.keys():
                counts_for_hist.word_to_bow[w] = None

        # lower order N-grams
        for n in range(0, self.ngram_order - 1):
            this_order_counts = self.counts[n]
            for hist, counts_for_hist in this_order_counts.items():
                for w in counts_for_hist.word_to_count.keys():
                    if w == self.eos_symbol:
                        counts_for_hist.word_to_bow[w] = None
                    else:
                        a_ = hist + (w,)

                        assert len(a_) < self.ngram_order
                        assert a_ in self.counts[len(a_)].keys()

                        a_counts_for_hist = self.counts[len(a_)][a_]

                        sum_z1_f_a_z = 0
                        for u in a_counts_for_hist.word_to_count.keys():
                            sum_z1_f_a_z += a_counts_for_hist.word_to_f[u]

                        sum_z1_f_z = 0
                        _ = a_[1:]
                        _counts_for_hist = self.counts[len(_)][_]
                        # Should be careful here: what is Z1
                        for u in a_counts_for_hist.word_to_count.keys():
                            sum_z1_f_z += _counts_for_hist.word_to_f[u]

                        if sum_z1_f_z < 1:
                            # assert sum_z1_f_a_z < 1
                            counts_for_hist.word_to_bow[w] = (1.0 - sum_z1_f_a_z) / (
                                1.0 - sum_z1_f_z
                            )
                        else:
                            counts_for_hist.word_to_bow[w] = None
        logger.info("`cal_bow` done.")

    def print_raw_counts(self, info_string):
        """Print the raw counts of each N-gram.
        
        Arguments
        ---------
        info_string : str
            The information string to be printed.
        """
        print(info_string)
        res = []
        for this_order_counts in self.counts:
            for hist, counts_for_hist in this_order_counts.items():
                for w in counts_for_hist.word_to_count.keys():
                    ngram = " ".join(hist) + " " + w
                    ngram = ngram.strip(self.strip_chars)

                    res.append(
                        "{0}\t{1}".format(ngram, counts_for_hist.word_to_count[w])
                    )
        res.sort(reverse=True)
        for r in res:
            print(r)

    def print_modified_counts(self, info_string):
        """Print the modified counts of each N-gram.

        Arguments
        ---------
        info_string : str
            The information string to be printed.
        """
        print(info_string)
        res = []
        for this_order_counts in self.counts:
            for hist, counts_for_hist in this_order_counts.items():
                for w in counts_for_hist.word_to_count.keys():
                    ngram = " ".join(hist) + " " + w
                    ngram = ngram.strip(self.strip_chars)

                    modified_count = len(counts_for_hist.word_to_context[w])
                    raw_count = counts_for_hist.word_to_count[w]

                    if modified_count == 0:
                        res.append("{0}\t{1}".format(ngram, raw_count))
                    else:
                        res.append("{0}\t{1}".format(ngram, modified_count))
        res.sort(reverse=True)
        for r in res:
            print(r)

    def print_f(self, info_string):
        """Print the discounted probability of each N-gram.
        
        Arguments
        ---------
        info_string : str
            The information string to be printed.
        """
        print(info_string)
        res = []
        for this_order_counts in self.counts:
            for hist, counts_for_hist in this_order_counts.items():
                for w in counts_for_hist.word_to_count.keys():
                    ngram = " ".join(hist) + " " + w
                    ngram = ngram.strip(self.strip_chars)

                    f = counts_for_hist.word_to_f[w]
                    if f == 0:  # f(<s>) is always 0
                        f = 1e-99

                    res.append("{0}\t{1}".format(ngram, math.log(f, 10)))
        res.sort(reverse=True)
        for r in res:
            print(r)

    def print_f_and_bow(self, info_string):
        """Print the discounted probability and back-off weight of each N-gram.
        
        Arguments
        ---------
        info_string : str
            The information string to be printed.
        """
        print(info_string)
        res = []
        for this_order_counts in self.counts:
            for hist, counts_for_hist in this_order_counts.items():
                for w in counts_for_hist.word_to_count.keys():
                    ngram = " ".join(hist) + " " + w
                    ngram = ngram.strip(self.strip_chars)

                    f = counts_for_hist.word_to_f[w]
                    if f == 0:  # f(<s>) is always 0
                        f = 1e-99

                    bow = counts_for_hist.word_to_bow[w]
                    if bow is None:
                        res.append("{1}\t{0}".format(ngram, math.log(f, 10)))
                    else:
                        res.append(
                            "{1}\t{0}\t{2}".format(
                                ngram, math.log(f, 10), math.log(bow, 10)
                            )
                        )
        res.sort(reverse=True)
        for r in res:
            print(r)

    def print_as_arpa(self, fout):
        """Print the language model in ARPA format.
        
        Arguments
        ---------
        fout : io.TextIOWrapper
            The file object to write the language model.
        """
        print("\\data\\", file=fout)
        for hist_len in range(self.ngram_order):
            # print the number of n-grams.
            print(
                "ngram {0}={1}".format(
                    hist_len + 1,
                    sum(
                        [
                            len(counts_for_hist.word_to_f)
                            for counts_for_hist in self.counts[hist_len].values()
                        ]
                    ),
                ),
                file=fout,
            )

        print("", file=fout)

        for hist_len in range(self.ngram_order):
            print("\\{0}-grams:".format(hist_len + 1), file=fout)

            this_order_counts = self.counts[hist_len]
            for hist, counts_for_hist in this_order_counts.items():
                for word in counts_for_hist.word_to_count.keys():
                    ngram = hist + (word,)
                    prob = counts_for_hist.word_to_f[word]
                    bow = counts_for_hist.word_to_bow[word]

                    if prob == 0:  # f(<s>) is always 0
                        prob = 1e-99

                    line = "{0}\t{1}".format("%.7f" % math.log10(prob), " ".join(ngram))
                    if bow is not None:
                        line += "\t{0}".format("%.7f" % math.log10(bow))
                    print(line, file=fout)
            print("", file=fout)
        print("\\end\\", file=fout)

    def save_as_arpa(self, out_file, encoding=DEFAULT_ENCODING):
        """Save the language model to file in ARPA format.

        Arguments
        ---------
        out_file : str
            The file to save the language model.
        encoding : str
            The encoding of the file. Default is latin-1.
        """
        with open(out_file, "w", encoding=encoding) as fw:
            # save as ARPA format.
            fw.write("\\data\\\n")
            for hist_len in range(self.ngram_order):
                # print the number of n-grams.
                fw.write(
                    "ngram {0}={1}\n".format(
                        hist_len + 1,
                        sum(
                            [
                                len(counts_for_hist.word_to_f)
                                for counts_for_hist in self.counts[hist_len].values()
                            ]
                        ),
                    ),
                )

            fw.write("\n")

            for hist_len in range(self.ngram_order):
                fw.write("\\{0}-grams:\n".format(hist_len + 1))

                this_order_counts = self.counts[hist_len]
                for hist, counts_for_hist in this_order_counts.items():
                    for word in counts_for_hist.word_to_count.keys():
                        ngram = hist + (word,)
                        prob = counts_for_hist.word_to_f[word]
                        bow = counts_for_hist.word_to_bow[word]

                        if prob == 0:  # f(<s>) is always 0
                            prob = 1e-99

                        line = "{0}\t{1}".format("%.7f" % math.log10(prob), " ".join(ngram))
                        if bow is not None:
                            line += "\t{0}".format("%.7f" % math.log10(bow))
                        fw.write(line + "\n")
                fw.write("\n")
            fw.write("\\end\\\n")


def _make_kn_lm(text, lm: str, ngram_order: int = 3, encoding: str = DEFAULT_ENCODING):
    """Make a Kneser-Ney smoothed language model. The language model is saved in ARPA format.

    Arguments
    ---------
    transcripts_path : Union[str, Iterable[str]]
        The text to train the language model. It can be a file containing 
        text sentences in each line, or a list of strings, where each string
        contains a sequence of integer word-ids.
    out_path : str
        The file to save the language model. If None, the language model 
        will be printed to stdout.
    ngram_order : int
        The order of n-gram.
    encoding : str
        The encoding of the file. Default is latin-1.
    """

    ngram_counts = NgramCounts(ngram_order)

    if text is None:
        ngram_counts.add_raw_counts_from_standard_input()
    elif (isinstance(text, str) or isinstance(text, Path)) and os.path.isfile(text):
        ngram_counts.add_raw_counts_from_file(text)
    elif isinstance(text, Iterable):
        ngram_counts.add_raw_counts_from_list_of_strs(text)
    else:
        raise ValueError("Unsupported type of text: {0}".format(type(text)))

    ngram_counts.cal_discounting_constants()
    ngram_counts.cal_f()
    ngram_counts.cal_bow()

    if lm is None:
        ngram_counts.print_as_arpa(io.TextIOWrapper(sys.stdout.buffer, encoding="latin-1"))
    else:
        logger.info(f"Saving ARPA LM to {lm}.")
        ngram_counts.save_as_arpa(out_file=lm, encoding=encoding)


def create_arpa_lm(
        transcripts_path: str,
        out_path: str,
        ngram_order: int = 3,
        encoding: str = DEFAULT_ENCODING,
    ):
    """Make a Kneser-Ney smoothed language model. The language model is saved in ARPA format.

    Arguments
    ---------
    transcripts_path : str
        The text to train the language model. It must be a file containing 
        text sentences in each line.
    out_path : str
        The file to save the language model.
    ngram_order : int
        The order of n-gram. Default is 3.
    encoding : str
        The encoding of the file. Default is latin-1.
    """

    if not os.path.isfile(transcripts_path):
        raise FileNotFoundError(f"File not found: {transcripts_path} (it must exist for training the LM)")

    if not os.path.isdir(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    _make_kn_lm(transcripts_path, out_path, ngram_order, encoding)
