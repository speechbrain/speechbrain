r"""
Tools for working with ARPA format N-gram models

Expects the ARPA format to have:
- a \data\ header
- counts of ngrams in the order that they are later listed
- line breaks between \data\ and \n-grams: sections
- \end\
E.G.
    ```
    \data\
    ngram 1=2
    ngram 2=1

    \1-grams:
    -1.0000 Hello -0.23
    -0.6990 world -0.2553

    \2-grams:
    -0.2553 Hello world

    \end\
    ```


Example
-------
>>> # This example loads an ARPA model and queries it with BackoffNgramLM
>>> import io
>>> from speechbrain.lm.ngram import BackoffNgramLM
>>> # First we'll put an ARPA format model in TextIO and load it:
>>> with io.StringIO() as f:
...     print("Anything can be here", file=f)
...     print("", file=f)
...     print("\\data\\", file=f)
...     print("ngram 1=2", file=f)
...     print("ngram 2=3", file=f)
...     print("", file=f)  # Ends data section
...     print("\\1-grams:", file=f)
...     print("-0.6931 a", file=f)
...     print("-0.6931 b 0.", file=f)
...     print("", file=f)  # Ends unigram section
...     print("\\2-grams:", file=f)
...     print("-0.6931 a a", file=f)
...     print("-0.6931 a b", file=f)
...     print("-0.6931 b a", file=f)
...     print("", file=f)  # Ends bigram section
...     print("\\end\\", file=f)  # Ends whole file
...     _ = f.seek(0)
...     num_grams, ngrams, backoffs = read_arpa(f)
>>> # The output of read arpa is already formatted right for the query class:
>>> lm = BackoffNgramLM(ngrams, backoffs)
>>> lm.logprob("a", context = tuple())
-0.6931
>>> # Query that requires a backoff:
>>> lm.logprob("b", context = ("b",))
-0.6931

Authors
 * Aku Rouhe 2020
"""
import collections
import tempfile
import logging
import math
import csv
import sys
import os
import re
import io

logger = logging.getLogger(__name__)


def read_arpa(fstream):
    r"""
    Reads an ARPA format N-gram language model from a stream

    Arguments
    ---------
    fstream : TextIO
        Text file stream (as commonly returned by open()) to read the model
        from.

    Returns
    -------
    dict
        Maps N-gram orders to the number ngrams of that order. Essentially the
        \data\ section of an ARPA format file.
    dict
        The log probabilities (first column) in the ARPA file.
        This is a triply nested dict.
        The first layer is indexed by N-gram order (integer).
        The second layer is indexed by the context (tuple of tokens).
        The third layer is indexed by tokens, and maps to the log prob.
        This format is compatible with `speechbrain.lm.ngram.BackoffNGramLM`
        Example:
        In ARPA format, log(P(fox|a quick red)) = -5.3 is expressed:
            `-5.3 a quick red fox`
        And to access that probability, use:
            `ngrams_by_order[4][('a', 'quick', 'red')]['fox']`
    dict
        The log backoff weights (last column) in the ARPA file.
        This is a doubly nested dict.
        The first layer is indexed by N-gram order (integer).
        The second layer is indexed by the backoff history (tuple of tokens)
        i.e. the context on which the probability distribution is conditioned
        on. This maps to the log weights.
        This format is compatible with `speechbrain.lm.ngram.BackoffNGramLM`
        Example:
        If log(P(fox|a quick red)) is not listed, we find
        log(backoff(a quick red)) = -23.4 which in ARPA format is:
            `<logp> a quick red -23.4`
        And to access that here, use:
            `backoffs_by_order[3][('a', 'quick', 'red')]`

    Raises
    ------
    ValueError
        If no LM is found or the file is badly formatted.
    """
    # Developer's note:
    # This is a long function.
    # It is because we support cases where a new section starts suddenly without
    # an empty line in between.
    #
    # \data\ section:
    _find_data_section(fstream)
    num_ngrams = {}
    for line in fstream:
        line = line.strip()
        if line[:5] == "ngram":
            lhs, rhs = line.split("=")
            order = int(lhs.split()[1])
            num_grams = int(rhs)
            num_ngrams[order] = num_grams
        elif not line:  # Normal case, empty line ends section
            ended, order = _next_section_or_end(fstream)
            break  # Good, proceed to next section
        elif _starts_ngrams_section(line):  # No empty line between sections
            ended = False
            order = _parse_order(line)
            break  # Good, proceed to next section
        else:
            raise ValueError("Not a properly formatted line")
    # At this point:
    # ended == False
    # type(order) == int
    #
    # \N-grams: sections
    # NOTE: This is the section that most time is spent on, so it's been written
    # with processing speed in mind.
    ngrams_by_order = {}
    backoffs_by_order = {}
    while not ended:
        probs = collections.defaultdict(dict)
        backoffs = {}
        backoff_line_length = order + 2
        # Use try-except because it is faster than always checking
        try:
            for line in fstream:
                line = line.strip()
                all_parts = tuple(line.split())
                prob = float(all_parts[0])
                if len(all_parts) == backoff_line_length:
                    context = all_parts[1:-2]
                    token = all_parts[-2]
                    backoff = float(all_parts[-1])
                    backoff_context = context + (token,)
                    backoffs[backoff_context] = backoff
                else:
                    context = all_parts[1:-1]
                    token = all_parts[-1]
                probs[context][token] = prob
        except (IndexError, ValueError):
            ngrams_by_order[order] = probs
            backoffs_by_order[order] = backoffs
            if not line:  # Normal case, empty line ends section
                ended, order = _next_section_or_end(fstream)
            elif _starts_ngrams_section(line):  # No empty line between sections
                ended = False
                order = _parse_order(line)
            elif _ends_arpa(line):  # No empty line before End of file
                ended = True
                order = None
            else:
                raise ValueError("Not a properly formatted ARPA file")
    # Got to the \end\. Still have to check whether all promised sections were
    # delivered.
    if not num_ngrams.keys() == ngrams_by_order.keys():
        raise ValueError("Not a properly formatted ARPA file")
    return num_ngrams, ngrams_by_order, backoffs_by_order


def _find_data_section(fstream):
    r"""
    Reads (lines) from the stream until the \data\ header is found.
    """
    for line in fstream:
        if line[:6] == "\\data\\":
            return
    # If we get here, no data header found
    raise ValueError("Not a properly formatted ARPA file")


def _next_section_or_end(fstream):
    """
    Returns
    -------
    bool
        Whether end was found.
    int
        The order of section that starts
    """
    for line in fstream:
        line = line.strip()
        if _starts_ngrams_section(line):
            order = _parse_order(line)
            return False, order
        if _ends_arpa(line):
            return True, None
    # If we got here, it's not a properly formatted file
    raise ValueError("Not a properly formatted ARPA file")


def _starts_ngrams_section(line):
    return line.strip().endswith("-grams:")


def _parse_order(line):
    order = int(line[1:].split("-")[0])
    return order


def _ends_arpa(line):
    return line == "\\end\\"


# The flowing code is copied from k2-fsa/icefall and is an implementation of
# computing Kneser-Ney smoothed language model in the same way as srilm. This
# is a back-off, unmodified version of Kneser-Ney smoothing, which produces the
# same results as the following command (as an example) of srilm:
#
# $ ngram-count -order 4 -kn-modify-counts-at-end -ukndiscount -gt1min 0 -gt2min 0 -gt3min 0 -gt4min 0 \
# -text corpus.txt -lm lm.arpa
#
# The data structure is based on: kaldi/egs/wsj/s5/utils/lang/make_phone_lm.py
# The smoothing algorithm is based on: http://www.speech.sri.com/projects/srilm/manpages/ngram-discount.7.html

# For encoding-agnostic scripts, we assume byte stream as input.
# Need to be very careful about the use of strip() and split()
# in this case, because there is a latin-1 whitespace character
# (nbsp) which is part of the unicode encoding range.
# Ref: kaldi/egs/wsj/s5/utils/lang/bpe/prepend_words.py @ 69cd717
default_encoding = "latin-1"

strip_chars = " \t\r\n"
whitespace = re.compile("[ \t]+")


class CountsForHistory:
    # This class (which is more like a struct) stores the counts seen in a
    # particular history-state.  It is used inside class NgramCounts.
    # It really does the job of a dict from int to float, but it also
    # keeps track of the total count.
    def __init__(self):
        # The 'lambda: defaultdict(float)' is an anonymous function taking no
        # arguments that returns a new defaultdict(float).
        self.word_to_count = collections.defaultdict(int)
        # using a set to count the number of unique contexts
        self.word_to_context = collections.defaultdict(set)
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
    # A note on data-structure.  Firstly, all words are represented as
    # integers.  We store n-gram counts as an array, indexed by (history-length
    # == n-gram order minus one) (note: python calls arrays "lists") of dicts
    # from histories to counts, where histories are arrays of integers and
    # "counts" are dicts from integer to float.  For instance, when
    # accumulating the 4-gram count for the '8' in the sequence '5 6 7 8', we'd
    # do as follows: self.counts[3][[5,6,7]][8] += 1.0 where the [3] indexes an
    # array, the [[5,6,7]] indexes a dict, and the [8] indexes a dict.
    def __init__(self, ngram_order, bos_symbol="<s>", eos_symbol="</s>"):
        assert ngram_order >= 2

        self.ngram_order = ngram_order
        self.bos_symbol = bos_symbol
        self.eos_symbol = eos_symbol

        self.counts = []
        for n in range(ngram_order):
            self.counts.append(
                collections.defaultdict(lambda: CountsForHistory())
            )

        self.d = []  # list of discounting factor for each order of ngram

    # adds a raw count (called while processing input data).
    # Suppose we see the sequence '6 7 8 9' and ngram_order=4, 'history'
    # would be (6,7,8) and 'predicted_word' would be 9; 'count' would be
    # 1.
    def add_count(self, history, predicted_word, context_word, count):
        self.counts[len(history)][history].add_count(
            predicted_word, context_word, count
        )

    # 'line' is a string containing a sequence of integer word-ids.
    # This function adds the un-smoothed counts from this line of text.
    def add_raw_counts_from_line(self, line):
        if line == "":
            words = [self.bos_symbol, self.eos_symbol]
        else:
            words = (
                [self.bos_symbol] + whitespace.split(line) + [self.eos_symbol]
            )

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
        lines_processed = 0
        # byte stream as input
        infile = io.TextIOWrapper(sys.stdin.buffer, encoding=default_encoding)
        for line in infile:
            line = line.strip(strip_chars)
            self.add_raw_counts_from_line(line)
            lines_processed += 1

    def add_raw_counts_from_file(self, filename):
        lines_processed = 0
        with open(filename, encoding=default_encoding) as fp:
            for line in fp:
                line = line.strip(strip_chars)
                self.add_raw_counts_from_line(line)
                lines_processed += 1

    def cal_discounting_constants(self):
        # For each order N of N-grams, we calculate discounting constant D_N = n1_N / (n1_N + 2 * n2_N),
        # where n1_N is the number of unique N-grams with count = 1 (counts-of-counts).
        # This constant is used similarly to absolute discounting.
        # Return value: d is a list of floats, where d[N+1] = D_N

        # for the lowest order, i.e., 1-gram, we do not need to discount, thus the constant is 0
        # This is a special case: as we currently assumed having seen all vocabularies in the dictionary,
        # but perhaps this is not the case for some other scenarios.
        self.d = [0]
        for n in range(1, self.ngram_order):
            this_order_counts = self.counts[n]
            n1 = 0
            n2 = 0
            for hist, counts_for_hist in this_order_counts.items():
                stat = collections.Counter(
                    counts_for_hist.word_to_count.values()
                )
                n1 += stat[1]
                n2 += stat[2]
            assert n1 + 2 * n2 > 0

            # We are doing this max(0.001, xxx) to avoid zero discounting constant D due to n1=0,
            # which could happen if the number of symbols is small.
            # Otherwise, zero discounting constant can cause division by zero in computing BOW.
            self.d.append(max(0.1, n1 * 1.0) / (n1 + 2 * n2))

    def cal_f(self):
        # f(a_z) is a probability distribution of word sequence a_z.
        # Typically f(a_z) is discounted to be less than the ML estimate so we have
        # some leftover probability for the z words unseen in the context (a_).
        #
        # f(a_z) = (c(a_z) - D0) / c(a_)    ;; for highest order N-grams
        # f(_z)  = (n(*_z) - D1) / n(*_*)	;; for lower order N-grams

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

    def cal_bow(self):
        # Backoff weights are only necessary for ngrams which form a prefix of a longer ngram.
        # Thus, two sorts of ngrams do not have a bow:
        # 1) highest order ngram
        # 2) ngrams ending in </s>
        #
        # bow(a_) = (1 - Sum_Z1 f(a_z)) / (1 - Sum_Z1 f(_z))
        # Note that Z1 is the set of all words with c(a_z) > 0

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
                            counts_for_hist.word_to_bow[w] = (
                                1.0 - sum_z1_f_a_z
                            ) / (1.0 - sum_z1_f_z)
                        else:
                            counts_for_hist.word_to_bow[w] = None

    def print_raw_counts(self, info_string):
        # these are useful for debug.
        print(info_string)
        res = []
        for this_order_counts in self.counts:
            for hist, counts_for_hist in this_order_counts.items():
                for w in counts_for_hist.word_to_count.keys():
                    ngram = " ".join(hist) + " " + w
                    ngram = ngram.strip(strip_chars)

                    res.append(
                        "{0}\t{1}".format(
                            ngram, counts_for_hist.word_to_count[w]
                        )
                    )
        res.sort(reverse=True)
        for r in res:
            print(r)

    def print_modified_counts(self, info_string):
        # these are useful for debug.
        print(info_string)
        res = []
        for this_order_counts in self.counts:
            for hist, counts_for_hist in this_order_counts.items():
                for w in counts_for_hist.word_to_count.keys():
                    ngram = " ".join(hist) + " " + w
                    ngram = ngram.strip(strip_chars)

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
        # these are useful for debug.
        print(info_string)
        res = []
        for this_order_counts in self.counts:
            for hist, counts_for_hist in this_order_counts.items():
                for w in counts_for_hist.word_to_count.keys():
                    ngram = " ".join(hist) + " " + w
                    ngram = ngram.strip(strip_chars)

                    f = counts_for_hist.word_to_f[w]
                    if f == 0:  # f(<s>) is always 0
                        f = 1e-99

                    res.append("{0}\t{1}".format(ngram, math.log(f, 10)))
        res.sort(reverse=True)
        for r in res:
            print(r)

    def print_f_and_bow(self, info_string):
        # these are useful for debug.
        print(info_string)
        res = []
        for this_order_counts in self.counts:
            for hist, counts_for_hist in this_order_counts.items():
                for w in counts_for_hist.word_to_count.keys():
                    ngram = " ".join(hist) + " " + w
                    ngram = ngram.strip(strip_chars)

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

    def print_as_arpa(
        self, fout=io.TextIOWrapper(sys.stdout.buffer, encoding="latin-1")
    ):
        # print as ARPA format.

        print("\\data\\", file=fout)
        for hist_len in range(self.ngram_order):
            # print the number of n-grams.
            print(
                "ngram {0}={1}".format(
                    hist_len + 1,
                    sum(
                        [
                            len(counts_for_hist.word_to_f)
                            for counts_for_hist in self.counts[
                                hist_len
                            ].values()
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

                    line = "{0}\t{1}".format(
                        "%.7f" % math.log10(prob), " ".join(ngram)
                    )
                    if bow is not None:
                        line += "\t{0}".format("%.7f" % math.log10(bow))
                    print(line, file=fout)
            print("", file=fout)
        print("\\end\\", file=fout)


def make(
    input_text_file, output_arpa_dir, ngram_order=3, prefix_name="", cache=True
):
    """
    Generate kneser-ney language model as arpa format.
    Read the corpus from `input_text_file`, and output to `output_arpa_dir`.

    Arguments
    ---------
    input_text_file : Path
        The text corpus line by line
    output_arpa_dir : Path
        Path to output arpa file for language models
        format ${output_arpa_dir}/${ngram_order}gram.arpa
    ngram_order: int
        Order of n-gram (default 3)
    """
    ngram_counts = NgramCounts(ngram_order)
    assert os.path.isfile(input_text_file)
    assert os.path.isdir(output_arpa_dir)
    out = os.path.join(output_arpa_dir, f"{prefix_name}{ngram_order}gram.arpa")
    if cache and os.path.exists(out):
        logger.critical(
            f"Ingoring '{out}' creation as the file already exists, "
            "Consider deleting the previous .pt file if this is not "
            "what you want."
        )
        return

    logger.info(f"Making '{out}' kneser-ney language model.")
    ngram_counts.add_raw_counts_from_file(input_text_file)
    ngram_counts.cal_discounting_constants()
    ngram_counts.cal_f()
    ngram_counts.cal_bow()
    with open(out, "w", encoding=default_encoding) as f:
        ngram_counts.print_as_arpa(fout=f)


def make_from_csv(
    input_csv_file,
    output_arpa_dir,
    column_name="wrd",
    ngram_order=3,
    prefix_name="",
    cache=True,
):
    """
    Generate kneser-ney language model as arpa format.
    Read the csv corpus from `input_csv_file`, and output to `output_arpa_dir`.
    Internaly cass `make`.

    Arguments
    ---------
    input_csv_file : Path
        The csv corpus line by line
    output_arpa_dir : Path
        Path to output arpa file for language models
        format ${output_arpa_dir}/${ngram_order}gram.arpa
    column_name : str
        The csv column name to get text from
    ngram_order: int
        Order of n-gram (default 3)
    """
    temp_dir = tempfile.gettempdir()
    output_file = tempfile.NamedTemporaryFile(
        delete=False, dir=temp_dir, mode="w"
    )

    with open(input_csv_file, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        # Check if the specified column name exists in the header
        if column_name not in reader.fieldnames:
            logger.critical(
                f"Column '{column_name}' not found in the CSV {input_csv_file} file."
            )
        else:
            for row in reader:
                text = row[column_name]
                output_file.write(text + "\n")

    make(
        output_file.name,
        output_arpa_dir,
        ngram_order=ngram_order,
        prefix_name=prefix_name,
        cache=cache,
    )

    output_file.close()
    os.remove(output_file.name)
