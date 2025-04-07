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
 * Pierre Champion 2023
"""

import collections
from pathlib import Path
from typing import Union

from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


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
    Arguments
    ---------
    fstream : stream
        Stream from which to read lines

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


def arpa_to_fst(
    words_txt: Union[str, Path],
    in_arpa: Union[str, Path],
    out_fst: Union[str, Path],
    ngram_order: int,
    disambig_symbol: str = "#0",
    cache: bool = True,
):
    r"""
    Use kaldilm to convert an ARPA LM to FST. For example, you could use
    speechbrain.lm.train_ngram to create an ARPA LM and then use this function
    to convert it to an FST.

    It is worth noting that if the fst already exists in the output_dir,
    then they will not be converted again (so you may need to delete them
    by hand if you, at any point, change your ARPA model).

    Arguments
    ---------
    words_txt: str | Path
        path to the words.txt file created by prepare_lang.
    in_arpa: str | Path
        Path to an ARPA LM to convert to an FST.
    out_fst: str | Path
        Path to where the fst will be saved.
    ngram_order: int
        ARPA (and FST) ngram order.
    disambig_symbol: str
        the disambiguation symbol to use.
    cache: bool
        Whether or not to re-create the fst.txt file if it already exist.

    Raises
    ------
    ImportError: If kaldilm is not installed.

    Returns
    -------
    None

    Example
    -------
    >>> from speechbrain.lm.arpa import arpa_to_fst

    >>> # Create a small arpa model
    >>> arpa_file = getfixture('tmpdir').join("bigram.arpa")
    >>> arpa_file.write(
    ...     "Anything can be here\n"
    ...     + "\n"
    ...     + "\\data\\\n"
    ...     + "ngram 1=3\n"
    ...     + "ngram 2=4\n"
    ...     + "\n"
    ...     + "\\1-grams:\n"
    ...     + "0 <s>\n"
    ...     + "-0.6931 a\n"
    ...     + "-0.6931 b 0.\n"
    ...     + "" # Ends unigram section
    ...     + "\\2-grams:\n"
    ...     + "-0.6931 <s> a\n"
    ...     + "-0.6931 a a\n"
    ...     + "-0.6931 a b\n"
    ...     + "-0.6931 b a\n"
    ...     + "\n"  # Ends bigram section
    ...     + "\\end\\\n")  # Ends whole file
    >>> # Create words vocab
    >>> vocav = getfixture('tmpdir').join("words.txt")
    >>> vocav.write(
    ...     "a 1\n"
    ...     + "b 2\n"
    ...     + "<s> 3\n"
    ...     + "#0 4")  # Ends whole file
    >>> out = getfixture('tmpdir').join("bigram.txt.fst")
    >>> arpa_to_fst(vocav, arpa_file, out, 2)
    """
    try:
        from kaldilm.arpa2fst import arpa2fst
    except ImportError:
        # This error will occur when there is fst LM in the provided lm_dir
        # and we are trying to create it by converting an ARPA LM to FST.
        # For this, we need to install kaldilm.
        raise ImportError(
            "Optional dependencies must be installed to use kaldilm.\n"
            "Install using `pip install kaldilm`."
        )

    if cache and out_fst.exists():
        return
    if not in_arpa.exists():
        raise FileNotFoundError(
            f"{in_arpa} not found while trying to create"
            f" the {ngram_order} FST."
        )
    try:
        logger.info(f"Converting arpa LM '{in_arpa}' to FST")
        s = arpa2fst(
            input_arpa=str(in_arpa),
            disambig_symbol=disambig_symbol,
            read_symbol_table=str(words_txt),
            max_order=ngram_order,
        )
    except Exception as e:
        logger.info(
            f"Failed to create {ngram_order}-gram FST from input={in_arpa}"
            f", disambig_symbol={disambig_symbol},"
            f" read_symbol_table={words_txt}"
        )
        raise e
    logger.info(f"Writing {out_fst}")
    with open(out_fst, "w", encoding="utf-8") as f:
        f.write(s)
