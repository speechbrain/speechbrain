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

Author
------
Aku Rouhe 2020
"""
import collections
import logging

logger = logging.getLogger(__name__)


def read_arpa(fin):
    r"""
    Reads an ARPA format N-gram language model from a stream

    Note that this returns a tuple of three dictno end tag found.

    Arguments
    ---------
    fin : TextIO
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
    find_data_section(fin)
    num_ngrams = read_num_ngrams(fin)
    ngrams_by_order = {}
    backoffs_by_order = {}
    for order in num_ngrams:
        logger.debug(f"Reading {order}-grams")
        try:
            probs, backoffs = read_ngrams_section(fin, order)
        except IndexError:
            raise ValueError("Not a properly formatted ARPA file")
        ngrams_by_order[order] = probs
        backoffs_by_order[order] = backoffs
    read_end(fin)
    return num_ngrams, ngrams_by_order, backoffs_by_order


def find_data_section(fstream):
    r"""
    Reads (lines) from the stream until the \data\ header is found.
    """
    for line in fstream:
        if line[:6] == "\\data\\":
            return
    # If we get here, no data header found
    raise ValueError("Not a properly formatted ARPA file")


def read_num_ngrams(fstream):
    r"""
    Reads the ARPA \data\ section from the stream.

    Assumes stream is at \data\ section.

    Arguments
    ---------
    fin : TextIO
        Text file stream (as commonly returned by open()) to read the model
        from.

    Returns
    -------
    num_ngrams : dict
        Maps N-gram orders to the number ngrams of that order. Essentially the
        \data\ section of an ARPA format file.
    """
    num_ngrams = {}
    for line in fstream:
        if line[:5] == "ngram":
            lhs, rhs = line.strip().split("=")
            order = int(lhs.split()[1])
            num_grams = int(rhs)
            num_ngrams[order] = num_grams
        else:
            break
    if not num_ngrams:
        raise ValueError("Empty ARPA file")
    return num_ngrams


def read_ngrams_section(fstream, order):
    r"""
    Reads one ARPA \N-grams: section (of order N)

    Assumes fstream is at section header.

    Arguments
    ---------
    fin : TextIO
        Text file stream (as commonly returned by open()) to read the model
        from.
    order : int
        The N

    Returns
    -------
    dict
        The log probabilities (first column) in the section.
        This is a doubly nested dict.
        The first layer is indexed by the context (tuple of tokens).
        The second layer is indexed by tokens, and maps to the log prob.
    dict
        The log backoff weights (last column) in the section.
        The dict is ndexed by the backoff history (tuple of tokens)
        i.e. the context on which the probability distribution is conditioned
        on. This maps to the log weights.

    NOTE
    ----
    This function is somewhat optimized in Python. Numerous experiments with
    attempts to parallelize reading yielded no improvement. The reading doesn't
    seem to be IO bound (at least on an SSD), and the Python reading code is
    still relatively slow.
    """
    section_header = fstream.readline()
    if not section_header.startswith(f"\\{order}-grams:"):
        raise ValueError("Not a properly formatted ARPA file")
    probs = collections.defaultdict(dict)
    backoffs = {}
    backoff_line_length = order + 2
    for line in fstream:
        line = line.strip()
        if not line:
            break
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
    return dict(probs), backoffs


def read_end(fstream):
    r"""
    Reads (lines) from the stream until the \end\ tag is found.
    """
    for line in fstream:
        if line[:5] == "\\end\\":
            return
    # If we get here, no end tag found.
    raise ValueError("Not a properly formatted ARPA file")
