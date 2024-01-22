#!/usr/bin/env python3
""" This module contains functions to prepare the lexicon and the language model
for k2 training. It is based on the script `prepare_lang.sh` from k2/icefall (work
of Fangjun Kuang). The original script is under Apache 2.0 license.
This script is modified to work with SpeechBrain.

Modified by:
  * Pierre Champion 2023
  * Zeyu Zhao 2023
  * Georgios Karakasidis 2023
"""


from . import k2  # import k2 from ./__init__.py
from .lexicon import read_lexicon, write_lexicon, EPS
import math
import os
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import torch

logger = logging.getLogger(__name__)

Lexicon = List[Tuple[str, List[str]]]


def write_mapping(filename: Union[str, Path], sym2id: Dict[str, int]) -> None:
    """
    Write a symbol to ID mapping to a file.

    NOTE: No need to implement `read_mapping` as it can be done through
      :func:`k2.SymbolTable.from_file`.

    Arguments
    ---------
    filename: str
        Filename to save the mapping.
    sym2id: Dict[str, int]
        A dict mapping symbols to IDs.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for sym, i in sym2id.items():
            f.write(f"{sym} {i}\n")


def get_tokens(
    lexicon: Lexicon, sil_token="SIL", manually_add_sil_to_tokens=False
) -> List[str]:
    """
    Get tokens from a lexicon.

    Arguments
    ---------
    lexicon: Lexicon
        It is the return value of :func:`read_lexicon`.
    sil_token: str
        The optional silence token between words. It should not appear in the lexicon,
        otherwise it will cause an error.
    manually_add_sil_to_tokens: bool
        If true, add `sil_token` to the tokens. This is useful when the lexicon
        does not contain `sil_token` but it is needed in the tokens.

    Returns
    -------
    sorted_ans: List[str]
        A list of unique tokens.
    """
    ans = set()
    if manually_add_sil_to_tokens:
        ans.add(sil_token)
    for _, tokens in lexicon:
        assert (
            sil_token not in tokens
        ), f"{sil_token} should not appear in the lexicon but it is found in {_}"
        ans.update(tokens)
    sorted_ans = sorted(list(ans))
    return sorted_ans


def get_words(lexicon: Lexicon) -> List[str]:
    """
    Get words from a lexicon.

    Arguments
    ---------
    lexicon: Lexicon
        It is the return value of :func:`read_lexicon`.

    Returns
    -------
    sorted_ans:
        Return a list of unique words.
    """
    ans = set()
    for word, _ in lexicon:
        ans.add(word)
    sorted_ans = sorted(list(ans))
    return sorted_ans


def add_disambig_symbols(lexicon: Lexicon) -> Tuple[Lexicon, int]:
    """
    It adds pseudo-token disambiguation symbols #1, #2 and so on
    at the ends of tokens to ensure that all pronunciations are different,
    and that none is a prefix of another.

    See also add_lex_disambig.pl from kaldi.

    Arguments
    ---------
    lexicon: Lexicon
        It is returned by :func:`read_lexicon`.

    Returns
    -------
    ans:
        The output lexicon with disambiguation symbols
    max_disambig:
        The ID of the max disambiguation symbol that appears
        in the lexicon
    """

    # (1) Work out the count of each token-sequence in the
    # lexicon.
    count = defaultdict(int)
    for _, tokens in lexicon:
        count[" ".join(tokens)] += 1

    # (2) For each left sub-sequence of each token-sequence, note down
    # that it exists (for identifying prefixes of longer strings).
    issubseq = defaultdict(int)
    for _, tokens in lexicon:
        tokens = tokens.copy()
        tokens.pop()
        while tokens:
            issubseq[" ".join(tokens)] = 1
            tokens.pop()

    # (3) For each entry in the lexicon:
    # if the token sequence is unique and is not a
    # prefix of another word, no disambig symbol.
    # Else output #1, or #2, #3, ... if the same token-seq
    # has already been assigned a disambig symbol.
    ans = []

    # We start with #1 since #0 has its own purpose
    first_allowed_disambig = 1
    max_disambig = first_allowed_disambig - 1
    last_used_disambig_symbol_of = defaultdict(int)

    for word, tokens in lexicon:
        tokenseq = " ".join(tokens)
        assert tokenseq != ""
        if issubseq[tokenseq] == 0 and count[tokenseq] == 1:
            ans.append((word, tokens))
            continue

        cur_disambig = last_used_disambig_symbol_of[tokenseq]
        if cur_disambig == 0:
            cur_disambig = first_allowed_disambig
        else:
            cur_disambig += 1

        if cur_disambig > max_disambig:
            max_disambig = cur_disambig
        last_used_disambig_symbol_of[tokenseq] = cur_disambig
        tokenseq += f" #{cur_disambig}"
        ans.append((word, tokenseq.split()))
    return ans, max_disambig


def generate_id_map(symbols: List[str]) -> Dict[str, int]:
    """
    Generate ID maps, i.e., map a symbol to a unique ID.

    Arguments
    ---------
    symbols: List[str]
        A list of unique symbols.

    Returns
    -------
    A dict containing the mapping between symbols and IDs.
    """
    return {sym: i for i, sym in enumerate(symbols)}


def add_self_loops(
    arcs: List[List[Any]], disambig_token: int, disambig_word: int
) -> List[List[Any]]:
    """
    Adds self-loops to states of an FST to propagate disambiguation symbols
    through it. They are added on each state with non-epsilon output symbols
    on at least one arc out of the state.

    See also fstaddselfloops.pl from Kaldi. One difference is that
    Kaldi uses OpenFst style FSTs and it has multiple final states.
    This function uses k2 style FSTs and it does not need to add self-loops
    to the final state.

    The input label of a self-loop is `disambig_token`, while the output
    label is `disambig_word`.

    Arguments
    ---------
    arcs: List[List[Any]]
        A list-of-list. The sublist contains
        `[src_state, dest_state, label, aux_label, score]`
    disambig_token: int
        It is the token ID of the symbol `#0`.
    disambig_word: int
        It is the word ID of the symbol `#0`.

    Returns
    -------
    Return new `arcs` containing self-loops.
    """
    states_needs_self_loops = set()
    for arc in arcs:
        src, dst, ilabel, olabel, score = arc
        if olabel != 0:
            states_needs_self_loops.add(src)

    ans = []
    for s in states_needs_self_loops:
        ans.append([s, s, disambig_token, disambig_word, 0])

    return arcs + ans


def lexicon_to_fst(
    lexicon: Lexicon,
    token2id: Dict[str, int],
    word2id: Dict[str, int],
    sil_token: str = "SIL",
    sil_prob: float = 0.5,
    need_self_loops: bool = False,
) -> k2.Fsa:
    """
    Convert a lexicon to an FST (in k2 format) with optional silence at the
    beginning and end of each word.

    Arguments
    ---------
    lexicon: Lexicon
        The input lexicon. See also :func:`read_lexicon`
    token2id: Dict[str, int]
        A dict mapping tokens to IDs.
    word2id: Dict[str, int]
        A dict mapping words to IDs.
    sil_token: str
        The silence token.
    sil_prob: float
        The probability for adding a silence at the beginning and end
        of the word.
    need_self_loops: bool
        If True, add self-loop to states with non-epsilon output symbols
        on at least one arc out of the state. The input label for this
        self loop is `token2id["#0"]` and the output label is `word2id["#0"]`.

    Returns
    -------
    fsa: k2.Fsa
        An FSA representing the given lexicon.
    """
    assert sil_prob > 0.0 and sil_prob < 1.0
    # CAUTION: we use score, i.e, negative cost.
    sil_score = math.log(sil_prob)
    no_sil_score = math.log(1.0 - sil_prob)

    start_state = 0
    loop_state = 1  # words enter and leave from here
    sil_state = 2  # words terminate here when followed by silence; this state
    # has a silence transition to loop_state.
    next_state = 3  # the next un-allocated state, will be incremented as we go.
    arcs = []

    assert token2id[EPS] == 0
    assert word2id[EPS] == 0

    eps = 0

    sil_token_id = token2id[sil_token]

    arcs.append([start_state, loop_state, eps, eps, no_sil_score])
    arcs.append([start_state, sil_state, eps, eps, sil_score])
    arcs.append([sil_state, loop_state, sil_token_id, eps, 0])

    for word, tokens in lexicon:
        assert len(tokens) > 0, f"{word} has no pronunciations"
        cur_state = loop_state

        word = word2id[word]
        tokens = [token2id[i] for i in tokens]

        for i in range(len(tokens) - 1):
            w = word if i == 0 else eps
            arcs.append([cur_state, next_state, tokens[i], w, 0])

            cur_state = next_state
            next_state += 1

        # now for the last token of this word
        # It has two out-going arcs, one to the loop state,
        # the other one to the sil_state.
        i = len(tokens) - 1
        w = word if i == 0 else eps
        arcs.append([cur_state, loop_state, tokens[i], w, no_sil_score])
        arcs.append([cur_state, sil_state, tokens[i], w, sil_score])

    if need_self_loops:
        disambig_token = token2id["#0"]
        disambig_word = word2id["#0"]
        arcs = add_self_loops(
            arcs, disambig_token=disambig_token, disambig_word=disambig_word,
        )

    final_state = next_state
    arcs.append([loop_state, final_state, -1, -1, 0])
    arcs.append([final_state])

    arcs = sorted(arcs, key=lambda arc: arc[0])
    arcs = [[str(i) for i in arc] for arc in arcs]
    arcs = [" ".join(arc) for arc in arcs]
    arcs = "\n".join(arcs)

    fsa = k2.Fsa.from_str(arcs, acceptor=False)
    return fsa


def lexicon_to_fst_no_sil(
    lexicon: Lexicon,
    token2id: Dict[str, int],
    word2id: Dict[str, int],
    need_self_loops: bool = False,
) -> k2.Fsa:
    """
    Convert a lexicon to an FST (in k2 format).

    Arguments
    ---------
    lexicon: Lexicon
        The input lexicon. See also :func:`read_lexicon`
    token2id: Dict[str, int]
        A dict mapping tokens to IDs.
    word2id: Dict[str, int]
        A dict mapping words to IDs.
    need_self_loops: bool
        If True, add self-loop to states with non-epsilon output symbols
        on at least one arc out of the state. The input label for this
        self loop is `token2id["#0"]` and the output label is `word2id["#0"]`.

    Returns
    -------
    fsa: k2.Fsa
        An FSA representing the given lexicon.
    """
    loop_state = 0  # words enter and leave from here
    next_state = 1  # the next un-allocated state, will be incremented as we go

    arcs = []

    assert token2id[EPS] == 0
    assert word2id[EPS] == 0

    eps = 0

    for word, pieces in lexicon:
        assert len(pieces) > 0, f"{word} has no pronunciations"
        cur_state = loop_state

        word = word2id[word]
        pieces = [token2id[i] for i in pieces]

        for i in range(len(pieces) - 1):
            w = word if i == 0 else eps
            arcs.append([cur_state, next_state, pieces[i], w, 0])

            cur_state = next_state
            next_state += 1

        # now for the last piece of this word
        i = len(pieces) - 1
        w = word if i == 0 else eps
        arcs.append([cur_state, loop_state, pieces[i], w, 0])

    if need_self_loops:
        disambig_token = token2id["#0"]
        disambig_word = word2id["#0"]
        arcs = add_self_loops(
            arcs, disambig_token=disambig_token, disambig_word=disambig_word,
        )

    final_state = next_state
    arcs.append([loop_state, final_state, -1, -1, 0])
    arcs.append([final_state])

    arcs = sorted(arcs, key=lambda arc: arc[0])
    arcs = [[str(i) for i in arc] for arc in arcs]
    arcs = [" ".join(arc) for arc in arcs]
    arcs = "\n".join(arcs)

    fsa = k2.Fsa.from_str(arcs, acceptor=False)
    return fsa


def prepare_lang(lang_dir, sil_token="SIL", sil_prob=0.5, cache=True):
    """
    This function takes as input a lexicon file "$lang_dir/lexicon.txt"
    consisting of words and tokens (i.e., phones) and does the following:

    1. Add disambiguation symbols to the lexicon and generate lexicon_disambig.txt

    2. Generate tokens.txt, the token table mapping a token to a unique integer.

    3. Generate words.txt, the word table mapping a word to a unique integer.

    4. Generate L.pt, in k2 format. It can be loaded by

            d = torch.load("L.pt")
            lexicon = k2.Fsa.from_dict(d)

    5. Generate L_disambig.pt, in k2 format.


    Arguments
    ---------
    lang_dir: str
        The directory to store the output files and read the input file lexicon.txt.
    sil_token: str
        The silence token. Default is "SIL".
    sil_prob: float
        The probability for adding a silence at the beginning and end of the word.
        Default is 0.5.
    cache: bool
        Whether or not to load/cache from/to the .pt format.

    Example
    -------
    >>> from speechbrain.k2_integration.prepare_lang import prepare_lang

    >>> # Create a small lexicon containing only two words and write it to a file.
    >>> lang_tmpdir = getfixture('tmpdir')
    >>> lexicon_sample = '''hello h e l l o\\nworld w o r l d'''
    >>> lexicon_file = lang_tmpdir.join("lexicon.txt")
    >>> lexicon_file.write(lexicon_sample)

    >>> prepare_lang(lang_tmpdir)
    >>> for expected_file in ["tokens.txt", "words.txt", "L.pt", "L_disambig.pt", "Linv.pt" ]:
    ...     assert os.path.exists(os.path.join(lang_tmpdir, expected_file))
    """

    out_dir = Path(lang_dir)
    lexicon_filename = out_dir / "lexicon.txt"

    # if source lexicon_filename has been re-created (only use 'Linv.pt' for date modification query)
    if (
        cache
        and (out_dir / "Linv.pt").exists()
        and (out_dir / "Linv.pt").stat().st_mtime
        < lexicon_filename.stat().st_mtime
    ):
        logger.warning(
            f"Skipping lang preparation of '{out_dir}'."
            " Set 'caching: False' in the yaml"
            " if this is not what you want."
        )
        return

    # backup L.pt, L_disambig.pt, tokens.txt and words.txt, Linv.pt and lexicon_disambig.txt
    for f in [
        "L.pt",
        "L_disambig.pt",
        "tokens.txt",
        "words.txt",
        "Linv.pt",
        "lexicon_disambig.txt",
    ]:
        if (out_dir / f).exists():
            os.makedirs(out_dir / "backup", exist_ok=True)
            logger.debug(f"Backing up {out_dir / f} to {out_dir}/backup/{f}")
            os.rename(out_dir / f, out_dir / "backup" / f)

    lexicon = read_lexicon(str(lexicon_filename))
    if sil_prob != 0:
        # add silence to the tokens
        tokens = get_tokens(
            lexicon, sil_token=sil_token, manually_add_sil_to_tokens=True
        )
    else:
        tokens = get_tokens(lexicon, manually_add_sil_to_tokens=False)
    words = get_words(lexicon)

    lexicon_disambig, max_disambig = add_disambig_symbols(lexicon)

    for i in range(max_disambig + 1):
        disambig = f"#{i}"
        assert disambig not in tokens
        tokens.append(f"#{i}")

    assert EPS not in tokens
    tokens = [EPS] + tokens

    assert EPS not in words
    assert "#0" not in words
    assert "<s>" not in words
    assert "</s>" not in words

    words = [EPS] + words + ["#0", "<s>", "</s>"]

    token2id = generate_id_map(tokens)
    word2id = generate_id_map(words)

    logger.info(
        f"Saving tokens.txt, words.txt, lexicon_disambig.txt to '{out_dir}'"
    )
    write_mapping(out_dir / "tokens.txt", token2id)
    write_mapping(out_dir / "words.txt", word2id)
    write_lexicon(out_dir / "lexicon_disambig.txt", lexicon_disambig)

    if sil_prob != 0:
        L = lexicon_to_fst(
            lexicon,
            token2id=token2id,
            word2id=word2id,
            sil_token=sil_token,
            sil_prob=sil_prob,
        )
    else:
        L = lexicon_to_fst_no_sil(lexicon, token2id=token2id, word2id=word2id,)

    if sil_prob != 0:
        L_disambig = lexicon_to_fst(
            lexicon_disambig,
            token2id=token2id,
            word2id=word2id,
            sil_token=sil_token,
            sil_prob=sil_prob,
            need_self_loops=True,
        )
    else:
        L_disambig = lexicon_to_fst_no_sil(
            lexicon_disambig,
            token2id=token2id,
            word2id=word2id,
            need_self_loops=True,
        )

    L_inv = k2.arc_sort(L.invert())
    logger.info(f"Saving L.pt, Linv.pt, L_disambig.pt to '{out_dir}'")
    torch.save(L.as_dict(), out_dir / "L.pt")
    torch.save(L_disambig.as_dict(), out_dir / "L_disambig.pt")
    torch.save(L_inv.as_dict(), out_dir / "Linv.pt")
