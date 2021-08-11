#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

"""
This script takes as input a lexicon file "data/lang/lexicon.txt"
consisting of words and phones and does the following:

1. Add disambiguation symbols to the lexicon and generate lexicon_disambig.txt

2. Generate phones.txt, the phones table mapping a phone to a unique integer.

3. Generate words.txt, the words table mapping a word to a unique integer.

4. Generate L.pt, in k2 format. It can be loaded by

        d = torch.load("L.pt")
        lexicon = k2.Fsa.from_dict(d)

5. Generate L_disambig.pt, in k2 format.
"""
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import k2
import torch

Lexicon = List[Tuple[str, List[str]]]


def read_lexicon(filename: str) -> Lexicon:
    """Read a lexicon.txt in `filename`.

    Each line in the lexicon contains "word p1 p2 p3 ...".
    That is, the first field is a word and the remaining
    fields are phones. Fields are separated by space(s).

    Args:
      filename:
        Path to the lexicon.txt

    Returns:
      A list of tuples., e.g., [('w', ['p1', 'p2']), ('w1', ['p3, 'p4'])]
    """
    ans = []

    with open(filename, "r", encoding="utf-8") as f:
        whitespace = re.compile("[ \t]+")
        for line in f:
            a = whitespace.split(line.strip(" \t\r\n"))
            if len(a) == 0:
                continue

            if len(a) < 2:
                print(f"Found bad line {line} in lexicon file {filename}")
                print("Every line is expected to contain at least 2 fields")
                sys.exit(1)
            word = a[0]
            if word == "<eps>":
                print(f"Found bad line {line} in lexicon file {filename}")
                print("<eps> should not be a valid word")
                sys.exit(1)

            prons = a[1:]
            ans.append((word, prons))

    return ans


def write_lexicon(filename: str, lexicon: Lexicon) -> None:
    """Write a lexicon to a file.

    Args:
      filename:
        Path to the lexicon file to be generated.
      lexicon:
        It can be the return value of :func:`read_lexicon`.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for word, prons in lexicon:
            f.write(f"{word} {' '.join(prons)}\n")


def write_mapping(filename: str, sym2id: Dict[str, int]) -> None:
    """Write a symbol to ID mapping to a file.

    Note:
      No need to implement `read_mapping` as it can be done
      through :func:`k2.SymbolTable.from_file`.

    Args:
      filename:
        Filename to save the mapping.
      sym2id:
        A dict mapping symbols to IDs.
    Returns:
      Return None.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for sym, i in sym2id.items():
            f.write(f"{sym} {i}\n")


def get_phones(lexicon: Lexicon) -> List[str]:
    """Get phones from a lexicon.

    Args:
      lexicon:
        It is the return value of :func:`read_lexicon`.
    Returns:
      Return a list of unique phones.
    """
    ans = set()
    for _, prons in lexicon:
        ans.update(prons)
    sorted_ans = sorted(list(ans))
    return sorted_ans


def get_words(lexicon: Lexicon) -> List[str]:
    """Get words from a lexicon.

    Args:
      lexicon:
        It is the return value of :func:`read_lexicon`.
    Returns:
      Return a list of unique words.
    """
    ans = set()
    for word, _ in lexicon:
        ans.add(word)
    sorted_ans = sorted(list(ans))
    return sorted_ans


def add_disambig_symbols(lexicon: Lexicon) -> Tuple[Lexicon, int]:
    """It adds pseudo-phone disambiguation symbols #1, #2 and so on
    at the ends of phones to ensure that all pronunciations are different,
    and that none is a prefix of another.

    See also add_lex_disambig.pl from kaldi.

    Args:
      lexicon:
        It is returned by :func:`read_lexicon`.
    Returns:
      Return a tuple with two elements:

        - The output lexicon with disambiguation symbols
        - The ID of the max disambiguation symbols that appears
          in the lexicon
    """

    # (1) Work out the count of each phone-sequence in the
    # lexicon.
    count = defaultdict(int)
    for _, prons in lexicon:
        count[" ".join(prons)] += 1

    # (2) For each left sub-sequence of each phone-sequence, note down
    # that it exists (for identifying prefixes of longer strings).
    issubseq = defaultdict(int)
    for _, prons in lexicon:
        prons = prons.copy()
        prons.pop()
        while prons:
            issubseq[" ".join(prons)] = 1
            prons.pop()

    # (3) For each entry in the lexicon:
    # if the phone sequence is unique and is not a
    # prefix of another word, no disambig symbol.
    # Else output #1, or #2, #3, ... if the same phone-seq
    # has already been assigned a disambig symbol.
    ans = []

    # We start with #1 since #0 has its own purpose
    first_allowed_disambig = 1
    max_disambig = first_allowed_disambig - 1
    last_used_disambig_symbol_of = defaultdict(int)

    for word, prons in lexicon:
        phnseq = " ".join(prons)
        assert phnseq != ""
        if issubseq[phnseq] == 0 and count[phnseq] == 1:
            ans.append((word, prons))
            continue

        cur_disambig = last_used_disambig_symbol_of[phnseq]
        if cur_disambig == 0:
            cur_disambig = first_allowed_disambig
        else:
            cur_disambig += 1

        if cur_disambig > max_disambig:
            max_disambig = cur_disambig
        last_used_disambig_symbol_of[phnseq] = cur_disambig
        phnseq += f" #{cur_disambig}"
        ans.append((word, phnseq.split()))
    return ans, max_disambig


def generate_id_map(symbols: List[str]) -> Dict[str, int]:
    """Generate ID maps, i.e., map a symbol to a unique ID.

    Args:
      symbols:
        A list of unique symbols.
    Returns:
      A dict containing the mapping between symbols and IDs.
    """
    return {sym: i for i, sym in enumerate(symbols)}


def add_self_loops(
    arcs: List[List[Any]], disambig_phone: int, disambig_word: int
) -> List[List[Any]]:
    """Adds self-loops to states of an FST to propagate disambiguation symbols
    through it. They are added on each state with non-epsilon output symbols
    on at least one arc out of the state.

    See also fstaddselfloops.pl from Kaldi. One difference is that
    Kaldi uses OpenFst style FSTs and it has multiple final states.
    This function uses k2 style FSTs and it does not need to add self-loops
    to the final state.

    Args:
      arcs:
        A list-of-list. The sublist contains
        `[src_state, dest_state, label, aux_label, score]`
      disambig_phone:
        It is the phone ID of the symbol `#0`.
      disambig_word:
        It is the word ID of the symbol `#0`.

    Return:
      Return new `arcs` containing self-loops.
    """
    states_needs_self_loops = set()
    for arc in arcs:
        src, dst, ilabel, olabel, score = arc
        if olabel != 0:
            states_needs_self_loops.add(src)

    ans = []
    for s in states_needs_self_loops:
        ans.append([s, s, disambig_phone, disambig_word, 0])

    return arcs + ans


def lexicon_to_fst(
    lexicon: Lexicon,
    phone2id: Dict[str, int],
    word2id: Dict[str, int],
    sil_phone: str = "SIL",
    sil_prob: float = 0.5,
    need_self_loops: bool = False,
) -> k2.Fsa:
    """Convert a lexicon to an FST (in k2 format) with optional silence at
    the beginning and end of the word.

    Args:
      lexicon:
        The input lexicon. See also :func:`read_lexicon`
      phone2id:
        A dict mapping phones to IDs.
      word2id:
        A dict mapping words to IDs.
      sil_phone:
        The silence phone.
      sil_prob:
        The probability for adding a silence at the beginning and end
        of the word.
      need_self_loops:
        If True, add self-loop to states with non-epsilon output symbols
        on at least one arc out of the state.
    Returns:
      Return an instance of `k2.Fsa` representing the given lexicon.
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

    assert phone2id["<eps>"] == 0
    assert word2id["<eps>"] == 0

    eps = 0

    sil_phone = phone2id[sil_phone]

    arcs.append([start_state, loop_state, eps, eps, no_sil_score])
    arcs.append([start_state, sil_state, eps, eps, sil_score])
    arcs.append([sil_state, loop_state, sil_phone, eps, 0])

    for word, prons in lexicon:
        assert len(prons) > 0, f"{word} has no pronunciations"
        cur_state = loop_state

        word = word2id[word]
        prons = [phone2id[i] for i in prons]

        for i in range(len(prons) - 1):
            if i == 0:
                arcs.append([cur_state, next_state, prons[i], word, 0])
            else:
                arcs.append([cur_state, next_state, prons[i], eps, 0])

            cur_state = next_state
            next_state += 1

        # now for the last phone of this word
        # It has two out-going arcs, one to the loop state,
        # the other one to the sil_state.
        i = len(prons) - 1
        w = word if i == 0 else eps
        arcs.append([cur_state, loop_state, prons[i], w, no_sil_score])
        arcs.append([cur_state, sil_state, prons[i], w, sil_score])

    if need_self_loops:
        disambig_phone = phone2id["#0"]
        disambig_word = word2id["#0"]
        arcs = add_self_loops(
            arcs, disambig_phone=disambig_phone, disambig_word=disambig_word,
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


def main():
    out_dir = Path("data/lang")
    lexicon_filename = out_dir / "lexicon.txt"
    sil_phone = "SIL"
    sil_prob = 0.5

    lexicon = read_lexicon(lexicon_filename)
    phones = get_phones(lexicon)
    words = get_words(lexicon)

    lexicon_disambig, max_disambig = add_disambig_symbols(lexicon)

    for i in range(max_disambig + 1):
        disambig = f"#{i}"
        assert disambig not in phones
        phones.append(f"#{i}")

    assert "<eps>" not in phones
    phones = ["<eps>"] + phones

    assert "<eps>" not in words
    assert "#0" not in words
    assert "<s>" not in words
    assert "</s>" not in words

    words = ["<eps>"] + words + ["#0", "<s>", "</s>"]

    phone2id = generate_id_map(phones)
    word2id = generate_id_map(words)

    write_mapping(out_dir / "phones.txt", phone2id)
    write_mapping(out_dir / "words.txt", word2id)
    write_lexicon(out_dir / "lexicon_disambig.txt", lexicon_disambig)

    L = lexicon_to_fst(
        lexicon,
        phone2id=phone2id,
        word2id=word2id,
        sil_phone=sil_phone,
        sil_prob=sil_prob,
    )

    L_disambig = lexicon_to_fst(
        lexicon_disambig,
        phone2id=phone2id,
        word2id=word2id,
        sil_phone=sil_phone,
        sil_prob=sil_prob,
        need_self_loops=True,
    )
    torch.save(L.as_dict(), out_dir / "L.pt")
    torch.save(L_disambig.as_dict(), out_dir / "L_disambig.pt")

    if False:
        # Just for debugging, will remove it
        L.labels_sym = k2.SymbolTable.from_file(out_dir / "phones.txt")
        L.aux_labels_sym = k2.SymbolTable.from_file(out_dir / "words.txt")
        L_disambig.labels_sym = L.labels_sym
        L_disambig.aux_labels_sym = L.aux_labels_sym
        L.draw(out_dir / "L.png", title="L")
        L_disambig.draw(out_dir / "L_disambig.png", title="L_disambig")


if __name__ == "__main__":
    main()
