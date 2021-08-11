#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

"""
This script takes as inputs the following files:
    - data/lang/bpe/bpe.model,
    - data/lang/bpe/tokens.txt (will remove it),
    - data/lang/bpe/words.txt

and generates the following files in the directory data/lang/bpe:

    - lexicon.txt
    - lexicon_disambig.txt
    - L.pt
    - L_disambig.pt
    - phones.txt
"""

from pathlib import Path
from typing import Dict, List

import k2
import sentencepiece as spm
import torch
from prepare_lang import (
    Lexicon,
    add_disambig_symbols,
    add_self_loops,
    write_lexicon,
)


def lexicon_to_fst_no_sil(
    lexicon: Lexicon,
    token2id: Dict[str, int],
    word2id: Dict[str, int],
    need_self_loops: bool = False,
) -> k2.Fsa:
    """Convert a lexicon to an FST (in k2 format).

    Args:
      lexicon:
        The input lexicon. See also :func:`read_lexicon`
      token2id:
        A dict mapping tokens to IDs.
      word2id:
        A dict mapping words to IDs.
      need_self_loops:
        If True, add self-loop to states with non-epsilon output symbols
        on at least one arc out of the state.
    Returns:
      Return an instance of `k2.Fsa` representing the given lexicon.
    """
    loop_state = 0  # words enter and leave from here
    next_state = 1  # the next un-allocated state, will be incremented as we go.

    arcs = []

    assert token2id["<unk>"] == 0
    assert word2id["<eps>"] == 0

    eps = 0

    for word, prons in lexicon:
        assert len(prons) > 0, f"{word} has no pronunciations"
        cur_state = loop_state

        word = word2id[word]
        prons = [token2id[i] for i in prons]

        for i in range(len(prons) - 1):
            if i == 0:
                arcs.append([cur_state, next_state, prons[i], word, 0])
            else:
                arcs.append([cur_state, next_state, prons[i], eps, 0])

            cur_state = next_state
            next_state += 1

        # now for the last phone of this word
        i = len(prons) - 1
        w = word if i == 0 else eps
        arcs.append([cur_state, loop_state, prons[i], w, 0])

    if need_self_loops:
        disambig_phone = token2id["#0"]
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


def generate_lexicon(model_file: str, words: List[str]) -> Lexicon:
    """Generate a lexicon from a BPE model.

    Args:
      model_file:
        Path to a sentencepiece model.
      words:
        A list of strings representing words.
    Returns:
      Return a dict whose keys are words and values are the corresponding
      word pieces.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_file))

    words_pieces: List[List[str]] = sp.encode(words, out_type=str)

    lexicon = []
    for word, pieces in zip(words, words_pieces):
        lexicon.append((word, pieces))

    lexicon.append(("<UNK>", ["<UNK>"]))
    return lexicon


def main():
    lang_dir = Path("data/lang/bpe")
    model_file = lang_dir / "bpe.model"

    word_sym_table = k2.SymbolTable.from_file(lang_dir / "words.txt")

    words = word_sym_table.symbols

    excluded = ["<eps>", "!SIL", "<SPOKEN_NOISE>", "<UNK>", "#0", "<s>", "</s>"]
    for w in excluded:
        if w in words:
            words.remove(w)

    lexicon = generate_lexicon(model_file, words)

    # TODO(fangjun): Remove tokens.txt and generate it from the model directly.
    #
    # We are using it since the IDs we are using in tokens.txt is
    # different from the one contained in the model
    token_sym_table = k2.SymbolTable.from_file(lang_dir / "tokens.txt")

    lexicon_disambig, max_disambig = add_disambig_symbols(lexicon)

    for i in range(max_disambig + 1):
        disambig = f"#{i}"
        assert disambig not in token_sym_table
        token_sym_table.add(f"#{i}")

    word_sym_table.add("#0")
    word_sym_table.add("<s>")
    word_sym_table.add("</s>")

    token_sym_table.to_file(lang_dir / "phones.txt")

    write_lexicon(lang_dir / "lexicon.txt", lexicon)
    write_lexicon(lang_dir / "lexicon_disambig.txt", lexicon_disambig)

    L = lexicon_to_fst_no_sil(
        lexicon, token2id=token_sym_table, word2id=word_sym_table,
    )

    L_disambig = lexicon_to_fst_no_sil(
        lexicon_disambig,
        token2id=token_sym_table,
        word2id=word_sym_table,
        need_self_loops=True,
    )
    torch.save(L.as_dict(), lang_dir / "L.pt")
    torch.save(L_disambig.as_dict(), lang_dir / "L_disambig.pt")

    if False:
        # Just for debugging, will remove it
        L.labels_sym = k2.SymbolTable.from_file(lang_dir / "phones.txt")
        L.aux_labels_sym = k2.SymbolTable.from_file(lang_dir / "words.txt")
        L_disambig.labels_sym = L.labels_sym
        L_disambig.aux_labels_sym = L.aux_labels_sym
        L.draw(lang_dir / "L.svg", title="L")
        L_disambig.draw(lang_dir / "L_disambig.svg", title="L_disambig")


if __name__ == "__main__":
    main()
