"""
compile Token, Lexicon and Grammar FST

Authors
 * Abdelwahab Heba 2021
"""

import torch
import logging

try:
    import k2
except ImportError:
    err_msg = "The optional dependency K2 is needed to use this module\n"
    err_msg += "Cannot import k2. To use WFSTs into autograd-based ML\n"
    err_msg += "Please follow the instructions below\n"
    err_msg += "=============================\n"
    err_msg += "pip install k2\n"
    err_msg += "for more information please refer to\n"
    err_msg += "https://github.com/k2-fsa/k2"
    raise ImportError(err_msg)

from k2 import Fsa




# Modified from snowfall project
# https://github.com/k2-fsa/snowfall/blob/master/snowfall/decoding/graph.py
def compile_TLG(
    L: Fsa,
    G: Fsa,
    ctc_topo: Fsa,
    labels_disambig_id_start: int,
    aux_labels_disambig_id_start: int,
) -> Fsa:
    """
    Creates a decoding graph using a lexicon fst ``L`` and language model fsa ``G``.
    Involves arc sorting, intersection, determinization, removal of disambiguation symbols
    and adding epsilon self-loops.
    Args:
        L:
            An ``Fsa`` that represents the lexicon (L), i.e. has phones as ``symbols``
                and words as ``aux_symbols``.
        G:
            An ``Fsa`` that represents the language model (G), i.e. it's an acceptor
            with words as ``symbols``.
        T:  An ```Fsa``` that represents the token, in which when 0 appears on the left side, it represents
                   the blank symbol; when it appears on the right side,
                   it indicates an epsilon.
        labels_disambig_id_start:
            An integer ID corresponding to the first disambiguation symbol in the
            phonetic alphabet.
        aux_labels_disambig_id_start:
            An integer ID corresponding to the first disambiguation symbol in the
            words vocabulary.
    :return:
    """
    L = k2.arc_sort(L)
    G = k2.arc_sort(G)
    logging.info("Intersecting L and G")
    LG = k2.compose(L, G)
    logging.info(f"LG shape = {LG.shape}")
    logging.info("Connecting L*G")
    LG = k2.connect(LG)
    logging.info(f"LG shape = {LG.shape}")
    logging.info("Determinizing L*G")
    LG = k2.determinize(LG)
    logging.info(f"LG shape = {LG.shape}")
    logging.info("Connecting det(L*G)")
    LG = k2.connect(LG)
    logging.info(f"LG shape = {LG.shape}")
    logging.info("Removing disambiguation symbols on L*G")
    LG.labels[LG.labels >= labels_disambig_id_start] = 0
    if isinstance(LG.aux_labels, torch.Tensor):
        LG.aux_labels[LG.aux_labels >= aux_labels_disambig_id_start] = 0
    else:
        LG.aux_labels.values()[
            LG.aux_labels.values() >= aux_labels_disambig_id_start
        ] = 0
    logging.info("Removing epsilons")
    LG = k2.remove_epsilon(LG)
    logging.info(f"LG shape = {LG.shape}")
    logging.info("Connecting rm-eps(det(L*G))")
    LG = k2.connect(LG)
    logging.info(f"LG shape = {LG.shape}")
    LG.aux_labels = k2.ragged.remove_values_eq(LG.aux_labels, 0)

    logging.info("Arc sorting LG")
    LG = k2.arc_sort(LG)

    logging.info("Composing T LG")
    TLG = k2.compose(T, LG, inner_labels="tokens")

    logging.info("Connecting TLG")
    TLG = k2.connect(TLG)

    logging.info("Arc sorting TLG")
    TLG = k2.arc_sort(TLG)
    logging.info(
        f"TLG is arc sorted: {(TLG.properties & k2.fsa_properties.ARC_SORTED) != 0}"
    )
    return TLG
