"""Utilities for k2 integration with SpeechBrain.

This code was adjusted from icefall (https://github.com/k2-fsa/icefall).


Authors:
  * Pierre Champion 2023
  * Zeyu Zhao 2023
  * Georgios Karakasidis 2023
"""

import os
import logging
from pathlib import Path
from typing import List, Union
import torch

from . import k2  # import k2 from ./__init__.py

logger = logging.getLogger(__name__)


def lattice_path_to_textid(
    best_paths: k2.Fsa, return_ragged: bool = False
) -> Union[List[List[int]], k2.RaggedTensor]:
    """
    Extract the texts (as word IDs) from the best-path FSAs.

    Arguments
    ---------
    best_paths: k2.Fsa
        A k2.Fsa with best_paths.arcs.num_axes() == 3, i.e.
        containing multiple FSAs, which is expected to be the result
        of k2.shortest_path (otherwise the returned values won't
        be meaningful).
    return_ragged: bool
        True to return a ragged tensor with two axes [utt][word_id].
        False to return a list-of-list word IDs.

    Returns
    -------
    Returns a list of lists of int, containing the label sequences we
    decoded.
    """
    if isinstance(best_paths.aux_labels, k2.RaggedTensor):
        # remove 0's and -1's.
        aux_labels = best_paths.aux_labels.remove_values_leq(0)
        # TODO: change arcs.shape() to arcs.shape
        aux_shape = best_paths.arcs.shape().compose(aux_labels.shape)

        # remove the states and arcs axes.
        aux_shape = aux_shape.remove_axis(1)
        aux_shape = aux_shape.remove_axis(1)
        aux_labels = k2.RaggedTensor(aux_shape, aux_labels.values)
    else:
        # remove axis corresponding to states.
        aux_shape = best_paths.arcs.shape().remove_axis(1)
        aux_labels = k2.RaggedTensor(aux_shape, best_paths.aux_labels)
        # remove 0's and -1's.
        aux_labels = aux_labels.remove_values_leq(0)

    assert aux_labels.num_axes == 2
    if return_ragged:
        return aux_labels
    else:
        return aux_labels.tolist()


def lattice_paths_to_text(best_paths: k2.Fsa, word_table) -> List[str]:
    """
    Convert the best path to a list of strings.

    Arguments
    ---------
    best_paths: k2.Fsa
        It is the path in the lattice with the highest score for a
        given utterance.
    word_table: List[str] or Dict[int,str]
        It is a list or dict that maps word IDs to words.

    Returns
    -------
    texts: List[str]
        A list of strings, each of which is the decoding result of the
        corresponding utterance.
    """
    hyps: List[List[int]] = lattice_path_to_textid(
        best_paths, return_ragged=False
    )
    texts = []
    for wids in hyps:
        texts.append(" ".join([word_table[wid] for wid in wids]))
    return texts


def load_G(path: Union[str, Path], cache: bool = True) -> k2.Fsa:
    """
    load a lm to be used in the decoding graph creation (or lm rescoring).

    Arguments
    ---------
    path: str
        The path to an FST LM (ending with .fst.txt) or a k2-converted
        LM (in pytorch .pt format).
    cache: bool
        Whether or not to load/cache the LM from/to the .pt format (in the same dir).

    Returns
    -------
    G: k2.Fsa
        An FSA representing the LM.
    """
    path = str(path)
    if os.path.exists(path.replace(".fst.txt", ".pt")) and cache:
        logger.warning(
            f"Loading '{path}' from its cached .pt format."
            " Set 'caching: False' in the yaml"
            " if this is not what you want."
        )
        G = k2.Fsa.from_dict(
            torch.load(path.replace(".fst.txt", ".pt"), map_location="cpu")
        )
        return G

    logger.info(f"Loading G LM: {path}")
    # If G_path is an fst.txt file then convert to .pt file
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"File {path} not found. " "You need to run arpa_to_fst to get it."
        )
    with open(path) as f:
        G = k2.Fsa.from_openfst(f.read(), acceptor=False)
        torch.save(G.as_dict(), path[:-8] + ".pt")
    return G


def prepare_rescoring_G(G: k2.Fsa) -> k2.Fsa:
    """
    Prepare a LM with the purpose of using it for LM rescoring.
    For instance, in the librispeech recipe this is a 4-gram LM (while a
    3gram LM is used for HLG construction).

    Arguments
    ---------
    G: k2.Fsa
        An FSA representing the LM.

    Returns
    -------
    G: k2.Fsa
        An FSA representing the LM, with the following modifications:
        - G.aux_labels is removed
        - G.lm_scores is set to G.scores
        - G is arc-sorted
    """
    if "_properties" in G.__dict__:
        G.__dict__["_properties"] = None
    del G.aux_labels
    G = k2.Fsa.from_fsas([G]).to("cpu")  # only used for decoding
    G = k2.arc_sort(G)
    G = k2.add_epsilon_self_loops(G)
    G = k2.arc_sort(G)
    # G.lm_scores is used to replace HLG.lm_scores during LM rescoring.
    if not hasattr(G, "lm_scores"):
        G.lm_scores = G.scores.clone()
    return G
