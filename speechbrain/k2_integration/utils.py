"""Utilities for k2 integration with SpeechBrain.

This code was adjusted from icefall (https://github.com/k2-fsa/icefall).


Authors:
  * Zeyu Zhao 2023
  * Georgios Karakasidis 2023
"""

import os
import logging
from pathlib import Path

import torch

from . import k2 # import k2 from ./__init__.py

from typing import Dict, List, Optional, Union


logger = logging.getLogger(__name__)


def get_texts(
    best_paths: k2.Fsa, return_ragged: bool = False
) -> Union[List[List[int]], k2.RaggedTensor]:
    """Extract the texts (as word IDs) from the best-path FSAs.

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


def texts_to_ids(
    texts: List[str],
    word_table: List[str],
    oov_token_id: int,
    add_sil_token_as_separator=False,
    sil_token_id:Optional[int]=None,
    log_unknown_warning=True
    ) -> List[List[int]]:
    """Convert a list of texts to a list-of-list of word IDs.

    Note that we only get the first spelling of a word in the lexicon if there
    are multiple spellings.

    Arguments
    ---------
    texts: List[str]
        It is a list of strings. Each string consists of space(s)
        separated words. An example containing two strings is given below:

            ['HELLO ICEFALL', 'HELLO k2']
    word_table: List[str]
        Word to id table.
    oov_token_id: int
        The OOV token ID
    add_sil_token_as_separator: bool
        If True, add `sil_token_id` as a separator between words.
        Argument sil_token_id must be set.
    log_unknown_warning: bool
        Log if word not found in word_table

    Returns
    -------
    word_ids_list:
        Return a list-of-list of word IDs.
    """
    word_ids_list = []
    for text in texts:
        word_ids = []
        words = text.split()
        for i, word in enumerate(words):
            if word in word_table:
                idword = word_table[word]
                if isinstance(idword, list):
                    idword = idword[0] # only first spelling
                word_ids.append(idword)
            else:
                word_ids.append(oov_token_id)
                if log_unknown_warning:
                    logger.warn(
                        f"Cannot find word {word} in the lexicon."
                        f" Replacing it with OOV token. "
                        f" Note that it is fine if you are testing."
                    )

            if add_sil_token_as_separator and i < len(words) - 1:
                assert sil_token_id != None, f"sil_token_id=None while add_sil_token_as_separator=True"
                word_ids.append(sil_token_id)

        word_ids_list.append(word_ids)
    return word_ids_list


def lattice_to_text(best_path: k2.Fsa, word_table) -> List[str]:
    """Convert the best path to a list of strings.

    Arguments
    ---------
    best_path: k2.Fsa
        It is the path in the lattice with the highest score for a
        given utterance.

    Returns
    -------
    A list of strings, each of which is the decoding result of the
    corresponding utterance.
    """
    hyps: List[List[int]] = utils.get_texts(best_path, return_ragged=False)
    texts = []
    for wids in hyps:
        texts.append(" ".join([word_table[wid] for wid in wids]))
    return texts


def arpa_to_fst(
    words_txt: Path,
    in_arpa_files: List[str],
    out_fst_files: List[str],
    lms_ngram_orders: List[int],
    disambig_symbol: str = "#0",
):
    """Use kaldilm to convert an ARPA LM to FST. For example, in librispeech
    you can find a 3-gram (pruned) and a 4-gram ARPA LM in the openslr
    website (https://www.openslr.org/11/). You can use this function to
    convert them to FSTs. The resulting FSTs can then be used to create a
    decoding graph (HLG) for k2 decoding.

    It is worth noting that if the fsts already exist in the output_dir,
    then we will not convert them again (so you may need to delete them
    by hand if you, at any point, change your arpa model).

    arguments
    ---------
        words_txt: str
            path to the words.txt file created by prepare_lang.
        in_arpa_names: List[str]
            List of arpa files to convert to fst.
        out_fst_files: List[Str]
            List of fst path where the fsts will be saved, len(in) == len(out).
        lms_ngram_orders: List[int]
            List of the arpa ngram orders, len(in) == len(out).
        disambig_symbol: str
            the disambiguation symbol to use.

    raises
    ---------
        importerror: if kaldilm is not installed.
    """
    assert len(in_arpa_names) == len(out_fst_files)
    assert len(in_arpa_names) == len(lms_ngram_orders)
    try:
        from kaldilm.arpa2fst import arpa2fst
    except importerror:
        # this error will occur when there is fst lm in the provided lm_dir
        # and we are trying to create it by converting an arpa lm to fst.
        # for this, we need to install kaldilm.
        raise importerror(
            "optional dependencies must be installed to use kaldilm.\n"
            "install using `pip install kaldilm`."
        )

    def _arpa_to_fst_single(
        arpa_path: path, out_fst_path: path, max_order: int
    ):
        """convert a single arpa lm to fst.

        arguments
        ---------
            arpa_path: str
                path to the arpa lm file.
            out_fst_path: str
                path to the output fst file.
            max_order: int
                the maximum order of the arpa lm.
        """
        if out_fst_path.exists():
            return
        if not arpa_path.exists():
            raise filenotfounderror(
                f"{arpa_path} not found while trying to create"
                f" the {max_order} fst."
            )
        try:
            s = arpa2fst(
                input_arpa=str(arpa_path),
                disambig_symbol=disambig_symbol,
                read_symbol_table=str(words_txt),
                max_order=max_order,
            )
        except exception as e:
            logger.info(
                f"failed to create {max_order}-gram fst from input={arpa_path}"
                f", disambig_symbol={disambig_symbol},"
                f" read_symbol_table={words_txt}"
            )
            raise e
        logger.info(f"writing {out_fst_path}")
        with open(out_fst_path, "w") as f:
            f.write(s)

    for a, f, n in zip(in_arpa_files, out_fst_files, lms_ngram_orders):
        _arpa_to_fst_single(a, f, max_order=n)


def load_G(
    path: Union[str, Path], device: str = "cpu", cache: bool = True
) -> k2.Fsa:
    """load a lm to be used in the decoding graph creation (or lm rescoring).
    note that it doesn't load g into memory.

    Arguments
    ---------
    path: str
        The path to an FST LM (ending with .fst.txt) or a k2-converted
        LM (in pytorch .pt format).
    device: str
        The device to load G on
    cache: bool
        Whether or not to cache the LM in .pt format (in the same dir).

    Returns
    -------
    G:
        An FSA representing the LM. The device is the same asked in argument
    """
    path = str(path)
    if os.path.exists(path.replace(".fst.txt", ".pt")):
        logger.warning(
            f"Loading '{path}' from its cached .pt format. Consider deleting the "
            "previous .pt file if this is not what you want."
        )
        path = path.replace(".fst.txt", ".pt")
    # If G_path is an fst.txt file then convert to .pt file
    if path.endswith(".fst.txt"):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"File {path} not found. "
                "You need to run the kaldilm to get it."
            )
        with open(path) as f:
            G = k2.Fsa.from_openfst(f.read(), acceptor=False).to(
                device
            )
    elif path.endswith(".pt"):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File {path} not found.")
        d = torch.load(path, map_location=device)
        G = k2.Fsa.from_dict(d).to(device)
    else:
        raise ValueError(f"File {path} is not a .fst.txt or .pt file.")
    if cache:
        torch.save(G.as_dict(), path[:-8] + ".pt")
    return G

def prepare_G(
    G: k2.Fsa, device: str = "cpu"
    ) -> k2.Fsa:
    """Prepare a LM with the purpose of using it for LM rescoring.
    For instance, in the librispeech recipe this is a 4-gram LM (while a
    3gram LM is used for HLG construction).

    Arguments
    ---------
    path: str
        The path to an FST LM (ending with .fst.txt) or a k2-converted
        LM (in pytorch .pt format).
    device: str
        The device to load G on

    Returns
    -------
    G:
        An FSA representing the LM. The device is the same as graph_compiler.device.
    """
    if "_properties" in G.__dict__:
        G.__dict__["_properties"] = None
    G = k2.Fsa.from_fsas([G]).to("cpu")  # only used for decoding
    G = k2.arc_sort(G)
    G = k2.add_epsilon_self_loops(G)
    G = k2.arc_sort(G)
    G = G.to(device)
    # G.lm_scores is used to replace HLG.lm_scores during
    # LM rescoring.
    if not hasattr(G, "lm_scores"):
        G.lm_scores = G.scores.clone()
    return G
