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


from typing import Dict, List, Optional, Union


logger = logging.getLogger(__name__)

try:
    import k2
except ImportError:
    MSG = "Cannot import k2, so training and decoding with k2 will not work.\n"
    MSG += "Please refer to https://k2-fsa.github.io/k2/installation/from_wheels.html for installation.\n"
    MSG += "You may also find the precompiled wheels for your platform at https://download.pytorch.org/whl/torch_stable.html"
    raise ImportError(MSG)


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


def one_best_decoding(
    lattice: k2.Fsa,
    use_double_scores: bool = True,
    lm_scale_list: Optional[List[float]] = None,
) -> Union[k2.Fsa, Dict[str, k2.Fsa]]:
    """Get the best path from a lattice.

    Arguments
    ---------
    lattice: k2.Fsa
        The decoding lattice returned by :func:`get_lattice`.
    use_double_scores: bool
        True to use double precision floating point in the computation.
        False to use single precision.
    lm_scale_list: Optional[List[float]]
        A list of floats representing LM score scales.

    Returns
    -------
    An FsaVec containing linear paths.
    """
    if lm_scale_list is not None:
        ans = dict()
        saved_am_scores = lattice.scores - lattice.lm_scores
        for lm_scale in lm_scale_list:
            am_scores = saved_am_scores / lm_scale
            lattice.scores = am_scores + lattice.lm_scores

            best_path = k2.shortest_path(
                lattice, use_double_scores=use_double_scores
            )
            key = f"lm_scale_{lm_scale}"
            ans[key] = best_path
        return ans

    return k2.shortest_path(lattice, use_double_scores=use_double_scores)


def rescore_with_whole_lattice(
    lattice: k2.Fsa,
    G_with_epsilon_loops: k2.Fsa,
    lm_scale_list: Optional[List[float]] = None,
    use_double_scores: bool = True,
) -> Union[k2.Fsa, Dict[str, k2.Fsa]]:
    """Intersect the lattice with an n-gram LM and use shortest path
    to decode.

    The input lattice is obtained by intersecting `HLG` with
    a DenseFsaVec, where the `G` in `HLG` is in general a 3-gram LM.
    The input `G_with_epsilon_loops` is usually a 4-gram LM. You can consider
    this function as a second pass decoding. In the first pass decoding, we
    use a small G, while we use a larger G in the second pass decoding.

    Arguments
    ---------
    lattice: k2.Fsa
        An FsaVec with axes [utt][state][arc]. Its `aux_labels` are word IDs.
        It must have an attribute `lm_scores`.
    G_with_epsilon_loops: k2.Fsa
        An FsaVec containing only a single FSA. It contains epsilon self-loops.
        It is an acceptor and its labels are word IDs.
    lm_scale_list: Optional[List[float]]
        If none, return the intersection of `lattice` and `G_with_epsilon_loops`.
        If not None, it contains a list of values to scale LM scores.
        For each scale, there is a corresponding decoding result contained in
        the resulting dict.
    use_double_scores: bool
        True to use double precision in the computation.
        False to use single precision.

    Returns
    -------
    If `lm_scale_list` is None, return a new lattice which is the intersection
    result of `lattice` and `G_with_epsilon_loops`.
    Otherwise, return a dict whose key is an entry in `lm_scale_list` and the
    value is the decoding result (i.e., an FsaVec containing linear FSAs).
    """
    assert G_with_epsilon_loops.shape == (1, None, None)

    device = lattice.device
    if hasattr(lattice, "lm_scores"):
        lattice.scores = lattice.scores - lattice.lm_scores
        # We will use lm_scores from G, so remove lats.lm_scores here
        del lattice.lm_scores

    assert hasattr(G_with_epsilon_loops, "lm_scores")

    # Now, lattice.scores contains only am_scores

    # inv_lattice has word IDs as labels.
    # Its `aux_labels` is token IDs
    inv_lattice = k2.invert(lattice)
    num_seqs = lattice.shape[0]

    b_to_a_map = torch.zeros(num_seqs, device=device, dtype=torch.int32)

    # NOTE: The choice of the threshold list is arbitrary here to avoid OOM.
    # You may need to fine tune it.
    prune_th_list = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
    prune_th_list += [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    max_loop_count = 10
    loop_count = 0
    while loop_count <= max_loop_count:
        try:
            if device == "cpu":
                rescoring_lattice = k2.intersect(
                    G_with_epsilon_loops,
                    inv_lattice,
                    treat_epsilons_specially=True,
                )
            else:
                rescoring_lattice = k2.intersect_device(
                    G_with_epsilon_loops,
                    inv_lattice,
                    b_to_a_map,
                    sorted_match_a=True,
                )
            rescoring_lattice = k2.top_sort(k2.connect(rescoring_lattice))
            break
        except RuntimeError as e:
            logger.info(f"Caught exception:\n{e}\n")
            if loop_count >= max_loop_count:
                logger.info(
                    "Return None as the resulting lattice is too large."
                )
                return None
            logger.info(
                f"num_arcs before pruning: {inv_lattice.arcs.num_elements()}"
            )
            logger.info(
                "This OOM is not an error. You can ignore it. "
                "If your model does not converge well, or --max-duration "
                "is too large, or the input sound file is difficult to "
                "decode, you will meet this exception."
            )
            inv_lattice = k2.prune_on_arc_post(
                inv_lattice,
                prune_th_list[loop_count],
                True,
            )
            logger.info(
                f"num_arcs after pruning: {inv_lattice.arcs.num_elements()}"
            )
        loop_count += 1

    # lat has token IDs as labels
    # and word IDs as aux_labels.
    lat = k2.invert(rescoring_lattice)

    if lm_scale_list is None:
        return lat

    ans = dict()
    saved_am_scores = lat.scores - lat.lm_scores
    for lm_scale in lm_scale_list:
        am_scores = saved_am_scores / lm_scale
        lat.scores = am_scores + lat.lm_scores

        best_path = k2.shortest_path(lat, use_double_scores=use_double_scores)
        key = f"lm_scale_{lm_scale:.1f}"
        ans[key] = best_path
    return ans


def arpa_to_fst(
    arpa_dir: Path,
    output_dir: Path,
    words_txt: Path,
    disambig_symbol: str = "#0",
    convert_4gram: bool = True,
    trigram_arpa_name: str = "3-gram.pruned.1e-7.arpa",
    fourgram_arpa_name: str = "4-gram.arpa",
    trigram_fst_output_name: str = "G_3_gram.fst.txt",
    fourgram_fst_output_name: str = "G_4_gram.fst.txt",
):
    """Use kaldilm to convert an ARPA LM to FST. For example, in librispeech
    you can find a 3-gram (pruned) and a 4-gram ARPA LM in the openslr
    website (https://www.openslr.org/11/). You can use this function to
    convert them to FSTs. The resulting FSTs can then be used to create a
    decoding graph (HLG) for k2 decoding.

    If `convert_4gram` is True, then we will convert the 4-gram ARPA LM to
    FST. Otherwise, we will only convert the 3-gram ARPA LM to FST.
    It is worth noting that if the fsts already exist in the output_dir,
    then we will not convert them again (so you may need to delete them
    by hand if you, at any point, change your ARPA model).

    Arguments
    ---------
        arpa_dir: str
            Path to the directory containing the ARPA LM (we expect a trigram
            ARPA LM to exist, and if `convert_4gram` is True, then a 4-gram
            ARPA LM should also exist).
        output_dir: str
            Path to the directory where the FSTs will be saved.
        words_txt: str
            Path to the words.txt file created by prepare_lang.
        disambig_symbol: str
            The disambiguation symbol to use.
        convert_4gram: bool
            If True, then we will convert the 4-gram ARPA LM to
            FST. Otherwise, we will only convert the 3-gram ARPA LM to FST.
        trigram_arpa_name: str
            The name of the 3-gram ARPA LM file. Defaults to the librispeech
            3-gram ARPA LM from openslr.
        fourgram_arpa_name: str
            The name of the 4-gram ARPA LM file. Defaults to the librispeech
            4-gram ARPA LM from openslr.
        trigram_fst_output_name: str
            The name of the 3-gram FST file that will be created with kaldilm.
            NOTE: This is just the name and not the whole path.
        fourgram_fst_output_name: str
            The name of the 4-gram FST file that will be created with kaldilm.
            NOTE: This is just the name and not the whole path.

    Raises
    ---------
        ImportError: If kaldilm is not installed.
    """
    assert arpa_dir.is_dir()
    assert output_dir.is_dir()
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

    def _arpa_to_fst_single(
        arpa_path: Path, out_fst_path: Path, max_order: int
    ):
        """Convert a single ARPA LM to FST.

        Arguments
        ---------
            arpa_path: str
                Path to the ARPA LM file.
            out_fst_path: str
                Path to the output FST file.
            max_order: int
                The maximum order of the ARPA LM.
        """
        if out_fst_path.exists():
            return
        if not arpa_path.exists():
            raise FileNotFoundError(
                f"{arpa_path} not found while trying to create"
                f" the {max_order} FST."
            )
        try:
            s = arpa2fst(
                input_arpa=str(arpa_path),
                disambig_symbol=disambig_symbol,
                read_symbol_table=str(words_txt),
                max_order=max_order,
            )
        except Exception as e:
            logger.info(
                f"Failed to create {max_order}-gram FST from input={arpa_path}"
                f", disambig_symbol={disambig_symbol},"
                f" read_symbol_table={words_txt}"
            )
            raise e
        logger.info(f"Writing {out_fst_path}")
        with open(out_fst_path, "w") as f:
            f.write(s)

    arpa_path = arpa_dir / trigram_arpa_name
    fst_path = output_dir / os.path.basename(trigram_fst_output_name)
    _arpa_to_fst_single(arpa_path, fst_path, max_order=3)
    if convert_4gram:
        arpa_path = arpa_dir / fourgram_arpa_name
        fst_path = output_dir / os.path.basename(fourgram_fst_output_name)
        _arpa_to_fst_single(arpa_path, fst_path, max_order=4)
