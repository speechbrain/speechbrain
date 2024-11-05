"""Different decoding graph algorithms for k2, be it HL or HLG (with G LM
and bigger rescoring LM).

This code was adjusted from icefall (https://github.com/k2-fsa/icefall/blob/master/icefall/decode.py).


Authors:
  * Pierre Champion 2023
  * Zeyu Zhao 2023
  * Georgios Karakasidis 2023
"""

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from speechbrain.lm.arpa import arpa_to_fst
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger

from . import k2  # import k2 from ./__init__.py
from . import graph_compiler, utils

logger = get_logger(__name__)


def get_decoding(
    hparams: Dict, graphCompiler: graph_compiler.GraphCompiler, device="cpu"
):
    """
    This function reads a config and creates the decoder for k2 graph compiler
    decoding.
    There are the following cases:
        - HLG is compiled and LM rescoring is used. In that case,
          compose_HL_with_G and use_G_rescoring are both True and we will
          create for example G_3_gram.fst.txt and G_4_gram.fst.txt. Note that
          the 3gram and 4gram ARPA lms will need to exist under
          `hparams['lm_dir']`.
        - HLG is compiled but LM rescoring is not used. In that case,
          compose_HL_with_G is True and use_G_rescoring is False and we will
          create for example G_3_gram.fst.txt. Note that the 3gram ARPA lm will
          need to exist under `hparams['lm_dir']`.
        - HLG is not compiled (only use HL graph) and LM rescoring used.
          In that case, compose_HL_with_G is False and use_G_rescoring is True.
          Note that the 4gram ARPA lms will need to exist under
          `hparams['lm_dir']`.
        - HLG is not compiled (only use HL graph) and LM rescoring is not used.
          In that case, compose_HL_with_G is False and use_G_rescoring is False
          and we will not convert LM to FST.

    Arguments
    ---------
    hparams: dict
        The hyperparameters.
    graphCompiler: graph_compiler.GraphCompiler
        The graphCompiler (H)
    device : torch.device
        The device to use.

    Returns
    -------
    Dict:
        decoding_graph: k2.Fsa
            A HL or HLG decoding graph.
            Used with a nnet output and the function `get_lattice` to
            obtain a decoding lattice `k2.Fsa`.
        decoding_method: Callable[[k2.Fsa], k2.Fsa]
            A function to call with a decoding lattice `k2.Fsa` (obtained
            after nnet output intersect with a HL or HLG).
            Returns an FsaVec containing linear FSAs

    Example
    -------
    >>> import torch
    >>> from speechbrain.k2_integration.losses import ctc_k2
    >>> from speechbrain.k2_integration.utils import lattice_paths_to_text
    >>> from speechbrain.k2_integration.graph_compiler import CtcGraphCompiler
    >>> from speechbrain.k2_integration.lexicon import Lexicon
    >>> from speechbrain.k2_integration.prepare_lang import prepare_lang
    >>> from speechbrain.k2_integration.lattice_decoder import get_decoding
    >>> from speechbrain.k2_integration.lattice_decoder import get_lattice

    >>> batch_size = 1

    >>> log_probs = torch.randn(batch_size, 40, 10)
    >>> log_probs.requires_grad = True
    >>> # Assume all utterances have the same length so no padding was needed.
    >>> input_lens = torch.ones(batch_size)
    >>> # Create a small lexicon containing only two words and write it to a file.
    >>> lang_tmpdir = getfixture('tmpdir')
    >>> lexicon_sample = "hello h e l l o\\nworld w o r l d\\n<UNK> <unk>"
    >>> lexicon_file = lang_tmpdir.join("lexicon.txt")
    >>> lexicon_file.write(lexicon_sample)
    >>> # Create a lang directory with the lexicon and L.pt, L_inv.pt, L_disambig.pt
    >>> prepare_lang(lang_tmpdir)
    >>> # Create a lexicon object
    >>> lexicon = Lexicon(lang_tmpdir)
    >>> # Create a random decoding graph
    >>> graph = CtcGraphCompiler(
    ...     lexicon,
    ...     log_probs.device,
    ... )

    >>> decode = get_decoding(
    ...     {"compose_HL_with_G": False,
    ...      "decoding_method": "onebest",
    ...      "lang_dir": lang_tmpdir},
    ...     graph)
    >>> lattice = get_lattice(log_probs, input_lens, decode["decoding_graph"])
    >>> path = decode["decoding_method"](lattice)['1best']
    >>> text = lattice_paths_to_text(path, lexicon.word_table)
    """

    compose_HL_with_G = hparams.get("compose_HL_with_G")
    use_G_rescoring = (
        hparams.get("decoding_method") == "whole-lattice-rescoring"
    )

    caching = (
        False if "caching" in hparams and hparams["caching"] is False else True
    )

    if compose_HL_with_G or use_G_rescoring:
        lm_dir = Path(hparams["lm_dir"])
        G_path = lm_dir / (hparams["G_arpa"].replace("arpa", "fst.txt"))
        G_rescoring_path = (
            lm_dir / (hparams["G_rescoring_arpa"].replace("arpa", "fst.txt"))
            if use_G_rescoring
            else None
        )
        if compose_HL_with_G:
            run_on_main(
                arpa_to_fst,
                kwargs={
                    "words_txt": Path(hparams["lang_dir"]) / "words.txt",
                    "in_arpa": lm_dir / hparams["G_arpa"],
                    "out_fst": G_path,
                    "ngram_order": 3,  # by default use 3-gram for HLG's LM
                    "cache": caching,
                },
            )
        if use_G_rescoring:
            run_on_main(
                arpa_to_fst,
                kwargs={
                    "words_txt": Path(hparams["lang_dir"]) / "words.txt",
                    "in_arpa": lm_dir / hparams["G_rescoring_arpa"],
                    "out_fst": G_rescoring_path,
                    "ngram_order": 4,  # by default use 4-gram for rescoring LM
                    "cache": caching,
                },
            )

    output_folder = None
    if "output_folder" in hparams:
        output_folder = output_folder

    if compose_HL_with_G:
        G = utils.load_G(G_path, cache=caching)
        decoding_graph = graphCompiler.compile_HLG(
            G, cache_dir=output_folder, cache=caching
        )
    else:
        decoding_graph = graphCompiler.compile_HL(
            cache_dir=output_folder, cache=caching
        )

    if hparams.get("decoding_method") == "whole-lattice-rescoring":
        G_rescoring = None
        if not isinstance(hparams["rescoring_lm_scale"], list):
            hparams["rescoring_lm_scale"] = [hparams["rescoring_lm_scale"]]

        def decoding_method(lattice: k2.Fsa) -> Dict[str, k2.Fsa]:
            """Get the best path from a lattice given rescoring_lm_scale."""

            # Lazy load rescoring G (takes a lot of time) for developer happiness
            nonlocal G_rescoring
            if G_rescoring is None:
                logger.info("Decoding method: whole-lattice-rescoring")
                logger.info(f"Loading rescoring LM: {G_rescoring_path}")
                G_rescoring_pt = utils.load_G(G_rescoring_path, cache=caching)
                graphCompiler.lexicon.remove_G_rescoring_disambig_symbols(
                    G_rescoring_pt
                )
                G_rescoring = utils.prepare_rescoring_G(G_rescoring_pt)

            # rescore_with_whole_lattice returns a list of paths depending on
            # lm_scale values.
            return rescore_with_whole_lattice(
                lattice,
                G_rescoring,
                lm_scale_list=hparams["rescoring_lm_scale"],
            )

    elif hparams.get("decoding_method") in ["1best", "onebest"]:
        logger.info("Decoding method: one-best-decoding")

        def decoding_method(lattice: k2.Fsa) -> Dict[str, k2.Fsa]:
            """Get the best path from a lattice."""
            return OrderedDict({"1best": one_best_decoding(lattice)})

    else:

        def decoding_method(lattice: k2.Fsa):
            """A dummy decoding method that raises an error."""
            raise NotImplementedError(
                f"{hparams.get('decoding_method')} not implemented as a decoding_method"
            )

    return {
        "decoding_graph": decoding_graph.to(device),
        "decoding_method": decoding_method,
    }


@torch.no_grad()
def get_lattice(
    log_probs_nnet_output: torch.Tensor,
    input_lens: torch.Tensor,
    decoder: k2.Fsa,
    search_beam: int = 5,
    output_beam: int = 5,
    min_active_states: int = 300,
    max_active_states: int = 1000,
    ac_scale: float = 1.0,
    subsampling_factor: int = 1,
) -> k2.Fsa:
    """
    Get the decoding lattice from a decoding graph and neural network output.

    Arguments
    ---------
    log_probs_nnet_output: torch.Tensor
        It is the output of a neural model of shape `(batch, seq_len, num_tokens)`.
    input_lens: torch.Tensor
        It is an int tensor of shape (batch,). It contains lengths of
        each sequence in `log_probs_nnet_output`.
    decoder: k2.Fsa
        It is an instance of :class:`k2.Fsa` that represents the decoding graph.
    search_beam: int
        Decoding beam, e.g. 20.  Ger is faster, larger is more exact
        (less pruning). This is the default value; it may be modified by
        `min_active_states` and `max_active_states`.
    output_beam: int
         Beam to prune output, similar to lattice-beam in Kaldi.  Relative
         to best path of output.
    min_active_states: int
        Minimum number of FSA states that are allowed to be active on any given
        frame for any given intersection/composition task. This is advisory,
        in that it will try not to have fewer than this number active.
        Set it to zero if there is no constraint.
    max_active_states: int
        Maximum number of FSA states that are allowed to be active on any given
        frame for any given intersection/composition task. This is advisory,
        in that it will try not to exceed that but may not always succeed.
        You can use a very large number if no constraint is needed.
    ac_scale: float
        acoustic scale applied to `log_probs_nnet_output`
    subsampling_factor: int
        The subsampling factor of the model.

    Returns
    -------
    lattice: k2.Fsa
        An FsaVec containing the decoding result. It has axes [utt][state][arc].
    """

    device = log_probs_nnet_output.device
    input_lens = input_lens.to(device)
    if decoder.device != device:
        logger.warn(
            "Decoding graph (HL or HLG) not loaded on the same device"
            "  as nnet, this will cause decoding speed degradation"
        )
        decoder = decoder.to(device)

    input_lens = (input_lens * log_probs_nnet_output.shape[1]).round().int()
    # NOTE: low ac_scales may results in very big lattices and OOM errors.
    log_probs_nnet_output *= ac_scale

    lattice = k2.get_lattice(
        log_probs_nnet_output,
        input_lens,
        decoder,
        search_beam=search_beam,
        output_beam=output_beam,
        min_active_states=min_active_states,
        max_active_states=max_active_states,
        subsampling_factor=subsampling_factor,
    )

    return lattice


@torch.no_grad()
def one_best_decoding(
    lattice: k2.Fsa,
    use_double_scores: bool = True,
) -> k2.Fsa:
    """
    Get the best path from a lattice.

    Arguments
    ---------
    lattice: k2.Fsa
        The decoding lattice returned by :func:`get_lattice`.
    use_double_scores: bool
        True to use double precision floating point in the computation.
        False to use single precision.

    Returns
    -------
    best_path: k2.Fsa
        An FsaVec containing linear paths.
    """
    best_path = k2.shortest_path(lattice, use_double_scores=use_double_scores)
    return best_path


@torch.no_grad()
def rescore_with_whole_lattice(
    lattice: k2.Fsa,
    G_with_epsilon_loops: k2.Fsa,
    lm_scale_list: Optional[List[float]] = None,
    use_double_scores: bool = True,
) -> Union[k2.Fsa, Dict[str, k2.Fsa]]:
    """
    Intersect the lattice with an n-gram LM and use shortest path to decode.
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
    G_with_epsilon_loops = G_with_epsilon_loops.to(lattice.device)
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
                "If your model does not converge well, or the segment length "
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

    ans = OrderedDict()
    saved_am_scores = lat.scores - lat.lm_scores
    for lm_scale in lm_scale_list:
        am_scores = saved_am_scores / lm_scale
        lat.scores = am_scores + lat.lm_scores

        best_path = k2.shortest_path(lat, use_double_scores=use_double_scores)
        key = f"whole_lattice_rescore_lm_scale_{lm_scale:.1f}"
        ans[key] = best_path
    return ans
