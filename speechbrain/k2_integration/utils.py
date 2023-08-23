# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                    Mingshuang Luo,
#                                                    Zengwei Yao)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import k2
from typing import Dict, Iterable, List, Optional, TextIO, Tuple, Union


logger = logging.getLogger(__name__)


def get_texts(
    best_paths: k2.Fsa, return_ragged: bool = False
) -> Union[List[List[int]], k2.RaggedTensor]:
    """Extract the texts (as word IDs) from the best-path FSAs.
    Args:
      best_paths:
        A k2.Fsa with best_paths.arcs.num_axes() == 3, i.e.
        containing multiple FSAs, which is expected to be the result
        of k2.shortest_path (otherwise the returned values won't
        be meaningful).
      return_ragged:
        True to return a ragged tensor with two axes [utt][word_id].
        False to return a list-of-list word IDs.
    Returns:
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

    Args:
      lattice:
        The decoding lattice returned by :func:`get_lattice`.
      use_double_scores:
        True to use double precision floating point in the computation.
        False to use single precision.
      lm_scale_list:
        A list of floats representing LM score scales.
    Return:
      An FsaVec containing linear paths.
    """
    if lm_scale_list is not None:
        ans = dict()
        saved_am_scores = lattice.scores - lattice.lm_scores
        for lm_scale in lm_scale_list:
            am_scores = saved_am_scores / lm_scale
            lattice.scores = am_scores + lattice.lm_scores

            best_path = k2.shortest_path(lattice, use_double_scores=use_double_scores)
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

    Args:
      lattice:
        An FsaVec with axes [utt][state][arc]. Its `aux_labels` are word IDs.
        It must have an attribute `lm_scores`.
      G_with_epsilon_loops:
        An FsaVec containing only a single FSA. It contains epsilon self-loops.
        It is an acceptor and its labels are word IDs.
      lm_scale_list:
        Optional. If none, return the intersection of `lattice` and
        `G_with_epsilon_loops`.
        If not None, it contains a list of values to scale LM scores.
        For each scale, there is a corresponding decoding result contained in
        the resulting dict.
      use_double_scores:
        True to use double precision in the computation.
        False to use single precision.
    Returns:
      If `lm_scale_list` is None, return a new lattice which is the intersection
      result of `lattice` and `G_with_epsilon_loops`.
      Otherwise, return a dict whose key is an entry in `lm_scale_list` and the
      value is the decoding result (i.e., an FsaVec containing linear FSAs).
    """
    # Nbest is not used in this function
    # assert hasattr(lattice, "lm_scores")
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
                    treat_epsilons_specially=True
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
                logger.info("Return None as the resulting lattice is too large.")
                return None
            logger.info(f"num_arcs before pruning: {inv_lattice.arcs.num_elements()}")
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
            logger.info(f"num_arcs after pruning: {inv_lattice.arcs.num_elements()}")
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