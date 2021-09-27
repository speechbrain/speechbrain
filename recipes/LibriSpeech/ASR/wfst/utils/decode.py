# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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
from typing import Dict, List, Optional, Union

import k2
import torch

from utils.utils import get_texts


def _intersect_device(
    a_fsas: k2.Fsa,
    b_fsas: k2.Fsa,
    b_to_a_map: torch.Tensor,
    sorted_match_a: bool,
    batch_size: int = 50,
) -> k2.Fsa:
    """This is a wrapper of k2.intersect_device and its purpose is to split
    b_fsas into several batches and process each batch separately to avoid
    CUDA OOM error.

    The arguments and return value of this function are the same as
    :func:`k2.intersect_device`.
    """
    num_fsas = b_fsas.shape[0]
    if num_fsas <= batch_size:
        return k2.intersect_device(
            a_fsas, b_fsas, b_to_a_map=b_to_a_map, sorted_match_a=sorted_match_a
        )

    num_batches = (num_fsas + batch_size - 1) // batch_size
    splits = []
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, num_fsas)
        splits.append((start, end))

    ans = []
    for start, end in splits:
        indexes = torch.arange(start, end).to(b_to_a_map)

        fsas = k2.index_fsa(b_fsas, indexes)
        b_to_a = k2.index_select(b_to_a_map, indexes)
        path_lattice = k2.intersect_device(
            a_fsas, fsas, b_to_a_map=b_to_a, sorted_match_a=sorted_match_a
        )
        ans.append(path_lattice)

    return k2.cat(ans)


def get_lattice(
    nnet_output: torch.Tensor,
    HLG: k2.Fsa,
    supervision_segments: torch.Tensor,
    search_beam: float,
    output_beam: float,
    min_active_states: int,
    max_active_states: int,
    subsampling_factor: int = 1,
) -> k2.Fsa:
    """Get the decoding lattice from a decoding graph and neural
    network output.
    Args:
      nnet_output:
        It is the output of a neural model of shape `(N, T, C)`.
      HLG:
        An Fsa, the decoding graph. See also `compile_HLG.py`.
      supervision_segments:
        A 2-D **CPU** tensor of dtype `torch.int32` with 3 columns.
        Each row contains information for a supervision segment. Column 0
        is the `sequence_index` indicating which sequence this segment
        comes from; column 1 specifies the `start_frame` of this segment
        within the sequence; column 2 contains the `duration` of this
        segment.
      search_beam:
        Decoding beam, e.g. 20.  Smaller is faster, larger is more exact
        (less pruning). This is the default value; it may be modified by
        `min_active_states` and `max_active_states`.
      output_beam:
         Beam to prune output, similar to lattice-beam in Kaldi.  Relative
         to best path of output.
      min_active_states:
        Minimum number of FSA states that are allowed to be active on any given
        frame for any given intersection/composition task. This is advisory,
        in that it will try not to have fewer than this number active.
        Set it to zero if there is no constraint.
      max_active_states:
        Maximum number of FSA states that are allowed to be active on any given
        frame for any given intersection/composition task. This is advisory,
        in that it will try not to exceed that but may not always succeed.
        You can use a very large number if no constraint is needed.
      subsampling_factor:
        The subsampling factor of the model.
    Returns:
      An FsaVec containing the decoding result. It has axes [utt][state][arc].
    """
    dense_fsa_vec = k2.DenseFsaVec(
        nnet_output,
        supervision_segments,
        allow_truncate=subsampling_factor - 1,
    )

    lattice = k2.intersect_dense_pruned(
        HLG,
        dense_fsa_vec,
        search_beam=search_beam,
        output_beam=output_beam,
        min_active_states=min_active_states,
        max_active_states=max_active_states,
    )

    return lattice


class Nbest(object):
    """
    An Nbest object contains two fields:

        (1) fsa. It is an FsaVec containing a vector of **linear** FSAs.
                 Its axes are [path][state][arc]
        (2) shape. Its type is :class:`k2.RaggedShape`.
                   Its axes are [utt][path]

    The field `shape` has two axes [utt][path]. `shape.dim0` contains
    the number of utterances, which is also the number of rows in the
    supervision_segments. `shape.tot_size(1)` contains the number
    of paths, which is also the number of FSAs in `fsa`.

    Caution:
      Don't be confused by the name `Nbest`. The best in the name `Nbest`
      has nothing to do with `best scores`. The important part is
      `N` in `Nbest`, not `best`.
    """

    def __init__(self, fsa: k2.Fsa, shape: k2.RaggedShape) -> None:
        """
        Args:
          fsa:
            An FsaVec with axes [path][state][arc]. It is expected to contain
            a list of **linear** FSAs.
          shape:
            A ragged shape with two axes [utt][path].
        """
        assert len(fsa.shape) == 3, f"fsa.shape: {fsa.shape}"
        assert shape.num_axes == 2, f"num_axes: {shape.num_axes}"

        if fsa.shape[0] != shape.tot_size(1):
            raise ValueError(
                f"{fsa.shape[0]} vs {shape.tot_size(1)}\n"
                "Number of FSAs in `fsa` does not match the given shape"
            )

        self.fsa = fsa
        self.shape = shape

    def __str__(self):
        s = "Nbest("
        s += f"Number of utterances:{self.shape.dim0}, "
        s += f"Number of Paths:{self.fsa.shape[0]})"
        return s

    @staticmethod
    def from_lattice(
        lattice: k2.Fsa,
        num_paths: int,
        use_double_scores: bool = True,
        lattice_score_scale: float = 0.5,
    ) -> "Nbest":
        """Construct an Nbest object by **sampling** `num_paths` from a lattice.

        Each sampled path is a linear FSA.

        We assume `lattice.labels` contains token IDs and `lattice.aux_labels`
        contains word IDs.

        Args:
          lattice:
            An FsaVec with axes [utt][state][arc].
          num_paths:
            Number of paths to **sample** from the lattice
            using :func:`k2.random_paths`.
          use_double_scores:
            True to use double precision in :func:`k2.random_paths`.
            False to use single precision.
          scale:
            Scale `lattice.score` before passing it to :func:`k2.random_paths`.
            A smaller value leads to more unique paths at the risk of being not
            to sample the path with the best score.
        Returns:
          Return an Nbest instance.
        """
        saved_scores = lattice.scores.clone()
        lattice.scores *= lattice_score_scale
        # path is a ragged tensor with dtype torch.int32.
        # It has three axes [utt][path][arc_pos]
        path = k2.random_paths(
            lattice, num_paths=num_paths, use_double_scores=use_double_scores
        )
        lattice.scores = saved_scores

        # word_seq is a k2.RaggedTensor sharing the same shape as `path`
        # but it contains word IDs. Note that it also contains 0s and -1s.
        # The last entry in each sublist is -1.
        # It axes is [utt][path][word_id]
        if isinstance(lattice.aux_labels, torch.Tensor):
            word_seq = k2.ragged.index(lattice.aux_labels, path)
        else:
            word_seq = lattice.aux_labels.index(path)
            word_seq = word_seq.remove_axis(word_seq.num_axes - 2)

        # Each utterance has `num_paths` paths but some of them transduces
        # to the same word sequence, so we need to remove repeated word
        # sequences within an utterance. After removing repeats, each utterance
        # contains different number of paths
        #
        # `new2old` is a 1-D torch.Tensor mapping from the output path index
        # to the input path index.
        _, _, new2old = word_seq.unique(
            need_num_repeats=False, need_new2old_indexes=True
        )

        # kept_path is a ragged tensor with dtype torch.int32.
        # It has axes [utt][path][arc_pos]
        kept_path, _ = path.index(new2old, axis=1, need_value_indexes=False)

        # utt_to_path_shape has axes [utt][path]
        utt_to_path_shape = kept_path.shape.get_layer(0)

        # Remove the utterance axis.
        # Now kept_path has only two axes [path][arc_pos]
        kept_path = kept_path.remove_axis(0)

        # labels is a ragged tensor with 2 axes [path][token_id]
        # Note that it contains -1s.
        labels = k2.ragged.index(lattice.labels.contiguous(), kept_path)

        # Remove -1 from labels as we will use it to construct a linear FSA
        labels = labels.remove_values_eq(-1)

        if isinstance(lattice.aux_labels, k2.RaggedTensor):
            # lattice.aux_labels is a ragged tensor with dtype torch.int32.
            # It has 2 axes [arc][word], so aux_labels is also a ragged tensor
            # with 2 axes [arc][word]
            aux_labels, _ = lattice.aux_labels.index(
                indexes=kept_path.values, axis=0, need_value_indexes=False
            )
        else:
            assert isinstance(lattice.aux_labels, torch.Tensor)
            aux_labels = k2.index_select(lattice.aux_labels, kept_path.values)
            # aux_labels is a 1-D torch.Tensor. It also contains -1 and 0.

        fsa = k2.linear_fsa(labels)
        fsa.aux_labels = aux_labels
        # Caution: fsa.scores are all 0s.
        # `fsa` has only one extra attribute: aux_labels.
        return Nbest(fsa=fsa, shape=utt_to_path_shape)

    def intersect(self, lattice: k2.Fsa, use_double_scores=True) -> "Nbest":
        """Intersect this Nbest object with a lattice, get 1-best
        path from the resulting FsaVec, and return a new Nbest object.

        The purpose of this function is to attach scores to an Nbest.

        Args:
          lattice:
            An FsaVec with axes [utt][state][arc]. If it has `aux_labels`, then
            we assume its `labels` are token IDs and `aux_labels` are word IDs.
            If it has only `labels`, we assume its `labels` are word IDs.
          use_double_scores:
            True to use double precision when computing shortest path.
            False to use single precision.
        Returns:
          Return a new Nbest. This new Nbest shares the same shape with `self`,
          while its `fsa` is the 1-best path from intersecting `self.fsa` and
          `lattice`. Also, its `fsa` has non-zero scores and inherits attributes
          for `lattice`.
        """
        # Note: We view each linear FSA as a word sequence
        # and we use the passed lattice to give each word sequence a score.
        #
        # We are not viewing each linear FSAs as a token sequence.
        #
        # So we use k2.invert() here.

        # We use a word fsa to intersect with k2.invert(lattice)
        word_fsa = k2.invert(self.fsa)

        if hasattr(lattice, "aux_labels"):
            # delete token IDs as it is not needed
            del word_fsa.aux_labels

        word_fsa.scores.zero_()
        word_fsa_with_epsilon_loops = k2.remove_epsilon_and_add_self_loops(
            word_fsa
        )

        path_to_utt_map = self.shape.row_ids(1)

        if hasattr(lattice, "aux_labels"):
            # lattice has token IDs as labels and word IDs as aux_labels.
            # inv_lattice has word IDs as labels and token IDs as aux_labels
            inv_lattice = k2.invert(lattice)
            inv_lattice = k2.arc_sort(inv_lattice)
        else:
            inv_lattice = k2.arc_sort(lattice)

        if inv_lattice.shape[0] == 1:
            path_lattice = _intersect_device(
                inv_lattice,
                word_fsa_with_epsilon_loops,
                b_to_a_map=torch.zeros_like(path_to_utt_map),
                sorted_match_a=True,
            )
        else:
            path_lattice = _intersect_device(
                inv_lattice,
                word_fsa_with_epsilon_loops,
                b_to_a_map=path_to_utt_map,
                sorted_match_a=True,
            )

        # path_lattice has word IDs as labels and token IDs as aux_labels
        path_lattice = k2.top_sort(k2.connect(path_lattice))

        one_best = k2.shortest_path(
            path_lattice, use_double_scores=use_double_scores
        )

        one_best = k2.invert(one_best)
        # Now one_best has token IDs as labels and word IDs as aux_labels

        return Nbest(fsa=one_best, shape=self.shape)

    def compute_am_scores(self) -> k2.RaggedTensor:
        """Compute AM scores of each linear FSA (i.e., each path within
        an utterance).

        Hint:
          `self.fsa.scores` contains two parts: acoustic scores (AM scores)
          and n-gram language model scores (LM scores).

        Caution:
          We require that ``self.fsa`` has an attribute ``lm_scores``.

        Returns:
          Return a ragged tensor with 2 axes [utt][path_scores].
          Its dtype is torch.float64.
        """
        saved_scores = self.fsa.scores

        # The `scores` of every arc consists of `am_scores` and `lm_scores`
        self.fsa.scores = self.fsa.scores - self.fsa.lm_scores

        am_scores = self.fsa.get_tot_scores(
            use_double_scores=True, log_semiring=False
        )
        self.fsa.scores = saved_scores

        return k2.RaggedTensor(self.shape, am_scores)

    def compute_lm_scores(self) -> k2.RaggedTensor:
        """Compute LM scores of each linear FSA (i.e., each path within
        an utterance).

        Hint:
          `self.fsa.scores` contains two parts: acoustic scores (AM scores)
          and n-gram language model scores (LM scores).

        Caution:
          We require that ``self.fsa`` has an attribute ``lm_scores``.

        Returns:
          Return a ragged tensor with 2 axes [utt][path_scores].
          Its dtype is torch.float64.
        """
        saved_scores = self.fsa.scores

        # The `scores` of every arc consists of `am_scores` and `lm_scores`
        self.fsa.scores = self.fsa.lm_scores

        lm_scores = self.fsa.get_tot_scores(
            use_double_scores=True, log_semiring=False
        )
        self.fsa.scores = saved_scores

        return k2.RaggedTensor(self.shape, lm_scores)

    def tot_scores(self) -> k2.RaggedTensor:
        """Get total scores of FSAs in this Nbest.

        Note:
          Since FSAs in Nbest are just linear FSAs, log-semiring
          and tropical semiring produce the same total scores.

        Returns:
          Return a ragged tensor with two axes [utt][path_scores].
          Its dtype is torch.float64.
        """
        scores = self.fsa.get_tot_scores(
            use_double_scores=True, log_semiring=False
        )
        return k2.RaggedTensor(self.shape, scores)

    def build_levenshtein_graphs(self) -> k2.Fsa:
        """Return an FsaVec with axes [utt][state][arc]."""
        word_ids = get_texts(self.fsa, return_ragged=True)
        return k2.levenshtein_graph(word_ids)


def one_best_decoding(
    lattice: k2.Fsa,
    use_double_scores: bool = True,
) -> k2.Fsa:
    """Get the best path from a lattice.

    Args:
      lattice:
        The decoding lattice returned by :func:`get_lattice`.
      use_double_scores:
        True to use double precision floating point in the computation.
        False to use single precision.
    Return:
      An FsaVec containing linear paths.
    """
    best_path = k2.shortest_path(lattice, use_double_scores=use_double_scores)
    return best_path


def nbest_decoding(
    lattice: k2.Fsa,
    num_paths: int,
    use_double_scores: bool = True,
    lattice_score_scale: float = 1.0,
) -> k2.Fsa:
    """It implements something like CTC prefix beam search using n-best lists.

    The basic idea is to first extract `num_paths` paths from the given lattice,
    build a word sequence from these paths, and compute the total scores
    of the word sequence in the tropical semiring. The one with the max score
    is used as the decoding output.

    Caution:
      Don't be confused by `best` in the name `n-best`. Paths are selected
      **randomly**, not by ranking their scores.

    Hint:
      This decoding method is for demonstration only and it does
      not produce a lower WER than :func:`one_best_decoding`.

    Args:
      lattice:
        The decoding lattice, e.g., can be the return value of
        :func:`get_lattice`. It has 3 axes [utt][state][arc].
      num_paths:
        It specifies the size `n` in n-best. Note: Paths are selected randomly
        and those containing identical word sequences are removed and only one
        of them is kept.
      use_double_scores:
        True to use double precision floating point in the computation.
        False to use single precision.
      lattice_score_scale:
        It's the scale applied to the `lattice.scores`. A smaller value
        leads to more unique paths at the risk of missing the correct path.
    Returns:
      An FsaVec containing **linear** FSAs. It axes are [utt][state][arc].
    """
    nbest = Nbest.from_lattice(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=use_double_scores,
        lattice_score_scale=lattice_score_scale,
    )
    # nbest.fsa.scores contains 0s

    nbest = nbest.intersect(lattice)
    # now nbest.fsa.scores gets assigned

    # max_indexes contains the indexes for the path with the maximum score
    # within an utterance.
    max_indexes = nbest.tot_scores().argmax()

    best_path = k2.index_fsa(nbest.fsa, max_indexes)
    return best_path


def nbest_oracle(
    lattice: k2.Fsa,
    num_paths: int,
    ref_texts: List[str],
    word_table: k2.SymbolTable,
    use_double_scores: bool = True,
    lattice_score_scale: float = 0.5,
    oov: str = "<UNK>",
) -> Dict[str, List[List[int]]]:
    """Select the best hypothesis given a lattice and a reference transcript.

    The basic idea is to extract `num_paths` paths from the given lattice,
    unique them, and select the one that has the minimum edit distance with
    the corresponding reference transcript as the decoding output.

    The decoding result returned from this function is the best result that
    we can obtain using n-best decoding with all kinds of rescoring techniques.

    This function is useful to tune the value of `lattice_score_scale`.

    Args:
      lattice:
        An FsaVec with axes [utt][state][arc].
        Note: We assume its `aux_labels` contains word IDs.
      num_paths:
        The size of `n` in n-best.
      ref_texts:
        A list of reference transcript. Each entry contains space(s)
        separated words
      word_table:
        It is the word symbol table.
      use_double_scores:
        True to use double precision for computation. False to use
        single precision.
      lattice_score_scale:
        It's the scale applied to the lattice.scores. A smaller value
        yields more unique paths.
      oov:
        The out of vocabulary word.
    Return:
      Return a dict. Its key contains the information about the parameters
      when calling this function, while its value contains the decoding output.
      `len(ans_dict) == len(ref_texts)`
    """
    device = lattice.device

    nbest = Nbest.from_lattice(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=use_double_scores,
        lattice_score_scale=lattice_score_scale,
    )

    hyps = nbest.build_levenshtein_graphs()

    oov_id = word_table[oov]
    word_ids_list = []
    for text in ref_texts:
        word_ids = []
        for word in text.split():
            if word in word_table:
                word_ids.append(word_table[word])
            else:
                word_ids.append(oov_id)
        word_ids_list.append(word_ids)

    refs = k2.levenshtein_graph(word_ids_list, device=device)

    levenshtein_alignment = k2.levenshtein_alignment(
        refs=refs,
        hyps=hyps,
        hyp_to_ref_map=nbest.shape.row_ids(1),
        sorted_match_ref=True,
    )

    tot_scores = levenshtein_alignment.get_tot_scores(
        use_double_scores=False, log_semiring=False
    )
    ragged_tot_scores = k2.RaggedTensor(nbest.shape, tot_scores)

    max_indexes = ragged_tot_scores.argmax()

    best_path = k2.index_fsa(nbest.fsa, max_indexes)
    return best_path


def rescore_with_n_best_list(
    lattice: k2.Fsa,
    G: k2.Fsa,
    num_paths: int,
    lm_scale_list: List[float],
    lattice_score_scale: float = 1.0,
    use_double_scores: bool = True,
) -> Dict[str, k2.Fsa]:
    """Rescore an n-best list with an n-gram LM.
    The path with the maximum score is used as the decoding output.

    Args:
      lattice:
        An FsaVec with axes [utt][state][arc]. It must have the following
        attributes: ``aux_labels`` and ``lm_scores``. Its labels are
        token IDs and ``aux_labels`` word IDs.
      G:
        An FsaVec containing only a single FSA. It is an n-gram LM.
      num_paths:
        Size of nbest list.
      lm_scale_list:
        A list of float representing LM score scales.
      lattice_score_scale:
        Scale to be applied to ``lattice.score`` when sampling paths
        using ``k2.random_paths``.
      use_double_scores:
        True to use double precision during computation. False to use
        single precision.
    Returns:
      A dict of FsaVec, whose key is an lm_scale and the value is the
      best decoding path for each utterance in the lattice.
    """
    device = lattice.device

    assert len(lattice.shape) == 3
    assert hasattr(lattice, "aux_labels")
    assert hasattr(lattice, "lm_scores")

    assert G.shape == (1, None, None)
    assert G.device == device
    assert hasattr(G, "aux_labels") is False

    nbest = Nbest.from_lattice(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=use_double_scores,
        lattice_score_scale=lattice_score_scale,
    )
    # nbest.fsa.scores are all 0s at this point

    nbest = nbest.intersect(lattice)
    # Now nbest.fsa has its scores set
    assert hasattr(nbest.fsa, "lm_scores")

    am_scores = nbest.compute_am_scores()

    nbest = nbest.intersect(G)
    # Now nbest contains only lm scores
    lm_scores = nbest.tot_scores()

    ans = dict()
    for lm_scale in lm_scale_list:
        tot_scores = am_scores.values / lm_scale + lm_scores.values
        tot_scores = k2.RaggedTensor(nbest.shape, tot_scores)
        max_indexes = tot_scores.argmax()
        best_path = k2.index_fsa(nbest.fsa, max_indexes)
        key = f"lm_scale_{lm_scale}"
        ans[key] = best_path
    return ans


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
        An FsaVec with axes [utt][state][arc]. Its `aux_lables` are word IDs.
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
    assert hasattr(lattice, "lm_scores")
    assert G_with_epsilon_loops.shape == (1, None, None)

    device = lattice.device
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

    max_loop_count = 10
    loop_count = 0
    while loop_count <= max_loop_count:
        loop_count += 1
        try:
            rescoring_lattice = k2.intersect_device(
                G_with_epsilon_loops,
                inv_lattice,
                b_to_a_map,
                sorted_match_a=True,
            )
            rescoring_lattice = k2.top_sort(k2.connect(rescoring_lattice))
            break
        except RuntimeError as e:
            logging.info(f"Caught exception:\n{e}\n")
            logging.info(
                f"num_arcs before pruning: {inv_lattice.arcs.num_elements()}"
            )

            # NOTE(fangjun): The choice of the threshold 1e-9 is arbitrary here
            # to avoid OOM. You may need to fine tune it.
            inv_lattice = k2.prune_on_arc_post(inv_lattice, 1e-9, True)
            logging.info(
                f"num_arcs after pruning: {inv_lattice.arcs.num_elements()}"
            )
    if loop_count > max_loop_count:
        logging.info("Return None as the resulting lattice is too large")
        return None

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
        key = f"lm_scale_{lm_scale}"
        ans[key] = best_path
    return ans


def rescore_with_attention_decoder(
    lattice: k2.Fsa,
    num_paths: int,
    model: torch.nn.Module,
    memory: torch.Tensor,
    memory_key_padding_mask: Optional[torch.Tensor],
    sos_id: int,
    eos_id: int,
    lattice_score_scale: float = 1.0,
    ngram_lm_scale: Optional[float] = None,
    attention_scale: Optional[float] = None,
    use_double_scores: bool = True,
) -> Dict[str, k2.Fsa]:
    """This function extracts `num_paths` paths from the given lattice and uses
    an attention decoder to rescore them. The path with the highest score is
    the decoding output.

    Args:
      lattice:
        An FsaVec with axes [utt][state][arc].
      num_paths:
        Number of paths to extract from the given lattice for rescoring.
      model:
        A transformer model. See the class "Transformer" in
        conformer_ctc/transformer.py for its interface.
      memory:
        The encoder memory of the given model. It is the output of
        the last torch.nn.TransformerEncoder layer in the given model.
        Its shape is `(T, N, C)`.
      memory_key_padding_mask:
        The padding mask for memory with shape `(N, T)`.
      sos_id:
        The token ID for SOS.
      eos_id:
        The token ID for EOS.
      lattice_score_scale:
        It's the scale applied to `lattice.scores`. A smaller value
        leads to more unique paths at the risk of missing the correct path.
      ngram_lm_scale:
        Optional. It specifies the scale for n-gram LM scores.
      attention_scale:
        Optional. It specifies the scale for attention decoder scores.
    Returns:
      A dict of FsaVec, whose key contains a string
      ngram_lm_scale_attention_scale and the value is the
      best decoding path for each utterance in the lattice.
    """
    nbest = Nbest.from_lattice(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=use_double_scores,
        lattice_score_scale=lattice_score_scale,
    )
    # nbest.fsa.scores are all 0s at this point

    nbest = nbest.intersect(lattice)
    # Now nbest.fsa has its scores set.
    # Also, nbest.fsa inherits the attributes from `lattice`.
    assert hasattr(nbest.fsa, "lm_scores")

    am_scores = nbest.compute_am_scores()
    ngram_lm_scores = nbest.compute_lm_scores()

    # The `tokens` attribute is set inside `compile_hlg.py`
    assert hasattr(nbest.fsa, "tokens")
    assert isinstance(nbest.fsa.tokens, torch.Tensor)

    path_to_utt_map = nbest.shape.row_ids(1).to(torch.long)
    # the shape of memory is (T, N, C), so we use axis=1 here
    expanded_memory = memory.index_select(1, path_to_utt_map)

    if memory_key_padding_mask is not None:
        # The shape of memory_key_padding_mask is (N, T), so we
        # use axis=0 here.
        expanded_memory_key_padding_mask = memory_key_padding_mask.index_select(
            0, path_to_utt_map
        )
    else:
        expanded_memory_key_padding_mask = None

    # remove axis corresponding to states.
    tokens_shape = nbest.fsa.arcs.shape().remove_axis(1)
    tokens = k2.RaggedTensor(tokens_shape, nbest.fsa.tokens)
    tokens = tokens.remove_values_leq(0)
    token_ids = tokens.tolist()

    nll = model.decoder_nll(
        memory=expanded_memory,
        memory_key_padding_mask=expanded_memory_key_padding_mask,
        token_ids=token_ids,
        sos_id=sos_id,
        eos_id=eos_id,
    )
    assert nll.ndim == 2
    assert nll.shape[0] == len(token_ids)

    attention_scores = -nll.sum(dim=1)

    if ngram_lm_scale is None:
        ngram_lm_scale_list = [0.01, 0.05, 0.08]
        ngram_lm_scale_list += [0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 1.0]
        ngram_lm_scale_list += [1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.0]
    else:
        ngram_lm_scale_list = [ngram_lm_scale]

    if attention_scale is None:
        attention_scale_list = [0.01, 0.05, 0.08]
        attention_scale_list += [0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 1.0]
        attention_scale_list += [1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.0]
    else:
        attention_scale_list = [attention_scale]

    ans = dict()
    for n_scale in ngram_lm_scale_list:
        for a_scale in attention_scale_list:
            tot_scores = (
                am_scores.values
                + n_scale * ngram_lm_scores.values
                + a_scale * attention_scores
            )
            ragged_tot_scores = k2.RaggedTensor(nbest.shape, tot_scores)
            max_indexes = ragged_tot_scores.argmax()
            best_path = k2.index_fsa(nbest.fsa, max_indexes)

            key = f"ngram_lm_scale_{n_scale}_attention_scale_{a_scale}"
            ans[key] = best_path
    return ans
