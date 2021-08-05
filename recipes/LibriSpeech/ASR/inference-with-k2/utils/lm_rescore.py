# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

from typing import Dict
from typing import List
from typing import Optional

import math

import k2
import torch


def _intersect_device(a_fsas: k2.Fsa, b_fsas: k2.Fsa, b_to_a_map: torch.Tensor,
                      sorted_match_a: bool):
    '''This is a wrapper of k2.intersect_device and its purpose is to split
    b_fsas into several batches and process each batch separately to avoid
    CUDA OOM error.

    The arguments and return value of this function are the same as
    k2.intersect_device.
    '''
    # NOTE: You can decrease batch_size in case of CUDA out of memory error.
    batch_size = 500
    num_fsas = b_fsas.shape[0]
    if num_fsas <= batch_size:
        return k2.intersect_device(a_fsas,
                                   b_fsas,
                                   b_to_a_map=b_to_a_map,
                                   sorted_match_a=sorted_match_a)

    num_batches = int(math.ceil(float(num_fsas) / batch_size))
    splits = []
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, num_fsas)
        splits.append((start, end))

    ans = []
    for start, end in splits:
        indexes = torch.arange(start, end).to(b_to_a_map)

        fsas = k2.index(b_fsas, indexes)
        b_to_a = k2.index(b_to_a_map, indexes)
        path_lats = k2.intersect_device(a_fsas,
                                        fsas,
                                        b_to_a_map=b_to_a,
                                        sorted_match_a=sorted_match_a)
        ans.append(path_lats)

    return k2.cat(ans)


def compute_am_scores(lats: k2.Fsa, word_fsas_with_epsilon_loops: k2.Fsa,
                      path_to_seq_map: torch.Tensor) -> torch.Tensor:
    '''Compute AM scores of n-best lists (represented as word_fsas).

    Args:
      lats:
        An FsaVec, which is the output of `k2.intersect_dense_pruned`.
        It must have the attribute `lm_scores`.
      word_fsas_with_epsilon_loops:
        An FsaVec representing a n-best list. Note that it has been processed
        by `k2.add_epsilon_self_loops`.
      path_to_seq_map:
        A 1-D torch.Tensor with dtype torch.int32. path_to_seq_map[i] indicates
        which sequence the i-th Fsa in word_fsas_with_epsilon_loops belongs to.
        path_to_seq_map.numel() == word_fsas_with_epsilon_loops.arcs.dim0().
    Returns:
      Return a 1-D torch.Tensor containing the AM scores of each path.
      `ans.numel() == word_fsas_with_epsilon_loops.shape[0]`
    '''
    device = lats.device
    assert len(lats.shape) == 3
    assert hasattr(lats, 'lm_scores')

    # k2.compose() currently does not support b_to_a_map. To void
    # replicating `lats`, we use k2.intersect_device here.
    #
    # lats has phone IDs as `labels` and word IDs as aux_labels, so we
    # need to invert it here.
    inverted_lats = k2.invert(lats)

    # Now the `labels` of inverted_lats are word IDs (a 1-D torch.Tensor)
    # and its `aux_labels` are phone IDs ( a k2.RaggedInt with 2 axes)

    # Remove its `aux_labels` since it is not needed in the
    # following computation
    del inverted_lats.aux_labels
    inverted_lats = k2.arc_sort(inverted_lats)

    am_path_lats = _intersect_device(inverted_lats,
                                     word_fsas_with_epsilon_loops,
                                     b_to_a_map=path_to_seq_map,
                                     sorted_match_a=True)

    am_path_lats = k2.top_sort(k2.connect(am_path_lats.to('cpu')))

    # The `scores` of every arc consists of `am_scores` and `lm_scores`
    am_path_lats.scores = am_path_lats.scores - am_path_lats.lm_scores

    am_scores = am_path_lats.get_tot_scores(True, True)

    return am_scores


@torch.no_grad()
def rescore_with_n_best_list(lats: k2.Fsa, G: k2.Fsa, num_paths: int,
                             lm_scale_list: List[float]) -> Dict[str, k2.Fsa]:
    '''Decode using n-best list with LM rescoring.

    `lats` is a decoding lattice, which has 3 axes. This function first
    extracts `num_paths` paths from `lats` for each sequence using
    `k2.random_paths`. The `am_scores` of these paths are computed.
    For each path, its `lm_scores` is computed using `G` (which is an LM).
    The final `tot_scores` is the sum of `am_scores` and `lm_scores`.
    The path with the greatest `tot_scores` within a sequence is used
    as the decoding output.

    Args:
      lats:
        An FsaVec. It can be the output of `k2.intersect_dense_pruned`.
      G:
        An FsaVec representing the language model (LM). Note that it
        is an FsaVec, but it contains only one Fsa.
      num_paths:
        It is the size `n` in `n-best` list.
      lm_scale_list:
        A list containing lm_scale values.
    Returns:
      A dict of FsaVec, whose key is a lm_scale and the value represents the
      best decoding path for each sequence in the lattice.
    '''
    device = lats.device

    assert len(lats.shape) == 3
    assert hasattr(lats, 'aux_labels')
    assert hasattr(lats, 'lm_scores')

    assert G.shape == (1, None, None)
    assert G.device == device
    assert hasattr(G, 'aux_labels') is False

    # First, extract `num_paths` paths for each sequence.
    # paths is a k2.RaggedInt with axes [seq][path][arc_pos]
    paths = k2.random_paths(lats, num_paths=num_paths, use_double_scores=True)
    if paths.shape().tot_size(1) == 0:
        print('Get None paths.')
        return None
    
    #print('Raggedint paths shape: ', paths.shape())
    # word_seqs is a k2.RaggedInt sharing the same shape as `paths`
    # but it contains word IDs. Note that it also contains 0s and -1s.
    # The last entry in each sublist is -1.
    word_seqs = k2.index(lats.aux_labels, paths)

    # Remove epsilons and -1 from word_seqs
    word_seqs = k2.ragged.remove_values_leq(word_seqs, 0)

    # Remove repeated sequences to avoid redundant computation later.
    #
    # unique_word_seqs is still a k2.RaggedInt with 3 axes [seq][path][word]
    # except that there are no repeated paths with the same word_seq
    # within a seq.
    #
    # num_repeats is also a k2.RaggedInt with 2 axes containing the
    # multiplicities of each path.
    # num_repeats.num_elements() == unique_word_seqs.num_elements()
    #
    # Since k2.ragged.unique_sequences will reorder paths within a seq,
    # `new2old` is a 1-D torch.Tensor mapping from the output path index
    # to the input path index.
    # new2old.numel() == unique_word_seqs.num_elements()
    unique_word_seqs, num_repeats, new2old = k2.ragged.unique_sequences(
        word_seqs, need_num_repeats=True, need_new2old_indexes=True)

    seq_to_path_shape = k2.ragged.get_layer(unique_word_seqs.shape(), 0)

    # path_to_seq_map is a 1-D torch.Tensor.
    # path_to_seq_map[i] is the seq to which the i-th path
    # belongs.
    path_to_seq_map = seq_to_path_shape.row_ids(1)

    # Remove the seq axis.
    # Now unique_word_seqs has only two axes [path][word]
    unique_word_seqs = k2.ragged.remove_axis(unique_word_seqs, 0)

    # word_fsas is an FsaVec with axes [path][state][arc]
    word_fsas = k2.linear_fsa(unique_word_seqs)

    word_fsas_with_epsilon_loops = k2.add_epsilon_self_loops(word_fsas)

    am_scores = compute_am_scores(lats, word_fsas_with_epsilon_loops,
                                  path_to_seq_map)

    # Now compute lm_scores
    b_to_a_map = torch.zeros_like(path_to_seq_map)
    lm_path_lats = _intersect_device(G,
                                     word_fsas_with_epsilon_loops,
                                     b_to_a_map=b_to_a_map,
                                     sorted_match_a=True)
    lm_path_lats = k2.top_sort(k2.connect(lm_path_lats))
    lm_scores = lm_path_lats.get_tot_scores(use_double_scores=True, log_semiring=False)

    ans = dict()
    for lm_scale in lm_scale_list:
        tot_scores = am_scores.to(device) / lm_scale + lm_scores.to(device)

        # Remember that we used `k2.ragged.unique_sequences` to remove repeated
        # paths to avoid redundant computation in `k2.intersect_device`.
        # Now we use `num_repeats` to correct the scores for each path.
        #
        # NOTE(fangjun): It is commented out as it leads to a worse WER
        # tot_scores = tot_scores * num_repeats.values()

        # TODO(fangjun): We may need to add `k2.RaggedDouble`
        ragged_tot_scores = k2.RaggedFloat(seq_to_path_shape,
                                           tot_scores.to(torch.float32))
        argmax_indexes = k2.ragged.argmax_per_sublist(ragged_tot_scores)

        # Use k2.index here since argmax_indexes' dtype is torch.int32
        best_path_indexes = k2.index(new2old, argmax_indexes)

        paths_2axes = k2.ragged.remove_axis(paths, 0)

        # best_path is a k2.RaggedInt with 2 axes [path][arc_pos]
        best_paths = k2.index(paths_2axes, best_path_indexes)

        # labels is a k2.RaggedInt with 2 axes [path][phone_id]
        # Note that it contains -1s.
        labels = k2.index(lats.labels.contiguous(), best_paths)

        labels = k2.ragged.remove_values_eq(labels, -1)

        # lats.aux_labels is a k2.RaggedInt tensor with 2 axes, so
        # aux_labels is also a k2.RaggedInt with 2 axes
        aux_labels = k2.index(lats.aux_labels, best_paths.values())

        best_path_fsas = k2.linear_fsa(labels)
        best_path_fsas.aux_labels = aux_labels

        key = f'lm_scale_{lm_scale}'
        ans[key] = best_path_fsas

    return ans

@torch.no_grad()
def rescore_with_whole_lattice(lats: k2.Fsa, G_with_epsilon_loops: k2.Fsa,
                               lm_scale_list: List[float]
                              ) -> Dict[str, k2.Fsa]:
    '''Use whole lattice to rescore.

    Args:
      lats:
        An FsaVec It can be the output of `k2.intersect_dense_pruned`.
      G_with_epsilon_loops:
        An FsaVec representing the language model (LM). Note that it
        is an FsaVec, but it contains only one Fsa.
      lm_scale_list:
        A list containing lm_scale values.
    Returns:
      A dict of FsaVec, whose key is a lm_scale and the value represents the
      best decoding path for each sequence in the lattice.
    '''
    assert len(lats.shape) == 3
    assert hasattr(lats, 'lm_scores')
    assert G_with_epsilon_loops.shape == (1, None, None)

    device = lats.device
    lats.scores = lats.scores - lats.lm_scores
    # We will use lm_scores from G, so remove lats.lm_scores here
    del lats.lm_scores
    assert hasattr(lats, 'lm_scores') is False

    #  lats.scores = scores / lm_scale
    # Now, lats.scores contains only am_scores

    # inverted_lats has word IDs as labels.
    # Its aux_labels are phone IDs, which is a ragged tensor k2.RaggedInt
    inverted_lats = k2.invert(lats)
    num_seqs = lats.shape[0]

    b_to_a_map = torch.zeros(num_seqs, device=device, dtype=torch.int32)
    try:
        rescoring_lats = k2.intersect_device(G_with_epsilon_loops,
                                             inverted_lats,
                                             b_to_a_map,
                                             sorted_match_a=True)
    except RuntimeError as e:
        print(f'Caught exception:\n{e}\n')
        print(f'Number of FSAs: {inverted_lats.shape[0]}')
        print('num_arcs before pruning: ', inverted_lats.arcs.num_elements())

        # NOTE(fangjun): The choice of the threshold 0.01 is arbitrary here
        # to avoid OOM. We may need to fine tune it.
        inverted_lats = k2.prune_on_arc_post(inverted_lats, 0.001, True)
        print('num_arcs after pruning: ', inverted_lats.arcs.num_elements())

        rescoring_lats = k2.intersect_device(G_with_epsilon_loops,
                                             inverted_lats,
                                             b_to_a_map,
                                             sorted_match_a=True)

    rescoring_lats = k2.top_sort(k2.connect(rescoring_lats.to('cpu')))

    # inv_lats has phone IDs as labels
    # and word IDs as aux_labels.
    inv_lats = k2.invert(rescoring_lats)

    ans = dict()
    #
    # The following implements
    # scores = (scores - lm_scores)/lm_scale + lm_scores
    #        = scores/lm_scale + lm_scores*(1 - 1/lm_scale)
    #
    saved_am_scores = inv_lats.scores - inv_lats.lm_scores
    for lm_scale in lm_scale_list:
        am_scores = saved_am_scores / lm_scale
        inv_lats.scores = am_scores + inv_lats.lm_scores

        best_paths = k2.shortest_path(inv_lats, use_double_scores=True)
        key = f'lm_scale_{lm_scale}'
        ans[key] = best_paths
    return ans
