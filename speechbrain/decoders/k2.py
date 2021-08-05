
# Copyright (c)  2021  Tsinghua University (Authors: Rong Fu)


import k2
import k2.ragged as k2r
import torch
from snowfall.training.ctc_graph import build_ctc_topo2
from speechbrain.pretrained import EncoderDecoderASR
from typing import List

def load_model():
    model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-transformer-transformerlm-librispeech",
        savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
        #  run_opts={'device': 'cuda:0'},
    )
    return model

def get_texts(best_paths: k2.Fsa) -> List[List[int]]:
    """Extract the texts (as word IDs) from the best-path FSAs.
    Args:
      best_paths:
        A k2.Fsa with best_paths.arcs.num_axes() == 3, i.e.
        containing multiple FSAs, which is expected to be the result
        of k2.shortest_path (otherwise the returned values won't
        be meaningful).
    Returns:
      Returns a list of lists of int, containing the label sequences we
      decoded.
    """
    if isinstance(best_paths.aux_labels, k2.RaggedInt):
        # remove 0's and -1's.
        aux_labels = k2r.remove_values_leq(best_paths.aux_labels, 0)
        aux_shape = k2r.compose_ragged_shapes(
            best_paths.arcs.shape(), aux_labels.shape()
        )

        # remove the states and arcs axes.
        aux_shape = k2r.remove_axis(aux_shape, 1)
        aux_shape = k2r.remove_axis(aux_shape, 1)
        aux_labels = k2.RaggedInt(aux_shape, aux_labels.values())
    else:
        # remove axis corresponding to states.
        aux_shape = k2r.remove_axis(best_paths.arcs.shape(), 1)
        aux_labels = k2.RaggedInt(aux_shape, best_paths.aux_labels)
        # remove 0's and -1's.
        aux_labels = k2r.remove_values_leq(aux_labels, 0)

    assert aux_labels.num_axes() == 2
    return k2r.to_list(aux_labels)

def ctc_decoding(log_probs: torch.Tensor,
                 ctc_topo: k2.Fsa,
                 search_beam: float,
                 output_beam: float,
                 min_active_states: int,
                 max_active_states: int
                 ) -> (List[List[int]], List[float]):
    '''building an FSA decoder based on ctc topology.
       Args:
           log_probs:
                    torch.Tensor of dimension [B, T, N].
                            where, B = Batchsize,
                                T = the number of frames,
                                N = number of tokens
                    It represents the probability distribution over tokens, which
                    is the output of an encoder network.
           ctc_topo:
                    a CTC topology fst that represents a specific topology used to
                    convert the network outputs to a sequence of tokens.
           search_beam:
                    Decoding beam, e.g. 20.  Smaller is faster, larger is more exact
                    (less pruning). This is the default value; it may be modified by
                    `min_active_states` and `max_active_states`.
           output_beam:
                    Pruning beam for the output of intersection (vs. best path);
                    equivalent to kaldi's lattice-beam.  E.g. 8.
           min_active_states:
                    Minimum number of FSA states that are allowed to be active on
                    any given frame for any given intersection/composition task.
                    This is advisory, in that it will try not to have fewer than
                    this number active. Set it to zero if there is no constraint.
           max_active_states:
                    Maximum number of FSA states that are allowed to be active on
                    any given frame for any given intersection/composition task.
                    This is advisory, in that it will try not to exceed that but
                    may not always succeed. You can use a very large number if no
                    constraint is needed.

       Return:
           predicted_tokens : a list of lists of int,
                This list contains batch_size number. Each inside list contains
                a list stores all the hypothesis for this sentence.
           scores : a list of float64
                This list contains the total score of each sequences.
    '''

    batchnum = log_probs.size(0)

    supervisions = []
    for i in range(batchnum):
        supervisions.append([i, 0, log_probs.size(1)])
    supervision_segments = torch.tensor(supervisions, dtype=torch.int32)

    supervision_segments = torch.clamp(supervision_segments, min=0)

    dense_fsa_vec = k2.DenseFsaVec(log_probs, supervision_segments)

    lattices = k2.intersect_dense_pruned(ctc_topo, dense_fsa_vec,
                                         search_beam=search_beam,
                                         output_beam=output_beam,
                                         min_active_states=min_active_states,
                                         max_active_states=max_active_states
                                         )

    best_paths = k2.shortest_path(lattices, True)

    predicted_tokens = get_texts(best_paths)

    # sum the scores for each sequence
    scores = best_paths.get_tot_scores(True, True)

    return predicted_tokens, scores.tolist()
