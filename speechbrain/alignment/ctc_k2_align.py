"""
Alignment for CTC models using k2.

Authors
 * Zeyu Zhao 2023
"""
import k2
import torch


def align_batch(log_probs, targets, input_lens, target_lens):
    """
    Args:
      log_probs:
        A 3-D tensor of shape (batch_size, max_seq_length, num_classes).
        It is the output of neural networks.
      targets:
        A 2-D tensor of shape (batch_size, max_target_length).
        targets[i] is a sequence of token IDs.
      input_lens:
        A 1-D tensor of shape (batch_size,). It is the lengths of log_probs.
      target_lens:
        A 1-D tensor of shape (batch_size,). It is the lengths of targets.
    Returns:
      alignment:
        A list of list of int. alignment[i] is the alignment (the token id sequence including <blank>) for the i-th sequence.
    """
    input_lens = (input_lens * log_probs.shape[1]).round().int()
    target_lens = (target_lens * targets.shape[1]).round().int()
    batch_size = log_probs.shape[0]

    max_token_id = log_probs.shape[-1] - 1
    ctc_topo = k2.ctc_topo(max_token_id, modified=False,
                           device=log_probs.device)

    # convert targets to k2.FsaVec
    labels = [targets[i, :target_lens[i]].tolist()
              for i in range(len(target_lens))]
    label_fsas = k2.linear_fsa(labels, device=log_probs.device)

    labels_fsas_with_self_loops = k2.remove_epsilon_and_add_self_loops(
        label_fsas)

    labels_fsas_with_self_loops = k2.arc_sort(labels_fsas_with_self_loops)

    graph = k2.compose(ctc_topo, labels_fsas_with_self_loops,
                       treat_epsilons_specially=False)

    assert graph.requires_grad is False

    supervision_segments = torch.tensor(
        [[i, 0, input_lens[i]] for i in range(batch_size)],
        device="cpu",
        dtype=torch.int32,
    )

    dense_fsa_vec = k2.DenseFsaVec(
        log_probs,
        supervision_segments,
    )

    lattice = k2.intersect_dense(graph, dense_fsa_vec, 10)
    one_best = k2.one_best_decoding(lattice)
    alignment = [one_best[i].labels.tolist()[:-1] for i in range(batch_size)]

    return alignment
