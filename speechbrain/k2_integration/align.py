"""Force alignment using k2 for CTC models.

Author:
    * Zeyu Zhao 2023
"""
import torch
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

try:
    import k2
except ImportError:
    MSG = "Cannot import k2, so training and decoding with k2 will not work.\n"
    MSG += "Please refer to https://k2-fsa.github.io/k2/installation/from_wheels.html for installation.\n"
    MSG += "You may also find the precompiled wheels for your platform at https://download.pytorch.org/whl/torch_stable.html"
    raise ImportError(MSG)


def align(
    log_prob: torch.Tensor,
    log_prob_len: torch.Tensor,
    targets: list,
) -> List[List[int]]:
    """Align targets to log_probs.

    Arguments
    ---------
        log_prob: torch.Tensor
            A tensor of shape (N, T, C) containing the log-probabilities.
            Please make sure that index 0 of the C dimension corresponds
            to the blank symbol.
        log_prob_len: torch.Tensor
            A tensor of shape (N,) containing the lengths of the log_probs.
            This is needed because the log_probs may have been padded.
            All elements in this tensor must be integers and <= T, and
            in descending order.
        targets: list
            A list of list of integers containing the targets.
            Note that the targets should not contain the blank symbol.
            The blank symbol is assumed to be index 0 in log_prob.
    Returns
    -------
        List
        List of lists of integers containing the alignments.
    """
    # Basic checks.
    assert log_prob.ndim == 3
    assert log_prob_len.ndim == 1
    assert log_prob.shape[0] == log_prob_len.shape[0]
    assert isinstance(targets, list)
    assert isinstance(targets[0], list)
    assert log_prob.shape[0] == len(targets)

    N, T, C = log_prob.shape

    graph = k2.ctc_graph(targets)

    lattice = k2.get_lattice(
        log_prob=log_prob,
        log_prob_len=log_prob_len,
        decoding_graph=graph,
    )

    best_path = k2.shortest_path(lattice, use_double_scores=True)
    labels = best_path.labels

    alignments = []
    alignment = []
    for e in labels.tolist():
        if e == -1:
            alignments.append(alignment)
            alignment = []
        else:
            alignment.append(e)

    return alignments


if __name__ == "__main__":
    # test align function
    log_prob = torch.tensor([[[0.1, 0.6, 0.1, 0.1, 0.1],
                              [0.1, 0.1, 0.6, 0.1, 0.1],
                              [0.1, 0.1, 0.1, 0.6, 0.1],
                              [0.6, 0.1, 0.1, 0.1, 0.1],
                              [0.1, 0.1, 0.1, 0.1, 0.6],
                              [0.6, 0.1, 0.1, 0.1, 0.1]]])
    log_prob = torch.log(log_prob)
    log_prob_len = torch.tensor([6])
    targets = [[1, 2, 3, 4]]
    alignment = align(log_prob, log_prob_len, targets)
    print("Simple test alignment:", alignment)

    # test alignment with different lengths
    log_prob = torch.tensor([[[0.1, 0.6, 0.1, 0.1, 0.1],
                              [0.1, 0.1, 0.6, 0.1, 0.1],
                              [0.1, 0.1, 0.1, 0.6, 0.1],
                              [0.6, 0.1, 0.1, 0.1, 0.1],
                              [0.1, 0.1, 0.1, 0.1, 0.6],
                              [0.6, 0.1, 0.1, 0.1, 0.1]],
                             [[0.1, 0.6, 0.1, 0.1, 0.1],
                                 [0.1, 0.1, 0.6, 0.1, 0.1],
                                 [0.1, 0.1, 0.1, 0.6, 0.1],
                                 [0.6, 0.1, 0.1, 0.1, 0.1],
                                 [0.1, 0.1, 0.1, 0.1, 0.1],
                                 [0.1, 0.1, 0.1, 0.1, 0.1]]])
    log_prob = torch.log(log_prob)
    log_prob_len = torch.tensor([6, 4])
    targets = [[1, 2, 3, 4], [1, 2, 3]]
    alignment = align(log_prob, log_prob_len, targets)
    print("Batch test alignment:", alignment)
