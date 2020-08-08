"""
Utils from the uis-rnn repo: https://github.com/google/uis-rnn/blob/master/uisrnn/utils.py
"""

import numpy as np
import torch
from torch import autograd

torch.manual_seed(1234)
np.random.seed(1234)


def estimate_transition_bias(cluster_ids, smooth=1):
    """Estimate the transition bias.

  Args:
    cluster_id: Either a list of cluster indicator sequences, or a single
      concatenated sequence. The former is strongly preferred, since the
      transition_bias estimated from the latter will be inaccurate.
    smooth: int or float - Smoothing coefficient, avoids -inf value in np.log
      in the case of a sequence with a single speaker and division by 0 in the
      case of empty sequences. Using a small value for smooth decreases the
      bias in the calculation of transition_bias but can also lead to underflow
      in some remote cases, larger values are safer but less accurate.

  Returns:
    bias: Flipping coin head probability.
    bias_denominator: The denominator of the bias, used for multiple calls to
      fit().
  """
    transit_num = smooth
    bias_denominator = 2 * smooth
    for cluster_id_seq in cluster_ids:
        for entry in range(len(cluster_id_seq) - 1):
            transit_num += cluster_id_seq[entry] != cluster_id_seq[entry + 1]
            bias_denominator += 1
    bias = transit_num / bias_denominator
    return bias, bias_denominator


def sample_sequences(xvectors, cluster_id, num_permutations):
    unique_id = np.unique(cluster_id)
    sub_sequences = []
    seq_lengths = []
    if num_permutations and num_permutations > 1:
        for i in unique_id:
            idx_set = np.where(cluster_id == i)[0]
            sampled_idx_sets = sample_permuted_segments(
                idx_set, num_permutations
            )
            for j in range(num_permutations):
                sub_sequences.append(xvectors[sampled_idx_sets[j], :])
                seq_lengths.append(len(idx_set) + 1)
    # else:
    #     for i in unique_id:
    #         idx_set = np.where(cluster_id == i)
    #         sub_sequences.append(sequence[idx_set, :][0])
    #         seq_lengths.append(len(idx_set[0]) + 1)
    return sub_sequences, seq_lengths


def sample_permuted_segments(index_sequence, number_samples):
    """Sample sequences with permuted blocks.

  Args:
    index_sequence: (integer array, size: L)
      - subsequence index
      For example, index_sequence = [1,2,6,10,11,12].
    number_samples: (integer)
      - number of subsampled block-preserving permuted sequences.
      For example, number_samples = 5

  Returns:
    sampled_index_sequences: (a list of numpy arrays) - a list of subsampled
      block-preserving permuted sequences. For example,
    ```
    sampled_index_sequences =
    [[10,11,12,1,2,6],
     [6,1,2,10,11,12],
     [1,2,10,11,12,6],
     [6,1,2,10,11,12],
     [1,2,6,10,11,12]]
    ```
      The length of "sampled_index_sequences" is "number_samples".
  """
    segments = []
    if len(index_sequence) == 1:
        segments.append(index_sequence)
    else:
        prev = 0
        for i in range(len(index_sequence) - 1):
            if index_sequence[i + 1] != index_sequence[i] + 1:
                segments.append(index_sequence[prev : (i + 1)])
                prev = i + 1
            if i + 1 == len(index_sequence) - 1:
                segments.append(index_sequence[prev:])
    # sample permutations
    sampled_index_sequences = []
    for _ in range(number_samples):
        segments_array = []
        permutation = np.random.permutation(len(segments))
        for permutation_item in permutation:
            segments_array.append(segments[permutation_item])
        sampled_index_sequences.append(np.concatenate(segments_array))
    return sampled_index_sequences


def pack_sequence(
    sub_sequences, seq_lengths, batch_size, observation_dim, device
):
    """Pack sequences for training.

  Args:
    sub_sequences: A list of numpy array, with obsevation vector from the same
      cluster in the same list.
    seq_lengths: The length of each cluster (+1).
    batch_size: int or None - Run batch learning if batch_size is None. Else,
      run online learning with specified batch size.
    observation_dim: int - dimension for observation vectors
    device: str - Your device. E.g., `cuda:0` or `cpu`.

  Returns:
    packed_rnn_input: (PackedSequence object) packed rnn input
    rnn_truth: ground truth
  """
    num_clusters = len(seq_lengths)
    sorted_seq_lengths = np.sort(seq_lengths)[::-1]
    permute_index = np.argsort(seq_lengths)[::-1]

    if batch_size is None:
        rnn_input = np.zeros(
            (sorted_seq_lengths[0], num_clusters, observation_dim)
        )
        for i in range(num_clusters):
            rnn_input[1 : sorted_seq_lengths[i], i, :] = sub_sequences[
                permute_index[i]
            ]
        rnn_input = autograd.Variable(torch.from_numpy(rnn_input).float()).to(
            device
        )
        packed_rnn_input = torch.nn.utils.rnn.pack_padded_sequence(
            rnn_input, sorted_seq_lengths, batch_first=False
        )
    else:
        mini_batch = np.sort(np.random.choice(num_clusters, batch_size))
        rnn_input = np.zeros(
            (sorted_seq_lengths[mini_batch[0]], batch_size, observation_dim)
        )
        for i in range(batch_size):
            rnn_input[1 : sorted_seq_lengths[mini_batch[i]], i, :] = (
                sub_sequences[permute_index[mini_batch[i]]].cpu().numpy()
            )
        rnn_input = autograd.Variable(torch.from_numpy(rnn_input).float()).to(
            device
        )
        packed_rnn_input = torch.nn.utils.rnn.pack_padded_sequence(
            rnn_input, sorted_seq_lengths[mini_batch], batch_first=False
        )
    # ground truth is the shifted input
    rnn_truth = rnn_input[1:, :, :]
    return packed_rnn_input, rnn_truth


def resize_sequence(sequence, cluster_id, num_permutations=None):
    """Resize sequences for packing and batching.

  Args:
    sequence: (real numpy matrix, size: seq_len*obs_size) - observed sequence
    cluster_id: (numpy vector, size: seq_len) - cluster indicator sequence
    num_permutations: int - Number of permutations per utterance sampled.

  Returns:
    sub_sequences: A list of numpy array, with obsevation vector from the same
      cluster in the same list.
    seq_lengths: The length of each cluster (+1).
  """
    # merge sub-sequences that belong to a single cluster to a single sequence
    unique_id = np.unique(cluster_id)
    sub_sequences = []
    seq_lengths = []
    if num_permutations and num_permutations > 1:
        for i in unique_id:
            idx_set = np.where(cluster_id == i)[0]
            sampled_idx_sets = sample_permuted_segments(
                idx_set, num_permutations
            )
            for j in range(num_permutations):
                sub_sequences.append(sequence[sampled_idx_sets[j], :])
                seq_lengths.append(len(idx_set) + 1)
    else:
        for i in unique_id:
            idx_set = np.where(cluster_id == i)
            sub_sequences.append(sequence[idx_set, :][0])
            seq_lengths.append(len(idx_set[0]) + 1)
    return sub_sequences, seq_lengths
