"""Tools for homograph disambiguation

Authors
 * Artem Ploujnikov 2021
"""
import torch
from torch import nn


class SubsequenceLoss(nn.Module):
    """
    A loss function for a specific word in the output, used in
    the homograph disambiguation task

    The approach is as follows:
    1. Arrange only the target words from the original batch into a
    single tensor
    2. Find the word index of each target word
    3. Compute the beginnings and endings of words in the predicted
    sequences. The assumption is that the model has been trained well
    enough to identify word boundaries with a simple argmax without
    having to perform a beam search.

    Important! This loss can be used for fine-tuning only
    The model is expected to be able to already be able
    to correctly predict word boundaries


    Arguments
    ---------
    seq_cost: callable
        the loss to be used on the extracted subsequences

    word_separator: int
        the index of the "space" character (in phonemes)
    """

    def __init__(self, seq_cost, word_separator=0):
        super().__init__()
        self.seq_cost = seq_cost
        self._subsequence_extractor = SubsequenceExtractor(word_separator)

    @property
    def word_separator(self):
        return self._subsequence_extractor.word_separator

    @word_separator.setter
    def word_separator(self, value):
        self._subsequence_extractor.word_separator = value

    def forward(
        self, phns, phn_lens, p_seq, subsequence_phn_start, subsequence_phn_end
    ):
        (
            p_seq_subsequence,
            phns_subsequence,
            subsequence_lengths,
        ) = self._subsequence_extractor(
            phns, phn_lens, p_seq, subsequence_phn_start, subsequence_phn_end
        )
        return self.seq_cost(
            p_seq_subsequence, phns_subsequence, subsequence_lengths
        )


class SubsequenceExtractor:
    """
    A utility class to help extract subsequences out of a batch
    of sequences

    Arguments
    ---------
    word_separator: int
        the index of the word separator
    """

    def __init__(self, word_separator=0):
        self.word_separator = word_separator

    def __call__(
        self, phns, phn_lens, p_seq, subsequence_phn_start, subsequence_phn_end
    ):
        return self.extract_seq(
            phns, phn_lens, p_seq, subsequence_phn_start, subsequence_phn_end
        )

    def extract_seq(
        self, phns, phn_lens, p_seq, subsequence_phn_start, subsequence_phn_end
    ):
        # Compute subsequence lengths and the longest length
        subsequence_lengths = subsequence_phn_end - subsequence_phn_start
        longest_subsequence = subsequence_lengths.max()

        # Pad the sequence axis to make sure the "distance" from the start of
        # each subsequence to the end of the sequence is at least as long
        # as the longest subsequence (e.g. subsequence = homograph)
        gap = (
            ((subsequence_phn_end + longest_subsequence).max() - phn_lens.max())
            .int()
            .item()
        )
        if gap > 0:
            phns = torch.nn.functional.pad(phns, (0, gap))
        p_seq_pad = (gap + longest_subsequence).item()
        p_seq_edge = p_seq.size(1)
        p_seq = torch.nn.functional.pad(p_seq, (0, 0, 0, p_seq_pad))

        # Copy only the subsequences from the targets and inputs
        # into new tensors
        subsequence_phn_start_unsq = subsequence_phn_start.unsqueeze(-1)
        range_phns = torch.arange(phns.size(1), device=phns.device).expand_as(
            phns
        )
        range_phns_subsequence = torch.arange(
            longest_subsequence, device=phns.device
        ).expand(phns.size(0), longest_subsequence)
        match = (range_phns >= subsequence_phn_start_unsq) & (
            range_phns < subsequence_phn_start_unsq + longest_subsequence
        )
        phns_subsequence = phns[match].reshape(range_phns_subsequence.shape)
        phns_subsequence[
            range_phns_subsequence >= subsequence_lengths.unsqueeze(-1)
        ] = 0.0

        # Count the words in predictions
        target_word_indexes = self._get_target_word_indexes(
            phns, range_phns, subsequence_phn_start_unsq
        )

        # Determine where the predicted subsequences start and end
        word_start, word_end = self._get_word_boundaries(
            p_seq, target_word_indexes, p_seq_edge
        )
        p_seq_range = (
            torch.arange(p_seq.size(1), device=p_seq.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand_as(p_seq)
        )
        word_start_unsq = word_start.unsqueeze(-1).unsqueeze(-1)
        word_end_unsq = word_end.unsqueeze(-1).unsqueeze(-1)
        phn_match = (p_seq_range >= word_start_unsq) & (
            p_seq_range < word_start_unsq + longest_subsequence
        )
        p_seq_subsequence = p_seq[phn_match].view(
            p_seq.size(0), longest_subsequence, p_seq.size(-1)
        )
        p_seq_subsequence_range = (
            torch.arange(
                p_seq_subsequence.size(1), device=p_seq_subsequence.device
            )
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand_as(p_seq_subsequence)
        )
        p_seq_subsequence[
            p_seq_subsequence_range >= (word_end_unsq - word_start_unsq)
        ] = 0.0
        return (
            p_seq_subsequence,
            phns_subsequence,
            subsequence_lengths / longest_subsequence,
        )

    def _get_target_word_indexes(self, phns, range_phns, start):
        word_boundaries = (range_phns < start) & (phns == self.word_separator)
        return word_boundaries.sum(dim=-1)

    def _get_word_boundaries(self, p_seq, word_indexes, edge):
        # Find all spaces in the tensor
        word_boundaries = p_seq.argmax(-1) == self.word_separator

        # Find which word a given position in the tensor belongs in
        words = word_boundaries.cumsum(dim=-1)

        # Compute an auxiliary range tensor to help determine
        # word boundaries
        words_range = torch.arange(
            words.size(-1), device=words.device
        ).expand_as(words)

        index_match = words == word_indexes.unsqueeze(-1)

        start = self._get_positions(index_match, words_range, torch.min, edge)
        end = self._get_positions(index_match, words_range, torch.max, 0)
        return start, end

    def _get_positions(
        self, index_match, words_range, aggregation, no_match_value
    ):
        positions = torch.where(index_match, words_range, no_match_value)
        positions = aggregation(positions, dim=-1).values
        return torch.where(positions == 0, 0, positions + 1)

    def extract_hyps(self, ref_seq, hyps, subsequence_phn_start):
        """
        Extracts a subsequnce from hypotheses (e.g. the result of a beam

        search) based on a refernece sequence, which can be either a sequence of phonemes (the target during training)

        Arguments
        ---------
        ref_seq: torch.Tensor
            a reference sequence (e.g. phoneme targets)
        hyps: list
            a batch of hypotheses, a list of list of
            integer indices (usually of phonemes)
        subsequence_phn_start: torch.tensor

        """
        range_phns = torch.arange(
            ref_seq.size(1), device=ref_seq.device
        ).expand_as(ref_seq)
        target_word_indexes = self._get_target_word_indexes(
            ref_seq, range_phns, subsequence_phn_start.unsqueeze(-1)
        )
        separator_indexes = [
            [-1]
            + [
                idx
                for idx, phn in enumerate(item_hyps)
                if phn == self.word_separator
            ]
            + [None]
            for item_hyps in hyps
        ]
        return [
            self._extract_hyp_word(
                item_hyps, item_separtaor_indexes, word_index
            )
            for item_hyps, item_separtaor_indexes, word_index in zip(
                hyps, separator_indexes, target_word_indexes
            )
        ]

    def _extract_hyp_word(self, hyps, separator_indexes, word_index):
        if word_index < len(separator_indexes):
            left = separator_indexes[word_index]
            if left is None:
                return ""
            left += 1
            right = separator_indexes[word_index + 1]
            result = hyps[left:right]
        else:
            result = []
        return result
