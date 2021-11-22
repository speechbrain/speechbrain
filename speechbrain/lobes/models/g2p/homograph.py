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
    word_separator_base: str
        the index of word separators used in unprocessed
        targets (if different, used with tokenizations)
    """

    def __init__(self, seq_cost, word_separator=0, word_separator_base=0):
        super().__init__()
        self.seq_cost = seq_cost
        self._subsequence_extractor = SubsequenceExtractor(
            word_separator, word_separator_base)

    @property
    def word_separator(self):
        """
        The word separator being used
        """
        return self._subsequence_extractor.word_separator

    @word_separator.setter
    def word_separator(self, value):
        """
        Sets the word separator
        """
        self._subsequence_extractor.word_separator = value

    @property
    def word_separator_base(self):
        """
        The word separator being used
        """
        return self._subsequence_extractor.word_separator_base

    @word_separator.setter
    def word_separator_base(self, value):
        """
        Sets the base word separator
        """
        self._subsequence_extractor.word_separator_base = value


    def forward(
        self,
        phns,
        phn_lens,
        p_seq,
        subsequence_phn_start,
        subsequence_phn_end,
        phns_base,
        phn_lens_base
    ):
        """
        Evaluates the subsequence loss

        Arguments
        ---------
        phns: torch.Tensor
            the phoneme tensor (batch x length)
        phn_lens: torch.Tensor
            the phoneme length tensor
        p_seq: torch.Tensor
            the output phoneme probability tensor
            (batch x length x phns)
        subsequence_phn_start: torch.Tensor
            the beginning of the target subsequence
            (i.e. the homograph)
        subsequence_phn_end: torch.Tensor
            the end of the target subsequence
            (i.e. the homograph)
        phns_base: torch.Tensor
            the phoneme tensor (not preprocessed)
        phn_lens_base: torch.Tensor
            the phoneme lengths (not preprocessed)

        Returns
        -------
        loss: torch.Tensor
            the loss tensor
        """
        (
            p_seq_subsequence,
            phns_subsequence,
            subsequence_lengths,
        ) = self._subsequence_extractor(
            phns,
            phn_lens,
            p_seq,
            subsequence_phn_start,
            subsequence_phn_end,
            phns_base,
            phn_lens_base
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
        the index of the word separator (used in p_seq)
    word_separator_base int
        the index of word separators used in unprocessed
        targets (if different)
    """

    def __init__(self, word_separator=0, word_separator_base=None):
        self.word_separator = word_separator
        if word_separator_base is None:
            word_separator_base = word_separator
        self.word_separator_base = word_separator_base

    def __call__(
        self, *args, **kwargs
    ):
        return self.extract_seq(*args, **kwargs)

    def extract_seq(
        self,
        phns,
        phn_lens,
        p_seq,
        subsequence_phn_start,
        subsequence_phn_end,
        phns_base=None,
        phn_base_lens=None
    ):
        """
        Extracts the subsequence from the complete sequence

        phns: torch.Tensor
            the phoneme tensor (batch x length)
        phn_lens: torch.Tensor
            the phoneme length tensor
        p_seq: torch.Tensor
            the output phoneme probability tensor
            (batch x length x phns)
        subsequence_phn_start: torch.Tensor
            the beginning of the target subsequence
            (i.e. the homograph)
        subsequence_phn_end: torch.Tensor
            the end of the target subsequence
            (i.e. the homograph)
        phns_base: torch.Tensor
            the phoneme tensor (not preprocessed)
        phn_base_lens: torch.Tensor
            the phoneme lengths (not preprocessed)

        """
        has_base = False
        if phns_base is None and phn_base_lens is None:
            phns_base = phns
            phn_base_lens = phn_lens
        elif phns_base is None or phn_base_lens is None:
            raise ValueError("phn_base and phn_lens_base, if provided, should be provided together")
        else:
            has_base = True

        p_seq_edge = p_seq.size(1)
        phns_edge = phns.size(1)

        # Compute subsequence lengths and the longest length
        subsequence_lengths = subsequence_phn_end - subsequence_phn_start
        longest_subsequence = subsequence_lengths.max()

        # Pad the sequence axis to make sure the "distance" from the start of
        # each subsequence to the end of the sequence is at least as long
        # as the longest subsequence (e.g. subsequence = homograph)
        phns = self._pad_subsequence(phns, longest_subsequence)
        phns_base = self._pad_subsequence(phns_base, longest_subsequence)
        #p_seq_pad = (gap + longest_subsequence + 1).item()
        p_seq_pad = p_seq.size(1)
        p_seq = torch.nn.functional.pad(p_seq, (0, 0, 0, p_seq_pad))

        # Copy only the subsequences from the targets and inputs
        # into new tensors
        subsequence_phn_start_unsq = subsequence_phn_start.unsqueeze(-1)
        range_phns_base = torch.arange(phns_base.size(1), device=phns_base.device).expand_as(
            phns_base
        )
        range_phns_subsequence = torch.arange(
            longest_subsequence, device=phns.device
        ).expand(phns.size(0), longest_subsequence)
        # Count the words in predictions
        target_word_indexes = self._get_target_word_indexes(
            phns_base, range_phns_base, subsequence_phn_start_unsq,
            self.word_separator_base
        )
        if has_base:
            # Needed if tokenization or any other transformation was used
            phns_subsequence = self._get_phns_subsequence(phns, target_word_indexes, longest_subsequence, phns_edge)
        else:
            # If phns and phns_base are the same, there is no need to re-detect word boundaries
            match = (range_phns_base >= subsequence_phn_start_unsq) & (
                range_phns_base < subsequence_phn_start_unsq + longest_subsequence
            )
            phns_subsequence = phns[match].reshape(range_phns_subsequence.shape)

            phns_subsequence[
                range_phns_subsequence >= subsequence_lengths.unsqueeze(-1)
            ] = 0.0

        p_seq_subsequence = self._get_p_seq_subsequence(
            p_seq, target_word_indexes, longest_subsequence, p_seq_edge)

        return (
            p_seq_subsequence,
            phns_subsequence,
            subsequence_lengths / longest_subsequence,
        )

    def _pad_subsequence(self, phns, longest_subsequence):
        if longest_subsequence > 0:
            phns = torch.nn.functional.pad(phns, (0, longest_subsequence))
        return phns


    def _get_phns_subsequence(self, phns, target_word_indexes, longest_subsequence, edge):
        word_start, word_end = self._get_word_boundaries(
            phns, target_word_indexes, edge
        )
        word_start_unsq = word_start.unsqueeze(-1)
        word_end_unsq = word_end.unsqueeze(-1)
        phns_range = (
            torch.arange(phns.size(1), device=phns.device)
            .unsqueeze(0)
            .expand_as(phns)
        )

        phn_match = (phns_range >= word_start_unsq) & (
            phns_range < word_start_unsq + longest_subsequence
        )
        phns_subsequence = phns[phn_match].view(
            phns.size(0), longest_subsequence
        )
        phns_subsequence_range = (
            torch.arange(
                phns_subsequence.size(1), device=phns_subsequence.device
            )
            .unsqueeze(0)
            .expand_as(phns_subsequence)
        )
        phns_subsequence[
            phns_subsequence_range >= (word_end_unsq - word_start_unsq)
        ] = 0.0
        return phns_subsequence


    def _get_p_seq_subsequence(self, p_seq, target_word_indexes, longest_subsequence, edge):
        # Determine where the predicted subsequences start and end

        word_start, word_end = self._get_word_boundaries(
            p_seq, target_word_indexes, edge
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
        try:
            p_seq_subsequence = p_seq[phn_match].view(
                p_seq.size(0), longest_subsequence, p_seq.size(-1)
            )
        except:
            breakpoint()
            raise
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
        return p_seq_subsequence


    def _get_target_word_indexes(self, phns, range_phns, start, word_separator):
        word_boundaries = (range_phns < start) & (phns == word_separator)
        word_indexes =  word_boundaries.sum(dim=-1)
        return word_indexes

    def _get_word_boundaries(self, seq, word_indexes, edge, word_separator=None):
        if word_separator is None:
            word_separator = self.word_separator
        # Find all spaces in the tensor
        tokens = seq.argmax(-1) if seq.dim() == 3 else seq
        word_boundaries = tokens == word_separator

        # Find which word a given position in the tensor belongs in
        words = word_boundaries.cumsum(dim=-1)

        # Compute an auxiliary range tensor to help determine
        # word boundaries
        words_range = torch.arange(
            words.size(-1), device=words.device
        ).expand_as(words)

        index_match = words == word_indexes.unsqueeze(-1)

        start = self._get_positions(
            index_match, words_range, torch.min, edge)
        end = self._get_positions(
            index_match, words_range, torch.max, 0)
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
            the index of the beginning of the subsequence
        """
        range_phns = torch.arange(
            ref_seq.size(1), device=ref_seq.device
        ).expand_as(ref_seq)
        target_word_indexes = self._get_target_word_indexes(
            ref_seq, range_phns, subsequence_phn_start.unsqueeze(-1),
            self.word_separator_base
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