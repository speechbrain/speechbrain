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

    Example
    -------
    >>> import torch
    >>> from speechbrain.lobes.models.g2p.homograph import SubsequenceLoss
    >>> from speechbrain.nnet.losses import nll_loss
    >>> loss = SubsequenceLoss(
    ...     seq_cost=nll_loss
    ... )
    >>> phns = torch.Tensor(
    ...     [[1, 2, 0, 1, 3, 0, 2, 1, 0],
    ...      [2, 1, 3, 0, 1, 2, 0, 3, 2]]
    ... )
    >>> phn_lens = torch.IntTensor([8, 9])
    >>> subsequence_phn_start = torch.IntTensor([3, 4])
    >>> subsequence_phn_end = torch.IntTensor([5, 7])
    >>> p_seq = torch.Tensor([
    ...     [[0., 1., 0., 0.],
    ...      [0., 0., 1., 0.],
    ...      [1., 0., 0., 0.],
    ...      [0., 1., 0., 0.],
    ...      [0., 0., 0., 1.],
    ...      [1., 0., 0., 0.],
    ...      [0., 0., 1., 0.],
    ...      [0., 1., 0., 0.],
    ...      [1., 0., 0., 0.]],
    ...     [[0., 0., 1., 0.],
    ...      [0., 1., 0., 0.],
    ...      [0., 0., 0., 1.],
    ...      [1., 0., 0., 0.],
    ...      [0., 1., 0., 0.],
    ...      [0., 0., 1., 0.],
    ...      [1., 0., 0., 0.],
    ...      [0., 0., 0., 1.],
    ...      [0., 0., 1., 0.]]
    ... ])
    >>> loss_value = loss(
    ...    phns,
    ...    phn_lens,
    ...    p_seq,
    ...    subsequence_phn_start,
    ...    subsequence_phn_end
    ... )
    >>> loss_value
    tensor(-0.8000)
    """

    def __init__(self, seq_cost, word_separator=0, word_separator_base=0):
        super().__init__()
        self.seq_cost = seq_cost
        self._subsequence_extractor = SubsequenceExtractor(
            word_separator, word_separator_base
        )

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
    def word_separator_base(self, value):  # noqa
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
        phns_base=None,
        phn_lens_base=None,
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
            phn_lens_base,
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

    Example
    -------
    >>> import torch
    >>> from speechbrain.lobes.models.g2p.homograph import SubsequenceExtractor
    >>> extractor = SubsequenceExtractor()
    >>> phns = torch.Tensor(
    ...     [[1, 2, 0, 1, 3, 0, 2, 1, 0],
    ...      [2, 1, 3, 0, 1, 2, 0, 3, 2]]
    ... )
    >>> phn_lens = torch.IntTensor([8, 9])
    >>> subsequence_phn_start = torch.IntTensor([3, 4])
    >>> subsequence_phn_end = torch.IntTensor([5, 7])
    >>> p_seq = torch.Tensor([
    ...     [[0., 1., 0., 0.],
    ...      [0., 0., 1., 0.],
    ...      [1., 0., 0., 0.],
    ...      [0., 1., 0., 0.],
    ...      [0., 0., 0., 1.],
    ...      [1., 0., 0., 0.],
    ...      [0., 0., 1., 0.],
    ...      [0., 1., 0., 0.],
    ...      [1., 0., 0., 0.]],
    ...     [[0., 0., 1., 0.],
    ...      [0., 1., 0., 0.],
    ...      [0., 0., 0., 1.],
    ...      [1., 0., 0., 0.],
    ...      [0., 1., 0., 0.],
    ...      [0., 0., 1., 0.],
    ...      [1., 0., 0., 0.],
    ...      [0., 0., 0., 1.],
    ...      [0., 0., 1., 0.]]
    ... ])
    >>> extractor.extract_seq(
    ...    phns,
    ...    phn_lens,
    ...    p_seq,
    ...    subsequence_phn_start,
    ...    subsequence_phn_end
    ... )
    (tensor([[[0., 1., 0., 0.],
             [0., 0., 0., 1.],
             [0., 0., 0., 0.]],
    <BLANKLINE>
            [[0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [0., 0., 0., 0.]]]), tensor([[1., 3., 0.],
            [1., 2., 0.]]), tensor([0.6667, 1.0000]))
    """

    def __init__(self, word_separator=0, word_separator_base=None):
        self.word_separator = word_separator
        if word_separator_base is None:
            word_separator_base = word_separator
        self.word_separator_base = word_separator_base

    def __call__(self, *args, **kwargs):
        return self.extract_seq(*args, **kwargs)

    def extract_seq(
        self,
        phns,
        phn_lens,
        p_seq,
        subsequence_phn_start,
        subsequence_phn_end,
        phns_base=None,
        phn_base_lens=None,
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

        Returns
        -------
        p_seq_subsequence: torch.Tensor
            the output subsequence (of probabilities)
        phns_subsequence: torch.Tensor
            the target subsequence
        subsequence_lengths: torch.Tensor
            subsequence lengths, expressed as a fraction
            of the tensor's last dimension

        """
        has_base = False
        if phns_base is None and phn_base_lens is None:
            phns_base = phns
            phn_base_lens = phn_lens
        elif phns_base is None or phn_base_lens is None:
            raise ValueError(
                "phn_base and phn_lens_base, if provided, should be provided together"
            )
        else:
            has_base = True

        p_seq_edge = p_seq.size(1)
        phns_edge = (phns.size(1) * phn_lens).long().unsqueeze(-1)

        # Compute subsequence lengths and the longest length
        subsequence_lengths = subsequence_phn_end - subsequence_phn_start
        longest_subsequence = subsequence_lengths.max()

        # Pad the sequence axis to make sure the "distance" from the start of
        # each subsequence to the end of the sequence is at least as long
        # as the longest subsequence (e.g. subsequence = homograph)
        phns = self._pad_subsequence(phns, longest_subsequence)
        phns_base = self._pad_subsequence(phns_base, longest_subsequence)
        # p_seq_pad = (gap + longest_subsequence + 1).item()
        p_seq_pad = p_seq.size(1)
        p_seq = torch.nn.functional.pad(p_seq, (0, 0, 0, p_seq_pad))

        # Copy only the subsequences from the targets and inputs
        # into new tensors
        subsequence_phn_start_unsq = subsequence_phn_start.unsqueeze(-1)
        range_phns_base = torch.arange(
            phns_base.size(1), device=phns_base.device
        ).expand_as(phns_base)
        range_phns_subsequence = torch.arange(
            longest_subsequence, device=phns.device
        ).expand(phns.size(0), longest_subsequence)
        # Count the words in predictions
        target_word_indexes = self._get_target_word_indexes(
            phns_base,
            range_phns_base,
            subsequence_phn_start_unsq,
            self.word_separator_base,
            phn_lens=phn_base_lens,
        )
        if has_base:
            # Needed if tokenization or any other transformation was used
            phns_subsequence, subsequence_lengths = self._get_phns_subsequence(
                phns, target_word_indexes, longest_subsequence, phns_edge
            )
        else:
            # If phns and phns_base are the same, there is no need to re-detect word boundaries
            match = (range_phns_base >= subsequence_phn_start_unsq) & (
                range_phns_base
                < subsequence_phn_start_unsq + longest_subsequence
            )
            phns_subsequence = phns[match].reshape(range_phns_subsequence.shape)

            phns_subsequence[
                range_phns_subsequence >= subsequence_lengths.unsqueeze(-1)
            ] = 0.0

        p_seq_subsequence = self._get_p_seq_subsequence(
            p_seq, target_word_indexes, longest_subsequence, p_seq_edge
        )

        return (
            p_seq_subsequence,
            phns_subsequence,
            subsequence_lengths / longest_subsequence,
        )

    def _pad_subsequence(self, sequence, longest_subsequence):
        """Pads a subsequence to the length of the longest subsequence

        Arguments
        ---------
        sequence: torch.tensor
            the sequence to be padded
        longest_subsequence: int
            the length of the longest subsequence
        """
        if longest_subsequence > 0:
            sequence = torch.nn.functional.pad(
                sequence, (0, longest_subsequence)
            )
        return sequence

    def _get_phns_subsequence(
        self, phns, target_word_indexes, longest_subsequence, edge
    ):
        """Extracts a subsequence

        Arguments
        ---------
        phns: torch.Tensor
            a tensor of phoneme indexes
        target_word_indexes: torch.Tensor
            a tensor of word indexes to extract, zero-based
            (e.g.) torch.IntTensor([2, 3])  means extracting
            the third word from the first sample and the
            fourth word from the second sample
        longest_subsequence: int
            the length of the longest subsequence
        edge: int
            the index of the "edge" of the sequence

        Returns
        -------
        phn_subsequence: torch.Tensor
            a tensor with only the target words
        subsequence_lengths: torch.Tensor
            the lengths of the extracted words
        """
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
        subsequence_lengths = torch.minimum(
            word_end - word_start, torch.tensor(phns_subsequence.size(1))
        )
        return phns_subsequence, subsequence_lengths

    def _get_p_seq_subsequence(
        self, p_seq, target_word_indexes, longest_subsequence, edge
    ):
        """Extracts a subsequence out of a tensor of probabilities

        Arguments
        ---------
        p_seq: torch.Tensor
            a tensor of phoneme probabilities
            (batch x sequence index x phoneme index)
        target_word_indexes: torch.Tensor
            a tensor of word indexes to extract, zero-based
            (e.g.) torch.IntTensor([2, 3])  means extracting
            the third word from the first sample and the
            fourth word from the second sample
        longest_subsequence: int
            the length of the longest subsequence
        edge: int
            the index of the "edge" of the sequence

        Returns
        -------
        p_seq_subsequence: torch.Tensor
            a probability tensor composed of the phoneme
            probabilities for target words only
        """
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
        return p_seq_subsequence

    def _get_target_word_indexes(
        self, phns, range_phns, start, word_separator, phn_lens=None
    ):
        """Computes the target word indexes

        Arguments
        ---------
        phns: torch.Tensor
            a phoneme batch tensor
        range_phns: torch.Tensor
            a range tensor over thephoneme sequence
        start: torch.Tensor
            the beginning of the subsequence
        word_separator: int
            the word separator being used

        Returns
        -------
        word_indexes: torch.Tensor
            the word index tensor

        """
        end_of_sequence = (
            (range_phns == ((phn_lens).unsqueeze(-1) * phns.size(1)).long())
            if phn_lens is not None
            else False
        )
        word_boundaries = (range_phns < start) & (
            (phns == word_separator) | end_of_sequence
        )
        word_indexes = word_boundaries.sum(dim=-1)
        return word_indexes

    def _get_word_boundaries(
        self, seq, word_indexes, edge, word_separator=None
    ):
        """Determines the word boundaries for the specified
        word indexes within a sequence

        Arguments
        ---------
        seq: torch.Tensor
            a sequence (phonemes or graphemes)
        word_indexes:
            the word indexes
        edge: int
            a tensor indicating the last position
        word_separator: int
            the word separator token

        Returns
        -------
        start: torch.Tensor
            word start indexes
        end: torch.Tensor
            word end indexes
        """
        if word_separator is None:
            word_separator = self.word_separator
        # Find all spaces in the tensor
        tokens = seq.argmax(-1) if seq.dim() == 3 else seq

        # Compute an auxiliary range tensor to help determine
        # word boundaries
        words_range = torch.arange(
            tokens.size(-1), device=tokens.device
        ).expand_as(tokens)

        word_boundaries = (tokens == word_separator) | (words_range == edge)

        # Find which word a given position in the tensor belongs in
        words = word_boundaries.cumsum(dim=-1)

        index_match = words == word_indexes.unsqueeze(-1)

        start = self._get_positions(index_match, words_range, torch.min, edge)
        end = self._get_positions(index_match, words_range, torch.max, 0)
        return start, end

    def _get_positions(
        self, index_match, words_range, aggregation, no_match_value
    ):
        """A helper method to calculate start or end positions corresponding
        to specific words

        Arguments
        ---------
        index_match: torch.Tensor
            a mask where positions matching the word index are
            indicated as a 1 and the remaining positions are 0

        words_range: torch.Tensor
            a range tensor over the tokens
        aggregation: callable
            the aggregation to use (torch.min or torch.max)
        no_match_value: int
            the value to output if no match is found (this could
            happen when searching in model outputs rather than
            in source data)

        """
        positions = torch.where(index_match, words_range, no_match_value)
        positions = aggregation(positions, dim=-1).values
        return torch.where(positions == 0, 0, positions + 1)

    def extract_hyps(
        self, ref_seq, hyps, subsequence_phn_start, use_base=False
    ):
        """Extracts a subsequnce from hypotheses (e.g. the result of a beam
        search) based on a refernece sequence, which can be either a sequence of phonemes (the target during training)
        Arguments
        ---------
        ref_seq: torch.Tensor
            a reference sequence (e.g. phoneme targets)
        hyps: list
            a batch of hypotheses, a list of list of
            integer indices (usually of phonemes)
        subsequence_phn_start: torch.tensor
            the index of the beginning of the subsequence to
        use_base: bool
            whether to use the raw (token) space for word separators
        """
        range_phns = torch.arange(
            ref_seq.size(1), device=ref_seq.device
        ).expand_as(ref_seq)
        target_word_indexes = self._get_target_word_indexes(
            ref_seq,
            range_phns,
            subsequence_phn_start.unsqueeze(-1),
            self.word_separator_base if use_base else self.word_separator,
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
        result = [
            self._extract_hyp_word(
                item_hyps, item_separtaor_indexes, word_index
            )
            for item_hyps, item_separtaor_indexes, word_index in zip(
                hyps, separator_indexes, target_word_indexes
            )
        ]
        return result

    def _extract_hyp_word(self, hyps, separator_indexes, word_index):
        """Extracts a single word out of a hypothesis sequence

        Arguments
        ---------
        hyps: list
            a hypotheses list (or tensor)
        separator_indexes: torch.Tensor
            a tensor of word separators
        word_index: int
            the index of the word to eb retrieved

        Returns
        -------
        result: list|str
            the extracted word
        """
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
