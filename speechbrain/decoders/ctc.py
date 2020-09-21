"""
Decoders and output normalization for CTC

Authors
 * Mirco Ravanelli 2020
 * Aku Rouhe 2020
"""
import torch
import numpy as np
from itertools import groupby
from speechbrain.data_io.data_io import length_to_mask


class CTCPrefixScorer:
    def __init__(self, x, enc_lens, batch_size, blank_index, eos_index):
        self.blank_index = blank_index
        self.eos_index = eos_index
        self.max_enc_len = x.size(1)
        self.batch_size = batch_size
        self.beam_size = x.size(0) // batch_size

        # mask frames > enc_lens
        mask = 1 - length_to_mask(enc_lens)
        mask = mask.unsqueeze(-1).expand(-1, -1, x.size(-1)) == 1
        x.masked_fill_(mask, -np.inf)
        x[:, :, 0] = x[:, :, 0].masked_fill_(mask[:, :, 0], 0)

        # xnb: dim=0, nonblank posteriors, xb: dim=1, blank posteriors
        xnb = x.transpose(0, 1)
        xb = xnb[:, :, self.blank_index].unsqueeze(2).expand(-1, -1, x.size(-1))
        # (2, L, batch_size * beam_size, vocab_size)
        self.x = torch.stack([xnb, xb])
        print(self.x.shape)

    def forward_step(self, g, memory, candidates=None):
        """g: prefix"""
        device = g.device
        prefix_length = g.size(1)
        num_candidates = candidates.size(-1)

        if memory is None:
            # r_prev: (max_enc_len, 2, batch_size * beam_size)
            r_prev = torch.Tensor(
                self.max_enc_len, 2, self.batch_size * self.beam_size
            ).to(device)
            r_prev.fill_(-np.inf)
            # Accumulate blank posteriors at each step
            r_prev[:, 1] = torch.cumsum(self.x[0, :, :, self.blank_index], 0)
            r_prev = r_prev.view(-1, 2, self.batch_size * self.beam_size)
        else:
            r_prev = memory

        r = torch.Tensor(
            prefix_length, 2, self.beam_size, num_candidates
        ).fill_(-np.inf)

        # scores for candidates
        # x_ = self.x.unsqueeze(3).repeat(1, 1, 1, beam_size, 1).view(2, -1, beam_size * beatch_size, vocab_size)

        # TODO: phi = (prev_nb, prev_b), note that phi only depends on prefix g.

        # TODO: if last token of prefix g in candidates

        # TODO: Compute forward probabilities log(r_t^n(h)) and log(r_t^b(h))
        # 1. p(h|cur step is nonblank) = [p(prev step=y) + phi] * p(c)
        # 2. p(h|cur step is blank) = [p(prev step is blank) + p(prev step is nonblank)] * p(blank)
        # 3. psi = psi + phi * p(c)

        print(self.batch_size, self.x.shape, self.beam_size, r_prev, r.shape)

        return None


def filter_ctc_output(string_pred, blank_id=-1):
    """Apply CTC output merge and filter rules.

    Removes the blank symbol and output repetitions.

    Parameters
    ----------
    string_pred : list
        a list containing the output strings/ints predicted by the CTC system
    blank_id : int, string
        the id of the blank

    Returns
    ------
    list
        The output predicted by CTC without the blank symbol and
        the repetitions

    Example
    -------
        >>> string_pred = ['a','a','blank','b','b','blank','c']
        >>> string_out = filter_ctc_output(string_pred, blank_id='blank')
        >>> print(string_out)
        ['a', 'b', 'c']
    """

    if isinstance(string_pred, list):
        # Filter the repetitions
        string_out = [
            v
            for i, v in enumerate(string_pred)
            if i == 0 or v != string_pred[i - 1]
        ]

        # Remove duplicates
        string_out = [i[0] for i in groupby(string_out)]

        # Filter the blank symbol
        string_out = list(filter(lambda elem: elem != blank_id, string_out))
    else:
        raise ValueError("filter_ctc_out can only filter python lists")
    return string_out


def ctc_greedy_decode(probabilities, seq_lens, blank_id=-1):
    """
    Greedy decode a batch of probabilities and apply CTC rules

    Parameters
    ----------
    probabilities : torch.tensor
        Output probabilities (or log-probabilities) from network with shape
        [batch, probabilities, time]
    seq_lens : torch.tensor
        Relative true sequence lengths (to deal with padded inputs),
        longest sequence has length 1.0, others a value betwee zero and one
        shape [batch, lengths]
    blank_id : int, string
        The blank symbol/index. Default: -1. If a negative number is given,
        it is assumed to mean counting down from the maximum possible index,
        so that -1 refers to the maximum possible index.

    Returns
    -------
    list
        Outputs as Python list of lists, with "ragged" dimensions; padding
        has been removed.

    Example
    -------
        >>> import torch
        >>> probs = torch.tensor([[[0.3, 0.7], [0.0, 0.0]],
        ...                       [[0.2, 0.8], [0.9, 0.1]]])
        >>> lens = torch.tensor([0.51, 1.0])
        >>> blank_id = 0
        >>> ctc_greedy_decode(probs, lens, blank_id)
        [[1], [1]]
    """
    if isinstance(blank_id, int) and blank_id < 0:
        blank_id = probabilities.shape[-1] + blank_id
    batch_max_len = probabilities.shape[1]
    batch_outputs = []
    for seq, seq_len in zip(probabilities, seq_lens):
        actual_size = int(torch.round(seq_len * batch_max_len))
        scores, predictions = torch.max(seq.narrow(0, 0, actual_size), dim=1)
        out = filter_ctc_output(predictions.tolist(), blank_id=blank_id)
        batch_outputs.append(out)
    return batch_outputs
