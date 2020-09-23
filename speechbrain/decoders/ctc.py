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
        self.vocab_size = x.size(-1)
        self.device = x.device
        self.last_frame_index = enc_lens - 1

        # mask frames > enc_lens
        mask = 1 - length_to_mask(enc_lens)
        mask = mask.unsqueeze(-1).expand(-1, -1, x.size(-1)) == 1
        x.masked_fill_(mask, -np.inf)
        x[:, :, 0] = x[:, :, 0].masked_fill_(mask[:, :, 0], 0)

        # xnb: dim=0, nonblank posteriors, xb: dim=1, blank posteriors
        xnb = x.transpose(0, 1)
        xb = (
            xnb[:, :, self.blank_index]
            .unsqueeze(2)
            .expand(-1, -1, self.vocab_size)
        )

        # (2, L, batch_size * beam_size, vocab_size)
        self.x = torch.stack([xnb, xb])

        # The first index of each sentence.
        # TODO: for candidates mode
        self.beam_offset = (torch.arange(batch_size) * self.beam_size).to(
            self.device
        )

    def forward_step(self, g, state, candidates=None):
        """g: prefix"""
        prefix_length = g.size(1) - 1  # TODO
        last_token = [gi[-1] for gi in g]
        num_candidates = (
            self.vocab_size
        )  # TODO support scoring for candidates, candidates.size(-1)

        if state is None:
            # r_prev: (max_enc_len, 2, batch_size * beam_size)
            r_prev = torch.Tensor(
                self.max_enc_len, 2, self.batch_size * self.beam_size
            ).to(self.device)
            r_prev.fill_(-np.inf)
            # Accumulate blank posteriors at each step
            r_prev[:, 1] = torch.cumsum(self.x[0, :, :, self.blank_index], 0)
            r_prev = r_prev.view(-1, 2, self.batch_size * self.beam_size)
            psi_prev = 0
        else:
            r_prev, psi_prev = state

        r = torch.Tensor(
            self.max_enc_len,
            2,
            self.batch_size * self.beam_size,
            num_candidates,
        ).to(self.device)
        r.fill_(-np.inf)

        if prefix_length == 0:
            r[0, 0] = self.x[0, 0]

        # TODO: scores for candidates

        # 0. phi = prev_nonblank + prev_blank = r_t-1^nb(g) + r_t-1^b(g), phi only depends on prefix g.
        r_sum = torch.logsumexp(r_prev, 1)
        phi = r_sum.unsqueeze(2).repeat(1, 1, num_candidates)

        # if last token of prefix g in candidates, phi = prev_b + 0
        for i in range(self.batch_size * self.beam_size):
            phi[:, i, last_token[i]] = r_prev[:, 1, i]

        # Define start, end, |g| < |h| for ctc decoding.
        start = max(1, prefix_length)
        end = self.max_enc_len

        # Compute forward prob log(r_t^nb(h)) and log(r_t^b(h))
        for t in range(start, end):
            # 1. p(h|cur step is nonblank) = [p(prev step=y) + phi] * p(c)
            r[t, 0] = torch.logsumexp(
                torch.stack((r[t - 1, 0], phi[t - 1]), dim=0), dim=0
            )
            r[t, 0] = r[t, 0] + self.x[0, t]
            # 2. p(h|cur step is blank) = [p(prev step is blank) + p(prev step is nonblank)] * p(blank)
            r[t, 1] = torch.logsumexp(
                torch.stack((r[t - 1, 0], r[t - 1, 1]), dim=0), dim=0
            )
            r[t, 1] = r[t, 1] + self.x[1, t]

        # Compute the predix prob
        psi = r[start - 1, 0].unsqueeze(0)
        # phi is prob at t-1 step, shift one frame then add it to current prob p(c)
        phix = torch.cat((phi[0].unsqueeze(0), phi[:-1]), dim=0) + self.x[0]
        # 3. psi = psi + phi * p(c)
        psi = torch.logsumexp(torch.cat((phix[start:end], psi), dim=0), dim=0,)
        for i in range(self.batch_size * self.beam_size):
            psi[i, self.eos_index] = r_sum[
                self.last_frame_index[i // self.beam_size], i
            ]

        # exclude blank probs for joint scoring
        # TODO: currently comment out this line since bos_index, eos_indx is the same as blank_index
        psi[:, self.blank_index] = -np.inf

        return psi - psi_prev, (r, psi)

    def permute_mem(self, memory, candidates):

        r, psi = memory
        best_index = (
            candidates
            + (
                self.beam_offset.unsqueeze(1).expand_as(candidates)
                * self.vocab_size
            )
        ).view(-1)
        r = torch.index_select(
            r.view(-1, 2, self.batch_size * self.beam_size * self.vocab_size),
            dim=-1,
            index=best_index,
        )
        r = r.view(-1, 2, self.batch_size * self.beam_size)

        psi = torch.index_select(psi.view(-1), dim=0, index=best_index)
        psi = (
            psi.view(-1, 1)
            .repeat(1, self.vocab_size)
            .view(self.batch_size * self.beam_size, self.vocab_size)
        )

        return r, psi


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
