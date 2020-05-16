"""
Decoding methods for for seq2seq model

Author:
    Ju-Chieh Chou 2020
"""
import torch
import numpy as np


class BaseSearcher(torch.nn.Module):
    """
    BaseSearcher class to be inherit by other decoding approches for seq2seq model.

    Parameters
    ----------
    modules : torch.nn.Module
        The modules that are used to perform auto-regressive decoding.
    bos_index : int
        The index of beginning-of-sequence token.
    eos_index : int
        The index of end-of-sequence token.
    min_decode_radio : float
        The ratio of minimum decoding steps to length of encoder states.
    max_decode_radio : float
        The ratio of maximum decoding steps to length of encoder states.

    Returns
    -------
    predictions:
        Outputs as Python list of lists, with "ragged" dimensions; padding
        has been removed.
    scores:
        The sum of log probabilities (and possible additional heuristic scores) for each prediction.

    Example
    -------
        >>> searcher = BaseSearcher()
        >>> probs = torch.tensor([[[0.3, 0.7], [0.0, 0.0]],
        ...                       [[0.2, 0.8], [0.9, 0.1]]])
        >>> lens = torch.tensor([0.51, 1.0])
        >>> blank_id = 0
        >>> ctc_greedy_decode(probs, lens, blank_id)
        [[1], [1]]

    Author:
        Aku Rouhe 2020
    """

    def __init__(
        self, modules, bos_index, eos_index, min_decode_ratio, max_decode_ratio
    ):
        super(BaseSearcher, self).__init__()
        self.modules = modules
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.min_decode_ratio = min_decode_ratio
        self.max_decode_ratio = max_decode_ratio

    def forward(self):
        raise NotImplementedError

    def forward_step(self):
        raise NotImplementedError

    def reset_mem(self, batch_size, device):
        raise NotImplementedError


class GreedySearcher(BaseSearcher):
    def forward(self, enc_states, wav_len):
        enc_len = torch.round(enc_states.shape[1] * wav_len).int()
        device = enc_states.device
        batch_size = enc_states.shape[0]

        memory = self.reset_mem(batch_size, device=device)
        inp_token = enc_states.new_ones(batch_size).long() * self.bos_index

        log_probs_lst = []
        max_decode_steps = enc_states.shape[1] * self.max_decode_ratio

        for t in range(max_decode_steps):
            log_probs, memory = self.forward_step(
                inp_token, memory, enc_states, enc_len
            )
            log_probs_lst.append(log_probs)
            inp_token = log_probs.argmax(dim=-1)

        log_probs = torch.stack(log_probs_lst, dim=1)
        scores, predictions = log_probs.max(dim=-1)
        scores = scores.sum(dim=1).tolist()
        predictions = batch_filter_seq2seq_output(
            predictions, eos_id=self.eos_index
        )

        return predictions, scores


class RNNGreedySearcher(GreedySearcher):
    def __init__(
        self, modules, bos_index, eos_index, min_decode_ratio, max_decode_ratio
    ):
        super(RNNGreedySearcher, self).__init__(
            modules, bos_index, eos_index, min_decode_ratio, max_decode_ratio
        )
        self.decoder = modules[0]
        self.linear = modules[1]
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def reset_mem(self, batch_size, device):
        hs = self.decoder.init_state(batch_size, device)
        self.decoder.attn.reset()
        c = torch.zeros(batch_size, self.decoder.attn_out_dim).to(device)
        return hs, c

    def forward_step(self, *args):
        inp_token, memory, enc_states, enc_len = args
        hs, c = memory
        dec_out, hs, c, w = self.decoder.forward_step(
            inp_token, hs, c, enc_states, enc_len
        )
        log_probs = self.softmax(self.linear(dec_out))
        return log_probs, (hs, c)


class BeamSearcher(BaseSearcher):
    def __init__(
        self,
        modules,
        bos_index,
        eos_index,
        min_decode_ratio,
        max_decode_ratio,
        beam_size,
        length_penalty,
        eos_penalty,
        minus_inf=-1e20,
    ):
        super(BeamSearcher, self).__init__(
            modules, bos_index, eos_index, min_decode_ratio, max_decode_ratio
        )
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.eos_penalty = eos_penalty
        self.minus_inf = minus_inf

    def forward(self, enc_states, wav_len):
        enc_len = torch.round(enc_states.shape[1] * wav_len).int()
        device = enc_states.device
        batch_size = enc_states.shape[0]

        enc_states = torch.repeat_interleave(enc_states, self.beam_size, dim=0)
        enc_len = torch.repeat_interleave(enc_len, self.beam_size, dim=0)

        memory = self.reset_mem(batch_size * self.beam_size, device=device)
        inp_tokens = (
            enc_states.new_ones(batch_size * self.beam_size).long()
            * self.bos_index
        )

        beam_offset = (torch.arange(batch_size) * self.beam_size).to(device)

        # initialize sequence scores variables
        sequence_scores = torch.Tensor(batch_size * self.beam_size).to(device)
        sequence_scores.fill_(-np.inf)
        sequence_scores.index_fill_(0, beam_offset, 0.0)

        hyps_and_scores = [[] for _ in range(batch_size)]

        alived_seq = (
            torch.empty(batch_size * self.beam_size, 0).long().to(device)
        )

        min_decode_steps = enc_states.shape[1] * self.min_decode_ratio
        max_decode_steps = enc_states.shape[1] * self.max_decode_ratio

        for t in range(max_decode_steps):
            log_probs, memory = self.forward_step(
                inp_tokens, memory, enc_states, enc_len
            )
            vocab_size = log_probs.shape[-1]

            if t < min_decode_steps:
                log_probs[:, self.eos_index] = self.minus_inf

            max_probs, _ = torch.max(log_probs[:, : self.eos_index], dim=-1)
            eos_probs = log_probs[:, self.eos_index]
            log_probs[:, self.eos_index] = torch.where(
                eos_probs > self.eos_penalty * max_probs,
                eos_probs,
                torch.Tensor([self.minus_inf]).to(device),
            )

            scores = sequence_scores.unsqueeze(1).expand(-1, vocab_size)
            scores = scores + log_probs

            # keep topk beams
            scores, candidates = scores.view(batch_size, -1).topk(
                self.beam_size, dim=-1
            )

            inp_tokens = (candidates % vocab_size).view(
                batch_size * self.beam_size
            )
            sequence_scores = scores.view(batch_size * self.beam_size)

            predecessors = (
                candidates // vocab_size
                + beam_offset.unsqueeze(1).expand_as(candidates)
            ).view(batch_size * self.beam_size)
            memory = self.permute_mem(memory, index=predecessors)

            alived_seq = torch.cat(
                [
                    torch.index_select(alived_seq, dim=0, index=predecessors),
                    inp_tokens.unsqueeze(1),
                ],
                dim=-1,
            )
            is_eos = inp_tokens.eq(self.eos_index)
            eos_indices = is_eos.nonzero()

            if eos_indices.shape[0] > 0:
                for index in eos_indices:
                    # convert to int
                    index = index.item()
                    batch_id = index // self.beam_size
                    if len(hyps_and_scores[batch_id]) == self.beam_size:
                        continue
                    hyp = alived_seq[index, :]
                    final_scores = sequence_scores[
                        index
                    ].item() + self.length_penalty * (t + 1)
                    hyps_and_scores[batch_id].append((hyp, final_scores))

                sequence_scores.masked_fill_(is_eos, -np.inf)

        # Check whether there are beam_size hypothesis
        for i in range(batch_size):
            batch_offset = i * self.beam_size
            n_hyps = len(hyps_and_scores[i])
            if n_hyps < self.beam_size:
                remains = self.beam_size - n_hyps
                hyps = alived_seq[batch_offset : batch_offset + remains, :]
                scores = (
                    sequence_scores[batch_offset : batch_offset + remains]
                    + self.length_penalty * max_decode_steps
                ).tolist()
                hyps_and_scores[i] += list(zip(hyps, scores))

        predictions, top_scores = [], []
        for i in range(batch_size):
            top_hyp, top_score = max(
                hyps_and_scores[i], key=lambda pair: pair[1]
            )
            predictions.append(top_hyp)
            top_scores.append(top_score)

        predictions = batch_filter_seq2seq_output(
            predictions, eos_id=self.eos_index
        )

        return predictions, top_scores

    def permute_mem(self, memory, index):
        raise NotImplementedError


class RNNBeamSearcher(BeamSearcher):
    def __init__(
        self,
        modules,
        bos_index,
        eos_index,
        min_decode_ratio,
        max_decode_ratio,
        beam_size,
        length_penalty,
        eos_penalty,
    ):
        super(RNNBeamSearcher, self).__init__(
            modules,
            bos_index,
            eos_index,
            min_decode_ratio,
            max_decode_ratio,
            beam_size,
            length_penalty,
            eos_penalty,
        )
        self.decoder = modules[0]
        self.linear = modules[1]
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def reset_mem(self, batch_size, device):
        hs = self.decoder.init_state(batch_size, device)
        self.decoder.attn.reset()
        c = torch.zeros(batch_size, self.decoder.attn_out_dim).to(device)
        return hs, c

    def forward_step(self, *args):
        inp_token, memory, enc_states, enc_len = args
        hs, c = memory
        dec_out, hs, c, w = self.decoder.forward_step(
            inp_token, hs, c, enc_states, enc_len
        )
        log_probs = self.softmax(self.linear(dec_out))
        return log_probs, (hs, c)

    def permute_mem(self, memory, index):
        hs, c = memory

        # shape of hs: [num_layers, batch_size, n_neurons]
        if isinstance(hs, tuple):
            hs_0 = torch.index_select(hs[0], dim=1, index=index)
            hs_1 = torch.index_select(hs[1], dim=1, index=index)
            hs = (hs_0, hs_1)
        else:
            hs = torch.index_select(hs, dim=1, index=index)

        c = torch.index_select(c, dim=0, index=index)
        if self.decoder.attn_type == "location":
            self.decoder.attn.prev_attn = torch.index_select(
                self.decoder.attn.prev_attn, dim=0, index=index
            )

        return (hs, c)


def batch_filter_seq2seq_output(prediction, eos_id=-1):
    outputs = []
    for p in prediction:
        res = filter_seq2seq_output(p.tolist(), eos_id=eos_id)
        outputs.append(res)
    return outputs


def filter_seq2seq_output(string_pred, eos_id=-1):
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

    Author
    ------
        Mirco Ravanelli 2020
    """
    if isinstance(string_pred, list):
        try:
            eos_index = next(
                i for i, v in enumerate(string_pred) if v == eos_id
            )
        except StopIteration:
            eos_index = len(string_pred)
        string_out = string_pred[:eos_index]
    return string_out
