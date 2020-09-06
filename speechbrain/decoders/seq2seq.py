"""
Decoding methods for seq2seq autoregressive model.

Authors
 * Ju-Chieh Chou 2020
"""
import torch
import numpy as np


class S2SBaseSearcher(torch.nn.Module):
    """
    S2SBaseSearcher class to be inherited by other
    decoding approches for seq2seq model.

    Parameters
    ----------
    modules : ModuleList
        The modules user uses to perform search algorithm.
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
        The sum of log probabilities (and possibly
        additional heuristic scores) for each prediction.

    """

    def __init__(
        self, modules, bos_index, eos_index, min_decode_ratio, max_decode_ratio
    ):
        super(S2SBaseSearcher, self).__init__()
        self.modules = modules
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.min_decode_ratio = min_decode_ratio
        self.max_decode_ratio = max_decode_ratio

    def forward(self, enc_states, wav_len):
        """This method should implement the forward algorithm of decoding method.

        Arguments
        ---------
        enc_states : torch.Tensor
            The precomputed encoder states to be used when decoding.
            (ex. the encoded speech representation to be attended).
        wav_len : torch.Tensor
            The speechbrain-style relative length.

        """
        raise NotImplementedError

    def forward_step(self, inp_tokens, memory, enc_states, enc_lens):
        """This method should implement one step of
        forwarding operation in autoregressive model.

        Arguments
        ---------
        inp_tokens : torch.Tensor
            The input tensor of current timestep.
        memory : No limit
            The momory variables input for this timestep.
            (ex. RNN hidden states).
        enc_states : torch.Tensor
            The encoder states to be attended.
        enc_lens : torch.Tensor
            The actual length of each enc_states sequence.

        Return
        ------
        log_probs : torch.Tensor
            Log-probilities of the current timestep output.
        memory : No limit
            The momory variables generated in this timestep.
            (ex. RNN hidden states).
        attn : torch.Tensor
            The attention weight for doing penalty.
        """
        raise NotImplementedError

    def reset_mem(self, batch_size, device):
        """This method should implement the reseting of
        memory variables for the seq2seq model.
        Ex. Initializing zero vector as initial hidden states.

        Arguments
        ---------
        batch_size : int
            The size of the batch.
        device : torch.device
            The device to put the initial variables.

        Return
        ------
        memory : No limit
            The initial memory variable.
        """
        raise NotImplementedError

    def lm_forward_step(self, inp_tokens, memory):
        """This method should implement one step of
        forwarding operation for language model.

        Arguments
        ---------
        inp_tokens : torch.Tensor
            The input tensor of current timestep.
        memory : No limit
            The momory variables input for this timestep.
            (ex. RNN hidden states).

        Return
        ------
        log_probs : torch.Tensor
            Log-probilities of the current timestep output.
        memory : No limit
            The momory variables generated in this timestep.
            (ex. RNN hidden states).
        """
        raise NotImplementedError

    def reset_lm_mem(self, batch_size, device):
        """This method should implement the reseting of
        memory variables in language model.
        Ex. Initializing zero vector as initial hidden states.

        Arguments
        ---------
        batch_size : int
            The size of the batch.
        device : torch.device
            The device to put the initial variables.

        Return
        ------
        memory : No limit
            The initial memory variable.
        """
        raise NotImplementedError


class S2SGreedySearcher(S2SBaseSearcher):
    """
    This class implements the general forward-pass of
    greedy decoding approach. See also S2SBaseSearcher().
    """

    def forward(self, enc_states, wav_len):
        enc_lens = torch.round(enc_states.shape[1] * wav_len).int()
        device = enc_states.device
        batch_size = enc_states.shape[0]

        memory = self.reset_mem(batch_size, device=device)

        # Using bos as the first input
        inp_tokens = (
            enc_states.new_zeros(batch_size).fill_(self.bos_index).long()
        )

        log_probs_lst = []
        max_decode_steps = int(enc_states.shape[1] * self.max_decode_ratio)

        for t in range(max_decode_steps):
            log_probs, memory, _ = self.forward_step(
                inp_tokens, memory, enc_states, enc_lens
            )
            log_probs_lst.append(log_probs)
            inp_tokens = log_probs.argmax(dim=-1)

        log_probs = torch.stack(log_probs_lst, dim=1)
        scores, predictions = log_probs.max(dim=-1)
        scores = scores.sum(dim=1).tolist()
        predictions = batch_filter_seq2seq_output(
            predictions, eos_id=self.eos_index
        )

        return predictions, scores


class S2SRNNGreedySearcher(S2SGreedySearcher):
    """
    This class implements the greedy decoding
    for AttentionalRNNDecoder (speechbrain/nnet/RNN.py).
    See also S2SBaseSearcher() and S2SGreedySearcher().

    Parameters
    ----------
    modules : list of torch.nn.Module
        The list should contain four items:
            1. Embedding layer
            2. Attentional RNN decoder
            3. Output layer
            4. LogSoftmax layer

    Example
    -------
    >>> import torch
    >>> import speechbrain as sb
    >>> emb = torch.nn.Embedding(5, 3)
    >>> dec = sb.nnet.RNN.AttentionalRNNDecoder("gru", "content", 3, 3, 1)
    >>> lin = sb.nnet.linear.Linear(5)
    >>> act = sb.nnet.activations.Softmax(apply_log=True)
    >>> inp = torch.randint(low=0, high=5, size=(2, 3))
    >>> enc = torch.rand([2, 6, 7])
    >>> wav_len = torch.rand([2])
    >>> e = emb(inp)
    >>> h, _ = dec(e, enc, wav_len, init_params=True)
    >>> log_probs = act(lin(h, init_params=True))
    >>> modules = [emb, dec, lin]
    >>> searcher = S2SRNNGreedySearcher(
    ... modules,
    ... bos_index=4,
    ... eos_index=4,
    ... min_decode_ratio=0,
    ... max_decode_ratio=1)
    >>> hyps, scores = searcher(enc, wav_len)
    """

    def __init__(
        self, modules, bos_index, eos_index, min_decode_ratio, max_decode_ratio,
    ):
        super(S2SRNNGreedySearcher, self).__init__(
            modules, bos_index, eos_index, min_decode_ratio, max_decode_ratio
        )
        self.emb = self.modules[0]
        self.dec = self.modules[1]
        self.fc = self.modules[2]
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def reset_mem(self, batch_size, device):
        """
        When doing greedy search, keep hidden state (hs) adn context vector (c)
        as memory.
        """
        hs = None
        self.dec.attn.reset()
        c = torch.zeros(batch_size, self.dec.attn_dim).to(device)
        return hs, c

    def forward_step(self, inp_tokens, memory, enc_states, enc_lens):
        hs, c = memory
        e = self.emb(inp_tokens)
        dec_out, hs, c, w = self.dec.forward_step(
            e, hs, c, enc_states, enc_lens
        )
        log_probs = self.softmax(self.fc(dec_out))
        return log_probs, (hs, c), w


class S2SBeamSearcher(S2SBaseSearcher):
    """
    This class implements the beam-search algorithm for seq2seq model.
    See also S2SBaseSearcher().

    Parameters
    ----------
    modules : ModuleList
        The modules user uses to perform search algorithm.
    bos_index : int
        The index of beginning-of-sequence token.
    eos_index : int
        The index of end-of-sequence token.
    min_decode_radio : float
        The ratio of minimum decoding steps to length of encoder states.
    max_decode_radio : float
        The ratio of maximum decoding steps to length of encoder states.
    beam_size : int
        The width of beam.
    topk : int
        Default : 1
        The number of hypothesis to return.
    return_log_probs : bool
        Default : False
        Whether to return log-probabilities.
    using_eos_threshold : bool
        Default : True
        Whether to use eos threshold.
    eos_threshold : float
        Default : 1.5
        The threshold coefficient for eos token. See 3.1.2 in
        reference: https://arxiv.org/abs/1904.02619
    length_normlization : bool
        Default : True
        Whether to divide the scores by the length.
    length_rewarding : float
        Default : 0.0
        The coefficient of length rewarding (γ).
        log P(y|x) + λ log P_LM(y) + γ*len(y)
    lm_weight : float
        Default : 0.0
        The weight of LM when performing beam search (λ).
        log P(y|x) + λ log P_LM(y)
    lm_modules : torch.nn.ModuleList
        neural networks modules for LM.
    using_max_attn_shift: bool
        Whether using the max_attn_shift constaint. Default: False
    max_attn_shift: int
        Beam search will block the beams that attention shift more
        than max_attn_shift.
        Reference: https://arxiv.org/abs/1904.02619
    minus_inf : float
        The value of minus infinity to block some path
        of the search (default : -1e20).
    """

    def __init__(
        self,
        modules,
        bos_index,
        eos_index,
        min_decode_ratio,
        max_decode_ratio,
        beam_size,
        topk=1,
        return_log_probs=False,
        using_eos_threshold=True,
        eos_threshold=1.5,
        length_normalization=True,
        length_rewarding=0,
        lm_weight=0.0,
        lm_modules=None,
        using_max_attn_shift=False,
        max_attn_shift=60,
        minus_inf=-1e20,
    ):
        super(S2SBeamSearcher, self).__init__(
            modules, bos_index, eos_index, min_decode_ratio, max_decode_ratio
        )
        self.beam_size = beam_size
        self.topk = topk
        self.return_log_probs = return_log_probs
        self.length_normalization = length_normalization
        self.length_rewarding = length_rewarding

        if self.length_normalization and self.length_rewarding > 0:
            raise ValueError(
                "length normalization is not compartiable with length rewarding."
            )

        self.using_eos_threshold = using_eos_threshold
        self.eos_threshold = eos_threshold
        self.using_max_attn_shift = using_max_attn_shift
        self.max_attn_shift = max_attn_shift
        self.lm_weight = lm_weight
        self.lm_modules = lm_modules

        # to initialize the params of LM modules
        self.init_lm_params = True
        self.minus_inf = minus_inf

    def _check_full_beams(self, hyps, beam_size):
        """
        This method checks whether hyps has been full.

        Parameters
        ----------
        hyps : List
            This list contains batch_size number of list.
            Each inside list contains a list stores all the hypothesis for this sentence.
        beam_size : int
            The number of beam_size.

        Return
        ------
        bool
            Whether the hyps has been full.
        """
        hyps_len = [len(lst) for lst in hyps]
        beam_size = [self.beam_size for _ in range(len(hyps_len))]
        if hyps_len == beam_size:
            return True
        else:
            return False

    def _check_attn_shift(self, attn, prev_attn_peak):
        """
        This method checks whether attention shift is more than attn_shift.

        Parameters
        ----------
        attn : torch.Tensor
            The attention to be checked.
        prev_attn_peak : torch.Tensor
            The previous attention peak place.

        Return
        ------
        cond : torch.BoolTensor
            Each element represent whether the beam is within the max_shift range.
        attn_peak : torch.Tensor
            The peak of the attn tensor.
        """
        # Block the candidates that exceed the max shift
        _, attn_peak = torch.max(attn, dim=1)
        lt_cond = attn_peak <= (prev_attn_peak + self.max_attn_shift)
        mt_cond = attn_peak > (prev_attn_peak - self.max_attn_shift)

        # True if not exceed limit
        # Multiplication equals to element-wise and for tensor
        cond = (lt_cond * mt_cond).unsqueeze(1)
        return cond, attn_peak

    def _check_eos_threshold(self, log_probs):
        """
        This method checks whether eos log-probabilities exceed threshold.

        Parameters
        ----------
        log_probs : torch.Tensor
            The log-probabilities.

        Return
        ------
        cond : torch.BoolTensor
            Each element represents whether the eos log-probabilities will be kept.
        """
        max_probs, _ = torch.max(log_probs, dim=-1)
        eos_probs = log_probs[:, self.eos_index]
        cond = eos_probs > (self.eos_threshold * max_probs)
        return cond

    def _update_hyp_and_scores(
        self,
        inp_tokens,
        alived_seq,
        alived_log_probs,
        hyps_and_scores,
        scores,
        timesteps,
    ):
        """
        This method will update hyps and scores if inp_tokens are eos.

        Parameters
        ----------
        inp_tokens : torch.Tensor
            The current output.
        alived_seq : torch.Tensor
            The tensor to store the alived_seq.
        alived_log_probs : torch.Tensor
            The tensor to store the alived_log_probs.
        hyps_and_scores : list
            To store generated hypothesis and scores.
        scores : torch.Tensor
            The final scores of beam search.
        timesteps : float
            The current timesteps. This is for length rewarding.

        Return
        ------
        is_eos : torch.BoolTensor
            Each element represents whether the token is eos.
        """
        is_eos = inp_tokens.eq(self.eos_index)
        (eos_indices,) = torch.nonzero(is_eos, as_tuple=True)

        # Store the hypothesis and their scores when reaching eos.
        if eos_indices.shape[0] > 0:
            for index in eos_indices:
                # convert to int
                index = index.item()
                batch_id = index // self.beam_size
                if len(hyps_and_scores[batch_id]) == self.beam_size:
                    continue
                hyp = alived_seq[index, :]
                log_probs = alived_log_probs[index, :]
                final_scores = scores[index].item() + self.length_rewarding * (
                    timesteps + 1
                )
                hyps_and_scores[batch_id].append((hyp, log_probs, final_scores))
        return is_eos

    def _get_top_score_prediction(self, hyps_and_scores, topk):
        """
        This method sort the scores and return corresponding hypothesis and log probs.

        Parameters
        ----------
        hyps_and_scores : list
            To store generated hypothesis and scores.
        topk : int
            Number of hypothesis to return.

        Return
        ------
        predictions : list
            This list contains the predicted hypothesis.
            The order will be the following:
            h_i_j, i is utterance id, and j is hypothesis id.
            When topk=2, and 3 sentences:
            [h_0_0, h_0_1,h_1_0, h_1_1, h_2_0, h_2_1]

        top_scores : list
            This list contains the final scores of hypothesis.
            The order is the same as predictions.

        top_log_probs : list
            This list contains the log probabilities of each hypothesis.
            The order is the same as predictions.
        """
        predictions, top_log_probs, top_scores = [], [], []
        for i in range(len(hyps_and_scores)):
            hyps, log_probs, scores = zip(*hyps_and_scores[i])

            # get topk indices and reverse it to make it descending
            indices = np.argsort(np.array(scores))[::-1][:topk]
            predictions += [hyps[index] for index in indices]
            top_scores += [scores[index] for index in indices]
            top_log_probs += [log_probs[index] for index in indices]
        return predictions, top_scores, top_log_probs

    def forward(self, enc_states, wav_len):
        enc_lens = torch.round(enc_states.shape[1] * wav_len).int()
        device = enc_states.device
        batch_size = enc_states.shape[0]

        # Inflate the enc_states and enc_len by beam_size times
        enc_states = inflate_tensor(enc_states, times=self.beam_size, dim=0)
        enc_lens = inflate_tensor(enc_lens, times=self.beam_size, dim=0)

        memory = self.reset_mem(batch_size * self.beam_size, device=device)

        if self.lm_weight > 0:
            lm_memory = self.reset_lm_mem(batch_size * self.beam_size, device)

        # Using bos as the first input
        inp_tokens = (
            torch.zeros(batch_size * self.beam_size)
            .to(device)
            .fill_(self.bos_index)
            .long()
        )

        # The first index of each sentence.
        beam_offset = (torch.arange(batch_size) * self.beam_size).to(device)

        # initialize sequence scores variables.
        sequence_scores = torch.Tensor(batch_size * self.beam_size).to(device)
        sequence_scores.fill_(-np.inf)

        # keep only the first to make sure no redundancy.
        sequence_scores.index_fill_(0, beam_offset, 0.0)

        # keep the hypothesis that reaches eos and their corresponding score and log_probs.
        hyps_and_scores = [[] for _ in range(batch_size)]

        # keep the sequences that still not reaches eos.
        alived_seq = (
            torch.empty(batch_size * self.beam_size, 0).long().to(device)
        )

        # Keep the log-probabilities of alived sequences.
        alived_log_probs = torch.empty(batch_size * self.beam_size, 0).to(
            device
        )

        min_decode_steps = int(enc_states.shape[1] * self.min_decode_ratio)
        max_decode_steps = int(enc_states.shape[1] * self.max_decode_ratio)

        # Initialize the previous attention peak to zero
        # This variable will be used when using_max_attn_shift=True
        prev_attn_peak = torch.zeros(batch_size * self.beam_size).to(device)

        for t in range(max_decode_steps):
            # terminate condition
            if self._check_full_beams(hyps_and_scores, self.beam_size):
                break

            log_probs, memory, attn = self.forward_step(
                inp_tokens, memory, enc_states, enc_lens
            )

            # Keep the original value
            log_probs_clone = log_probs.clone().reshape(batch_size, -1)
            vocab_size = log_probs.shape[-1]

            if self.using_max_attn_shift:
                # Block the candidates that exceed the max shift
                cond, attn_peak = self._check_attn_shift(attn, prev_attn_peak)
                log_probs = mask_by_condition(
                    log_probs, cond, fill_value=self.minus_inf
                )
                prev_attn_peak = attn_peak

            # Set eos to minus_inf when less than minimum steps.
            if t < min_decode_steps:
                log_probs[:, self.eos_index] = self.minus_inf

            # Set the eos prob to minus_inf when it doesn't exceed threshold.
            if self.using_eos_threshold:
                cond = self._check_eos_threshold(log_probs)
                log_probs[:, self.eos_index] = mask_by_condition(
                    log_probs[:, self.eos_index],
                    cond,
                    fill_value=self.minus_inf,
                )

            # adding LM scores to log_prob if lm_weight > 0
            if self.lm_weight > 0:
                lm_log_probs, lm_memory = self.lm_forward_step(
                    inp_tokens, lm_memory
                )
                log_probs = log_probs + self.lm_weight * lm_log_probs

            scores = sequence_scores.unsqueeze(1).expand(-1, vocab_size)
            scores = scores + log_probs

            # length normalization
            if self.length_normalization:
                scores = scores / (t + 1)

            # keep topk beams
            scores, candidates = scores.view(batch_size, -1).topk(
                self.beam_size, dim=-1
            )

            # The input for the next step, also the output of current step.
            inp_tokens = (candidates % vocab_size).view(
                batch_size * self.beam_size
            )

            scores = scores.view(batch_size * self.beam_size)
            sequence_scores = scores

            # recover the length normalization
            if self.length_normalization:
                sequence_scores = sequence_scores * (t + 1)

            # The index of which beam the current top-K output came from in (t-1) timesteps.
            predecessors = (
                candidates // vocab_size
                + beam_offset.unsqueeze(1).expand_as(candidates)
            ).view(batch_size * self.beam_size)

            # Permute the memory to synchoronize with the output.
            memory = self.permute_mem(memory, index=predecessors)
            if self.lm_weight > 0:
                lm_memory = self.permute_lm_mem(lm_memory, index=predecessors)

            # If using_max_attn_shift, thne the previous attn peak has to be permuted too.
            if self.using_max_attn_shift:
                prev_attn_peak = torch.index_select(
                    prev_attn_peak, dim=0, index=predecessors
                )

            # Update alived_seq
            alived_seq = torch.cat(
                [
                    torch.index_select(alived_seq, dim=0, index=predecessors),
                    inp_tokens.unsqueeze(1),
                ],
                dim=-1,
            )

            # Takes the log-probabilities
            beam_log_probs = log_probs_clone[
                torch.arange(batch_size).unsqueeze(1), candidates
            ].reshape(batch_size * self.beam_size)
            alived_log_probs = torch.cat(
                [
                    torch.index_select(
                        alived_log_probs, dim=0, index=predecessors
                    ),
                    beam_log_probs.unsqueeze(1),
                ],
                dim=-1,
            )

            is_eos = self._update_hyp_and_scores(
                inp_tokens,
                alived_seq,
                alived_log_probs,
                hyps_and_scores,
                scores,
                timesteps=t,
            )

            # Block the pathes that have reached eos.
            sequence_scores.masked_fill_(is_eos, -np.inf)

        if not self._check_full_beams(hyps_and_scores, self.beam_size):
            # Using all eos to fill-up the hyps.
            eos = (
                torch.zeros(batch_size * self.beam_size)
                .to(device)
                .fill_(self.eos_index)
                .long()
            )
            _ = self._update_hyp_and_scores(
                eos,
                alived_seq,
                alived_log_probs,
                hyps_and_scores,
                scores,
                timesteps=max_decode_steps,
            )

        predictions, top_scores, log_probs = self._get_top_score_prediction(
            hyps_and_scores, topk=self.topk,
        )
        predictions = batch_filter_seq2seq_output(
            predictions, eos_id=self.eos_index
        )

        if self.return_log_probs:
            return predictions, top_scores, log_probs
        else:
            return predictions, top_scores

    def permute_mem(self, memory, index):
        """
        This method permutes the seq2seq model memory
        to synchronize the memory index with the current output.

        Arguments
        ---------

        memory : No limit
            The memory variable to be permuted.
        index : torch.Tensor
            The index of the previous path.

        Return
        ------
        The variable of the memory being permuted.

        """
        raise NotImplementedError

    def permute_lm_mem(self, memory, index):
        """
        This method permutes the language model memory
        to synchronize the memory index with the current output.

        Arguments
        ---------

        memory : No limit
            The memory variable to be permuted.
        index : torch.Tensor
            The index of the previous path.

        Return
        ------
        The variable of the memory being permuted.

        """
        raise NotImplementedError


class S2SRNNBeamSearcher(S2SBeamSearcher):
    """
    This class implements the beam search decoding
    for AttentionalRNNDecoder (speechbrain/nnet/RNN.py).
    See also S2SBaseSearcher(), S2SBeamSearcher().

    Parameters
    ----------
    modules : list of torch.nn.Module
        The list should contain three items:
            1. Embedding layer
            2. Attentional RNN decoder
            3. Output layer

    bos_index : int
        The index of beginning-of-sequence token.
    eos_index : int
        The index of end-of-sequence token.
    min_decode_radio : float
        The ratio of minimum decoding steps to length of encoder states.
    max_decode_radio : float
        The ratio of maximum decoding steps to length of encoder states.
    beam_size : int
        The width of beam.
    topk : int
        Default : 1
        The number of hypothesis to return.
    return_log_probs : bool
        Default : False
        Whether to return log-probabilities.
    using_eos_threshold : bool
        Whether to use eos threshold.
    eos_threshold : float
        The threshold coefficient for eos token. See 3.1.2 in
        reference: https://arxiv.org/abs/1904.02619
    length_normlization : bool
        Default : True
        Whether to divide the scores by the length.
    length_rewarding : float
        Default : 0.0
        The coefficient of length rewarding (γ).
        log P(y|x) + λ log P_LM(y) + γ*len(y
    using_max_attn_shift: bool
        Whether using the max_attn_shift constaint. Default: False
    max_attn_shift: int
        Beam search will block the beams that attention shift more
        than max_attn_shift.
        Reference: https://arxiv.org/abs/1904.02619
    minus_inf : float
        The value of minus infinity to block some path
        of the search (default : -1e20).
    Example
    -------
    >>> import torch
    >>> import speechbrain as sb
    >>> emb = torch.nn.Embedding(5, 3)
    >>> dec = sb.nnet.RNN.AttentionalRNNDecoder("gru", "content", 3, 3, 1)
    >>> lin = sb.nnet.linear.Linear(5)
    >>> act = sb.nnet.activations.Softmax(apply_log=True)
    >>> inp = torch.randint(low=0, high=5, size=(2, 3))
    >>> enc = torch.rand([2, 6, 7])
    >>> wav_len = torch.rand([2])
    >>> e = emb(inp)
    >>> h, _ = dec(e, enc, wav_len, init_params=True)
    >>> log_probs = act(lin(h, init_params=True))
    >>> modules = [emb, dec, lin]
    >>> searcher = S2SRNNBeamSearcher(
    ... modules,
    ... bos_index=4,
    ... eos_index=4,
    ... min_decode_ratio=0,
    ... max_decode_ratio=1,
    ... beam_size=2)
    >>> hyps, scores = searcher(enc, wav_len)
    """

    def __init__(
        self,
        modules,
        bos_index,
        eos_index,
        min_decode_ratio,
        max_decode_ratio,
        beam_size,
        topk=1,
        return_log_probs=False,
        using_eos_threshold=True,
        eos_threshold=1.5,
        length_normalization=True,
        length_rewarding=0,
        lm_weight=0.0,
        lm_modules=None,
        using_max_attn_shift=False,
        max_attn_shift=60,
        minus_inf=-1e20,
    ):
        super(S2SRNNBeamSearcher, self).__init__(
            modules,
            bos_index,
            eos_index,
            min_decode_ratio,
            max_decode_ratio,
            beam_size,
            topk,
            return_log_probs,
            using_eos_threshold,
            eos_threshold,
            length_normalization,
            length_rewarding,
            lm_weight,
            lm_modules,
            using_max_attn_shift,
            max_attn_shift,
        )
        self.emb = self.modules[0]
        self.dec = self.modules[1]
        self.fc = self.modules[2]
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def reset_mem(self, batch_size, device):
        hs = None
        self.dec.attn.reset()
        c = torch.zeros(batch_size, self.dec.attn_dim).to(device)
        return hs, c

    def forward_step(self, inp_tokens, memory, enc_states, enc_lens):
        hs, c = memory
        e = self.emb(inp_tokens)
        dec_out, hs, c, w = self.dec.forward_step(
            e, hs, c, enc_states, enc_lens
        )
        log_probs = self.softmax(self.fc(dec_out))
        return log_probs, (hs, c), w

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
        if self.dec.attn_type == "location":
            self.dec.attn.prev_attn = torch.index_select(
                self.dec.attn.prev_attn, dim=0, index=index
            )
        return (hs, c)


def inflate_tensor(tensor, times, dim):
    """
    This function inflate the tensor for times along dim.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to be inflated.
    times : int
        The tensor will inflate for this number of times.
    dim : int
        The dim to be inflated.

    Returns
    -------
    torch.Tensor
        The inflated tensor.

    Example
    -------
    >>> tensor = torch.Tensor([[1,2,3], [4,5,6]])
    >>> new_tensor = inflate_tensor(tensor, 2, dim=0)
    >>> new_tensor
    tensor([[1., 2., 3.],
            [1., 2., 3.],
            [4., 5., 6.],
            [4., 5., 6.]])
    """
    return torch.repeat_interleave(tensor, times, dim=dim)


def mask_by_condition(tensor, cond, fill_value):
    """
    This function will mask some element in the tensor with fill_value, if condition=False.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to be masked.
    cond : torch.BoolTensor
        This tensor has to be the same size as tensor.
        Each element represent whether to keep the value in tensor.
    fill_value : float
        The value to fill in the masked element.

    Returns
    -------
    torch.Tensor
        The masked tensor.

    Example
    -------
    >>> tensor = torch.Tensor([[1,2,3], [4,5,6]])
    >>> cond = torch.BoolTensor([[True, True, False], [True, False, False]])
    >>> mask_by_condition(tensor, cond, 0)
    tensor([[1., 2., 0.],
            [4., 0., 0.]])
    """
    tensor = torch.where(
        cond, tensor, torch.Tensor([fill_value]).to(tensor.device)
    )
    return tensor


def _update_mem(inp_tokens, memory):
    """This function is for updating the memory for transformer searches.
    it is code at each decode step. When being called, it appends the predicted token of the previous step to existing memory.

    Arguements:
    -----------
    inp_tokens: tensor
        predicted token of the previous decoding step
    memory: tensor
        Contains all the predicted tokens
    """
    if memory is None:
        return inp_tokens.unsqueeze(1)
    return torch.cat([memory, inp_tokens.unsqueeze(1)], dim=-1)


def _model_decode(model, softmax, fc, inp_tokens, memory, enc_states):
    """This function implements 1 decode step for the transformer searches

    Arguements:
    -----------
    model: torch class
        Transformer model
    softmax: torch class
        softmax fuction
    fc: torch class
        output linear layer
    inp_token: tensor
        predicted token from t-1 step
    memory: tensor
        contains all predicted tokens
    enc_states: tensor
        encoder states
    """
    memory = _update_mem(inp_tokens, memory)
    pred = model.decode(memory, enc_states)
    prob_dist = softmax(fc(pred))
    return prob_dist, memory


class S2STransformerBeamSearch(S2SBeamSearcher):
    """This class implements the beam search decoding
    for Transformer.
    See also S2SBaseSearcher(), S2SBeamSearcher().

    Parameters
    ----------
    modules : list of torch.nn.Module
        The list should contain two items:
            1. Transformer model
            2. Output layer

    Example:
    --------
    >>> # see recipes/LibriSpeech/ASR_transformer/experiment.py
    """

    def __init__(
        self,
        modules,
        bos_index,
        eos_index,
        min_decode_ratio,
        max_decode_ratio,
        beam_size,
        topk=1,
        return_log_probs=False,
        using_eos_threshold=False,
        eos_threshold=1.5,
        length_normalization=False,
        length_rewarding=0,
        lm_weight=0.0,
        lm_modules=None,
        using_max_attn_shift=False,
        max_attn_shift=60,
        minus_inf=-1e20,
    ):
        super(S2STransformerBeamSearch, self).__init__(
            modules,
            bos_index,
            eos_index,
            min_decode_ratio,
            max_decode_ratio,
            beam_size,
            topk,
            return_log_probs,
            using_eos_threshold,
            eos_threshold,
            length_normalization,
            length_rewarding,
            lm_weight,
            lm_modules,
            using_max_attn_shift,
            max_attn_shift,
        )

        self.model = modules[0]
        self.fc = modules[1]
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def reset_mem(self, batch_size, device):
        return None

    def permute_mem(self, memory, index):
        memory = torch.index_select(memory, dim=0, index=index)
        return memory

    def forward_step(self, inp_tokens, memory, enc_states, enc_lens):
        prob_dist, memory = _model_decode(
            self.model, self.softmax, self.fc, inp_tokens, memory, enc_states
        )
        return prob_dist[:, -1, :], memory, None


class S2STransformerGreedySearch(S2SGreedySearcher):
    """This class implements the greedy decoding
    for AttentionalRNNDecoder (speechbrain/nnet/RNN.py).
    See also S2SBaseSearcher() and S2SGreedySearcher().

    Parameters
    ----------
    modules : list of torch.nn.Module
        The list should contain two items:
            1. Transformer model
            2. Output layer

    Example:
    --------
    >>> # see recipes/LibriSpeech/ASR_transformer/experiment.py
    """

    def __init__(
        self, modules, bos_index, eos_index, min_decode_ratio, max_decode_ratio,
    ):
        super().__init__(
            modules, bos_index, eos_index, min_decode_ratio, max_decode_ratio,
        )
        self.model = modules[0]
        self.fc = modules[1]
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def reset_mem(self, batch_size, device):
        return None

    def forward_step(self, inp_tokens, memory, enc_states, enc_lens):
        prob_dist, memory = _model_decode(
            self.model, self.softmax, self.fc, inp_tokens, memory, enc_states
        )
        return prob_dist[:, -1, :], memory, None


def batch_filter_seq2seq_output(prediction, eos_id=-1):
    """Calling batch_size times of filter_seq2seq_output.

    Parameters
    ----------
    prediction : list of torch.Tensor
        a list containing the output ints predicted by the seq2seq system.
    eos_id : int, string
        the id of the eos.

    Returns
    ------
    list
        The output predicted by seq2seq model.

    Example
    -------
        >>> predictions = [torch.IntTensor([1,2,3,4]), torch.IntTensor([2,3,4,5,6])]
        >>> predictions = batch_filter_seq2seq_output(predictions, eos_id=4)
        >>> predictions
        [[1, 2, 3], [2, 3]]
    """
    outputs = []
    for p in prediction:
        res = filter_seq2seq_output(p.tolist(), eos_id=eos_id)
        outputs.append(res)
    return outputs


def filter_seq2seq_output(string_pred, eos_id=-1):
    """Filter the output until the first eos occurs (exclusive).

    Parameters
    ----------
    string_pred : list
        a list containing the output strings/ints predicted by the seq2seq system.
    eos_id : int, string
        the id of the eos.

    Returns
    ------
    list
        The output predicted by seq2seq model.

    Example
    -------
        >>> string_pred = ['a','b','c','d','eos','e']
        >>> string_out = filter_seq2seq_output(string_pred, eos_id='eos')
        >>> string_out
        ['a', 'b', 'c', 'd']
    """
    if isinstance(string_pred, list):
        try:
            eos_index = next(
                i for i, v in enumerate(string_pred) if v == eos_id
            )
        except StopIteration:
            eos_index = len(string_pred)
        string_out = string_pred[:eos_index]
    else:
        raise ValueError("The input must be a list.")
    return string_out
