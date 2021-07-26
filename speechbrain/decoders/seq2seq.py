"""Decoding methods for seq2seq autoregressive model.

Authors
 * Ju-Chieh Chou 2020
 * Peter Plantinga 2020
 * Mirco Ravanelli 2020
 * Sung-Lin Yeh 2020
"""
import torch


class S2SBaseSearcher(torch.nn.Module):
    """S2SBaseSearcher class to be inherited by other
    decoding approaches for seq2seq model.

    Arguments
    ---------
    bos_index : int
        The index of the beginning-of-sequence (bos) token.
    eos_index : int
        The index of end-of-sequence token.
    min_decode_radio : float
        The ratio of minimum decoding steps to the length of encoder states.
    max_decode_radio : float
        The ratio of maximum decoding steps to the length of encoder states.

    Returns
    -------
    predictions
        Outputs as Python list of lists, with "ragged" dimensions; padding
        has been removed.
    scores
        The sum of log probabilities (and possibly
        additional heuristic scores) for each prediction.

    """

    def __init__(
        self, bos_index, eos_index, min_decode_ratio, max_decode_ratio,
    ):
        super(S2SBaseSearcher, self).__init__()
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
        forwarding operation in the autoregressive model.

        Arguments
        ---------
        inp_tokens : torch.Tensor
            The input tensor of the current timestep.
        memory : No limit
            The memory variables input for this timestep.
            (ex. RNN hidden states).
        enc_states : torch.Tensor
            The encoder states to be attended.
        enc_lens : torch.Tensor
            The actual length of each enc_states sequence.

        Returns
        -------
        log_probs : torch.Tensor
            Log-probabilities of the current timestep output.
        memory : No limit
            The memory variables generated in this timestep.
            (ex. RNN hidden states).
        attn : torch.Tensor
            The attention weight for doing penalty.
        """
        raise NotImplementedError

    def reset_mem(self, batch_size, device):
        """This method should implement the resetting of
        memory variables for the seq2seq model.
        E.g., initializing zero vector as initial hidden states.

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
    """This class implements the general forward-pass of
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

        (
            top_hyps,
            top_lengths,
            top_scores,
            top_log_probs,
        ) = self._get_top_prediction(predictions, scores, log_probs)

        return top_hyps, top_lengths, top_scores, top_log_probs

    def _get_top_prediction(self, hyps, scores, log_probs):
        """This method sorts the scores and return corresponding hypothesis and log probs.

        Arguments
        ---------
        hyps_and_scores : list
            To store generated hypotheses and scores.
        topk : int
            Number of hypothesis to return.

        Returns
        -------
        topk_hyps : torch.Tensor (batch, topk, max length of token_id sequences)
            This tensor stores the topk predicted hypothesis.
        topk_scores : torch.Tensor (batch, topk)
            The length of each topk sequence in the batch.
        topk_lengths : torch.Tensor (batch, topk)
            This tensor contains the final scores of topk hypotheses.
        topk_log_probs : torch.Tensor (batch, topk, max length of token_id sequences)
            The log probabilities of each hypotheses.
        """
        batch_size = hyps.size(0)
        max_length = hyps.size(1)
        top_lengths = [max_length] * batch_size

        # Collect lengths of top hyps
        for pred_index in range(batch_size):
            pred = hyps[pred_index]
            pred_length = (pred == self.eos_index).nonzero(as_tuple=False)
            if len(pred_length) > 0:
                top_lengths[pred_index] = pred_length[0].item()
        # Convert lists to tensors
        top_lengths = torch.tensor(
            top_lengths, dtype=torch.float, device=hyps.device
        )

        # Pick top log probabilities
        top_log_probs = log_probs

        # Use SpeechBrain style lengths
        top_lengths = (top_lengths - 1) / max_length

        return (
            hyps.unsqueeze(1),
            top_lengths.unsqueeze(1),
            scores.unsqueeze(1),
            top_log_probs.unsqueeze(1),
        )


class S2SRNNGreedySearcher(S2SGreedySearcher):
    """
    This class implements the greedy decoding
    for AttentionalRNNDecoder (speechbrain/nnet/RNN.py).
    See also S2SBaseSearcher() and S2SGreedySearcher().

    Arguments
    ---------
    embedding : torch.nn.Module
        An embedding layer.
    decoder : torch.nn.Module
        Attentional RNN decoder.
    linear : torch.nn.Module
        A linear output layer.
    **kwargs
        see S2SBaseSearcher, arguments are directly passed.

    Example
    -------
    >>> import speechbrain as sb
    >>> emb = torch.nn.Embedding(5, 3)
    >>> dec = sb.nnet.RNN.AttentionalRNNDecoder(
    ...     "gru", "content", 3, 3, 1, enc_dim=7, input_size=3
    ... )
    >>> lin = sb.nnet.linear.Linear(n_neurons=5, input_size=3)
    >>> searcher = S2SRNNGreedySearcher(
    ...     embedding=emb,
    ...     decoder=dec,
    ...     linear=lin,
    ...     bos_index=4,
    ...     eos_index=4,
    ...     min_decode_ratio=0,
    ...     max_decode_ratio=1,
    ... )
    >>> enc = torch.rand([2, 6, 7])
    >>> wav_len = torch.rand([2])
    >>> top_hyps, top_lengths, _, _ = searcher(enc, wav_len)
    """

    def __init__(self, embedding, decoder, linear, **kwargs):
        super(S2SRNNGreedySearcher, self).__init__(**kwargs)
        self.emb = embedding
        self.dec = decoder
        self.fc = linear
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def reset_mem(self, batch_size, device):
        """When doing greedy search, keep hidden state (hs) adn context vector (c)
        as memory.
        """
        hs = None
        self.dec.attn.reset()
        c = torch.zeros(batch_size, self.dec.attn_dim, device=device)
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
    """This class implements the beam-search algorithm for the seq2seq model.
    See also S2SBaseSearcher().

    Arguments
    ---------
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
    vocab_size : int
        The size of vocabulary.
    scorer: speechbrain.decoders.scorers.ScorerBuilder
        Scorer instance.
    topk : int
        The number of hypothesis to return. (default: 1)
    using_eos_threshold : bool
        Whether to use eos threshold. (default: true)
    eos_threshold : float
        The threshold coefficient for eos token (default: 1.5). See 3.1.2 in
        reference: https://arxiv.org/abs/1904.02619
    length_normalization : bool
        Whether to divide the scores by the length. (default: True)
    using_max_attn_shift: bool
        Whether using the max_attn_shift constraint. (default: False)
    max_attn_shift: int
        Beam search will block the beams that attention shift more
        than max_attn_shift.
        Reference: https://arxiv.org/abs/1904.02619
    minus_inf : float
        The value of minus infinity to block some path
        of the search. (default: -1e20)
    """

    def __init__(
        self,
        bos_index,
        eos_index,
        min_decode_ratio,
        max_decode_ratio,
        beam_size,
        vocab_size,
        scorer=None,
        topk=1,
        using_eos_threshold=True,
        eos_threshold=1.5,
        length_normalization=True,
        using_max_attn_shift=False,
        max_attn_shift=60,
        minus_inf=-1e20,
    ):
        super(S2SBeamSearcher, self).__init__(
            bos_index, eos_index, min_decode_ratio, max_decode_ratio,
        )
        self.beam_size = beam_size
        self.vocab_size = vocab_size
        self.scorer = scorer
        self.topk = topk
        self.length_normalization = length_normalization
        self.using_eos_threshold = using_eos_threshold
        self.eos_threshold = eos_threshold
        self.using_max_attn_shift = using_max_attn_shift
        self.max_attn_shift = max_attn_shift
        self.attn_weight = 1.0
        self.ctc_weight = 0.0
        self.minus_inf = minus_inf

        if self.scorer is not None:
            # Check length normalization
            if length_normalization and self.scorer.weights["length"] > 0.0:
                raise ValueError(
                    "Length normalization is not compatible with length rewarding."
                )
            if self.scorer.weights["ctc"] > 0.0:
                # Check indices for ctc
                all_scorers = {
                    **self.scorer.full_scorers,
                    **self.scorer.partial_scorers,
                }
                blank_index = all_scorers["ctc"].blank_index
                if len({bos_index, eos_index, blank_index}) < 3:
                    raise ValueError(
                        "Set blank, eos and bos to different indexes for joint ATT/CTC or CTC decoding"
                    )

                self.ctc_weight = self.scorer.weights["ctc"]
                self.attn_weight = 1.0 - self.ctc_weight

    def _check_full_beams(self, hyps, beam_size):
        """This method checks whether hyps has been full.

        Arguments
        ---------
        hyps : List
            This list contains batch_size number.
            Each inside list contains a list stores all the hypothesis for this sentence.
        beam_size : int
            The number of beam_size.

        Returns
        -------
        bool
            Whether the hyps has been full.
        """
        hyps_len = [len(lst) for lst in hyps]
        beam_size = [self.beam_size for _ in range(len(hyps_len))]
        if hyps_len == beam_size:
            return True

        return False

    def _check_attn_shift(self, attn, prev_attn_peak):
        """This method checks whether attention shift is more than attn_shift.

        Arguments
        ---------
        attn : torch.Tensor
            The attention to be checked.
        prev_attn_peak : torch.Tensor
            The previous attention peak place.

        Returns
        -------
        cond : torch.BoolTensor
            Each element represents whether the beam is within the max_shift range.
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

        Arguments
        ---------
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
        """This method will update hyps and scores if inp_tokens are eos.

        Arguments
        ---------
        inp_tokens : torch.Tensor
            The current output.
        alived_seq : torch.Tensor
            The tensor to store the alived_seq.
        alived_log_probs : torch.Tensor
            The tensor to store the alived_log_probs.
        hyps_and_scores : list
            To store generated hypotheses and scores.
        scores : torch.Tensor
            The final scores of beam search.
        timesteps : float
            The current timesteps. This is for length rewarding.

        Returns
        -------
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
                batch_id = torch.div(
                    index, self.beam_size, rounding_mode="floor"
                )
                if len(hyps_and_scores[batch_id]) == self.beam_size:
                    continue
                hyp = alived_seq[index, :]
                log_probs = alived_log_probs[index, :]
                final_scores = scores[index]
                hyps_and_scores[batch_id].append((hyp, log_probs, final_scores))
        return is_eos

    def _get_topk_prediction(self, hyps_and_scores, topk):
        """This method sorts the scores and return corresponding hypothesis and log probs.

        Arguments
        ---------
        hyps_and_scores : list
            To store generated hypotheses and scores.
        topk : int
            Number of hypothesis to return.

        Returns
        -------
        topk_hyps : torch.Tensor (batch, topk, max length of token_id sequences)
            This tensor stores the topk predicted hypothesis.
        topk_lengths : torch.Tensor (batch, topk)
            This tensor contains the final scores of topk hypotheses.
        topk_scores : torch.Tensor (batch, topk)
            The length of each topk sequence in the batch.
        topk_log_probs : torch.Tensor (batch, topk, max length of token_id sequences)
            The log probabilities of each hypotheses.
        """
        top_hyps, top_log_probs, top_scores, top_lengths = [], [], [], []
        batch_size = len(hyps_and_scores)

        # Collect hypotheses
        for i in range(len(hyps_and_scores)):
            hyps, log_probs, scores = zip(*hyps_and_scores[i])
            top_hyps += hyps
            top_scores += scores
            top_log_probs += log_probs
            top_lengths += [len(hyp) for hyp in hyps]
        # Convert lists to tensors
        top_hyps = torch.nn.utils.rnn.pad_sequence(
            top_hyps, batch_first=True, padding_value=0
        )
        top_log_probs = torch.nn.utils.rnn.pad_sequence(
            top_log_probs, batch_first=True, padding_value=0
        )
        top_lengths = torch.tensor(
            top_lengths, dtype=torch.float, device=top_hyps.device
        )
        top_scores = torch.stack((top_scores), dim=0).view(batch_size, -1)

        # Use SpeechBrain style lengths
        top_lengths = (top_lengths - 1) / top_hyps.size(1)

        # Get topk indices
        topk_scores, indices = top_scores.topk(self.topk, dim=-1)
        indices = (indices + self.beam_offset.unsqueeze(1)).view(
            batch_size * self.topk
        )
        # Select topk hypotheses
        topk_hyps = torch.index_select(top_hyps, dim=0, index=indices,).view(
            batch_size, self.topk, -1
        )
        topk_lengths = torch.index_select(
            top_lengths, dim=0, index=indices,
        ).view(batch_size, self.topk)
        topk_log_probs = torch.index_select(
            top_log_probs, dim=0, index=indices,
        ).view(batch_size, self.topk, -1)

        return topk_hyps, topk_lengths, topk_scores, topk_log_probs

    def forward(self, enc_states, wav_len):  # noqa: C901
        enc_lens = torch.round(enc_states.shape[1] * wav_len).int()
        device = enc_states.device
        batch_size = enc_states.shape[0]
        n_bh = batch_size * self.beam_size

        memory = self.reset_mem(n_bh, device=device)

        if self.scorer is not None:
            scorer_memory = self.scorer.reset_scorer_mem(enc_states, enc_lens)

        # Inflate the enc_states and enc_len by beam_size times
        enc_states = inflate_tensor(enc_states, times=self.beam_size, dim=0)
        enc_lens = inflate_tensor(enc_lens, times=self.beam_size, dim=0)

        # Using bos as the first input
        inp_tokens = (
            torch.zeros(n_bh, device=device).fill_(self.bos_index).long()
        )

        # The first index of each sentence.
        self.beam_offset = (
            torch.arange(batch_size, device=device) * self.beam_size
        )

        # initialize sequence scores variables.
        sequence_scores = torch.empty(n_bh, device=device)
        sequence_scores.fill_(float("-inf"))

        # keep only the first to make sure no redundancy.
        sequence_scores.index_fill_(0, self.beam_offset, 0.0)

        # keep the hypothesis that reaches eos and their corresponding score and log_probs.
        hyps_and_scores = [[] for _ in range(batch_size)]

        # keep the sequences that still not reaches eos.
        alived_seq = torch.empty(n_bh, 0, device=device).long()

        # Keep the log-probabilities of alived log_probs
        alived_log_probs = torch.empty(n_bh, 0, device=device)

        min_decode_steps = int(enc_states.shape[1] * self.min_decode_ratio)
        max_decode_steps = int(enc_states.shape[1] * self.max_decode_ratio)

        # Initialize the previous attention peak to zero
        # This variable will be used when using_max_attn_shift=True
        prev_attn_peak = torch.zeros(n_bh, device=device)
        attn = None

        log_probs = torch.full((n_bh, self.vocab_size), 0.0, device=device)

        for t in range(max_decode_steps):
            # terminate condition
            if self._check_full_beams(hyps_and_scores, self.beam_size):
                break

            if self.attn_weight > 0:
                log_probs, memory, attn = self.forward_step(
                    inp_tokens, memory, enc_states, enc_lens
                )
            log_probs = self.attn_weight * log_probs

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

            # Adding Scorer scores to log_prob if Scorer is not None.
            if self.scorer is not None:
                # score with scorers
                log_probs, scorer_memory = self.scorer.score(
                    inp_tokens, scorer_memory, attn, log_probs, self.beam_size
                )

            # Set the eos prob to minus_inf when it doesn't exceed threshold.
            if self.using_eos_threshold:
                cond = self._check_eos_threshold(log_probs)
                log_probs[:, self.eos_index] = mask_by_condition(
                    log_probs[:, self.eos_index],
                    cond,
                    fill_value=self.minus_inf,
                )

            scores = sequence_scores.unsqueeze(1).expand(-1, self.vocab_size)
            scores = scores + log_probs

            # Keep the original value
            log_probs_clone = log_probs.clone().reshape(batch_size, -1)

            # length normalization
            if self.length_normalization:
                scores = scores / (t + 1)

            # keep topk beams
            scores, candidates = scores.view(batch_size, -1).topk(
                self.beam_size, dim=-1
            )

            # The input for the next step, also the output of current step.
            inp_tokens = (candidates % self.vocab_size).view(n_bh)

            scores = scores.view(n_bh)
            sequence_scores = scores

            # recover the length normalization
            if self.length_normalization:
                sequence_scores = sequence_scores * (t + 1)

            # The index of which beam the current top-K output came from in (t-1) timesteps.
            predecessors = (
                torch.div(candidates, self.vocab_size, rounding_mode="floor")
                + self.beam_offset.unsqueeze(1).expand_as(candidates)
            ).view(n_bh)

            # Permute the memory to synchoronize with the output.
            if self.attn_weight:
                memory = self.permute_mem(memory, index=predecessors)

            if self.scorer is not None:
                scorer_memory = self.scorer.permute_scorer_mem(
                    scorer_memory, index=predecessors, candidates=candidates
                )

            # If using_max_attn_shift, then the previous attn peak has to be permuted too.
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
            ].reshape(n_bh)

            # Update alived_log_probs
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

            # Block the paths that have reached eos.
            sequence_scores.masked_fill_(is_eos, float("-inf"))

        if not self._check_full_beams(hyps_and_scores, self.beam_size):
            # Using all eos to fill-up the hyps.
            eos = torch.zeros(n_bh, device=device).fill_(self.eos_index).long()
            _ = self._update_hyp_and_scores(
                eos,
                alived_seq,
                alived_log_probs,
                hyps_and_scores,
                scores,
                timesteps=max_decode_steps,
            )

        (
            topk_hyps,
            topk_lengths,
            topk_scores,
            topk_log_probs,
        ) = self._get_topk_prediction(hyps_and_scores, topk=self.topk,)

        return topk_hyps, topk_lengths, topk_scores, topk_log_probs

    def permute_mem(self, memory, index):
        """This method permutes the seq2seq model memory
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

    Arguments
    ---------
    embedding : torch.nn.Module
        An embedding layer.
    decoder : torch.nn.Module
        Attentional RNN decoder.
    linear : torch.nn.Module
        A linear output layer.
    temperature : float
        Temperature factor applied to softmax. It changes the probability
        distribution, being softer when T>1 and sharper with T<1.
    **kwargs
        see S2SBeamSearcher, arguments are directly passed.

    Example
    -------
    >>> import speechbrain as sb
    >>> vocab_size = 5
    >>> emb = torch.nn.Embedding(vocab_size, 3)
    >>> dec = sb.nnet.RNN.AttentionalRNNDecoder(
    ...     "gru", "content", 3, 3, 1, enc_dim=7, input_size=3
    ... )
    >>> lin = sb.nnet.linear.Linear(n_neurons=vocab_size, input_size=3)
    >>> coverage_scorer = sb.decoders.scorer.CoverageScorer(vocab_size)
    >>> scorer = sb.decoders.scorer.ScorerBuilder(
    ...     full_scorers = [coverage_scorer],
    ...     partial_scorers = [],
    ...     weights= dict(coverage=1.5)
    ... )
    >>> searcher = S2SRNNBeamSearcher(
    ...     embedding=emb,
    ...     decoder=dec,
    ...     linear=lin,
    ...     bos_index=4,
    ...     eos_index=4,
    ...     min_decode_ratio=0,
    ...     max_decode_ratio=1,
    ...     beam_size=2,
    ...     vocab_size=vocab_size,
    ...     scorer=scorer,
    ... )
    >>> enc = torch.rand([2, 6, 7])
    >>> wav_len = torch.rand([2])
    >>> topk_hyps, topk_lengths, _, _ = searcher(enc, wav_len)
    """

    def __init__(
        self, embedding, decoder, linear, temperature=1.0, **kwargs,
    ):
        super(S2SRNNBeamSearcher, self).__init__(**kwargs)
        self.emb = embedding
        self.dec = decoder
        self.fc = linear
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.temperature = temperature

    def reset_mem(self, batch_size, device):
        hs = None
        self.dec.attn.reset()
        c = torch.zeros(batch_size, self.dec.attn_dim, device=device)
        return hs, c

    def forward_step(self, inp_tokens, memory, enc_states, enc_lens):
        with torch.no_grad():
            hs, c = memory
            e = self.emb(inp_tokens)
            dec_out, hs, c, w = self.dec.forward_step(
                e, hs, c, enc_states, enc_lens
            )
            log_probs = self.softmax(self.fc(dec_out) / self.temperature)
            # average attn weight of heads when attn_type is multiheadlocation
            if self.dec.attn_type == "multiheadlocation":
                w = torch.mean(w, dim=1)
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


class S2STransformerBeamSearcher(S2SBeamSearcher):
    """This class implements the beam search decoding
    for Transformer.
    See also S2SBaseSearcher(), S2SBeamSearcher().

    Arguments
    ---------
    model : torch.nn.Module
        The model to use for decoding.
    linear : torch.nn.Module
        A linear output layer.
    **kwargs
        Arguments to pass to S2SBeamSearcher

    Example:
    --------
    >>> # see recipes/LibriSpeech/ASR_transformer/experiment.py
    """

    def __init__(
        self, modules, temperature=1.0, **kwargs,
    ):
        super(S2STransformerBeamSearcher, self).__init__(**kwargs)

        self.model = modules[0]
        self.fc = modules[1]
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.temperature = temperature

    def reset_mem(self, batch_size, device):
        return None

    def permute_mem(self, memory, index):
        memory = torch.index_select(memory, dim=0, index=index)
        return memory

    def forward_step(self, inp_tokens, memory, enc_states, enc_lens):
        if memory is None:
            memory = torch.empty(
                inp_tokens.size(0), 0, device=inp_tokens.device
            )
        # Append the predicted token of the previous step to existing memory.
        memory = torch.cat([memory, inp_tokens.unsqueeze(1)], dim=-1)
        pred, attn = self.model.decode(memory, enc_states)
        prob_dist = self.softmax(self.fc(pred) / self.temperature)
        return prob_dist[:, -1, :], memory, attn


def inflate_tensor(tensor, times, dim):
    """This function inflates the tensor for times along dim.

    Arguments
    ---------
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
    """This function will mask some element in the tensor with fill_value, if condition=False.

    Arguments
    ---------
    tensor : torch.Tensor
        The tensor to be masked.
    cond : torch.BoolTensor
        This tensor has to be the same size as tensor.
        Each element represents whether to keep the value in tensor.
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
