"""Decoders and output normalization for CTC.

Authors
 * Mirco Ravanelli 2020
 * Aku Rouhe 2020
 * Sung-Lin Yeh 2020
 * Adel Moumen 2023
"""
import torch
from itertools import groupby
from speechbrain.dataio.dataio import length_to_mask
import math
import dataclasses
import numpy as np
import heapq
import logging

logger = logging.getLogger(__name__)
import torch
import math
import dataclasses
import numpy as np
import heapq
import logging

from typing import (
    Dict,
    List,
    Optional,
    Union,
    Any
)

from speechbrain.decoders.language_model import (
    LanguageModel,
    load_unigram_set_from_arpa,
)


from typing import (
    Dict,
    Optional,
)

from speechbrain.decoders.language_model import (
    LanguageModel,
    load_unigram_set_from_arpa,
)


class CTCPrefixScore:
    """This class implements the CTC prefix score of Algorithm 2 in
    reference: https://www.merl.com/publications/docs/TR2017-190.pdf.
    Official implementation: https://github.com/espnet/espnet/blob/master/espnet/nets/ctc_prefix_score.py

    Arguments
    ---------
    x : torch.Tensor
        The encoder states.
    enc_lens : torch.Tensor
        The actual length of each enc_states sequence.
    batch_size : int
        The size of the batch.
    beam_size : int
        The width of beam.
    blank_index : int
        The index of the blank token.
    eos_index : int
        The index of the end-of-sequence (eos) token.
    ctc_window_size: int
        Compute the ctc scores over the time frames using windowing based on attention peaks.
        If 0, no windowing applied.
    """

    def __init__(
        self, x, enc_lens, blank_index, eos_index, ctc_window_size=0,
    ):
        self.blank_index = blank_index
        self.eos_index = eos_index
        self.batch_size = x.size(0)
        self.max_enc_len = x.size(1)
        self.vocab_size = x.size(-1)
        self.device = x.device
        self.minus_inf = -1e20
        self.last_frame_index = enc_lens - 1
        self.ctc_window_size = ctc_window_size
        self.prefix_length = 0

        # mask frames > enc_lens
        mask = 1 - length_to_mask(enc_lens)
        mask = mask.unsqueeze(-1).expand(-1, -1, x.size(-1)).eq(1)
        x.masked_fill_(mask, self.minus_inf)
        x[:, :, 0] = x[:, :, 0].masked_fill_(mask[:, :, 0], 0)

        # dim=0: xnb, nonblank posteriors, dim=1: xb, blank posteriors
        xnb = x.transpose(0, 1)
        xb = (
            xnb[:, :, self.blank_index]
            .unsqueeze(2)
            .expand(-1, -1, self.vocab_size)
        )

        # (2, L, batch_size * beam_size, vocab_size)
        self.x = torch.stack([xnb, xb])

        # indices of batch.
        self.batch_index = torch.arange(self.batch_size, device=self.device)

    @torch.no_grad()
    def forward_step(self, inp_tokens, states, candidates=None, attn=None):
        """This method if one step of forwarding operation
        for the prefix ctc scorer.

        Arguments
        ---------
        inp_tokens : torch.Tensor
            The last chars of prefix label sequences g, where h = g + c.
        states : tuple
            Previous ctc states.
        candidates : torch.Tensor
            (batch_size * beam_size, ctc_beam_size), The topk candidates for rescoring.
            If given, performing partial ctc scoring.
        """

        n_bh = inp_tokens.size(0)
        beam_size = n_bh // self.batch_size
        last_char = inp_tokens
        self.prefix_length += 1
        self.num_candidates = (
            self.vocab_size if candidates is None else candidates.size(-1)
        )
        if states is None:
            # r_prev: (L, 2, batch_size * beam_size)
            r_prev = torch.full(
                (self.max_enc_len, 2, self.batch_size, beam_size),
                self.minus_inf,
                device=self.device,
            )

            # Accumulate blank posteriors at each step
            r_prev[:, 1] = torch.cumsum(
                self.x[0, :, :, self.blank_index], 0
            ).unsqueeze(2)
            r_prev = r_prev.view(-1, 2, n_bh)
            psi_prev = torch.full(
                (n_bh, self.vocab_size), 0.0, device=self.device,
            )
        else:
            r_prev, psi_prev = states

        # for partial search
        if candidates is not None:
            # The first index of each candidate.
            cand_offset = self.batch_index * self.vocab_size
            scoring_table = torch.full(
                (n_bh, self.vocab_size),
                -1,
                dtype=torch.long,
                device=self.device,
            )
            # Assign indices of candidates to their positions in the table
            col_index = torch.arange(n_bh, device=self.device).unsqueeze(1)
            scoring_table[col_index, candidates] = torch.arange(
                self.num_candidates, device=self.device
            )
            # Select candidates indices for scoring
            scoring_index = (
                candidates
                + cand_offset.unsqueeze(1).repeat(1, beam_size).view(-1, 1)
            ).view(-1)
            x_inflate = torch.index_select(
                self.x.view(2, -1, self.batch_size * self.vocab_size),
                2,
                scoring_index,
            ).view(2, -1, n_bh, self.num_candidates)
        # for full search
        else:
            scoring_table = None
            x_inflate = (
                self.x.unsqueeze(3)
                .repeat(1, 1, 1, beam_size, 1)
                .view(2, -1, n_bh, self.num_candidates)
            )

        # Prepare forward probs
        r = torch.full(
            (self.max_enc_len, 2, n_bh, self.num_candidates,),
            self.minus_inf,
            device=self.device,
        )
        r.fill_(self.minus_inf)

        # (Alg.2-6)
        if self.prefix_length == 0:
            r[0, 0] = x_inflate[0, 0]
        # (Alg.2-10): phi = prev_nonblank + prev_blank = r_t-1^nb(g) + r_t-1^b(g)
        r_sum = torch.logsumexp(r_prev, 1)
        phi = r_sum.unsqueeze(2).repeat(1, 1, self.num_candidates)

        # (Alg.2-10): if last token of prefix g in candidates, phi = prev_b + 0
        if candidates is not None:
            for i in range(n_bh):
                pos = scoring_table[i, last_char[i]]
                if pos != -1:
                    phi[:, i, pos] = r_prev[:, 1, i]
        else:
            for i in range(n_bh):
                phi[:, i, last_char[i]] = r_prev[:, 1, i]

        # Start, end frames for scoring (|g| < |h|).
        # Scoring based on attn peak if ctc_window_size > 0
        if self.ctc_window_size == 0 or attn is None:
            start = max(1, self.prefix_length)
            end = self.max_enc_len
        else:
            _, attn_peak = torch.max(attn, dim=1)
            max_frame = torch.max(attn_peak).item() + self.ctc_window_size
            min_frame = torch.min(attn_peak).item() - self.ctc_window_size
            start = max(max(1, self.prefix_length), int(min_frame))
            end = min(self.max_enc_len, int(max_frame))

        # Compute forward prob log(r_t^nb(h)) and log(r_t^b(h)):
        for t in range(start, end):
            # (Alg.2-11): dim=0, p(h|cur step is nonblank) = [p(prev step=y) + phi] * p(c)
            rnb_prev = r[t - 1, 0]
            # (Alg.2-12): dim=1, p(h|cur step is blank) = [p(prev step is blank) + p(prev step is nonblank)] * p(blank)
            rb_prev = r[t - 1, 1]
            r_ = torch.stack([rnb_prev, phi[t - 1], rnb_prev, rb_prev]).view(
                2, 2, n_bh, self.num_candidates
            )
            r[t] = torch.logsumexp(r_, 1) + x_inflate[:, t]

        # Compute the predix prob, psi
        psi_init = r[start - 1, 0].unsqueeze(0)
        # phi is prob at t-1 step, shift one frame and add it to the current prob p(c)
        phix = torch.cat((phi[0].unsqueeze(0), phi[:-1]), dim=0) + x_inflate[0]
        # (Alg.2-13): psi = psi + phi * p(c)
        if candidates is not None:
            psi = torch.full(
                (n_bh, self.vocab_size), self.minus_inf, device=self.device,
            )
            psi_ = torch.logsumexp(
                torch.cat((phix[start:end], psi_init), dim=0), dim=0
            )
            # only assign prob to candidates
            for i in range(n_bh):
                psi[i, candidates[i]] = psi_[i]
        else:
            psi = torch.logsumexp(
                torch.cat((phix[start:end], psi_init), dim=0), dim=0
            )

        # (Alg.2-3): if c = <eos>, psi = log(r_T^n(g) + r_T^b(g)), where T is the length of max frames
        for i in range(n_bh):
            psi[i, self.eos_index] = r_sum[
                self.last_frame_index[i // beam_size], i
            ]

        # Exclude blank probs for joint scoring
        psi[:, self.blank_index] = self.minus_inf

        return psi - psi_prev, (r, psi, scoring_table)

    def permute_mem(self, memory, index):
        """This method permutes the CTC model memory
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

        r, psi, scoring_table = memory

        beam_size = index.size(1)
        n_bh = self.batch_size * beam_size

        # The first index of each batch.
        beam_offset = self.batch_index * beam_size
        # The index of top-K vocab came from in (t-1) timesteps at batch * beam * vocab dimension.
        cand_index = (
            index + beam_offset.unsqueeze(1).expand_as(index) * self.vocab_size
        ).view(n_bh)
        # synchronize forward prob
        psi = torch.index_select(psi.view(-1), dim=0, index=cand_index)
        psi = (
            psi.view(-1, 1)
            .repeat(1, self.vocab_size)
            .view(n_bh, self.vocab_size)
        )
        # The index of top-K vocab came from in (t-1) timesteps at batch * beam dimension.
        hyp_index = (
            torch.div(index, self.vocab_size, rounding_mode="floor")
            + beam_offset.unsqueeze(1).expand_as(index)
        ).view(n_bh)
        # synchronize ctc states
        if scoring_table is not None:
            selected_vocab = (index % self.vocab_size).view(-1)
            score_index = scoring_table[hyp_index, selected_vocab]
            score_index[score_index == -1] = 0
            cand_index = score_index + hyp_index * self.num_candidates

        r = torch.index_select(
            r.view(-1, 2, n_bh * self.num_candidates), dim=-1, index=cand_index,
        )
        r = r.view(-1, 2, n_bh)

        return r, psi


def filter_ctc_output(string_pred, blank_id=-1):
    """Apply CTC output merge and filter rules.

    Removes the blank symbol and output repetitions.

    Arguments
    ---------
    string_pred : list
        A list containing the output strings/ints predicted by the CTC system.
    blank_id : int, string
        The id of the blank.

    Returns
    -------
    list
        The output predicted by CTC without the blank symbol and
        the repetitions.

    Example
    -------
    >>> string_pred = ['a','a','blank','b','b','blank','c']
    >>> string_out = filter_ctc_output(string_pred, blank_id='blank')
    >>> print(string_out)
    ['a', 'b', 'c']
    """

    if isinstance(string_pred, list):
        # Filter the repetitions
        string_out = [i[0] for i in groupby(string_pred)]

        # Filter the blank symbol
        string_out = list(filter(lambda elem: elem != blank_id, string_out))
    else:
        raise ValueError("filter_ctc_out can only filter python lists")
    return string_out


def ctc_greedy_decode(probabilities, seq_lens, blank_id=-1):
    """Greedy decode a batch of probabilities and apply CTC rules.

    Arguments
    ---------
    probabilities : torch.tensor
        Output probabilities (or log-probabilities) from the network with shape
        [batch, probabilities, time]
    seq_lens : torch.tensor
        Relative true sequence lengths (to deal with padded inputs),
        the longest sequence has length 1.0, others a value between zero and one
        shape [batch, lengths].
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


@dataclasses.dataclass
class CTCBeam:
    """Contains all the info needed for decoding a beam."""

    text: str
    full_text: str
    next_word: str
    partial_word: str
    last_token: Optional[str]
    last_token_index: Optional[int]
    p: float = -math.inf
    p_b: float = -math.inf
    p_nb: float = -math.inf
    n_p_b: float = -math.inf
    n_p_nb: float = -math.inf
    score: float = -math.inf
    score_ctc: float = -math.inf

    @classmethod
    def from_lm_beam(cls, lm_beam):
        """ Create a CTCBeam from a LMCTCBeam"""
        return CTCBeam(
            text=lm_beam.text,
            full_text=lm_beam.full_text,
            next_word=lm_beam.next_word,
            partial_word=lm_beam.partial_word,
            last_token=lm_beam.last_token,
            last_token_index=lm_beam.last_token_index,
            p=lm_beam.p,
            p_b=lm_beam.p_b,
            p_nb=lm_beam.p_nb,
            n_p_b=lm_beam.n_p_b,
            n_p_nb=lm_beam.n_p_nb,
            score=lm_beam.score,
            score_ctc=lm_beam.score_ctc,
        )

    def step(self):
        self.p_b, self.p_nb = self.n_p_b, self.n_p_nb
        self.n_p_b = self.n_p_nb = -math.inf
        self.score_ctc = np.logaddexp(self.p_b, self.p_nb)
        self.score = self.score_ctc  # + self.lm_score


@dataclasses.dataclass
class LMCTCBeam(CTCBeam):
    lm_score: float = -math.inf


@dataclasses.dataclass
class CTCHypothesis:
    text: str
    last_lm_state: None
    score: float  # Cumulative logit score
    lm_score: float  # Cumulative language model + logit score
    timesteps: list = None


class CTCBaseSearcher(torch.nn.Module):
    """CTCBaseSearcher class to be inherited by other
    CTC beam searchers.

    Arguments
    ---------
    blank_index : int
        The index of the blank token.
    vocab_list : list
        The list of the vocabulary tokens.
    space_index : int
        The index of the space token.
    kenlm_model_path : str
        The path to the kenlm model. Use .bin for a faster loading.
    unigrams : dict
        The unigram dictionary.
    beam_size : int
        The width of the beam.
    beam_prune_logp : float
        The pruning threshold for the beam.
    token_prune_min_logp : float
        The pruning threshold for the tokens.
    prune_history : bool
        Whether to prune the history.
    blank_skip_threshold : float
        The threshold for skipping the frames when the log probability
        of the blank token is higher than this value.
    topk : int
        The number of top hypotheses to return.
    """

    def __init__(
        self,
        blank_index,
        vocab_list,
        space_index=-1,
        kenlm_model_path=None,
        unigrams=None,
        beam_size=100,
        beam_prune_logp=-10.0,
        token_prune_min_logp=-5.0,
        prune_history=True,
        blank_skip_threshold=None,  # by default the pruning is not applied
        topk=1,
    ):
        super().__init__()

        self.blank_index = blank_index
        self.space_index = space_index
        self.kenlm_model_path = kenlm_model_path
        self.unigrams = unigrams
        self.vocab_list = vocab_list
        self.beam_size = beam_size
        self.beam_prune_logp = beam_prune_logp
        self.token_prune_min_logp = token_prune_min_logp
        self.prune_history = prune_history
        self.blank_skip_threshold = blank_skip_threshold
        self.topk = topk

        # sentencepiece
        self.spm_token = "▁"
        self.is_spm = any([s.startswith(self.spm_token) for s in vocab_list])

        self.kenlm_model = None
        if kenlm_model_path is not None:
            try:
                import kenlm  # type: ignore
            except ImportError:
                raise ImportError(
                    "kenlm python bindings are not installed. To install it use: "
                    "pip install https://github.com/kpu/kenlm/archive/master.zip"
                )

            self.kenlm_model = kenlm.Model(kenlm_model_path)

        if kenlm_model_path is not None and kenlm_model_path.endswith(".arpa"):
            logger.info(
                "Using arpa instead of binary LM file, decoder instantiation might be slow."
            )

        if unigrams is None and kenlm_model_path is not None:
            if kenlm_model_path.endswith(".arpa"):
                unigrams = load_unigram_set_from_arpa(kenlm_model_path)
            else:
                logger.warning(
                    "Unigrams not provided and cannot be automatically determined from LM file (only "
                    "arpa format). Decoding accuracy might be reduced."
                )

        if self.kenlm_model is not None:
            self.lm = LanguageModel(self.kenlm_model, unigrams)
        else:
            self.lm = None

        if not self.is_spm and space_index == -1 and self.lm is not None:
            raise ValueError("space_index must be set")
        
    def partial_decoding(
        self,
        log_probs,
        beams,
        cached_lm_scores,
        cached_p_lm_scores,
        processed_frames=0,
    ):
        """ Perform a single step of decoding.

        Arguments
        ---------
        log_probs : torch.Tensor
            The log probabilities of the CTC output.
        beams : list
            The list of the beams.
        cached_lm_scores : dict
            The cached language model scores.
        cached_p_lm_scores : dict
            The cached prefix language model scores.
        processed_frames : int
            The start frame of the current decoding step.
        """
        raise NotImplementedError

    def normalize_whitespace(self, text):
        """Efficiently normalize whitespace.

        Arguments
        ---------
        text : str
            The text to normalize.
        """
        return " ".join(text.split())

    def merge_tokens(self, token_1, token_2):
        """ Merge two tokens, and avoid empty ones.

        Arguments
        ---------
        token_1 : str
            The first token.
        token_2 : str
            The second token.
        """
        if len(token_2) == 0:
            text = token_1
        elif len(token_1) == 0:
            text = token_2
        else:
            text = token_1 + " " + token_2
        return text

    def merge_beams(self, beams):
        """ Merge beams with the same text.

        Arguments
        ---------
        beams : list
            The list of the beams.
        """
        beam_dict = {}
        for beam in beams:
            new_text = self.merge_tokens(beam.text, beam.next_word)
            hash_idx = (new_text, beam.partial_word, beam.last_token)
            if hash_idx not in beam_dict:
                beam_dict[hash_idx] = beam
            else:
                # We've already seen this text - we want to combine the scores
                beam_dict[hash_idx] = dataclasses.replace(
                    beam,
                    score=np.logaddexp(beam_dict[hash_idx].score, beam.score),
                )
        return list(beam_dict.values())

    def sort_beams(self, beams):
        """ Sort beams by lm_score.

        Arguments
        ---------
        beams : list
            The list of the beams.
        """
        return heapq.nlargest(self.beam_size, beams, key=lambda x: x.lm_score)

    def _prune_history(self, beams: List[CTCBeam], lm_order: int):
        """Filter out beams that are the same over max_ngram history.

        Since n-gram language models have a finite history when scoring a new token, we can use that
        fact to prune beams that only differ early on (more than n tokens in the past) and keep only the
        higher scoring ones. Note that this helps speed up the decoding process but comes at the cost of
        some amount of beam diversity. If more than the top beam is used in the output it should
        potentially be disabled.

        Arguments
        ---------
        beams : list
            The list of the beams.
        lm_order : int
            The order of the language model.
        """
        # let's keep at least 1 word of history
        min_n_history = max(1, lm_order - 1)
        seen_hashes = set()
        filtered_beams = []
        # for each beam after this, check if we need to add it
        for lm_beam in beams:
            # hash based on history that can still affect lm scoring going forward
            hash_idx = (
                tuple(lm_beam.text.split()[-min_n_history:]),
                lm_beam.partial_word,
                lm_beam.last_token,
            )
            if hash_idx not in seen_hashes:
                filtered_beams.append(CTCBeam.from_lm_beam(lm_beam))
                seen_hashes.add(hash_idx)
        return filtered_beams

    def finalize_decoding(
        self,
        beams,
        cached_lm_scores,
        cached_p_lm_scores,
        force_next_word=False,
        is_end=False,
    ):
        """ Finalize the decoding process.

        Arguments
        ---------
        beams : list
            The list of the beams.
        cached_lm_scores : dict
            The cached language model scores.
        cached_p_lm_scores : dict
            The cached prefix language model scores.
        force_next_word : bool
            Whether to force the next word.
        is_end : bool
            Whether the end of the sequence has been reached.
        """
        if force_next_word or is_end:
            new_beams = []
            for beam in beams:
                new_beams.append(
                    CTCBeam(
                        text=beam.text,
                        full_text=beam.full_text,
                        next_word=beam.partial_word,
                        partial_word="",
                        last_token=None,
                        last_token_index=None,
                        score=beam.score,
                    )
                )

            new_beams = self.merge_beams(new_beams)
        else:
            new_beams = list(beams)

        scored_beams = self.get_lm_beams(
            new_beams, cached_lm_scores, cached_p_lm_scores,
        )
        # remove beam outliers
        max_score = max([b.lm_score for b in scored_beams])
        scored_beams = [
            b
            for b in scored_beams
            if b.lm_score >= max_score + self.beam_prune_logp
        ]
        return self.sort_beams(scored_beams)

    def decode_beams(self, log_probs: torch.Tensor, wav_lens: Optional[torch.Tensor] = None, lm_start_state: Any =None):
        """Decodes the log probabilities of the CTC output.

        Arguments
        ---------
        log_probs : torch.Tensor
            The log probabilities of the CTC output.
            The expected shape is [batch_size, seq_length, vocab_size].
        wav_lens : torch.Tensor, optional (default: None)
            The SpeechBrain's relative length of the wav input.
        lm_start_state : Any, optional (default: None)
            The start state of the language model.

        Returns
        -------
        list of list
            The list of topk list of the decoded hypotheses.
        """
        # compute wav_lens and cast to numpy as it is faster
        if wav_lens is not None:
            wav_lens = log_probs.size(1) * wav_lens
            wav_lens = wav_lens.cpu().numpy().astype(int)

        log_probs = log_probs.cpu().numpy()

        hyps = [
            self.decode_log_probs(log_prob, wav_len, lm_start_state)
            for log_prob, wav_len in zip(log_probs, wav_lens)
        ]
        return hyps

    def __call__(self, log_probs: torch.Tensor, wav_lens: Optional[torch.Tensor] = None, lm_start_state: Any =None):
        """Decodes the log probabilities of the CTC output.

        Arguments
        ---------
        log_probs : torch.Tensor
            The log probabilities of the CTC output.
            The expected shape is [batch_size, seq_length, vocab_size].
        wav_lens : torch.Tensor, optional (default: None)
            The SpeechBrain's relative length of the wav input.
        lm_start_state : Any, optional (default: None)
            The start state of the language model.

        Returns
        -------
        list of list
            The list of topk list of the decoded hypotheses.
        """
        return self.decode_beams(log_probs, wav_lens, lm_start_state)

    def partial_decode_beams(
        self,
        log_probs: torch.Tensor,
        cached_lm_scores: dict,
        cached_p_lm_scores: dict,
        beams: List[CTCBeam],
        processed_frames: int,
        force_next_word=False,
        is_end=False,
    ):
        """ Perform a single step of decoding.

        Arguments
        ---------
        log_probs : torch.Tensor
            The log probabilities of the CTC output.
        cached_lm_scores : dict
            The cached language model scores.
        cached_p_lm_scores : dict
            The cached prefix language model scores.
        beams : list
            The list of the beams.
        processed_frames : int
            The start frame of the current decoding step.
        force_next_word : bool, optional (default: False)
            Whether to force the next word.
        is_end : bool, optional (default: False)
            Whether the end of the sequence has been reached.

        Returns
        -------
        list
            The list of the beams.
        """
        beams = self.partial_decoding(
            log_probs,
            beams,
            cached_lm_scores,
            cached_p_lm_scores,
            processed_frames=processed_frames,
        )

        trimmed_beams = self.finalize_decoding(
            beams,
            cached_lm_scores,
            cached_p_lm_scores,
            force_next_word=force_next_word,
            is_end=is_end,
        )

        return trimmed_beams

    def decode_log_probs(self, log_probs: torch.Tensor, wav_len: int, lm_start_state: Optional[Any] = None) -> List[CTCHypothesis]:
        """Decodes the log probabilities of the CTC output.

        Arguments
        ---------
        log_probs : torch.Tensor
            The log probabilities of the CTC output.
            The expected shape is [seq_length, vocab_size].
        wav_len : int
            The length of the wav input.
        lm_start_state : Any, optional (default: None)
            The start state of the language model. 

        Returns
        -------
        list
            The topk list of the decoded hypotheses.
        """
        language_model = self.lm
        if language_model is None:
            cached_lm_scores = {}
        else:
            if lm_start_state is None:
                start_state = language_model.get_start_state()
            else:
                start_state = lm_start_state
            cached_lm_scores = {("", False): (0.0, start_state)}
        cached_p_lm_scores: Dict[str, float] = {}

        beams = [
            CTCBeam(
                text="",
                full_text="",
                next_word="",
                partial_word="",
                last_token=None,
                last_token_index=None,
                score=0.0,
                score_ctc=0.0,
                p_b=0.0,
            )
        ]

        beams = self.partial_decoding(
            log_probs, wav_len, beams, cached_lm_scores, cached_p_lm_scores,
        )

        trimmed_beams = self.finalize_decoding(
            beams,
            cached_lm_scores,
            cached_p_lm_scores,
            force_next_word=True,
            is_end=True,
        )

        # remove unnecessary information from beams
        output_beams = [
            CTCHypothesis(
                text=self.normalize_whitespace(lm_beam.text),
                last_lm_state=(
                    cached_lm_scores[(lm_beam.text, True)][-1]
                    if (lm_beam.text, True) in cached_lm_scores
                    else None
                ),
                score=lm_beam.score,
                lm_score=lm_beam.lm_score,
            )
            for lm_beam in trimmed_beams
        ][: self.topk]
        return output_beams


class CTCBeamSearch(CTCBaseSearcher):
    """CTC Beam Search is a Beam Search for CTC which does not keep track of
    the blank and non-blank probabilities. Each new token probability is 
    added to the general score, and each beams that share the same text are
    merged together.

    The input is expected to be a log-probabilities tensor of shape [batch, time, vocab_size].

    The main advantage of this CTCBeamSearch over the CTCPrefixBeamSearch is that it is
    relatively faster, and obtains slightly better results. However, the implementation is
    based on the one from the PyCTCDecode toolkit, adpated for the SpeechBrain's needs and does
    not follow a specific paper. We do recommand to use the CTCPrefixBeamSearch if you want
    to cite the appropriate paper for the decoding method.

    Several heuristics are implemented to speed up the decoding process:
    - pruning of the beam : the beams are pruned if their score is lower than
        the best beam score minus the beam_prune_logp
    - pruning of the tokens : the tokens are pruned if their score is lower than
        the token_prune_min_logp
    - pruning of the history : the beams are pruned if they are the same over
        max_ngram history
    - skipping of the blank : the frame is skipped if the blank probability is
        higher than the blank_skip_threshold
        
    Arguments
    ---------
    **kwargs
        see CTCBaseSearcher, arguments are directly passed.

    Example
    -------
    >>> import torch
    >>> from speechbrain.decoders import CTCBeamSearch
    >>> probs = torch.tensor([[[0.2, 0.0, 0.8], 
    ...                   [0.4, 0.0, 0.6]]])
    >>> log_probs = torch.log(probs)
    >>> lens = torch.tensor([1.0])
    >>> blank_index = 2
    >>> vocab_list = ['a', 'b', '-']
    >>> decoder = CTCBeamSearch(blank_index=blank_index, vocab_list=vocab_list)
    >>> decoder(probs, lens)
    [[CTCHypothesis(text='a', last_lm_state=None, score=1.9971160035663789, lm_score=1.9971160035663789, timesteps=None)]]
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_lm_beams(
        self,
        beams: List[CTCBeam],
        cached_lm_scores: dict,
        cached_partial_token_scores: dict,
        is_eos=False,
    ) -> List[LMCTCBeam]:
        """Score the beams with the language model if not None, and 
        return the new beams.

        This function is modified and adapted from
        https://github.com/kensho-technologies/pyctcdecode

        Arguments
        ---------
        beams : list
            The list of the beams.
        cached_lm_scores : dict
            The cached language model scores.
        cached_partial_token_scores : dict
            The cached partial token scores.
        is_eos : bool (default: False)
            Whether the end of the sequence has been reached.

        Returns
        -------
        new_beams : list
            The list of the new beams.
        """
        if self.lm is None:
            new_beams = []
            for beam in beams:
                new_text = self.merge_tokens(beam.text, beam.next_word)
                new_beams.append(
                    LMCTCBeam(
                        text=new_text,
                        full_text=beam.full_text,
                        next_word="",
                        partial_word=beam.partial_word,
                        last_token=beam.last_token,
                        last_token_index=beam.last_token,
                        score=beam.score,
                        lm_score=beam.score,
                    )
                )
            return new_beams
        else:
            new_beams = []
            for beam in beams:
                # fast token merge
                new_text = self.merge_tokens(beam.text, beam.next_word)
                cache_key = (new_text, is_eos)
                if cache_key not in cached_lm_scores:
                    prev_raw_lm_score, start_state = cached_lm_scores[
                        (beam.text, False)
                    ]
                    score, end_state = self.lm.score(
                        start_state, beam.next_word, is_last_word=is_eos
                    )
                    raw_lm_score = prev_raw_lm_score + score
                    cached_lm_scores[cache_key] = (raw_lm_score, end_state)
                lm_score, _ = cached_lm_scores[cache_key]
                word_part = beam.partial_word
                if len(word_part) > 0:
                    if word_part not in cached_partial_token_scores:

                        cached_partial_token_scores[
                            word_part
                        ] = self.lm.score_partial_token(word_part)
                    lm_score += cached_partial_token_scores[word_part]

                new_beams.append(
                    LMCTCBeam(
                        text=new_text,
                        full_text=beam.full_text,
                        next_word="",
                        partial_word=word_part,
                        last_token=beam.last_token,
                        last_token_index=beam.last_token,
                        score=beam.score,
                        lm_score=beam.score + lm_score,
                    )
                )
            return new_beams

    def partial_decoding(
        self,
        log_probs: torch.Tensor,
        wav_len: int,
        beams: List[CTCBeam],
        cached_lm_scores: dict,
        cached_p_lm_scores: dict,
        processed_frames: int = 0,
    ) -> List[CTCBeam]:
        """Perform CTC Prefix Beam Search decoding. 

        If self.lm is not None, the language model scores are computed and added to the CTC scores.

        Arguments
        ---------
        log_probs : torch.Tensor
            The log probabilities of the CTC input.
            Shape: (seq_length, vocab_size)
        wav_len : int
            The length of the input sequence.
        beams : list
            The list of CTCBeam objects.
        cached_lm_scores : dict
            The cached language model scores.
        cached_p_lm_scores : dict
            The cached prefix language model scores.
        processed_frames : int
            The start frame of the current decoding step. (default: 0)

        Returns
        -------
        beams : list
            The list of CTCBeam objects.
        """
        # select only the valid frames i.e. the frames that are not padded
        log_probs = log_probs[:wav_len]

        for _, logit_col in enumerate(log_probs, start=processed_frames):

            if (
                self.blank_skip_threshold is not None
                and logit_col[self.blank_index] >= self.blank_skip_threshold
            ):
                continue

            max_index = logit_col.argmax()
            tokens_index_list = set(
                np.where(logit_col > self.token_prune_min_logp)[0]
            ) | {max_index}
            new_beams = []

            for token_index in tokens_index_list:
                p_token = logit_col[token_index]
                token = self.vocab_list[token_index]

                for beam in beams:

                    if (
                        token_index == self.blank_index
                        or beam.last_token == token
                    ):

                        new_beams.append(
                            CTCBeam(
                                text=beam.text,
                                full_text=beam.full_text,
                                next_word=beam.next_word,
                                partial_word=beam.partial_word,
                                last_token=token,
                                last_token_index=token_index,
                                score=beam.score + p_token,
                            )
                        )

                    elif self.is_spm and token[:1] == self.spm_token:
                        clean_token = token[1:]

                        new_beams.append(
                            CTCBeam(
                                text=beam.text,
                                full_text=beam.full_text,
                                next_word=beam.partial_word,
                                partial_word=clean_token,
                                last_token=token,
                                last_token_index=token_index,
                                score=beam.score + p_token,
                            )
                        )

                    elif not self.is_spm and token_index == self.space_index:
                        new_beams.append(
                            CTCBeam(
                                text=beam.text,
                                full_text=beam.full_text,
                                next_word=beam.partial_word,
                                partial_word="",
                                last_token=token,
                                last_token_index=token_index,
                                score=beam.score + p_token,
                            )
                        )
                    else:

                        new_beams.append(
                            CTCBeam(
                                text=beam.text,
                                full_text=beam.full_text,
                                next_word=beam.next_word,
                                partial_word=beam.partial_word + token,
                                last_token=token,
                                last_token_index=token_index,
                                score=beam.score + p_token,
                            )
                        )

            new_beams = self.merge_beams(new_beams)
            scored_beams = self.get_lm_beams(
                new_beams, cached_lm_scores, cached_p_lm_scores,
            )
            # remove beam outliers
            max_score = max([b.lm_score for b in scored_beams])
            scored_beams = [
                b
                for b in scored_beams
                if b.lm_score >= max_score + self.beam_prune_logp
            ]

            trimmed_beams = self.sort_beams(scored_beams)

            if self.prune_history:
                lm_order = 1 if self.lm is None else self.lm.order
                beams = self._prune_history(trimmed_beams, lm_order=lm_order)
            else:
                beams = [CTCBeam.from_lm_beam(b) for b in trimmed_beams]

        return beams


class CTCPrefixBeamSearch(CTCBaseSearcher):
    """CTC Prefix Beam Search is based on the paper
    `First-Pass Large Vocabulary Continuous Speech Recognition using Bi-Directional Recurrent DNNs`
    by Awni Y. Hannun and al (https://arxiv.org/abs/1408.2873). 

    The implementation keep tracks of the blank and non-blank probabilities. 
    It also suppors n-gram scoring on words and SentencePiece tokens. The input
    is expected to be a log-probabilities tensor of shape [batch, time, vocab_size].

    Several heuristics are implemented to speed up the decoding process:
    - pruning of the beam : the beams are pruned if their score is lower than
        the best beam score minus the beam_prune_logp
    - pruning of the tokens : the tokens are pruned if their score is lower than
        the token_prune_min_logp
    - pruning of the history : the beams are pruned if they are the same over
        max_ngram history
    - skipping of the blank : the frame is skipped if the blank probability is
        higher than the blank_skip_threshold
        
    Arguments
    ---------
    **kwargs
        see CTCBaseSearcher, arguments are directly passed.

    Example
    -------
    >>> import torch
    >>> from speechbrain.decoders import CTCPrefixBeamSearch
    >>> probs = torch.tensor([[[0.2, 0.0, 0.8], 
    ...                   [0.4, 0.0, 0.6]]])
    >>> log_probs = torch.log(probs)
    >>> lens = torch.tensor([1.0])
    >>> blank_index = 2
    >>> vocab_list = ['a', 'b', '-']
    >>> decoder = CTCPrefixBeamSearch(blank_index=blank_index, vocab_list=vocab_list)
    >>> decoder(probs, lens)
    [[CTCHypothesis(text='a', last_lm_state=None, score=-0.6539264320473421, lm_score=-0.6539264320473421, timesteps=None)]]
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_lm_beams(
        self,
        beams: List[CTCBeam],
        cached_lm_scores: dict,
        cached_partial_token_scores: dict,
        is_eos=False,
    ) -> List[LMCTCBeam]:
        """Score the beams with the language model if not None, and 
        return the new beams.

        This function is modified and adapted from
        https://github.com/kensho-technologies/pyctcdecode

        Arguments
        ---------
        beams : list
            The list of the beams.
        cached_lm_scores : dict
            The cached language model scores.
        cached_partial_token_scores : dict
            The cached partial token scores.
        is_eos : bool (default: False)
            Whether the end of the sequence has been reached.

        Returns
        -------
        new_beams : list
            The list of the new beams.
        """
        if self.lm is None:
            new_beams = []
            for beam in beams:
                new_text = self.merge_tokens(beam.full_text, beam.next_word)
                new_beams.append(
                    LMCTCBeam(
                        text=beam.text,
                        full_text=new_text,
                        next_word="",
                        partial_word=beam.partial_word,
                        last_token=beam.last_token,
                        last_token_index=beam.last_token_index,
                        p=beam.p,
                        p_b=beam.p_b,
                        p_nb=beam.p_nb,
                        n_p_b=beam.n_p_b,
                        n_p_nb=beam.n_p_nb,
                        score=beam.score,
                        score_ctc=beam.score_ctc,
                        lm_score=beam.score,
                    )
                )
            return new_beams
        else:
            new_beams = []
            for beam in beams:
                # fast token merge
                new_text = self.merge_tokens(beam.full_text, beam.next_word)
                cache_key = (new_text, is_eos)
                if cache_key not in cached_lm_scores:
                    prev_raw_lm_score, start_state = cached_lm_scores[
                        (beam.full_text, False)
                    ]
                    score, end_state = self.lm.score(
                        start_state, beam.next_word, is_last_word=is_eos
                    )
                    raw_lm_score = prev_raw_lm_score + score
                    cached_lm_scores[cache_key] = (raw_lm_score, end_state)
                lm_score, _ = cached_lm_scores[cache_key]
                word_part = beam.partial_word

                if len(word_part) > 0:
                    if word_part not in cached_partial_token_scores:

                        cached_partial_token_scores[
                            word_part
                        ] = self.lm.score_partial_token(word_part)
                    lm_score += cached_partial_token_scores[word_part]

                new_beams.append(
                    LMCTCBeam(
                        text=beam.text,
                        full_text=new_text,
                        next_word="",
                        partial_word=beam.partial_word,
                        last_token=beam.last_token,
                        last_token_index=beam.last_token_index,
                        p=beam.p,
                        p_b=beam.p_b,
                        p_nb=beam.p_nb,
                        n_p_b=beam.n_p_b,
                        n_p_nb=beam.n_p_nb,
                        score=beam.score,
                        score_ctc=beam.score_ctc,
                        lm_score=beam.score + lm_score,
                    )
                )
            return new_beams

    def _get_new_beam(
        self,
        new_prefix: str,
        new_token: str,
        new_token_index: int,
        beams: List[CTCBeam],
        p: float,
        previous_beam: CTCBeam,
    ) -> CTCBeam:
        """Create a new beam and add it to the list of beams.

        Arguments
        ---------
        new_prefix : str
            The new prefix.
        new_token : str
            The new token.
        new_token_index : int
            The index of the new token.
        beams : list
            The list of beams.
        p : float
            The probability of the new token.
        previous_beam : CTCBeam
            The previous beam.

        Returns
        -------
        new_beam : CTCBeam
            The new beam.
        """
        for beam in beams:
            if beam.text == new_prefix:
                if p and p > beam.p:
                    beam.p = p
                return beam

        if not self.is_spm and new_token_index == self.space_index:
            new_beam = CTCBeam(
                text=new_prefix,
                full_text=previous_beam.full_text,
                next_word=previous_beam.partial_word,
                partial_word="",
                last_token=new_token,
                last_token_index=new_token_index,
                score=-math.inf,
                score_ctc=-math.inf,
                p_b=-math.inf,
            )
        elif self.is_spm and new_token[:1] == self.spm_token:
            clean_token = new_token[1:]
            new_prefix = previous_beam.text + " " + clean_token
            new_beam = CTCBeam(
                text=new_prefix,
                full_text=previous_beam.full_text,
                next_word=previous_beam.partial_word,
                partial_word=clean_token,
                last_token=new_token,
                last_token_index=new_token_index,
                score=-math.inf,
                score_ctc=-math.inf,
                p_b=-math.inf,
            )
        elif new_token_index == previous_beam.last_token_index:
            new_beam = CTCBeam(
                text=new_prefix,
                full_text=previous_beam.full_text,
                next_word="",
                partial_word=previous_beam.partial_word,
                last_token=new_token,
                last_token_index=new_token_index,
                score=-math.inf,
                score_ctc=-math.inf,
                p_b=-math.inf,
            )
        else:
            new_beam = CTCBeam(
                text=new_prefix,
                full_text=previous_beam.full_text,
                next_word="",
                partial_word=previous_beam.partial_word + new_token,
                last_token=new_token,
                last_token_index=new_token_index,
                score=-math.inf,
                score_ctc=-math.inf,
                p_b=-math.inf,
            )
        beams.append(new_beam)
        if previous_beam:
            new_beam.p = previous_beam.p
        return new_beam

    def partial_decoding(
        self,
        log_probs: torch.Tensor,
        wav_len: int,
        beams: List[CTCBeam],
        cached_lm_scores: dict,
        cached_p_lm_scores: dict,
        processed_frames: int = 0,
    ) -> List[CTCBeam]:
        """Perform CTC Prefix Beam Search decoding. 

        If self.lm is not None, the language model scores are computed and added to the CTC scores.

        Arguments
        ---------
        log_probs : torch.Tensor
            The log probabilities of the CTC input.
            Shape: (seq_length, vocab_size)
        wav_len : int
            The length of the input sequence.
        beams : list
            The list of CTCBeam objects.
        cached_lm_scores : dict
            The cached language model scores.
        cached_p_lm_scores : dict
            The cached prefix language model scores.
        processed_frames : int
            The start frame of the current decoding step. (default: 0)

        Returns
        -------
        beams : list
            The list of CTCBeam objects.
        """
        # select only the valid frames, i.e., the frames that are not padded
        log_probs = log_probs[:wav_len]
        for _, logit_col in enumerate(log_probs, start=processed_frames):
            
            # skip blank frames
            if (
                self.blank_skip_threshold is not None
                and logit_col[self.blank_index] >= self.blank_skip_threshold
            ):
                continue

            max_index = logit_col.argmax()
            tokens_index_list = set(
                np.where(logit_col > self.token_prune_min_logp)[0]
            ) | {max_index}

            curr_beams = beams.copy()

            for token_index in tokens_index_list:
                p_token = logit_col[token_index]
                token = self.vocab_list[token_index]

                for beam in curr_beams:
                    p_b, p_nb = beam.p_b, beam.p_nb

                    # blank case
                    if token_index == self.blank_index:
                        beam.n_p_b = np.logaddexp(
                            beam.n_p_b, beam.score_ctc + p_token
                        )
                        continue

                    if token == beam.last_token:
                        beam.n_p_nb = np.logaddexp(beam.n_p_nb, p_nb + p_token)

                    new_text = beam.text + token

                    new_beam = self._get_new_beam(
                        new_text,
                        token,
                        token_index,
                        beams,
                        p=p_token,
                        previous_beam=beam,
                    )

                    n_p_nb = new_beam.n_p_nb

                    if token_index == beam.last_token_index and p_b > -math.inf:
                        n_p_nb = np.logaddexp(n_p_nb, p_b + p_token)
                    elif token_index != beam.last_token_index:
                        n_p_nb = np.logaddexp(n_p_nb, beam.score_ctc + p_token)
                    new_beam.n_p_nb = n_p_nb

            # update the CTC probabilities
            for beam in beams:
                beam.step()

            # kenLM scores
            scored_beams = self.get_lm_beams(
                beams, cached_lm_scores, cached_p_lm_scores,
            )

            # remove beams outliers
            max_score = max([b.lm_score for b in scored_beams])
            scored_beams = [
                b
                for b in scored_beams
                if b.lm_score >= max_score + self.beam_prune_logp
            ]
            trimmed_beams = self.sort_beams(scored_beams)

            if self.prune_history:
                lm_order = 1 if self.lm is None else self.lm.order
                beams = self._prune_history(trimmed_beams, lm_order=lm_order)
            else:
                beams = [CTCBeam.from_lm_beam(b) for b in trimmed_beams]

        return beams


class TorchAudioCTCBeamSearch:
    """TorchAudio CTC Beam Search Decoder.

    This class is a wrapper around the CTC decoder from TorchAudio. It provides a simple interface
    where you can either use the CPU or CUDA CTC decoder. The CPU decoder is slower but uses less
    memory. The CUDA decoder is faster but uses more memory. The CUDA decoder is also only available
    in the nightly version of torchaudio. A lot of features are missing in the CUDA decoder, such as
    the ability to use a language model, constraint search, and more. If you want to use those features,
    you have to use the CPU decoder.

    For more information about the CPU decoder, please refer to the documentation of TorchAudio:
    https://pytorch.org/audio/main/generated/torchaudio.models.decoder.ctc_decoder.html

    If you want to use the language model, or the lexicon search, please make sure that your 
    tokenizer/acoustic model uses the same tokens as the language model/lexicon. Otherwise, the decoding will fail. 

    Note: When using CUDA CTC decoder, the blank_index has to be 0. Furthermore, using CUDA CTC decoder 
    requires the nightly version of torchaudio and a lot of VRAM memory. Overall, we do recommand to use
    the CTCBeamSearch or CTCPrefixBeamSearch in SpeechBrain if you wants to use n-gram + beam search decoding.
    If you wants to have constraint search, please use the CPU version of torchaudio, and if you want to speedup
    as much as possible the decoding, please use the CUDA version.

    Arguments
    ---------
    tokens : list or str
        The list of tokens or the path to the tokens file.
        If this is a path, then the file should contain one token per line.
    lexicon : str, optional 
        Lexicon file containing the possible words and corresponding spellings. Each line consists of a word and its space separated spelling. 
        If None, uses lexicon-free decoding. (default: None)
    lm : str, optional
        A path containing KenLM language model or None if not using a language model. (default: None)
    lm_dict : str, optional
        File consisting of the dictionary used for the LM, with a word per line sorted by LM index. 
        If decoding with a lexicon, entries in lm_dict must also occur in the lexicon file. 
        If None, dictionary for LM is constructed using the lexicon file. (Default: None)
    topk : int, optional
        Number of top CTCHypothesis to return. (default: 1)
    beam_size : int, optional
        Numbers of hypotheses to hold after each decode step. (default: 50)
    beam_size_token : int, optional 
        Max number of tokens to consider at each decode step. If None, it is set to the total number of tokens. (default: None)
    beam_threshold : float, optional
        Threshold for pruning hypothesis. (default: 50)
    lm_weight : float, optional
        Weight of language model. (default: 2)
    word_score : float, optional
        Word insertion score. (default: 0)
    unk_score : float, optional
        Unknown word insertion score. (default: float("-inf"))
    sil_score : float, optional
        Silence insertion score. (default: 0)
    log_add : bool, optional
        Whether to use use logadd when merging hypotheses. (default: False)
    blank_index : int or str, optional
        Index of the blank token. If tokens is a file path, then this should be an str. Otherwise, this should be a int. (default: 0)
    sil_index : int or str, optional
        Index of the silence token. If tokens is a file path, then this should be an str. Otherwise, this should be a int. (default: 0)
    unk_word : str, optional
        Unknown word token. (default: "<unk>")
    using_cpu_decoder : bool, optional
        Whether to use the CPU decoder. If False, then the CUDA decoder is used. (default: True)
    blank_skip_threshold : float, optional
        Skip frames if log_prob(blank) > blank_skip_threshold, to speed up decoding (Default: log(1.0)).
        Note: This is only used when using the CUDA decoder, and it might worsen the results. Use it at your own risk.
    
    Example
    -------
    >>> import torch
    >>> from speechbrain.decoders import TorchAudioCTCBeamSearch
    >>> probs = torch.tensor([[[0.2, 0.0, 0.8], 
    ...                   [0.4, 0.0, 0.6]]])
    >>> log_probs = torch.log(probs)
    >>> lens = torch.tensor([1.0])
    >>> blank_index = 2
    >>> vocab_list = ['a', 'b', '-']
    >>> decoder = TorchAudioCTCBeamSearch(tokens=vocab_list, blank_index=blank_index, sil_index=blank_index)
    >>> decoder(probs, lens)
    [[CTCHypothesis(text='', last_lm_state=None, score=-0.7339691072702408, lm_score=-0.7339691072702408, timesteps=[])]]
    """
    def __init__(
        self,
        tokens: Union[list, str], 
        lexicon: Optional[str] = None, 
        lm: Optional[str] = None, 
        lm_dict: Optional[str] = None,
        topk: int = 1,
        beam_size: int = 50,
        beam_size_token: Optional[int] = None,
        beam_threshold: float = 50,
        lm_weight: float = 2,
        word_score: float = 0,
        unk_score: float = float("-inf"),
        sil_score: float = 0,
        log_add: bool = False,
        blank_index: int = 0, 
        sil_index: int = 0, 
        unk_word: str = "<unk>",
        using_cpu_decoder: bool = True,
        blank_skip_threshold: float = math.log(1.0),
    ):
        self.lexicon = lexicon
        self.tokens = tokens
        self.lm = lm
        self.lm_dict = lm_dict
        self.topk = topk
        self.beam_size = beam_size
        self.beam_size_token = beam_size_token
        self.beam_threshold = beam_threshold
        self.lm_weight = lm_weight
        self.word_score = word_score
        self.unk_score = unk_score
        self.sil_score = sil_score
        self.log_add = log_add
        self.blank_index = blank_index
        self.sil_index = sil_index
        self.unk_word = unk_word
        self.using_cpu_decoder = using_cpu_decoder
        self.blank_skip_threshold = blank_skip_threshold

        if self.using_cpu_decoder:
            try:
                from torchaudio.models.decoder import ctc_decoder
            except ImportError:
                raise ImportError(
                    "ctc_decoder not found. Please install torchaudio and flashlight to use this decoder"
                )
            
            # if this is a path, then torchaudio expect to be an index
            # while its a list then it expects to be a token
            if isinstance(self.tokens, str):
                blank_token = self.blank_index
                sil_token = self.sil_index
            else:
                blank_token = self.tokens[self.blank_index]
                sil_token = self.tokens[self.sil_index]

            self._ctc_decoder = ctc_decoder(
                lexicon=self.lexicon,
                tokens=self.tokens,
                lm=self.lm,
                lm_dict=self.lm_dict,
                nbest=self.topk,
                beam_size=self.beam_size,
                beam_size_token=self.beam_size_token,
                beam_threshold=self.beam_threshold,
                lm_weight=self.lm_weight,
                word_score=self.word_score,
                unk_score=self.unk_score,
                sil_score=self.sil_score,
                log_add=self.log_add,
                blank_token=blank_token,
                sil_token=sil_token,
                unk_word=self.unk_word,
            )
        else:
            try:
                from torchaudio.models.decoder import cuda_ctc_decoder
            except ImportError:
                raise ImportError(
                    "cuda_ctc_decoder not found. Please install the nightly version of torchaudio to use this decoder"
                )
            assert self.blank_index == 0, "Index of blank token has to be 0 when using CUDA CTC decoder."

            self._ctc_decoder = cuda_ctc_decoder(
                tokens=self.tokens,
                nbest=self.topk,
                beam_size=self.beam_size,
                blank_skip_threshold=self.blank_skip_threshold,
            )

    def decode_beams(self, log_probs: torch.Tensor, wav_len: Union[torch.Tensor, None] = None) -> List[List[CTCHypothesis]]:
        """Decode log_probs using TorchAudio CTC decoder.

        If `using_cpu_decoder=True` then log_probs and wav_len are moved to CPU before decoding.
        When using CUDA CTC decoder, the timestep information is not available. Therefore, the timesteps
        in the returned hypotheses are set to None.

        Arguments
        ---------
        log_probs : torch.Tensor
            The log probabilities of the input audio. 
            Shape: (batch_size, seq_length, vocab_size)
        wav_len : torch.Tensor, optional
            The speechbrain-style relative length. Shape: (batch_size,)
            If None, then the length of each audio is assumed to be seq_length.
        
        Returns
        -------
        list of list of CTCHypothesis
            The decoded hypotheses. The outer list is over the batch dimension, and the inner list is over the topk dimension.
        """
        if wav_len is not None:
            wav_len = log_probs.size(1) * wav_len
        else:
            wav_len = torch.tensor([log_probs.size(1)] * log_probs.size(0), 
                                       device=log_probs.device, 
                                       dtype=torch.int32)

        if wav_len.dtype != torch.int32:
            wav_len = wav_len.to(torch.int32)

        if log_probs.dtype != torch.float32:
            raise ValueError("log_probs must be float32.")

        # When using CPU decoder, we need to move the log_probs and wav_len to CPU
        if self.using_cpu_decoder == True and log_probs.is_cuda:
            log_probs = log_probs.cpu()

        if self.using_cpu_decoder == True and wav_len.is_cuda:
            wav_len = wav_len.cpu()

        if not log_probs.is_contiguous():
            raise RuntimeError("log_probs must be contiguous.")

        results = self._ctc_decoder(log_probs, wav_len)

        tokens_preds = []
        words_preds = []
        scores_preds = []
        timesteps_preds = []

        # over batch dim
        for i in range(len(results)):

            if self.using_cpu_decoder:

                preds = [
                    results[i][j].tokens.tolist()
                    for j in range(len(results[i]))
                ]
                preds = [
                    [self.tokens[token] for token in tokens] for tokens in preds
                ]
                tokens_preds.append(preds)

                timesteps = [
                    results[i][j].timesteps.tolist()
                    for j in range(len(results[i]))
                ]
                timesteps_preds.append(timesteps)

            else:
                # no timesteps is available for CUDA CTC decoder
                timesteps = [None for _ in range(len(results[i]))]
                timesteps_preds.append(timesteps)

                preds = [results[i][j].tokens for j in range(len(results[i]))]
                preds = [
                    [self.tokens[token] for token in tokens] for tokens in preds
                ]
                tokens_preds.append(preds)

            words = [results[i][j].words for j in range(len(results[i]))]
            words_preds.append(words)

            scores = [results[i][j].score for j in range(len(results[i]))]
            scores_preds.append(scores)

        hyps = []
        for batch_index, (batch_text, batch_score, batch_timesteps) in enumerate(zip(
            tokens_preds, scores_preds, timesteps_preds
        )):
            hyps.append([])
            for text, score, timestep in zip(
                batch_text, batch_score, batch_timesteps
            ):
                hyps[batch_index].append(
                    CTCHypothesis(
                        text=''.join(text),
                        last_lm_state=None,
                        score=score,
                        lm_score=score,
                        timesteps=timestep,
                    )
                )
        return hyps

    def __call__(self, log_probs: torch.Tensor, wav_len: Union[torch.Tensor, None] = None) -> List[List[CTCHypothesis]]:
        """Decode log_probs using TorchAudio CTC decoder.

        If `using_cpu_decoder=True` then log_probs and wav_len are moved to CPU before decoding.
        When using CUDA CTC decoder, the timestep information is not available. Therefore, the timesteps
        in the returned hypotheses are set to None.

        Arguments
        ---------
        log_probs : torch.Tensor
            The log probabilities of the input audio. 
            Shape: (batch_size, seq_length, vocab_size)
        wav_len : torch.Tensor, optional
            The speechbrain-style relative length. Shape: (batch_size,)
            If None, then the length of each audio is assumed to be seq_length.
        
        Returns
        -------
        list of list of CTCHypothesis
            The decoded hypotheses. The outer list is over the batch dimension, and the inner list is over the topk dimension.
        """
        return self.decode_beams(log_probs, wav_len)