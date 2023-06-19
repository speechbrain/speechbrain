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
import copy
from collections.abc import MutableMapping
from dataclasses import dataclass
from operator import itemgetter
from typing import Tuple

from typing import (
    Dict,
    List,
    Optional,
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
class Beam:
    """Contains all the info needed for decoding a beam."""

    text: str
    next_word: str
    partial_word: str
    last_token: Optional[str]
    last_idx_token: Optional[int]

    p : float = - math.inf
    p_b : float = - math.inf
    p_nb : float = - math.inf

    n_p_b: float = - math.inf
    n_p_nb : float = - math.inf

    score : float = - math.inf
    score_lm : float = 0.0
    score_ctc : float = - math.inf

    def step(self):
        self.p_b, self.p_nb = self.n_p_b, self.n_p_nb
        self.n_p_b = self.n_p_nb = - math.inf
        self.score_ctc = np.logaddexp(self.p_b, self.p_nb)
        self.score = self.score_ctc + self.score_lm

class Beams(MutableMapping):
    def __init__(self):
        self.beams = {(): Beam("", "", "", None, None)}
        self.beams[()].p_b = 0.0
        self.beams[()].score_ctc = 0.0

    def __getitem__(self, key):
        return self.getitem(key)
    
    def getitem(self, key, p=None, previous_beam=None):
        if key in self.beams:
            beam = self.beams[key]
            if p and p > beam.p:
                beam.p = p
            return beam

        new_beam = Beam("", "", "", None, None)
        if previous_beam:
            new_beam.p = p
        self.beams[key] = new_beam
        return new_beam

    def __setitem__(self, key, value):
        self.beams[key] = value

    def __delitem__(self, key):
        del self.beams[key]

    def __len__(self):
        return len(self.beams)

    def __iter__(self):
        return iter(self.beams)

    def step(self):

        for beam in self.beams.values():
            beam.step()

    def topk_(self, k):
        """ Keep only the top k prefixes """
        if len(self.beams) <= k:
            return self

        beams = list(self.beams.items())
        indexes = np.argpartition([-v.score for k, v in beams], k)[:k].tolist()

        self.beams = {k: v for k, v in itemgetter(*indexes)(beams)}

        return self

    def sort(self):
        return sorted(self.beams.items(), key=lambda x: x[1].score, reverse=True)

class LMBeam(Beam):
    lm_score: float

    def __repr__(self):
        return f"LMBeam(text={self.text}, next_word={self.next_word}, partial_word={self.partial_word}, last_token={self.last_token}, last_idx_token={self.last_idx_token}, logit_score={self.logit_score}, lm_score={self.lm_score}, p_blank={self.p_blank}, p_non_blank={self.p_non_blank})"

def _sort_and_trim_beams(beams: List[LMBeam], beam_width: int) -> List[LMBeam]:
    """Take top N beams by score."""
    return heapq.nlargest(beam_width, beams, key=lambda x: x.lm_score)

def _merge_tokens(token_1, token_2):
    """Fast, whitespace safe merging of tokens."""
    if len(token_2) == 0:
        text = token_1
    elif len(token_1) == 0:
        text = token_2
    else:
        text = token_1 + " " + token_2
    return text

def _prune_history(beams: List[LMBeam], lm_order: int) -> List[Beam]:
    """Filter out beams that are the same over max_ngram history.

    Since n-gram language models have a finite history when scoring a new token, we can use that
    fact to prune beams that only differ early on (more than n tokens in the past) and keep only the
    higher scoring ones. Note that this helps speed up the decoding process but comes at the cost of
    some amount of beam diversity. If more than the top beam is used in the output it should
    potentially be disabled.

    Args:
        beams: list of LMBeam
        lm_order: int, the order of the n-gram model

    Returns:
        list of Beam
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
            filtered_beams.append(Beam.from_lm_beam(lm_beam))
            seen_hashes.add(hash_idx)
    return filtered_beams

def _merge_beams(beams):
    """Merge beams with same prefix together."""
    beam_dict = {}
    for beam in beams:
        new_text = _merge_tokens(beam.text, beam.next_word)
        hash_idx = (new_text, beam.partial_word, beam.last_token)
        if hash_idx not in beam_dict:
            beam_dict[hash_idx] = beam
        else:
            # merge same prefix beams
            beam_dict[hash_idx] = dataclasses.replace(
                beam, logit_score=np.logaddexp(beam_dict[hash_idx].logit_score, beam.logit_score)
            )
    return list(beam_dict.values())

class BeamSearchDecoderCTC:
    def __init__(
            self, 
            blank_id, 
            vocab, 
            space_id=-1, 
            beam_size=100, 
            topk=1, 
            kenlm_model_path=None,
            unigrams=None,
            prune_frames=False, 
            beam_size_token=None,
            prune_frames_thresh=0.95, 
            prune_vocab=-5.0, 
            prune_beams=-10.0,
            prune_history=False,
        ):
        from speechbrain.decoders.language_model import (
            LanguageModel,
            load_unigram_set_from_arpa,
        )

        self.blank_id = blank_id
        self.beam_size = beam_size
        self.vocab = vocab
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
            logger.info("Using arpa instead of binary LM file, decoder instantiation might be slow.")
        
        if unigrams is None and kenlm_model_path is not None:
            if kenlm_model_path.endswith(".arpa"):
                unigrams = load_unigram_set_from_arpa(kenlm_model_path)
            else:
                logger.warning(
                    "Unigrams not provided and cannot be automatically determined from LM file (only "
                    "arpa format). Decoding accuracy might be reduced."
                )
        
        if self.kenlm_model is not None:
            self.lm = LanguageModel(
                self.kenlm_model, 
                unigrams
            )
        else:
            self.lm = None
            
        self.prune_vocab = prune_vocab
        self.prune_beams = prune_beams
        self.space_id = space_id
        self.topk = topk
        self.prune_frames = prune_frames
        self.prune_frames_thresh = math.log(prune_frames_thresh)
        self.beam_size_token = beam_size_token
        self.prune_history = prune_history

        # sentencepiece
        self.spm_token = "▁"
        self.is_spm = any([s.startswith(self.spm_token) for s in vocab])

        if not self.is_spm and space_id == -1:
            raise ValueError("Space id must be set")
        
        self.NUM_FLT_INF = math.inf

    def _decode_logits(
            self, 
            logits: torch.Tensor,
            lm_start_state = None,
        ):

        beams = Beams()

        # blank skip threshold
        if self.prune_frames:
            valid_frames = np.where(logits[:, self.blank_id] <= self.prune_frames_thresh)[0]
        else:
            valid_frames = range(logits.shape[0])

        for frame_idx in valid_frames:
            log_probs = logits[frame_idx]
            
            # pruning step 
            max_idx = log_probs.argmax()
            
            log_prob_idx = set(np.where(log_probs >= math.log(0.50))[0]) | {max_idx}
            
            curr_beams = list(beams.sort())
            full_beam = False 
            min_cutoff = -self.NUM_FLT_INF
            num_prefixes = len(curr_beams)
            
            for token_index in log_prob_idx:
                c = self.vocab[token_index]
                p = log_probs[token_index]

                for prefix, beam in curr_beams:
                    p_b, p_nb = beam.p_b, beam.p_nb
                    
                    if full_beam and p + beam.score_ctc < min_cutoff:
                        break
                        
                    # blank case
                    if token_index == self.blank_id:
                        beam.n_p_b = np.logaddexp(beam.n_p_b, beam.score_ctc + p)
                        continue

                    last_token_index = prefix[-1] if prefix else None
                  
                    # repeated token
                    if token_index == last_token_index:
                        beam.n_p_nb = np.logaddexp(beam.n_p_nb, p_nb + p)

                    n_prefix = prefix + (token_index, )
                    # Must update state for prefix search
                    n_beam = beams.getitem(n_prefix, p=p, previous_beam=beam)

                    n_p_nb = n_beam.n_p_nb

                    if token_index == last_token_index and p_b > -self.NUM_FLT_INF:
                        # We don't include the previous probability of not ending in blank (p_nb)
                        # if s is repeated at the end. The CTC algorithm merges characters not
                        # separated by a blank.
                        n_p_nb = np.logaddexp(n_p_nb, p_b + p)
                    elif token_index != last_token_index:
                        n_p_nb = np.logaddexp(n_p_nb, beam.score_ctc + p)
                    n_beam.n_p_nb = n_p_nb

            # Update the probabilities
            beams.step()
            # Trim the beam before moving on to the next time-step.
            beams.topk_(self.beam_size)

        for p, beam in beams.sort():
            for token in p:
                beam.text += self.vocab[token]
            return beam.text

    def __call__(self, logits):
        return self._decode_logits(logits)   

    def batch_decode(self, logits):
        """ Decode logits in batch mode.
        Trigger lattice rescoring at the end in a batched fashion.
        """
        ...

    