"""Decoders and output normalization for CTC.

Authors
 * Mirco Ravanelli 2020
 * Aku Rouhe 2020
 * Sung-Lin Yeh 2020
"""
import torch
from itertools import groupby
from speechbrain.dataio.dataio import length_to_mask
import torch.nn as nn
import math
import dataclasses
import numpy as np
import heapq 
import logging 
logger = logging.getLogger(__name__)
from numba import jit

from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

# TODO: move CTCPrefixScore in scorer.py
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

@dataclasses.dataclass(frozen=True)
class Beam:
    """Contains all the info needed for decoding a beam."""

    text: str
    next_word: str
    partial_word: str
    last_token: Optional[str]
    last_idx_token: Optional[int]
    logit_score: float
        
    def __repr__(self):
        return f"Beam(text={self.text}, next_word={self.next_word}, partial_word={self.partial_word}, last_token={self.last_token},last_idx_token={self.last_idx_token}, logit_score={self.logit_score})"
    
    @classmethod
    def from_lm_beam(cls, lm_beam):
        return Beam(
            text=lm_beam.text,
            next_word=lm_beam.next_word,
            partial_word=lm_beam.partial_word,
            last_token=lm_beam.last_token,
            last_idx_token=lm_beam.last_idx_token,
            logit_score=lm_beam.logit_score,
        )

@dataclasses.dataclass(frozen=True)
class LMBeam(Beam):
    lm_score: float

    def __repr__(self):
        return f"LMBeam(text={self.text}, next_word={self.next_word}, partial_word={self.partial_word}, last_token={self.last_token}, last_idx_token={self.last_idx_token}, logit_score={self.logit_score}, lm_score={self.lm_score})"
    
EMPTY_START_BEAM: Beam = Beam("", "", "", None, None, 0.0)


def _sort_and_trim_beams(beams: List[LMBeam], beam_width: int) -> List[LMBeam]:
    """Take top N beams by score."""
    return heapq.nlargest(beam_width, beams, key=lambda x: x.lm_score)

def _merge_tokens(token_1: str, token_2: str) -> str:
    """Fast, whitespace safe merging of tokens."""
    if len(token_2) == 0:
        text = token_1
    elif len(token_1) == 0:
        text = token_2
    else:
        text = token_1 + " " + token_2
    return text

def _merge_beams(beams: List[Beam]) -> List[Beam]:
    """Merge beams with same prefix together."""
    beam_dict = {}
    for beam in beams:
        new_text = _merge_tokens(beam.text, beam.next_word)
        hash_idx = (new_text, beam.partial_word, beam.last_token)
        if hash_idx not in beam_dict:
            beam_dict[hash_idx] = beam
        else:
            # We've already seen this text - we want to combine the scores
            beam_dict[hash_idx] = dataclasses.replace(
                beam, logit_score=np.logaddexp(beam_dict[hash_idx].logit_score, beam.logit_score)
            )
    return list(beam_dict.values())


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
            scorer=None,
            prune_history=False,
        ):
        from .language_model import (
            AbstractLanguageModel,
            AbstractLMState,
            HotwordScorer,
            LanguageModel,
            load_unigram_set_from_arpa,
        )

        self.blank_id = blank_id
        self.beam_size = beam_size
        self.vocab = vocab

        if kenlm_model_path is not None:
            try:
                import kenlm  # type: ignore
            except ImportError:
                logger.warning(
                    "kenlm python bindings are not installed. To install it use: "
                    "pip install https://github.com/kpu/kenlm/archive/master.zip"
                )
        self.kenlm_model = None if kenlm_model_path is None else kenlm.Model(kenlm_model_path)
        
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
        print(f"BeamSearchDecoderCTC: lm={self.lm}")
        
        self.prune_vocab = prune_vocab
        self.prune_beams = prune_beams
        self.space_id = space_id
        self.topk = topk
        self.prune_frames = prune_frames
        self.prune_frames_thresh = math.log(prune_frames_thresh)
        self.beam_size_token = beam_size_token
        # sentencepiece
        self.spm_token = "▁"
        self.is_spm = any([s.startswith(self.spm_token) for s in vocab])
        self.prune_history = prune_history

        self.scorer = scorer
        if self.scorer is not None:
            print(f"BeamSearchDecoderCTC: scorer={self.scorer}")

        if self.scorer is not None and len(self.scorer.lattice_scorers) > 0 and self.prune_history is True:
            raise ValueError("BeamSearchDecoderCTC: prune_history is not compatible with scorer as prune_history is elaging too much")

        if not self.is_spm and space_id == -1:
            raise ValueError("Space id must be set")
        print(f"BeamSearchDecoderCTC: is_spm={self.is_spm}")

    def _get_lm_beams(
        self,
        beams: list,
        cached_lm_scores,
        cached_partial_token_scores: Dict[str, float],
        is_eos: bool = False,
    ):
        if self.lm is None:
            new_beams = []
            
            
            for beam in beams:
                new_text = _merge_tokens(beam.text, beam.next_word)

                new_beams.append(
                    LMBeam(
                        text=new_text,
                        next_word="",
                        partial_word=beam.partial_word,
                        last_token=beam.last_token,
                        last_idx_token=beam.last_idx_token,
                        logit_score=beam.logit_score,
                        lm_score=beam.logit_score,
                    )
                )
            return new_beams
        else:
            new_beams = []
            for beam in beams:
                # fast token merge
                new_text = _merge_tokens(beam.text, beam.next_word)
                cache_key = (new_text, is_eos)
                if cache_key not in cached_lm_scores:
                    prev_raw_lm_score, start_state = cached_lm_scores[(beam.text, False)]
                    score, end_state = self.lm.score(
                        start_state, beam.next_word, is_last_word=is_eos
                    )
                    raw_lm_score = prev_raw_lm_score + score
                    cached_lm_scores[cache_key] = (raw_lm_score, end_state)
                lm_score, _ = cached_lm_scores[cache_key]
                word_part = beam.partial_word
                if len(word_part) > 0:
                    if word_part not in cached_partial_token_scores:
        
                        cached_partial_token_scores[word_part] = self.lm.score_partial_token(
                            word_part
                        )
                    lm_score += cached_partial_token_scores[word_part]

                new_beams.append(
                    LMBeam(
                        text=new_text,
                        next_word="",
                        partial_word=word_part,
                        last_token=beam.last_token,
                        last_idx_token=beam.last_idx_token,
                        logit_score=beam.logit_score,
                        lm_score=beam.logit_score + lm_score,
                    )
                )

            return new_beams
        
    def _decode_logits(
            self, 
            logits: torch.Tensor,
            lm_start_state=None,
        ):
        if self.lm is None:
            cached_lm_scores = {}
        else:
            if lm_start_state is None:
                start_state = self.lm.get_start_state()
            else:
                start_state = lm_start_state
            cached_lm_scores = {("", False): (0.0, start_state)}
        cached_p_lm_scores: Dict[str, float] = {}

        # Initialize beams
        beams = [Beam("", "", "", None, None, 0.0)]

        # blank skip threshold
        if self.prune_frames:
            valid_frames = np.where(logits[:, self.blank_id] <= self.prune_frames_thresh)[0]
        else:
            valid_frames = range(logits.shape[0])

        for frame_idx in valid_frames:
            logit = logits[frame_idx]

            max_idx = logit.argmax()
            idx_list = set(np.where(logit >= self.prune_vocab)[0]) | {max_idx}
            new_beams = []
     

            for idx_token in idx_list:
                p_token = logit[idx_token]
                token = self.vocab[idx_token]
                for beam in beams:
                    if idx_token == self.blank_id or beam.last_token == token:
                        new_beams.append(
                            Beam(
                                text=beam.text,
                                next_word=beam.next_word,
                                partial_word=beam.partial_word,
                                last_token=token,
                                last_idx_token=idx_token,
                                logit_score=beam.logit_score + p_token,
                            )
                        )
                    elif self.is_spm and token[:1] == self.spm_token:
                        
                        clean_token = token[1:]
                        new_beams.append(
                            Beam(
                                text=beam.text,
                                next_word=beam.partial_word,
                                partial_word=clean_token,
                                last_token=token,
                                last_idx_token=idx_token,
                                logit_score=beam.logit_score + p_token,
                            )
                        )

                    elif not self.is_spm and idx_token == self.space_id:
                        new_beams.append(
                            Beam(
                                text=beam.text,
                                next_word=beam.partial_word,
                                partial_word="",
                                last_token=token,
                                last_idx_token=idx_token,
                                logit_score=beam.logit_score + p_token,
                            )
                        )
                    else:
                        new_beams.append(
                            Beam(
                                text=beam.text,
                                next_word=beam.next_word ,
                                partial_word=beam.partial_word + token,
                                last_token=token,
                                last_idx_token=idx_token,
                                logit_score=beam.logit_score + p_token,
                            )
                        )
            
            new_beams = _merge_beams(new_beams)
            
            # scorer here
            scored_beams = self._get_lm_beams(
                new_beams,
                cached_lm_scores,
                cached_p_lm_scores,
            )
            # remove beam outliers
            max_score = max([b.lm_score for b in scored_beams])
            scored_beams = [b for b in scored_beams if b.lm_score >= max_score + self.prune_beams]
           
            trimmed_beams = _sort_and_trim_beams(scored_beams, self.beam_size)

            if self.prune_history:
                lm_order = 1 if self.lm is None else self.lm.order
                beams = _prune_history(trimmed_beams, lm_order)
            else:
                beams = [Beam.from_lm_beam(b) for b in trimmed_beams]

        new_beams = []
        for beam in beams:
            # we need to merge the last partial word
            new_beams.append(
                Beam(
                    text=beam.text,
                    next_word=beam.partial_word,
                    partial_word="",
                    last_token=None,
                    last_idx_token=None,
                    logit_score=beam.logit_score,
                )
            )

        new_beams = _merge_beams(new_beams)
        scored_beams = self._get_lm_beams(
            new_beams,
            cached_lm_scores,
            cached_p_lm_scores,
            is_eos=True,
        )

        if self.scorer is not None and len(self.scorer.lattice_scorers) > 0:
            beams_to_score = [[b.text] for b in scored_beams]
            beams_scores = torch.tensor([[b.lm_score] for b in scored_beams])
            new_lm_scores = self.scorer.lattice_rescoring(beams_scores, beams_to_score)
            
            scored_beams = [LMBeam(
                text=b.text,
                next_word=b.next_word,
                partial_word=b.partial_word,
                last_token=b.last_token,
                last_idx_token=b.last_idx_token,
                logit_score=b.logit_score,
                lm_score=new_lm_scores[i].item()) 
                for i, b in enumerate(scored_beams)
            ]

        beams = _sort_and_trim_beams(scored_beams, self.beam_size)
        return beams

    def __call__(self, logits):
        return self._decode_logits(logits)   

    def batch_decode(self, logits):
        """ Decode logits in batch mode.
        Trigger lattice rescoring at the end in a batched fashion.
        """
        ...

    