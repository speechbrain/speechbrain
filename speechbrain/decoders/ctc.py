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
import functools
import multiprocessing
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


@dataclasses.dataclass(frozen=True)
class CTCBeam:
    """Contains all the info needed for decoding a beam."""

    text: str
    next_word: str
    partial_word: str
    last_token: Optional[str]
    last_token_index: Optional[int]
    text_frames: Tuple[int, int]
    partial_frames: Tuple[int, int]

    p : float=  -math.inf
    p_b : float =  -math.inf
    p_nb: float =  -math.inf

    p_b_prev: float =  -math.inf
    p_nb_prev : float=  -math.inf

    score: float =  -math.inf
    score_ctc: float =  -math.inf

    @classmethod
    def from_lm_beam(cls, lm_beam):
        return CTCBeam(
            text=lm_beam.text,
            next_word=lm_beam.next_word,
            partial_word=lm_beam.partial_word,
            last_token=lm_beam.last_token,
            last_token_index=lm_beam.last_token_index,
            text_frames=lm_beam.text_frames,
            partial_frames=lm_beam.partial_frames,
            p=lm_beam.p,
            p_b=lm_beam.p_b,
            p_nb=lm_beam.p_nb,
            p_b_prev=lm_beam.p_b_prev,
            p_nb_prev=lm_beam.p_nb_prev,
            score=lm_beam.score,
            score_ctc=lm_beam.score_ctc,
        )

@dataclasses.dataclass(frozen=True)
class LMCTCBeam(CTCBeam):
    lm_score: float = -math.inf

@dataclasses.dataclass(frozen=True)
class CTCHypothesis:
    text: str
    last_lm_state: None
    text_frames: List[Tuple[str, Tuple[int, int]]]
    score: float  # Cumulative logit score
    lm_score: float  # Cumulative language model + logit score

    def get_mp_safe_beam(self):
        """Get a multiprocessing-safe version of the beam."""
        if self.last_lm_state is None:
            last_lm_state = None
        else:
            last_lm_state = self.last_lm_state.get_mp_safe_state()
        return dataclasses.replace(self, last_lm_state=last_lm_state)


class CTCBaseSearcher(torch.nn.Module):
    """ TODO: docstring
    TODO: integrate scorers for N-Gram (as it is already the case of n-best rescorers)
    """

    def __init__(
        self,
        blank_index,
        vocab_list,
        space_index=-1,
        kenlm_model_path=None,
        unigrams=None,
        beam_width=100,
        beam_prune_logp=-10.0,
        token_prune_min_logp=-5.0,
        frames_prune_min_blank_logp=math.log(0.99),
        history_prune=False,
        topk=1,
    ):
        super().__init__()

        self.blank_index = blank_index
        self.space_index = space_index
        self.kenlm_model_path = kenlm_model_path
        self.unigrams = unigrams
        self.vocab_list = vocab_list
        self.beam_width = beam_width
        self.beam_prune_logp = beam_prune_logp
        self.token_prune_min_logp = token_prune_min_logp
        self.frames_prune_min_blank_logp = frames_prune_min_blank_logp
        self.history_prune = history_prune
        self.topk = topk

        # sentencepiece
        self.spm_token = "▁"
        self.is_spm = any([s.startswith(self.spm_token) for s in vocab_list])

        if not self.is_spm and space_index == -1:
            raise ValueError("space_index must be set")
        

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

    def get_valid_pool(self, pool):
        """Return the pool if the pool is appropriate for multiprocessing."""
        if pool is not None and isinstance(
            pool._ctx, multiprocessing.context.SpawnContext  # type: ignore [attr-defined] # pylint: disable=W0212
        ):
            logger.warning(
                "Specified pool object has a spawn context, which is not currently supported. "
                "See https://github.com/kensho-technologies/pyctcdecode/issues/65."
                "\nFalling back to sequential decoding."
            )
            return None
        return pool

    def partial_decoding(
            self, 
            log_probs, 
            beams, 
            cached_lm_scores, 
            cached_p_lm_scores,
            processed_frames = 0,
        ):
        raise NotImplementedError
    
    def on_start_decode(self, **kwargs):
        raise NotImplementedError

    def one_decode_step(
            self, 
            log_probs, 
            beams, 
            cached_lm_scores, 
            cached_p_lm_scores
        ):
            raise NotImplementedError

    def finalize_decoding(
            self, 
            beams, 
            cached_lm_scores,
            cached_p_lm_scores,
            force_next_word=False, 
            is_end=False
            ):
        raise NotImplementedError

    def partial_decode(self, **kwargs):
        raise NotImplementedError

    def full_decode(self, log_probs):
        raise NotImplementedError
    
    def normalize_whitespace(self, text: str) -> str:
        """Efficiently normalize whitespace."""
        return " ".join(text.split()) 

    def merge_tokens(self, token_1: str, token_2: str) -> str:
        if len(token_2) == 0:
            text = token_1
        elif len(token_1) == 0:
            text = token_2
        else:
            text = token_1 + " " + token_2
        return text

    def merge_beams(self, beams):
        beam_dict = {}
        for beam in beams:
            new_text = self.merge_tokens(beam.text, beam.next_word)
            hash_idx = (new_text, beam.partial_word, beam.last_token)
            if hash_idx not in beam_dict:
                beam_dict[hash_idx] = beam
            else:
                # We've already seen this text - we want to combine the scores
                beam_dict[hash_idx] = dataclasses.replace(
                    beam, score=np.logaddexp(beam_dict[hash_idx].score, beam.score)
                )
        return list(beam_dict.values())

    def sort_beams(self, beams):
        return heapq.nlargest(self.beam_width, beams, key=lambda x: x.lm_score)

    def get_lm_beams(
        self,
        beams,
        cached_lm_scores,
        cached_partial_token_scores,
        is_eos= False,
    ):
        if self.lm is None:
            new_beams = []
            for beam in beams:
                new_text = self.merge_tokens(beam.text, beam.next_word)
                new_beams.append(
                    LMCTCBeam(
                        text=new_text,
                        next_word="",
                        partial_word=beam.partial_word,
                        last_token=beam.last_token,
                        last_token_index=beam.last_token,                    
                        text_frames=beam.text_frames,
                        partial_frames=beam.partial_frames,
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
                        next_word="",
                        partial_word=word_part,
                        last_token=beam.last_token,
                        last_token_index=beam.last_token,  
                        text_frames=beam.text_frames,
                        partial_frames=beam.partial_frames,                  
                        score=beam.score,
                        lm_score=beam.score + lm_score,
                    )
                )
            return new_beams

class CTCBeamSearch(CTCBaseSearcher):
    def __init__(self, blank_index, vocab_list, kenlm_model_path=None, unigrams=None, space_index=-1, beam_width=100, beam_prune_logp=-10, token_prune_min_logp=-5, frames_prune_min_blank_logp=-0.01, history_prune=False, topk=1):
        super().__init__(blank_index, vocab_list, space_index, kenlm_model_path, unigrams, beam_width, beam_prune_logp, token_prune_min_logp, frames_prune_min_blank_logp, history_prune, topk)

    def partial_decoding(
        self, 
        log_probs,
        beams,
        cached_lm_scores,
        cached_p_lm_scores,
        processed_frames = 0,
    ):        
        for frame_index, logit_col in enumerate(log_probs, start=processed_frames):
            max_index = logit_col.argmax()
            tokens_index_list = set(np.where(logit_col > self.token_prune_min_logp)[0]) | {max_index}
            new_beams = []

            for token_index in tokens_index_list:
                p_token = logit_col[token_index]
                token = self.vocab_list[token_index]

                for beam in beams:
                    
                    if token_index == self.blank_index or beam.last_token == token:
                        if token_index == self.blank_index:
                            new_end_frame = beam.partial_frames[0]
                        else:
                            new_end_frame = frame_index + 1

                        new_part_frames = (
                            beam.partial_frames if token_index == self.blank_index else (beam.partial_frames[0], new_end_frame)
                        )

                        new_beams.append(
                            CTCBeam(
                                text=beam.text,
                                next_word=beam.next_word,
                                partial_word=beam.partial_word,
                                last_token=token,
                                last_token_index=token_index,
                                text_frames=beam.text_frames,
                                partial_frames=new_part_frames,
                                score=beam.score + p_token,
                            )
                        )

                    elif self.is_spm and token[:1] == self.spm_token:
                        clean_token = token[1:]    

                        new_frame_list = (
                            beam.text_frames
                            if beam.partial_word == ""
                            else beam.text_frames + [beam.partial_frames]
                        )

                        new_beams.append(
                            CTCBeam(
                                text=beam.text,
                                next_word=beam.partial_word,
                                partial_word=clean_token,
                                last_token=token,
                                last_token_index=token_index,
                                text_frames=new_frame_list,
                                partial_frames=(frame_index, frame_index + 1),
                                score=beam.score + p_token,
                            )
                        )

                    elif not self.is_spm and token_index == self.space_index:
                        new_frame_list = (
                            beam.text_frames
                            if beam.partial_word == ""
                            else beam.text_frames + [beam.partial_frames]
                        )
                        new_beams.append(
                            CTCBeam(
                                text=beam.text,
                                next_word=beam.partial_word,
                                partial_word="",
                                last_token=token,
                                last_token_index=token_index,
                                text_frames=new_frame_list,
                                partial_frames=(-1, -1),
                                score=beam.score + p_token,
                            )
                        )
                    else:
                        new_part_frames = (
                            (frame_index, frame_index + 1)
                            if beam.partial_frames[0] < 0
                            else (beam.partial_frames[0], frame_index + 1)
                        )
                                                
                        new_beams.append(
                            CTCBeam(
                                text=beam.text,
                                next_word=beam.next_word,
                                partial_word=beam.partial_word + token,
                                last_token=token,
                                last_token_index=token_index,
                                text_frames=beam.text_frames,
                                partial_frames=new_part_frames,
                                score=beam.score + p_token,
                            )
                        )

            new_beams = self.merge_beams(new_beams)
            scored_beams = self.get_lm_beams(
                new_beams,
                cached_lm_scores,
                cached_p_lm_scores,
            )
            # remove beam outliers
            max_score = max([b.lm_score for b in scored_beams])
            scored_beams = [b for b in scored_beams if b.lm_score >= max_score + self.beam_prune_logp]

            trimmed_beams = self.sort_beams(scored_beams)

            beams = [CTCBeam.from_lm_beam(b) for b in trimmed_beams]
        return beams
    

    def finalize_decoding(
            self, 
            beams, 
            cached_lm_scores,
            cached_p_lm_scores,
            force_next_word=False, 
            is_end=False
        ):
        if force_next_word or is_end:
            new_beams = []
            for beam in beams:
                new_token_times = (
                    beam.text_frames
                    if beam.partial_word == ""
                    else beam.text_frames + [beam.partial_frames]
                )
                new_beams.append(
                    CTCBeam(
                        text=beam.text,
                        next_word=beam.partial_word,
                        partial_word="",
                        last_token=None,
                        last_token_index=None,
                        text_frames=new_token_times,
                        partial_frames=(-1, -1),
                        score=beam.score,
                    )
                )
            new_beams = self.merge_beams(new_beams)
        else:
            new_beams = list(beams)
        
        scored_beams = self.get_lm_beams(
            new_beams,
            cached_lm_scores,
            cached_p_lm_scores,
        )
        # remove beam outliers
        max_score = max([b.lm_score for b in scored_beams])
        scored_beams = [b for b in scored_beams if b.lm_score >= max_score + self.beam_prune_logp]
        return self.sort_beams(scored_beams)
    
    def decode_beams_batch(self, log_probs, pool=None):
        valid_pool = self.get_valid_pool(pool)
        if valid_pool is None:
            return [
                 self.decode_beams_mp_safe(log_prob) for log_prob in log_probs
            ]
        
        p_decode = functools.partial(
            self.decode_beams_mp_safe,
        )
        decoded_beams_list = valid_pool.map(p_decode, log_probs)
        return decoded_beams_list

    def decode_beams_mp_safe(self, log_probs):

        decoded_beams = self.decode_beams(log_probs)

        decoded_beams_mp_safe = [output_beam.get_mp_safe_beam() for output_beam in decoded_beams]

        return decoded_beams_mp_safe[:self.topk]
    

    def partial_decode_beams(
            self, 
            log_probs,
            cached_lm_scores,
            cached_p_lm_scores,
            beams,
            processed_frames,
            force_next_word = False, 
            is_end = False, 
        ):

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

    def decode_beams(self, log_probs, lm_start_state=None):
        return self.decode_log_probs(log_probs, lm_start_state)

    def decode_log_probs(self, log_probs, lm_start_state = None):
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
                next_word="",
                partial_word="",
                last_token=None,
                last_token_index=None,
                text_frames=[],
                partial_frames=(-1, -1),
                score=0.0,
            )
        ]

        beams = self.partial_decoding(
            log_probs,
            beams,
            cached_lm_scores,
            cached_p_lm_scores,
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
                text_frames=list(zip(lm_beam.text.split(), lm_beam.text_frames)),
                score=lm_beam.score,
                lm_score=lm_beam.lm_score,
            )
            for lm_beam in trimmed_beams
        ][:self.topk]

        return output_beams