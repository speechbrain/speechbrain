"""Decoders and output normalization for CTC.

Authors
 * Adel Moumen 2023
"""
import torch
import math
import dataclasses
import numpy as np
import heapq
import logging
import functools
import multiprocessing
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



logger = logging.getLogger(__name__)


@dataclasses.dataclass
class CTCBeam:
    """Contains all the info needed for decoding a beam."""
    text: str
    next_word: str
    partial_word: str
    last_token: Optional[str]
    last_token_index: Optional[int]
    text_frames: Tuple[int, int]
    partial_frames: Tuple[int, int]
    
    p: float = -math.inf
    p_b: float = -math.inf
    p_nb: float =  -math.inf

    n_p_b: float =  -math.inf
    n_p_nb : float=  -math.inf

    score: float =  -math.inf
    score_ctc: float = -math.inf

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
            n_p_b=lm_beam.n_p_b,
            n_p_nb=lm_beam.n_p_nb,
            score=lm_beam.score,
            score_ctc=lm_beam.score_ctc,
        )
    
    def step(self):
        self.p_b, self.p_nb = self.n_p_b, self.n_p_nb
        self.n_p_b = self.n_p_nb = -math.inf
        self.score_ctc = np.logaddexp(self.p_b, self.p_nb)
        self.score = self.score_ctc + self.score

@dataclasses.dataclass
class LMCTCBeam(CTCBeam):
    lm_score: float = -math.inf

@dataclasses.dataclass
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
        history_prune=True,
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
            print("LOADING unigram set")
            if kenlm_model_path.endswith(".arpa"):
                unigrams = load_unigram_set_from_arpa(kenlm_model_path)
            else:
                logger.warning(
                    "Unigrams not provided and cannot be automatically determined from LM file (only "
                    "arpa format). Decoding accuracy might be reduced."
                )

        if self.kenlm_model is not None:
            print("LOADING lm")
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
    
    def prune_history(self, beams, lm_order: int):
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
                filtered_beams.append(CTCBeam.from_lm_beam(lm_beam))
                seen_hashes.add(hash_idx)
        return filtered_beams
    
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
    
    def decode_beams(self, log_probs, lm_start_state=None):
        return [
            self.decode_log_probs(log_prob, lm_start_state)[:self.topk] for log_prob in log_probs
        ]

    
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
                score_ctc=0.0,
                p_b=0.0,
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
    
class CTCBeamSearch(CTCBaseSearcher):
    def __init__(self, blank_index, vocab_list, kenlm_model_path=None, unigrams=None, space_index=-1, beam_width=100, beam_prune_logp=-10, token_prune_min_logp=-5, history_prune=True, topk=1):
        super().__init__(blank_index, vocab_list, space_index, kenlm_model_path, unigrams, beam_width, beam_prune_logp, token_prune_min_logp, history_prune, topk)

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

            if self.history_prune:
                lm_order = 1 if self.lm is None else self.lm.order
                beams = self.prune_history(trimmed_beams, lm_order=lm_order)
            else:
                beams = [CTCBeam.from_lm_beam(b) for b in trimmed_beams]

        return beams