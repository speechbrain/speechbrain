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

from typing import (
    Dict,
    List,
    Optional,
)

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

class BeamSearchDecoderCTCV1:
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
            lm_start_state = None,
        ):
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

        beams = _sort_and_trim_beams(scored_beams, self.beam_size)
        return beams

    def __call__(self, logits):
        return self._decode_logits(logits)   

    def batch_decode(self, logits):
        """ Decode logits in batch mode.
        Trigger lattice rescoring at the end in a batched fashion.
        """
        ...

    