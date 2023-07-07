from speechbrain.decoders.ctc import CTCBaseSearcher
from speechbrain.decoders.ctc import CTCBeam
from speechbrain.decoders.ctc import LMCTCBeam

import torch
import math
import dataclasses
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CTCPrefixBeamSearch(CTCBaseSearcher):
    def __init__(self, blank_index, vocab_list, kenlm_model_path=None, unigrams=None, space_index=-1, beam_width=100, beam_prune_logp=-10, token_prune_min_logp=-5, history_prune=True, topk=1):
        super().__init__(blank_index, vocab_list, space_index, kenlm_model_path, unigrams, beam_width, beam_prune_logp, token_prune_min_logp, history_prune, topk)

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
                new_beams.append(
                    LMCTCBeam(
                        text=beam.text,
                        next_word=beam.next_word,
                        partial_word=beam.partial_word,
                        last_token=beam.last_token,
                        last_token_index=beam.last_token_index,                    
                        text_frames=beam.text_frames,
                        partial_frames=beam.partial_frames,
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
        
    def _get_new_beam(self, new_prefix, new_token, new_token_index, beams, p=None, previous_beam=None):
        for beam in beams:
            if beam.text == new_prefix:
                if p and p > beam.p:
                    beam.p = p 
                beam.last_token = new_token
                beam.last_token_index = new_token_index
                return beam 
        
        if new_token_index == self.space_index:
            new_beam = CTCBeam(
                text=new_prefix,
                next_word=beam.partial_word,
                partial_word="",
                last_token=new_token,
                last_token_index=new_token_index,
                text_frames=[],
                partial_frames=(-1, -1),
                score=-math.inf,
                score_ctc=-math.inf,
                p_b=-math.inf,
            )     
            #print("next word = ", new_beam.next_word)  
        elif new_token_index == beam.last_token_index:
            new_beam = CTCBeam(
                text=new_prefix,
                next_word="",
                partial_word=beam.partial_word,
                last_token=new_token,
                last_token_index=new_token_index,
                text_frames=[],
                partial_frames=(-1, -1),
                score=-math.inf,
                score_ctc=-math.inf,
                p_b=-math.inf,
            )
        else: 
            new_beam = CTCBeam(
                text=new_prefix,
                next_word="",
                partial_word=beam.partial_word + new_token,
                last_token=new_token,
                last_token_index=new_token_index,
                text_frames=[],
                partial_frames=(-1, -1),
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
        log_probs,
        beams,
        cached_lm_scores,
        cached_p_lm_scores,
        processed_frames = 0,
    ):  
        for frame_index, logit_col in enumerate(log_probs, start=processed_frames):
            max_index = logit_col.argmax()
            tokens_index_list = set(np.where(logit_col > self.token_prune_min_logp)[0]) | {max_index}
            
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
                        previous_beam=beam
                    )

                    n_p_nb = new_beam.n_p_nb

                    if token_index == beam.last_token_index and p_b > -math.inf:
                        n_p_nb = np.logaddexp(n_p_nb, p_b + p_token)
                    elif token_index != beam.last_token_index:
                        n_p_nb = np.logaddexp(n_p_nb, beam.score_ctc + p_token)
                    new_beam.n_p_nb = n_p_nb 

            # here kenlm scorer
            for beam in beams:
                beam.step()


            # beams = sorted(beams, key=lambda x: x.score, reverse=True)[:self.beam_width]

            scored_beams = self.get_lm_beams(
                beams, 
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