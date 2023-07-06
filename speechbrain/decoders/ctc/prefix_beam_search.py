from speechbrain.decoders.ctc import CTCBaseSearcher
from speechbrain.decoders.ctc import CTCBeam
import torch
import math
import dataclasses
import numpy as np
import logging

logger = logging.getLogger(__name__)


from typing import (
    Dict,
    List,
    Optional,
)

class CTCPrefixBeamSearch(CTCBaseSearcher):
    def __init__(self, blank_index, vocab_list, kenlm_model_path=None, unigrams=None, space_index=-1, beam_width=100, beam_prune_logp=-10, token_prune_min_logp=-5, history_prune=True, topk=1):
        super().__init__(blank_index, vocab_list, space_index, kenlm_model_path, unigrams, beam_width, beam_prune_logp, token_prune_min_logp, history_prune, topk)
    
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
                lm_score=0.0,
                p_b=-math.inf,
            )     
            print("partial wrd = ", beam.partial_word)       
        else: 
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
                lm_score=0.0,
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

            for beam in beams:
                beam.step()

            beams = sorted(beams, key=lambda x: x.score, reverse=True)[:self.beam_width]
        exit()
        print(beams)
        exit()
        return beams