"""Decoders and output normalization for CTC.

Authors
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


@dataclasses.dataclass
class Beam:
    """Contains all the info needed for decoding a beam."""

    text: str
    next_word: str
    partial_word: str
    last_token: Optional[str]
    last_idx_token: Optional[int]

    p: float = -math.inf
    p_b: float = -math.inf
    p_nb: float = -math.inf

    n_p_b: float = -math.inf
    n_p_nb: float = -math.inf

    score: float = -math.inf
    score_lm: float = 0.0
    score_ctc: float = -math.inf

    def step(self):
        self.p_b, self.p_nb = self.n_p_b, self.n_p_nb
        self.n_p_b = self.n_p_nb = -math.inf
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
        return sorted(
            self.beams.items(), key=lambda x: x[1].score, reverse=True
        )

class LMBeam(Beam):
    lm_score: float

    def __repr__(self):
        return f"LMBeam(text={self.text}, next_word={self.next_word}, partial_word={self.partial_word}, last_token={self.last_token}, last_idx_token={self.last_idx_token}, logit_score={self.logit_score}, lm_score={self.lm_score}, p_blank={self.p_blank}, p_non_blank={self.p_non_blank})"

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
        self, logits: torch.Tensor, lm_start_state=None,
    ):

        beams = Beams()

        # blank skip threshold
        if self.prune_frames:
            valid_frames = np.where(
                logits[:, self.blank_id] <= self.prune_frames_thresh
            )[0]
        else:
            valid_frames = range(logits.shape[0])

        for frame_idx in valid_frames:
            log_probs = logits[frame_idx]

            # pruning step
            max_idx = log_probs.argmax()

            log_prob_idx = set(np.where(log_probs >= math.log(0.50))[0]) | {
                max_idx
            }

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
                        beam.n_p_b = np.logaddexp(
                            beam.n_p_b, beam.score_ctc + p
                        )
                        continue

                    last_token_index = prefix[-1] if prefix else None

                    # repeated token
                    if token_index == last_token_index:
                        beam.n_p_nb = np.logaddexp(beam.n_p_nb, p_nb + p)

                    n_prefix = prefix + (token_index,)
                    # Must update state for prefix search
                    n_beam = beams.getitem(n_prefix, p=p, previous_beam=beam)
                    n_p_nb = n_beam.n_p_nb

                    if (
                        token_index == last_token_index
                        and p_b > -self.NUM_FLT_INF
                    ):
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
