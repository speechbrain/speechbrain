
import torch
import math 
import numpy as np
from typing import (
    Dict,
    List,
    Optional,
)
import dataclasses
from typing import Tuple

from speechbrain.decoders.language_model import (
    LanguageModel,
    load_unigram_set_from_arpa,
)

import logging 
from itertools import groupby

logger = logging.getLogger(__name__)

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