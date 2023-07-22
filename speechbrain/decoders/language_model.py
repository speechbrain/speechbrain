"""Language model wrapper for kenlm n-gram.

This file is based on the implementation of the kenLM wrapper from
PyCTCDecode (see: https://github.com/kensho-technologies/pyctcdecode) and
is used in CTC decoders.

See: speechbrain.decoders.ctc.py

Authors
 * Adel Moumen 2023
"""
import logging
from typing import (
    Collection,
    Optional,
    Set,
    Tuple,
    cast,
)

from pygtrie import CharTrie

import math

logger = logging.getLogger(__name__)

try:
    import kenlm
except ImportError:
    raise ImportError(
        "kenlm python bindings are not installed. To install it use: "
        "pip install https://github.com/kpu/kenlm/archive/master.zip"
    )


def load_unigram_set_from_arpa(arpa_path: str) -> Set[str]:
    """Read unigrams from arpa file.

    Taken from: https://github.com/kensho-technologies/pyctcdecode

    Arguments
    ---------
    arpa_path : str
        Path to arpa file.

    Returns
    -------
    unigrams : set
        Set of unigrams.
    """
    unigrams = set()
    with open(arpa_path) as f:
        start_1_gram = False
        for line in f:
            line = line.strip()
            if line == "\\1-grams:":
                start_1_gram = True
            elif line == "\\2-grams:":
                break
            if start_1_gram and len(line) > 0:
                parts = line.split("\t")
                if len(parts) == 3:
                    unigrams.add(parts[1])
    if len(unigrams) == 0:
        raise ValueError(
            "No unigrams found in arpa file. Something is wrong with the file."
        )
    return unigrams


class KenlmState:
    """Wrapper for kenlm state.

    This is a wrapper for the kenlm state object. It is used to make sure that the
    state is not modified outside of the language model class.

    Taken from: https://github.com/kensho-technologies/pyctcdecode

    Arguments
    ---------
    state : kenlm.State
        Kenlm state object.
    """

    def __init__(self, state: "kenlm.State"):
        self._state = state

    @property
    def state(self) -> "kenlm.State":
        """Get the raw state object."""
        return self._state


def _prepare_unigram_set(
    unigrams: Collection[str], kenlm_model: "kenlm.Model"
) -> Set[str]:
    """Filter unigrams down to vocabulary that exists in kenlm_model.

    Taken from: https://github.com/kensho-technologies/pyctcdecode

    Arguments
    ---------
    unigrams : list
        List of unigrams.
    kenlm_model : kenlm.Model
        Kenlm model.

    Returns
    -------
    unigram_set : set
        Set of unigrams.
    """
    if len(unigrams) < 1000:
        logger.warning(
            "Only %s unigrams passed as vocabulary. Is this small or artificial data?",
            len(unigrams),
        )
    unigram_set = set(unigrams)
    unigram_set = set([t for t in unigram_set if t in kenlm_model])
    retained_fraction = (
        1.0 if len(unigrams) == 0 else len(unigram_set) / len(unigrams)
    )
    if retained_fraction < 0.1:
        logger.warning(
            "Only %s%% of unigrams in vocabulary found in kenlm model-- this might mean that your "
            "vocabulary and language model are incompatible. Is this intentional?",
            round(retained_fraction * 100, 1),
        )
    return unigram_set


def _get_empty_lm_state() -> "kenlm.State":
    """Get unintialized kenlm state.

    Taken from: https://github.com/kensho-technologies/pyctcdecode

    Returns
    -------
    kenlm_state : kenlm.State
        Empty kenlm state.
    """
    try:
        kenlm_state = kenlm.State()
    except ImportError:
        raise ValueError("To use a language model, you need to install kenlm.")
    return kenlm_state


class LanguageModel:
    """Language model container class to consolidate functionality.

    This class is a wrapper around the kenlm language model. It provides
    functionality to score tokens and to get the initial state.

    Taken from: https://github.com/kensho-technologies/pyctcdecode

    Arguments
    ---------
    kenlm_model : kenlm.Model
        Kenlm model.
    unigrams : list
        List of known word unigrams.
    alpha : float
        Weight for language model during shallow fusion.
    beta : float
        Weight for length score adjustment of during scoring.
    unk_score_offset : float
        Amount of log score offset for unknown tokens.
    score_boundary : bool
        Whether to have kenlm respect boundaries when scoring.
    """

    def __init__(
        self,
        kenlm_model: "kenlm.Model",
        unigrams: Optional[Collection[str]] = None,
        alpha: float = 0.5,
        beta: float = 1.5,
        unk_score_offset: float = -10.0,
        score_boundary: bool = True,
    ) -> None:
        self._kenlm_model = kenlm_model
        if unigrams is None:
            logger.warning(
                "No known unigrams provided, decoding results might be a lot worse."
            )
            unigram_set = set()
            char_trie = None
        else:
            unigram_set = _prepare_unigram_set(unigrams, self._kenlm_model)
            char_trie = CharTrie.fromkeys(unigram_set)
        self._unigram_set = unigram_set
        self._char_trie = char_trie
        self.alpha = alpha
        self.beta = beta
        self.unk_score_offset = unk_score_offset
        self.score_boundary = score_boundary

    @property
    def order(self) -> int:
        """Get the order of the n-gram language model."""
        return cast(int, self._kenlm_model.order)

    def get_start_state(self) -> KenlmState:
        """Get initial lm state."""
        start_state = _get_empty_lm_state()
        if self.score_boundary:
            self._kenlm_model.BeginSentenceWrite(start_state)
        else:
            self._kenlm_model.NullContextWrite(start_state)
        return KenlmState(start_state)

    def _get_raw_end_score(self, start_state: "kenlm.State") -> float:
        """Calculate final lm score."""
        if self.score_boundary:
            end_state = _get_empty_lm_state()
            score: float = self._kenlm_model.BaseScore(
                start_state, "</s>", end_state
            )
        else:
            score = 0.0
        return score

    def score_partial_token(self, partial_token: str) -> float:
        """Get partial token score."""
        if self._char_trie is None:
            is_oov = 1.0
        else:
            is_oov = int(self._char_trie.has_node(partial_token) == 0)
        unk_score = self.unk_score_offset * is_oov
        # if unk token length exceeds expected length then additionally decrease score
        if len(partial_token) > 6:
            unk_score = unk_score * len(partial_token) / 6
        return unk_score

    def score(
        self, prev_state, word: str, is_last_word: bool = False
    ) -> Tuple[float, KenlmState]:
        """Score word conditional on start state."""
        if not isinstance(prev_state, KenlmState):
            raise AssertionError(
                f"Wrong input state type found. Expected KenlmState, got {type(prev_state)}"
            )
        end_state = _get_empty_lm_state()
        lm_score = self._kenlm_model.BaseScore(
            prev_state.state, word, end_state
        )
        # override UNK prob. use unigram set if we have because it's faster
        if (
            len(self._unigram_set) > 0
            and word not in self._unigram_set
            or word not in self._kenlm_model
        ):
            lm_score += self.unk_score_offset
        # add end of sentence context if needed
        if is_last_word:
            # note that we want to return the unmodified end_state to keep extension capabilities
            lm_score = lm_score + self._get_raw_end_score(end_state)
        lm_score = self.alpha * lm_score * 1.0 / math.log10(math.e) + self.beta
        return lm_score, KenlmState(end_state)
