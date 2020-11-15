from itertools import chain
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple
from typing import Union

import torch
import six
import warnings
import numpy as np


class ScorerInterface:
    """Scorer interface for beam search.
    The scorer performs scoring of the all tokens in vocabulary.
    Examples:
        * Search heuristics
            * :class:`espnet.nets.scorers.length_bonus.LengthBonus`
        * Decoder networks of the sequence-to-sequence models
            * :class:`espnet.nets.pytorch_backend.nets.transformer.decoder.Decoder`
            * :class:`espnet.nets.pytorch_backend.nets.rnn.decoders.Decoder`
        * Neural language models
            * :class:`espnet.nets.pytorch_backend.lm.transformer.TransformerLM`
            * :class:`espnet.nets.pytorch_backend.lm.default.DefaultRNNLM`
            * :class:`espnet.nets.pytorch_backend.lm.seq_rnn.SequentialRNNLM`
    """

    def init_state(self, x: torch.Tensor) -> Any:
        """Get an initial state for decoding (optional).
        Args:
            x (torch.Tensor): The encoded feature tensor
        Returns: initial state
        """
        return None

    def select_state(self, state: Any, i: int, new_id: int = None) -> Any:
        """Select state with relative ids in the main beam search.
        Args:
            state: Decoder state for prefix tokens
            i (int): Index to select a state in the main beam search
            new_id (int): New label index to select a state if necessary
        Returns:
            state: pruned state
        """
        return None if state is None else state[i]

    def score(
        self, y: torch.Tensor, state: Any, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        """Score new token (required).
        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (torch.Tensor): The encoder feature that generates ys.
        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                scores for next token that has a shape of `(n_vocab)`
                and next state for ys
        """
        raise NotImplementedError

    def final_score(self, state: Any) -> float:
        """Score eos (optional).
        Args:
            state: Scorer state for prefix tokens
        Returns:
            float: final score
        """
        return 0.0


class BatchScorerInterface(ScorerInterface):
    """Batch scorer interface."""

    def batch_init_state(self, x: torch.Tensor) -> Any:
        """Get an initial state for decoding (optional).
        Args:
            x (torch.Tensor): The encoded feature tensor
        Returns: initial state
        """
        return self.init_state(x)

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch (required).
        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).
        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.
        """
        warnings.warn(
            "{} batch score is implemented through for loop not parallelized".format(
                self.__class__.__name__
            )
        )
        scores = list()
        outstates = list()
        for i, (y, state, x) in enumerate(zip(ys, states, xs)):
            score, outstate = self.score(y, state, x)
            outstates.append(outstate)
            scores.append(score)
        scores = torch.cat(scores, 0).view(ys.shape[0], -1)
        return scores, outstates


class PartialScorerInterface(ScorerInterface):
    """Partial scorer interface for beam search.
    The partial scorer performs scoring when non-partial scorer finished scoring,
    and recieves pre-pruned next tokens to score because it is too heavy to score
    all the tokens.
    Examples:
         * Prefix search for connectionist-temporal-classification models
             * :class:`espnet.nets.scorers.ctc.CTCPrefixScorer`
    """

    def score_partial(
        self,
        y: torch.Tensor,
        next_tokens: torch.Tensor,
        state: Any,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Any]:
        """Score new token (required).
        Args:
            y (torch.Tensor): 1D prefix token
            next_tokens (torch.Tensor): torch.int64 next token to score
            state: decoder state for prefix tokens
            x (torch.Tensor): The encoder feature that generates ys
        Returns:
            tuple[torch.Tensor, Any]:
                Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                and next state for ys
        """
        raise NotImplementedError


class BatchPartialScorerInterface(BatchScorerInterface, PartialScorerInterface):
    """Batch partial scorer interface for beam search."""

    def batch_score_partial(
        self,
        ys: torch.Tensor,
        next_tokens: torch.Tensor,
        states: List[Any],
        xs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Any]:
        """Score new token (required).
        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            next_tokens (torch.Tensor): torch.int64 tokens to score (n_batch, n_token).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).
        Returns:
            tuple[torch.Tensor, Any]:
                Tuple of a score tensor for ys that has a shape `(n_batch, n_vocab)`
                and next states for ys
        """
        raise NotImplementedError


class PartialScorerInterface(ScorerInterface):
    """Partial scorer interface for beam search.
    The partial scorer performs scoring when non-partial scorer finished scoring,
    and recieves pre-pruned next tokens to score because it is too heavy to score
    all the tokens.
    Examples:
         * Prefix search for connectionist-temporal-classification models
             * :class:`espnet.nets.scorers.ctc.CTCPrefixScorer`
    """

    def score_partial(
        self,
        y: torch.Tensor,
        next_tokens: torch.Tensor,
        state: Any,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Any]:
        """Score new token (required).
        Args:
            y (torch.Tensor): 1D prefix token
            next_tokens (torch.Tensor): torch.int64 next token to score
            state: decoder state for prefix tokens
            x (torch.Tensor): The encoder feature that generates ys
        Returns:
            tuple[torch.Tensor, Any]:
                Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                and next state for ys
        """
        raise NotImplementedError


class CTCPrefixScore(object):
    """Compute CTC label sequence scores
    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    but extended to efficiently compute the probablities of multiple labels
    simultaneously
    """

    def __init__(self, x, blank, eos, xp):
        self.xp = xp
        self.logzero = -10000000000.0
        self.blank = blank
        self.eos = eos
        self.input_length = len(x)
        self.x = x

    def initial_state(self):
        """Obtain an initial CTC state
        :return: CTC state
        """
        # initial CTC state is made of a frame x 2 tensor that corresponds to
        # r_t^n(<sos>) and r_t^b(<sos>), where 0 and 1 of axis=1 represent
        # superscripts n and b (non-blank and blank), respectively.
        r = self.xp.full((self.input_length, 2), self.logzero, dtype=np.float32)
        r[0, 1] = self.x[0, self.blank]
        for i in six.moves.range(1, self.input_length):
            r[i, 1] = r[i - 1, 1] + self.x[i, self.blank]
        return r

    def __call__(self, y, cs, r_prev):
        """Compute CTC prefix scores for next labels
        :param y     : prefix label sequence
        :param cs    : array of next labels
        :param r_prev: previous CTC state
        :return ctc_scores, ctc_states
        """
        # initialize CTC states
        output_length = len(y) - 1  # ignore sos
        # new CTC states are prepared as a frame x (n or b) x n_labels tensor
        # that corresponds to r_t^n(h) and r_t^b(h).
        r = self.xp.ndarray((self.input_length, 2, len(cs)), dtype=np.float32)
        xs = self.x[:, cs]
        if output_length == 0:
            r[0, 0] = xs[0]
            r[0, 1] = self.logzero
        else:
            r[output_length - 1] = self.logzero

        # prepare forward probabilities for the last label
        r_sum = self.xp.logaddexp(
            r_prev[:, 0], r_prev[:, 1]
        )  # log(r_t^n(g) + r_t^b(g))
        last = y[-1]
        if output_length > 0 and last in cs:
            log_phi = self.xp.ndarray(
                (self.input_length, len(cs)), dtype=np.float32
            )
            for i in six.moves.range(len(cs)):
                log_phi[:, i] = r_sum if cs[i] != last else r_prev[:, 1]
        else:
            log_phi = r_sum

        # compute forward probabilities log(r_t^n(h)), log(r_t^b(h)),
        # and log prefix probabilites log(psi)
        start = max(output_length, 1)
        log_psi = r[start - 1, 0]
        for t in six.moves.range(start, self.input_length):
            r[t, 0] = self.xp.logaddexp(r[t - 1, 0], log_phi[t - 1]) + xs[t]
            r[t, 1] = (
                self.xp.logaddexp(r[t - 1, 0], r[t - 1, 1])
                + self.x[t, self.blank]
            )
            log_psi = self.xp.logaddexp(log_psi, log_phi[t - 1] + xs[t])

        # get P(...eos|X) that ends with the prefix itself
        eos_pos = self.xp.where(cs == self.eos)[0]
        if len(eos_pos) > 0:
            log_psi[eos_pos] = r_sum[-1]  # log(r_T^n(g) + r_T^b(g))

        # exclude blank probs
        blank_pos = self.xp.where(cs == self.blank)[0]
        if len(blank_pos) > 0:
            log_psi[blank_pos] = self.logzero

        # return the log prefix probability and CTC states, where the label axis
        # of the CTC states is moved to the first axis to slice it easily
        return log_psi, self.xp.rollaxis(r, 2)


class CTCPrefixScorer(PartialScorerInterface):
    """Decoder interface wrapper for CTCPrefixScore."""

    def __init__(self, ctc: torch.nn.Module, eos: int):
        """Initialize class.
        Args:
            ctc (torch.nn.Module): The CTC implementaiton.
                For example, :class:`espnet.nets.pytorch_backend.ctc.CTC`
            eos (int): The end-of-sequence id.
        """
        self.ctc = ctc
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.eos = eos
        self.impl = None

    def init_state(self, x: torch.Tensor):
        """Get an initial state for decoding.
        Args:
            x (torch.Tensor): The encoded feature tensor
        Returns: initial state
        """
        logp = (
            self.log_softmax(self.ctc(x.unsqueeze(0)))
            .detach()
            .squeeze(0)
            .cpu()
            .numpy()
        )
        # TODO(karita): use CTCPrefixScoreTH
        self.impl = CTCPrefixScore(logp, 0, self.eos, np)
        return 0, self.impl.initial_state()

    def select_state(self, state, i, new_id=None):
        """Select state with relative ids in the main beam search.
        Args:
            state: Decoder state for prefix tokens
            i (int): Index to select a state in the main beam search
            new_id (int): New label id to select a state if necessary
        Returns:
            state: pruned state
        """
        if type(state) == tuple:
            if len(state) == 2:  # for CTCPrefixScore
                sc, st = state
                return sc[i], st[i]
            else:  # for CTCPrefixScoreTH (need new_id > 0)
                r, log_psi, f_min, f_max, scoring_idmap = state
                s = log_psi[i, new_id].expand(log_psi.size(1))
                if scoring_idmap is not None:
                    return r[:, :, i, scoring_idmap[i, new_id]], s, f_min, f_max
                else:
                    return r[:, :, i, new_id], s, f_min, f_max
        return None if state is None else state[i]

    def score_partial(self, y, ids, state, x):
        """Score new token.
        Args:
            y (torch.Tensor): 1D prefix token
            next_tokens (torch.Tensor): torch.int64 next token to score
            state: decoder state for prefix tokens
            x (torch.Tensor): 2D encoder feature that generates ys
        Returns:
            tuple[torch.Tensor, Any]:
                Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                and next state for ys
        """
        prev_score, state = state
        presub_score, new_st = self.impl(y.cpu(), ids.cpu(), state)
        tscore = torch.as_tensor(
            presub_score - prev_score, device=x.device, dtype=x.dtype
        )
        return tscore, (presub_score, new_st)

    # def batch_init_state(self, x: torch.Tensor):
    #     """Get an initial state for decoding.
    #     Args:
    #         x (torch.Tensor): The encoded feature tensor
    #     Returns: initial state
    #     """
    #     logp = self.ctc.log_softmax(x.unsqueeze(0))  # assuming batch_size = 1
    #     xlen = torch.tensor([logp.size(1)])
    #     self.impl = CTCPrefixScoreTH(logp, xlen, 0, self.eos)
    #     return None

    # def batch_score_partial(self, y, ids, state, x):
    #     """Score new token.
    #     Args:
    #         y (torch.Tensor): 1D prefix token
    #         ids (torch.Tensor): torch.int64 next token to score
    #         state: decoder state for prefix tokens
    #         x (torch.Tensor): 2D encoder feature that generates ys
    #     Returns:
    #         tuple[torch.Tensor, Any]:
    #             Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
    #             and next state for ys
    #     """
    #     batch_state = (
    #         (
    #             torch.stack([s[0] for s in state], dim=2),
    #             torch.stack([s[1] for s in state]),
    #             state[0][2],
    #             state[0][3],
    #         )
    #         if state[0] is not None
    #         else None
    #     )
    #     return self.impl(y, batch_state, ids)


class TransformerAMScorer(ScorerInterface):
    def __init__(self, transformer, linear):
        self.transformer = transformer
        self.linear = linear
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

    def score(
        self, y: torch.Tensor, state: Any, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        """Score new token (required).
        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (torch.Tensor): The encoder feature that generates ys.
        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                scores for next token that has a shape of `(n_vocab)`
                and next state for ys
        """
        logit = self.transformer.decode(y.unsqueeze(0), x.unsqueeze(0))
        log_prob = self.log_softmax(self.linear(logit))[:, -1, :]
        return log_prob.squeeze(0), state


class TransformerLMScorer(ScorerInterface):
    def __init__(self, lm):
        self.lm = lm
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

    def score(
        self, y: torch.Tensor, state: Any, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        self.lm.to(y.device)
        logit = self.lm(y.unsqueeze(0))
        log_prob = self.log_softmax(logit)[:, -1, :]
        return log_prob.squeeze(0), state


def end_detect(ended_hyps, i, M=3, D_end=np.log(1 * np.exp(-10))):
    """End detection.
    desribed in Eq. (50) of S. Watanabe et al
    "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition"
    :param ended_hyps:
    :param i:
    :param M:
    :param D_end:
    :return:
    """
    if len(ended_hyps) == 0:
        return False
    count = 0
    best_hyp = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[0]
    for m in six.moves.range(M):
        # get ended_hyps with their length is i - m
        hyp_length = i - m
        hyps_same_length = [
            x for x in ended_hyps if len(x["yseq"]) == hyp_length
        ]
        if len(hyps_same_length) > 0:
            best_hyp_same_length = sorted(
                hyps_same_length, key=lambda x: x["score"], reverse=True
            )[0]
            if best_hyp_same_length["score"] - best_hyp["score"] < D_end:
                count += 1

    if count == M:
        return True
    else:
        return False


class Hypothesis(NamedTuple):
    """Hypothesis data type."""

    yseq: torch.Tensor
    score: Union[float, torch.Tensor] = 0
    scores: Dict[str, Union[float, torch.Tensor]] = dict()
    states: Dict[str, Any] = dict()

    def asdict(self) -> dict:
        """Convert data to JSON-friendly dict."""
        return self._replace(
            yseq=self.yseq.tolist(),
            score=float(self.score),
            scores={k: float(v) for k, v in self.scores.items()},
        )._asdict()


class BeamSearch(torch.nn.Module):
    """Beam search implementation."""

    def __init__(
        self,
        scorers: Dict[str, ScorerInterface],
        weights: Dict[str, float],
        beam_size: int,
        vocab_size: int,
        sos: int,
        eos: int,
        token_list: List[str] = None,
        pre_beam_ratio: float = 1.5,
        pre_beam_score_key: str = None,
    ):
        """Initialize beam search.
        Args:
            scorers (dict[str, ScorerInterface]): Dict of decoder modules
                e.g., Decoder, CTCPrefixScorer, LM
                The scorer will be ignored if it is `None`
            weights (dict[str, float]): Dict of weights for each scorers
                The scorer will be ignored if its weight is 0
            beam_size (int): The number of hypotheses kept during search
            vocab_size (int): The number of vocabulary
            sos (int): Start of sequence id
            eos (int): End of sequence id
            token_list (list[str]): List of tokens for debug log
            pre_beam_score_key (str): key of scores to perform pre-beam search
            pre_beam_ratio (float): beam size in the pre-beam search
                will be `int(pre_beam_ratio * beam_size)`
        """
        super().__init__()
        # set scorers
        self.weights = weights
        self.scorers = dict()
        self.full_scorers = dict()
        self.part_scorers = dict()
        # this module dict is required for recursive cast
        # `self.to(device, dtype)` in `recog.py`
        self.nn_dict = torch.nn.ModuleDict()
        for k, v in scorers.items():
            w = weights.get(k, 0)
            if w == 0 or v is None:
                continue
            assert isinstance(
                v, ScorerInterface
            ), f"{k} ({type(v)}) does not implement ScorerInterface"
            self.scorers[k] = v
            if isinstance(v, PartialScorerInterface):
                self.part_scorers[k] = v
            else:
                self.full_scorers[k] = v
            if isinstance(v, torch.nn.Module):
                self.nn_dict[k] = v

        # set configurations
        self.sos = sos
        self.eos = eos
        self.token_list = token_list
        self.pre_beam_size = int(pre_beam_ratio * beam_size)
        self.beam_size = beam_size
        self.n_vocab = vocab_size
        if (
            pre_beam_score_key is not None
            and pre_beam_score_key != "full"
            and pre_beam_score_key not in self.full_scorers
        ):
            raise KeyError(
                f"{pre_beam_score_key} is not found in {self.full_scorers}"
            )
        self.pre_beam_score_key = pre_beam_score_key
        self.do_pre_beam = (
            self.pre_beam_score_key is not None
            and self.pre_beam_size < self.n_vocab
            and len(self.part_scorers) > 0
        )

    def init_hyp(self, x: torch.Tensor) -> List[Hypothesis]:
        """Get an initial hypothesis data.
        Args:
            x (torch.Tensor): The encoder output feature
        Returns:
            Hypothesis: The initial hypothesis.
        """
        init_states = dict()
        init_scores = dict()
        for k, d in self.scorers.items():
            init_states[k] = d.init_state(x)
            init_scores[k] = 0.0
        return [
            Hypothesis(
                score=0.0,
                scores=init_scores,
                states=init_states,
                yseq=torch.tensor([self.sos], device=x.device),
            )
        ]

    @staticmethod
    def append_token(xs: torch.Tensor, x: int) -> torch.Tensor:
        """Append new token to prefix tokens.
        Args:
            xs (torch.Tensor): The prefix token
            x (int): The new token to append
        Returns:
            torch.Tensor: New tensor contains: xs + [x] with xs.dtype and xs.device
        """
        x = torch.tensor([x], dtype=xs.dtype, device=xs.device)
        return torch.cat((xs, x))

    def score_full(
        self, hyp: Hypothesis, x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by `self.full_scorers`.
        Args:
            hyp (Hypothesis): Hypothesis with prefix tokens to score
            x (torch.Tensor): Corresponding input feature
        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.full_scorers`
                and tensor score values of shape: `(self.n_vocab,)`,
                and state dict that has string keys
                and state values of `self.full_scorers`
        """
        scores = dict()
        states = dict()
        for k, d in self.full_scorers.items():
            scores[k], states[k] = d.score(hyp.yseq, hyp.states[k], x)
        return scores, states

    def score_partial(
        self, hyp: Hypothesis, ids: torch.Tensor, x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by `self.part_scorers`.
        Args:
            hyp (Hypothesis): Hypothesis with prefix tokens to score
            ids (torch.Tensor): 1D tensor of new partial tokens to score
            x (torch.Tensor): Corresponding input feature
        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.part_scorers`
                and tensor score values of shape: `(len(ids),)`,
                and state dict that has string keys
                and state values of `self.part_scorers`
        """
        scores = dict()
        states = dict()
        for k, d in self.part_scorers.items():
            scores[k], states[k] = d.score_partial(
                hyp.yseq, ids, hyp.states[k], x
            )
        return scores, states

    def beam(
        self, weighted_scores: torch.Tensor, ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute topk full token ids and partial token ids.
        Args:
            weighted_scores (torch.Tensor): The weighted sum scores for each tokens.
            Its shape is `(self.n_vocab,)`.
            ids (torch.Tensor): The partial token ids to compute topk
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                The topk full token ids and partial token ids.
                Their shapes are `(self.beam_size,)`
        """
        # no pre beam performed
        if weighted_scores.size(0) == ids.size(0):
            top_ids = weighted_scores.topk(self.beam_size)[1]
            return top_ids, top_ids

        # mask pruned in pre-beam not to select in topk
        tmp = weighted_scores[ids]
        weighted_scores[:] = -float("inf")
        weighted_scores[ids] = tmp
        top_ids = weighted_scores.topk(self.beam_size)[1]
        local_ids = weighted_scores[ids].topk(self.beam_size)[1]
        return top_ids, local_ids

    @staticmethod
    def merge_scores(
        prev_scores: Dict[str, float],
        next_full_scores: Dict[str, torch.Tensor],
        full_idx: int,
        next_part_scores: Dict[str, torch.Tensor],
        part_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Merge scores for new hypothesis.
        Args:
            prev_scores (Dict[str, float]):
                The previous hypothesis scores by `self.scorers`
            next_full_scores (Dict[str, torch.Tensor]): scores by `self.full_scorers`
            full_idx (int): The next token id for `next_full_scores`
            next_part_scores (Dict[str, torch.Tensor]):
                scores of partial tokens by `self.part_scorers`
            part_idx (int): The new token id for `next_part_scores`
        Returns:
            Dict[str, torch.Tensor]: The new score dict.
                Its keys are names of `self.full_scorers` and `self.part_scorers`.
                Its values are scalar tensors by the scorers.
        """
        new_scores = dict()
        for k, v in next_full_scores.items():
            new_scores[k] = prev_scores[k] + v[full_idx]
        for k, v in next_part_scores.items():
            new_scores[k] = prev_scores[k] + v[part_idx]
        return new_scores

    def merge_states(self, states: Any, part_states: Any, part_idx: int) -> Any:
        """Merge states for new hypothesis.
        Args:
            states: states of `self.full_scorers`
            part_states: states of `self.part_scorers`
            part_idx (int): The new token id for `part_scores`
        Returns:
            Dict[str, torch.Tensor]: The new score dict.
                Its keys are names of `self.full_scorers` and `self.part_scorers`.
                Its values are states of the scorers.
        """
        new_states = dict()
        for k, v in states.items():
            new_states[k] = v
        for k, d in self.part_scorers.items():
            new_states[k] = d.select_state(part_states[k], part_idx)
        return new_states

    def search(
        self, running_hyps: List[Hypothesis], x: torch.Tensor
    ) -> List[Hypothesis]:
        """Search new tokens for running hypotheses and encoded speech x.
        Args:
            running_hyps (List[Hypothesis]): Running hypotheses on beam
            x (torch.Tensor): Encoded speech feature (T, D)
        Returns:
            List[Hypotheses]: Best sorted hypotheses
        """
        best_hyps = []
        part_ids = torch.arange(self.n_vocab, device=x.device)  # no pre-beam
        for hyp in running_hyps:
            # scoring
            weighted_scores = torch.zeros(
                self.n_vocab, dtype=x.dtype, device=x.device
            )
            scores, states = self.score_full(hyp, x)
            for k in self.full_scorers:
                weighted_scores += self.weights[k] * scores[k]
            # partial scoring
            if self.do_pre_beam:
                pre_beam_scores = (
                    weighted_scores
                    if self.pre_beam_score_key == "full"
                    else scores[self.pre_beam_score_key]
                )
                part_ids = torch.topk(pre_beam_scores, self.pre_beam_size)[1]
            part_scores, part_states = self.score_partial(hyp, part_ids, x)
            for k in self.part_scorers:
                weighted_scores[part_ids] += self.weights[k] * part_scores[k]
            # add previous hyp score
            weighted_scores += hyp.score

            # update hyps
            for j, part_j in zip(*self.beam(weighted_scores, part_ids)):
                # will be (2 x beam at most)
                best_hyps.append(
                    Hypothesis(
                        score=weighted_scores[j],
                        yseq=self.append_token(hyp.yseq, j),
                        scores=self.merge_scores(
                            hyp.scores, scores, j, part_scores, part_j
                        ),
                        states=self.merge_states(states, part_states, part_j),
                    )
                )

            # sort and prune 2 x beam -> beam
            best_hyps = sorted(best_hyps, key=lambda x: x.score, reverse=True)[
                : min(len(best_hyps), self.beam_size)
            ]
        return best_hyps

    def forward(
        self,
        x: torch.Tensor,
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
    ) -> List[Hypothesis]:
        """Perform beam search.
        Args:
            x (torch.Tensor): Encoded speech feature (T, D)
            maxlenratio (float): Input length ratio to obtain max output length.
                If maxlenratio=0.0 (default), it uses a end-detect function
                to automatically find maximum hypothesis lengths
            minlenratio (float): Input length ratio to obtain min output length.
        Returns:
            list[Hypothesis]: N-best decoding results
        """
        # set length bounds
        if maxlenratio == 0:
            maxlen = x.shape[0]
        else:
            maxlen = max(1, int(maxlenratio * x.size(0)))
        minlen = int(minlenratio * x.size(0))
        logging.info("decoder input length: " + str(x.shape[0]))
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # main loop of prefix search
        running_hyps = self.init_hyp(x)
        ended_hyps = []
        for i in range(maxlen):
            logging.debug("position " + str(i))
            best = self.search(running_hyps, x)
            # post process of one iteration
            running_hyps = self.post_process(
                i, maxlen, maxlenratio, best, ended_hyps
            )
            # end detection
            if maxlenratio == 0.0 and end_detect(
                [h.asdict() for h in ended_hyps], i
            ):
                logging.info(f"end detected at {i}")
                break
            if len(running_hyps) == 0:
                logging.info("no hypothesis. Finish decoding.")
                break
            else:
                logging.debug(f"remained hypotheses: {len(running_hyps)}")

        nbest_hyps = sorted(ended_hyps, key=lambda x: x.score, reverse=True)
        # check the number of hypotheses reaching to eos
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            return (
                []
                if minlenratio < 0.1
                else self.forward(x, maxlenratio, max(0.0, minlenratio - 0.1))
            )

        # report the best result
        best = nbest_hyps[0]
        for k, v in best.scores.items():
            logging.info(
                f"{v:6.2f} * {self.weights[k]:3} = {v * self.weights[k]:6.2f} for {k}"
            )
        logging.info(f"total log probability: {best.score:.2f}")
        logging.info(
            f"normalized log probability: {best.score / len(best.yseq):.2f}"
        )
        logging.info(f"total number of ended hypotheses: {len(nbest_hyps)}")
        if self.token_list is not None:
            logging.info(
                "best hypo: "
                + "".join([self.token_list[x] for x in best.yseq[1:-1]])
                + "\n"
            )
        return nbest_hyps

    def post_process(
        self,
        i: int,
        maxlen: int,
        maxlenratio: float,
        running_hyps: List[Hypothesis],
        ended_hyps: List[Hypothesis],
    ) -> List[Hypothesis]:
        """Perform post-processing of beam search iterations.
        Args:
            i (int): The length of hypothesis tokens.
            maxlen (int): The maximum length of tokens in beam search.
            maxlenratio (int): The maximum length ratio in beam search.
            running_hyps (List[Hypothesis]): The running hypotheses in beam search.
            ended_hyps (List[Hypothesis]): The ended hypotheses in beam search.
        Returns:
            List[Hypothesis]: The new running hypotheses.
        """
        logging.debug(f"the number of running hypotheses: {len(running_hyps)}")
        if self.token_list is not None:
            logging.debug(
                "best hypo: "
                + "".join(
                    [self.token_list[x] for x in running_hyps[0].yseq[1:]]
                )
            )
        # add eos in the final loop to avoid that there are no ended hyps
        if i == maxlen - 1:
            logging.info("adding <eos> in the last position in the loop")
            running_hyps = [
                h._replace(yseq=self.append_token(h.yseq, self.eos))
                for h in running_hyps
            ]

        # add ended hypotheses to a final list, and removed them from current hypotheses
        # (this will be a problem, number of hyps < beam)
        remained_hyps = []
        for hyp in running_hyps:
            if hyp.yseq[-1] == self.eos:
                # e.g., Word LM needs to add final <eos> score
                for k, d in chain(
                    self.full_scorers.items(), self.part_scorers.items()
                ):
                    s = d.final_score(hyp.states[k])
                    hyp.scores[k] += s
                    hyp = hyp._replace(score=hyp.score + self.weights[k] * s)
                ended_hyps.append(hyp)
            else:
                remained_hyps.append(hyp)
        return remained_hyps


class TransformerBeamSearch:
    def __init__(
        self,
        modules: list,
        lm_weight: float,
        ctc_weight: float,
        lm_modules: torch.nn.Module,
        sos: int,
        eos: int,
        beam_size: int,
        vocab_size: int,
        token_list: List[str] = None,
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        pre_beam_ratio: float = 1.5,
        pre_beam_score_key: str = "full",
    ):
        self.transformer = modules[0]
        self.seq_linear = modules[1]
        self.ctc_linear = modules[2]
        self.lm = lm_modules
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio

        # TODO: Scorer Dict & Weight Dict & Beamsearch
        decoder = TransformerAMScorer(
            transformer=self.transformer, linear=self.seq_linear
        )

        lm = TransformerLMScorer(lm=self.lm)

        ctc = CTCPrefixScorer(ctc=self.ctc_linear, eos=eos)

        self.scorers = dict(decoder=decoder, lm=lm, ctc=ctc)

        self.weights = dict(
            decoder=1.0 - ctc_weight, ctc=ctc_weight, lm=lm_weight,
        )

        self.beam_search = BeamSearch(
            beam_size=beam_size,
            vocab_size=vocab_size,
            weights=self.weights,
            scorers=self.scorers,
            sos=sos,
            eos=eos,
            token_list=token_list,
            pre_beam_score_key=None if ctc_weight == 1 else "full",
        )

    def __call__(self, x, length):
        best_hyp = []
        scores = []
        for i in range(x.shape[0]):
            self.beam_search.to(x.device).eval()
            nbest_hyps = self.beam_search(
                x[i, :, :],
                maxlenratio=self.maxlenratio,
                minlenratio=self.minlenratio,
            )
            best = nbest_hyps[0].asdict()
            best_hyp.append(best["yseq"])
            scores.append(best["score"])
        return best_hyp, scores
