import torch
import numpy as np
import kenlm
import speechbrain as sb
from speechbrain.decoders.ctc import CTCPrefixScore
import sentencepiece as spm


class BaseScorer:
    # an abstraction for scorers:
    #   ctc scorer, ngram scorer, coverage penalty

    def score(self, g, states, candidates, attn):
        raise NotImplementedError

    def permute_mem(self, memory, index):
        return None, None

    def reset_mem(self, x, enc_lens):
        return None


class CTCPrefixScorer(BaseScorer):
    def __init__(
        self, ctc_fc, blank_index, eos_index, ctc_window_size=0,
    ):
        self.ctc_fc = ctc_fc
        self.softmax = sb.nnet.activations.Softmax(apply_log=True)
        self.blank_index = blank_index
        self.eos_index = eos_index
        self.ctc_window_size = ctc_window_size

    def score(self, g, states, candidates, attn):
        scores, states = self.ctc_score.forward_step(
            g, states, candidates, attn
        )
        return scores, states

    def permute_mem(self, memory, index):
        r, psi = self.ctc_score.permute_mem(memory, index)
        return r, psi

    def reset_mem(self, x, enc_lens):
        logits = self.ctc_fc(x)
        x = self.softmax(logits)
        self.ctc_score = CTCPrefixScore(
            x, enc_lens, self.blank_index, self.eos_index, self.ctc_window_size
        )
        return None


class NGramScorer(BaseScorer):
    def __init__(
        self, lm_path, tokenizer_path, vocab_size, bos_index, eos_index
    ):
        self.lm = kenlm.Model(lm_path)
        self.vocab_size = vocab_size
        self.full_candidates = np.arange(self.vocab_size)
        self.minus_inf = -1e20

        # Create token list
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        self.id2char = [
            tokenizer.id_to_piece([i])[0].replace("\u2581", "_")
            for i in range(vocab_size)
        ]
        self.id2char[bos_index] = "<s>"
        self.id2char[eos_index] = "</s>"

    def score(self, g, states, candidates, attn):
        """
        Returns:
        new_memory: [B * Num_hyps, Vocab_size]

        """
        n_bh = g.size(0)
        scale = 1.0 / np.log10(np.e)

        if states is None:
            state = kenlm.State()
            state = np.array([state] * n_bh)
            scoring_table = np.ones(n_bh)
        else:
            state, scoring_table = states

        # Perform full scorer mode, not recommend
        if candidates is None:
            candidates = [self.full_candidates] * n_bh

        # Store new states and scores
        scores = np.ones((n_bh, self.vocab_size)) * self.minus_inf
        new_memory = np.zeros((n_bh, self.vocab_size), dtype=object)
        new_scoring_table = np.ones((n_bh, self.vocab_size)) * -1
        # Scoring
        for i in range(n_bh):
            if scoring_table[i] == -1:
                continue
            parent_state = state[i]
            for token_id in candidates[i]:
                char = self.id2char[token_id.item()]
                out_state = kenlm.State()
                score = scale * self.lm.BaseScore(parent_state, char, out_state)
                scores[i, token_id] = score
                new_memory[i, token_id] = out_state
                new_scoring_table[i, token_id] = 1
        scores = torch.from_numpy(scores).float().to(g.device)
        return scores, (new_memory, new_scoring_table)

    def permute_mem(self, memory, index):
        """
        Returns:
        new_memory: [B, Num_hyps]

        """
        state, scoring_table = memory

        index = index.cpu().numpy()
        # The first index of each sentence.
        beam_size = index.shape[1]
        beam_offset = self.batch_index * beam_size
        hyp_index = (
            index
            + np.broadcast_to(np.expand_dims(beam_offset, 1), index.shape)
            * self.vocab_size
        )
        hyp_index = hyp_index.reshape(-1)
        # Update states
        state = state.reshape(-1)
        state = state[hyp_index]
        scoring_table = scoring_table.reshape(-1)
        scoring_table = scoring_table[hyp_index]
        return state, scoring_table

    def reset_mem(self, x, enc_lens):
        state = kenlm.State()
        self.lm.NullContextWrite(state)
        self.batch_index = np.arange(x.size(0))
        return None


class CoveragePenalty(BaseScorer):
    def __init__(self, vocab_size, threshold=0.5):
        self.threshold = threshold
        self.vocab_size = vocab_size

    def score(self, g, coverage, candidates, attn):
        n_bh = attn.size(0)

        if coverage is None:
            coverage = torch.zeros_like(attn, device=attn.device)

        # the attn of transformer is [batch_size*beam_size, current_step, source_len]
        if len(attn.size()) > 2:
            coverage = torch.sum(attn, dim=1)

        # Current coverage
        coverage = coverage + attn
        # Compute coverage penalty and add it to scores
        penalty = torch.max(
            coverage, coverage.clone().fill_(self.threshold)
        ).sum(-1)
        penalty = penalty - coverage.size(-1) * self.threshold
        penalty = penalty.view(n_bh).unsqueeze(1).expand(-1, self.vocab_size)
        return -1 * penalty, coverage

    def permute_mem(self, coverage, index):
        # Update coverage
        n_bh = coverage.size(0)
        beam_size = index.size(1)
        beam_offset = self.batch_index * beam_size
        hyp_index = (
            index // self.vocab_size + beam_offset.unsqueeze(1).expand_as(index)
        ).view(n_bh)
        coverage = torch.index_select(coverage, dim=0, index=hyp_index)
        return coverage

    def reset_mem(self, x, enc_lens):
        self.batch_index = torch.arange(x.size(0), device=x.device)
        return None


class ScorerBuilder:
    def __init__(
        self,
        bos_index,
        eos_index,
        blank_index,
        vocab_size,
        ctc_weight=0.0,
        ngram_weight=0.0,
        coverage_weight=0.0,
        ctc_score_mode="partial",
        ngram_score_mode="partial",
        ctc_linear=None,
        lm_path=None,
        tokenizer=None,
    ):
        """
        weights: Dict
        score_mode: Dict
        """
        self.weights = dict(
            ctc=ctc_weight, ngram=ngram_weight, coverage=coverage_weight
        )
        self.score_mode = dict(ctc=ctc_score_mode, ngram=ngram_score_mode,)
        self.scorers = {}

        if self.score_mode["ctc"] == "full" or ctc_weight == 1.0:
            ctc_weight = 1.0
            self.score_mode["ctc"] = "full"
            coverage_weight = 0.0

        if ctc_weight > 0.0:
            self.scorers["ctc"] = CTCPrefixScorer(
                ctc_linear, blank_index, eos_index
            )

        if ngram_weight > 0.0:

            self.scorers["ngram"] = NGramScorer(
                lm_path, tokenizer, vocab_size, bos_index, eos_index
            )

        if coverage_weight > 0.0:
            self.scorers["coverage"] = CoveragePenalty(vocab_size)
            # Must be full scorer model
            self.score_mode["coverage"] = "full"

    def score(self, g, states, attn, log_probs, beam_size):
        new_states = dict()
        # score full candidates
        for k, impl in self.scorers.items():
            if self.score_mode[k] != "full":
                continue
            score, new_states[k] = impl.score(g, states[k], None, attn)
            log_probs += score * self.weights[k]
        # select candidates for partial scorers
        _, candidates = log_probs.topk(int(beam_size * 1.5), dim=-1)
        # score patial candidates
        for k, impl in self.scorers.items():
            if self.score_mode[k] != "partial":
                continue
            score, new_states[k] = impl.score(g, states[k], candidates, attn)
            log_probs += score * self.weights[k]

        return log_probs, new_states

    def permute_scorer_mem(self, states, index):
        for k, impl in self.scorers.items():
            states[k] = impl.permute_mem(states[k], index)
        return states

    def reset_scorer_mem(self, x, enc_lens):
        states = dict()
        for k, impl in self.scorers.items():
            states[k] = impl.reset_mem(x, enc_lens)
        return states
