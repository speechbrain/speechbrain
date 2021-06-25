import torch
import numpy as np
import kenlm
import sentencepiece as spm
import speechbrain as sb
from speechbrain.decoders.ctc import CTCPrefixScore


class BaseScorer:
    """A scorer abstraction to be inherited by other
    scoring approaches for beam search.

    See:
        - speechbrain.decoders.scorer.CTCPrefixScorer
        - speechbrain.decoders.scorer.RNNLMScorer
        - speechbrain.decoders.scorer.TransformerLMScorer
        - speechbrain.decoders.scorer.NGramLMScoer
        - speechbrain.decoders.scorer.CoverageScorer
    """

    def score(self, inp_tokens, memory, candidates, attn):
        """This method scores tokens in vocabulary.

        Arguments
        ---------
        inp_tokens : torch.Tensor
            The input tensor of the current timestep.
        memory : No limit
            The memory variables input for this timestep.
        """
        raise NotImplementedError

    def permute_mem(self, memory, index):
        """This method permutes the scorer memory
        to synchronize the memory index with the current output.
        """
        return None, None

    def reset_mem(self, x, enc_lens):
        """This method should implement the resetting of
        memory variables for the scorer.

        Arguments
        ---------
        x : torch.Tensor
            The precomputed encoder states to be used when decoding.
            (ex. the encoded speech representation to be attended).
        wav_len : torch.Tensor
            The speechbrain-style relative length.
        """
        return None


class CTCScorer(BaseScorer):
    def __init__(
        self, ctc_fc, blank_index, eos_index, ctc_window_size=0,
    ):
        self.ctc_fc = ctc_fc
        self.softmax = sb.nnet.activations.Softmax(apply_log=True)
        self.blank_index = blank_index
        self.eos_index = eos_index
        self.ctc_window_size = ctc_window_size

    def score(self, inp_tokens, memory, candidates, attn):
        scores, memory = self.ctc_score.forward_step(
            inp_tokens, memory, candidates, attn
        )
        return scores, memory

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


class RNNLMScorer(BaseScorer):
    def __init__(self, language_model, temperature=1.0):
        self.lm = language_model
        self.lm.eval()
        self.temperature = temperature
        self.softmax = sb.nnet.activations.Softmax(apply_log=True)

    def score(self, inp_tokens, memory, candidates, attn):
        with torch.no_grad():
            logits, hs = self.lm(inp_tokens, hx=memory)
            log_probs = self.softmax(logits / self.temperature)

        return log_probs, hs

    def permute_mem(self, memory, index):
        """This is to permute lm memory to synchronize with current index
        during beam search. The order of beams will be shuffled by scores
        every timestep to allow batched beam search.
        Further details please refer to speechbrain/decoder/seq2seq.py.
        """
        if isinstance(memory, tuple):
            memory_0 = torch.index_select(memory[0], dim=1, index=index)
            memory_1 = torch.index_select(memory[1], dim=1, index=index)
            memory = (memory_0, memory_1)
        else:
            memory = torch.index_select(memory, dim=1, index=index)
        return memory

    def reset_mem(self, x, enc_lens):
        return None


class TransformerLMScorer(BaseScorer):
    def __init__(self, language_model, temperature=1.0):
        self.lm = language_model
        self.lm.eval()
        self.temperature = temperature
        self.softmax = sb.nnet.activations.Softmax(apply_log=True)

    def score(self, inp_tokens, memory, candidates, attn):
        if memory is None:
            memory = torch.empty(
                inp_tokens.size(0), 0, device=inp_tokens.device
            )
        # Append the predicted token of the previous step to existing memory.
        memory = torch.cat([memory, inp_tokens.unsqueeze(1)], dim=-1)
        if not next(self.lm.parameters()).is_cuda:
            self.lm.to(inp_tokens.device)
        logits = self.lm(memory)
        log_probs = self.softmax(logits / self.temperature)
        return log_probs[:, -1, :], memory

    def permute_mem(self, memory, index):
        memory = torch.index_select(memory, dim=0, index=index)
        return memory

    def reset_mem(self, x, enc_lens):
        return None


class NGramLMScorer(BaseScorer):
    def __init__(self, lm_path, tokenizer_path, vocab_size):
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

    def score(self, inp_tokens, memory, candidates, attn):
        """
        Returns:
        new_memory: [B * Num_hyps, Vocab_size]

        """
        n_bh = inp_tokens.size(0)
        scale = 1.0 / np.log10(np.e)

        if memory is None:
            state = kenlm.State()
            state = np.array([state] * n_bh)
            scoring_table = np.ones(n_bh)
        else:
            state, scoring_table = memory

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
        scores = torch.from_numpy(scores).float().to(inp_tokens.device)
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


class CoverageScorer(BaseScorer):
    def __init__(self, vocab_size, threshold=0.5):
        self.vocab_size = vocab_size
        self.threshold = threshold
        self.time_step = 0

    def score(self, inp_tokens, coverage, candidates, attn):
        n_bh = attn.size(0)
        self.time_step += 1

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
        return -1 * penalty / self.time_step, coverage

    def permute_mem(self, coverage, index):
        # Update coverage
        coverage = torch.index_select(coverage, dim=0, index=index)
        return coverage

    def reset_mem(self, x, enc_lens):
        self.time_step = 0
        return None


class ScorerBuilder:
    def __init__(
        self,
        eos_index,
        blank_index,
        vocab_size,
        ctc_weight=0.0,
        ngramlm_weight=0.0,
        coverage_weight=0.0,
        rnnlm_weight=0.0,
        transformerlm_weight=0.0,
        ctc_score_mode="partial",
        ctc_linear=None,
        rnnlm=None,
        transformerlm=None,
        lm_path=None,
        tokenizer=None,
    ):
        """
        weights: Dict
        score_mode: Dict
        """
        self.weights = dict(
            ctc=ctc_weight,
            rnnlm=rnnlm_weight,
            ngramlm=ngramlm_weight,
            transformerlm=transformerlm_weight,
            coverage=coverage_weight,
        )
        self.full_scorers = {}
        self.partial_scorers = {}

        if ctc_score_mode == "full" or ctc_weight == 1.0:
            ctc_weight = 1.0
            coverage_weight = 0.0

        if ctc_weight > 0.0:
            if ctc_score_mode == "full":
                self.full_scorers["ctc"] = CTCScorer(
                    ctc_linear, blank_index, eos_index
                )
            elif ctc_score_mode == "partial":
                self.partial_scorers["ctc"] = CTCScorer(
                    ctc_linear, blank_index, eos_index
                )
            else:
                raise NotImplementedError("Must be full or partial mode.")

        if ngramlm_weight > 0.0:
            self.partial_scorers["ngram"] = NGramLMScorer(
                lm_path, tokenizer, vocab_size
            )

        if coverage_weight > 0.0:
            self.full_scorers["coverage"] = CoverageScorer(vocab_size)
            print(self.full_scorers["coverage"].__class__.__name__)

        if rnnlm_weight > 0.0:
            self.full_scorers["rnnlm"] = RNNLMScorer(rnnlm)

        if transformerlm_weight > 0.0:
            self.full_scorers["transformerlm"] = TransformerLMScorer(
                transformerlm
            )

    def score(self, inp_tokens, memory, attn, log_probs, beam_size):
        new_memory = dict()
        # score full candidates
        for k, impl in self.full_scorers.items():
            score, new_memory[k] = impl.score(inp_tokens, memory[k], None, attn)
            log_probs += score * self.weights[k]

        # select candidates for partial scorers
        _, candidates = log_probs.topk(int(beam_size * 1.5), dim=-1)

        # score patial candidates
        for k, impl in self.partial_scorers.items():
            score, new_memory[k] = impl.score(
                inp_tokens, memory[k], candidates, attn
            )
            log_probs += score * self.weights[k]

        return log_probs, new_memory

    def permute_scorer_mem(self, memory, index, candidates):
        for k, impl in self.full_scorers.items():
            memory[k] = impl.permute_mem(memory[k], index)
        for k, impl in self.partial_scorers.items():
            memory[k] = impl.permute_mem(memory[k], candidates)
        return memory

    def reset_scorer_mem(self, x, enc_lens):
        memory = dict()
        for k, impl in {**self.full_scorers, **self.partial_scorers}.items():
            memory[k] = impl.reset_mem(x, enc_lens)
        return memory


class ScorerBuilder2:
    def __init__(
        self,
        ctc_weight=0.0,
        ngramlm_weight=0.0,
        coverage_weight=0.0,
        rnnlm_weight=0.0,
        transformerlm_weight=0.0,
        full_scorers=None,
        partial_scorers=None,
    ):
        """
        weights: Dict
        score_mode: Dict
        """
        self.weights = dict(
            ctc=ctc_weight,
            ngram=ngramlm_weight,
            coverage=coverage_weight,
            rnnlm=rnnlm_weight,
            transformerlm=transformerlm_weight,
        )
        full_scorer_names = [
            impl.__class__.__name__.lower().split("scorer")[0]
            for impl in full_scorers
        ]
        partial_scorer_names = [
            impl.__class__.__name__.lower().split("scorer")[0]
            for impl in partial_scorers
        ]

        self.full_scorers = dict(zip(full_scorer_names, full_scorers))
        self.partial_scorers = dict(zip(partial_scorer_names, partial_scorers))

        if "ctc" in full_scorer_names:
            self.weights["ctc"] = 1.0
            self.weights["coverage"] = 0.0

    def score(self, inp_tokens, memory, attn, log_probs, beam_size):
        new_memory = dict()
        # score full candidates
        for k, impl in self.full_scorers.items():
            score, new_memory[k] = impl.score(inp_tokens, memory[k], None, attn)
            log_probs += score * self.weights[k]

        # select candidates for partial scorers
        _, candidates = log_probs.topk(int(beam_size * 1.5), dim=-1)

        # score patial candidates
        for k, impl in self.partial_scorers.items():
            score, new_memory[k] = impl.score(
                inp_tokens, memory[k], candidates, attn
            )
            log_probs += score * self.weights[k]

        return log_probs, new_memory

    def permute_scorer_mem(self, memory, index, candidates):
        for k, impl in self.full_scorers.items():
            memory[k] = impl.permute_mem(memory[k], index)
        for k, impl in self.partial_scorers.items():
            memory[k] = impl.permute_mem(memory[k], candidates)
        return memory

    def reset_scorer_mem(self, x, enc_lens):
        memory = dict()
        for k, impl in {**self.full_scorers, **self.partial_scorers}.items():
            memory[k] = impl.reset_mem(x, enc_lens)
        return memory
