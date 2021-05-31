import kenlm
import torch
import numpy as np


class NgramScorer:
    def __init__(
        self, lm_path, bos_index, eos_index, batch_size, beam_size, device
    ):
        self.lm = kenlm.Model(lm_path)
        state = kenlm.State()
        self.lm.NullContextWrite(state)
        self.id2char = np.load("token_list.npy")  # token_list
        # print(self.id2char)
        self.id2char[bos_index] = "<s>"
        self.id2char[eos_index] = "</s>"
        self.vocab_size = len(self.id2char)
        self.candidates = np.tile(
            np.arange(self.vocab_size).reshape(1, -1),
            (batch_size * beam_size, 1),
        )
        # The first index of each sentence.
        self.beam_offset = torch.arange(batch_size, device=device) * beam_size

    def forward_step(self, g, states, candidates=None):
        """
        Returns:
        new_memory: [B * Num_hyps, Vocab_size]

        """

        n_bh = g.size(0)
        state, scoring_table = states

        if state is None:
            state = kenlm.State()
            state = np.array([state] * n_bh)

        if candidates is None:
            candidates = self.candidates

        if scoring_table is None:
            scoring_table = np.ones(n_bh)

        scores = torch.full(
            (n_bh, self.vocab_size),
            -(2 ** 32),
            dtype=torch.float,
            device=g.device,
        )
        new_memory = np.zeros((n_bh, self.vocab_size), dtype=object)
        new_scoring_table = np.ones((n_bh, self.vocab_size)) * -1
        for i in range(n_bh):
            parent_state = state[i]
            if scoring_table[i] == -1:
                continue
            for token_id in candidates[i]:
                char = str(self.id2char[token_id.item()])  # TODO
                out_state = kenlm.State()
                score = self.lm.BaseScore(parent_state, char, out_state)
                scores[i, token_id] = score
                new_memory[i, token_id] = out_state
                new_scoring_table[i, token_id] = 1
        return scores, (new_memory, new_scoring_table)

    def permute_mem(self, memory, index):
        """
        Returns:
        new_memory: [B, Num_hyps]

        """
        state, scoring_table = memory
        best_index = (
            index
            + (self.beam_offset.unsqueeze(1).expand_as(index) * self.vocab_size)
        ).view(-1)
        best_index = best_index.cpu().numpy()
        state = state.reshape(-1)
        state = state[best_index]
        scoring_table = scoring_table.reshape(-1)
        scoring_table = scoring_table[best_index]
        return state, scoring_table

    def reset_mem(self):
        state = kenlm.State()
        self.lm.NullContextWrite(state)
        return None, None
