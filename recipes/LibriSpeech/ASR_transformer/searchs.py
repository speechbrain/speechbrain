#!/usr/bin/env python3
import torch
from speechbrain.decoders.seq2seq import S2SBeamSearcher, S2SGreedySearcher


def update_mem(inp_tokens, memory):
    if memory is None:
        return inp_tokens.unsqueeze(1)
    return torch.cat([memory, inp_tokens.unsqueeze(1)], dim=-1)


def model_decode(model, softmax, fc, inp_tokens, memory, enc_states):
    memory = update_mem(inp_tokens, memory)
    pred = model.decode(memory, enc_states)
    prob_dist = softmax(fc(pred))
    return prob_dist, memory


# Define a beam search according to this recipe
class BeamSearch(S2SBeamSearcher):
    def __init__(
        self,
        modules,
        bos_index,
        eos_index,
        min_decode_ratio,
        max_decode_ratio,
        beam_size,
        length_penalty,
        eos_threshold,
        using_max_attn_shift=False,
        max_attn_shift=30,
        minus_inf=-1e20,
    ):
        super().__init__(
            modules,
            bos_index,
            eos_index,
            min_decode_ratio,
            max_decode_ratio,
            beam_size,
            length_penalty,
            eos_threshold,
            using_max_attn_shift,
            max_attn_shift,
        )

        self.model = modules[0]
        self.fc = modules[1]
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def reset_mem(self, batch_size, device):
        return None

    def permute_mem(self, memory, index):
        memory = torch.index_select(memory, dim=0, index=index)
        return memory

    def forward_step(self, inp_tokens, memory, enc_states, enc_lens):
        prob_dist, memory = model_decode(
            self.model, self.softmax, self.fc, inp_tokens, memory, enc_states
        )
        return prob_dist[:, -1, :], memory, None


# define a greedy search w.r.t this task
class GreedySearch(S2SGreedySearcher):
    def __init__(
        self, modules, bos_index, eos_index, min_decode_ratio, max_decode_ratio,
    ):
        super().__init__(
            modules, bos_index, eos_index, min_decode_ratio, max_decode_ratio,
        )
        self.model = modules[0]
        self.fc = modules[1]
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def reset_mem(self, batch_size, device):
        return None

    def forward_step(self, inp_tokens, memory, enc_states, enc_lens):
        prob_dist, memory = model_decode(
            self.model, self.softmax, self.fc, inp_tokens, memory, enc_states
        )
        return prob_dist[:, -1, :], memory, None
