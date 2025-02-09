import torch
import torch.nn as nn

from speechbrain.lobes.models.huggingface_transformers.huggingface import (
    HFTransformersInterface,
)
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)

from dataclasses import dataclass

from transformers import AutoModelForCausalLM

# TODO: move this file OUTSIDE. instatniate PT model directly in yaml


@dataclass
class DiscreteSpeechLMConfig:
    source: str
    cache_dir: str
    block_size: int
    n_codebooks: int
    vocabsize: int
    tie_embds: bool


class InterleavedCodebookPattern:
    def __init__(self, audio_pad_token):
        self.audio_pad_token = audio_pad_token

    def apply_delay_pattern(self, tokens):
        # TODO: use moshi interleaving pattern instead of this one. AND UNDERSTAND WHAT IS THE PATTERN
        # todo: undertsansd this
        B, K, T = tokens.shape
        result = torch.full(
            (B, K, T),
            self.audio_pad_token,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        for i in range(K):
            result[:, i, i:T] = tokens[:, i, : T - i]
        return result

    def undelay_logits(self, logits):
        B, K, T, D = logits.shape
        unpadded_length = T - K
        # Create an empty tensor to store the reconstructed sequence
        undelayed_logits = torch.full(
            (B, K, T, D), float("nan"), dtype=logits.dtype, device=logits.device
        )
        undelayed_logits_mask = torch.ones(
            (B, K, T), dtype=bool, device=logits.device
        )
        undelayed_logits_mask[..., -K:] = False
        # Reconstruct the original sequence by removing the delays
        for i in range(K):
            undelayed_logits[:, i, :-K] = logits[
                :, i, i : i + unpadded_length, :
            ]
        return undelayed_logits, undelayed_logits_mask


class DiscreteSpeechLM(nn.Module):
    def __init__(
        self,
        config,
        model_backbone,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model_backbone
        self.source = config.source
        self.cache_dir = config.cache_dir

        self.block_size = config.block_size
        self.n_codebooks = config.n_codebooks
        self.vocabsize = config.vocabsize
        config = self.model.config
        self.dim = config.hidden_size
        self.n_layers = config.num_hidden_layers
        self.n_heads = config.num_attention_heads

        # remove embeddings and output projections from the backbone
        self.model.set_input_embeddings(None)
        self.model.set_output_embeddings(None)

        self.audio_in_embds = nn.ModuleList(
            [
                nn.Embedding(self.vocabsize, self.dim)
                for _ in range(self.n_codebooks)
            ]
        )
        self.audio_out = nn.ModuleList(
            [
                nn.Linear(self.dim, self.vocabsize, bias=False)
                for _ in range(self.n_codebooks)
            ]
        )

        if self.config.tie_embds:
            # share the unembedding parameters with the embedding parameters
            for k in range(self.n_codebooks):
                self.audio_in_embds[k].weight = self.audio_out[
                    k
                ].weight  # https://paperswithcode.com/method/weight-tying

        for module in [self.audio_in_embds, self.audio_out]:
            module.apply(self._init_weights)

    def _init_weights(self, module):
        # regular inits
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_audio_tokens: torch.Tensor,
    ):
        batch_size, n_codebooks, seq_length = input_audio_tokens.size()
        assert (
            seq_length <= self.block_size
        ), f"Sequence beyond maximum length of {self.block_size}"
        assert (
            n_codebooks == self.n_codebooks
        ), "Sequence shape must match the specified number of codebooks"

        # compute the frame audio embeddings as the sum of codebook embeddings
        h = sum(
            [
                self.audio_in_embds[k](input_audio_tokens[:, k])
                for k in range(n_codebooks)
            ]
        )

        # obtain contextual embeddings
        # todo: allows to retrieve all the hidden states of the model.
        h = self.model.model(inputs_embeds=h, use_cache=False)[
            "last_hidden_state"
        ]

        logits_audio = torch.stack(
            [self.audio_out[k](h) for k in range(self.n_codebooks)], dim=1
        )

        return logits_audio
