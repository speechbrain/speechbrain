"""Configuration and utility classes for classes for Dynamic Chunk Training, as
often used for the training of streaming-capable models in speech recognition.

Authors
* Sylvain de Langen 2023
"""

from speechbrain.core import Stage
from dataclasses import dataclass
from typing import Optional

import torch

# NOTE: this configuration object is intended to be relatively specific to DCT;
# if you want to implement a different similar type of chunking different from
# DCT you should consider using a different object.
@dataclass
class DCTConfig:
    """Dynamic Chunk Training configuration object for use with transformers,
    often in ASR for streaming.

    This object may be used both to configure masking at training time and for
    run-time configuration of DCT-ready models."""

    chunk_size: int
    """Size in frames of a single chunk, always `>0`.
    If chunkwise streaming should be disabled at some point, pass an optional
    streaming config parameter."""

    left_context_size: Optional[int]
    """Number of *chunks* (not frames) visible to the left, always `>=0`.
    If zero, then chunks can never attend to any past chunk.
    If `None`, the left context is infinite (but use
    `.is_fininite_left_context` for such a check)."""

    def is_infinite_left_context(self) -> bool:
        """Returns true if the left context is infinite (i.e. any chunk can
        attend to any past frame)."""
        return self.left_context_size is not None


@dataclass
class DCTConfigRandomSampler:
    """Helper class to generate a DCTConfig at runtime depending on the current
    stage."""

    dct_prob: float
    """When sampling (during `Stage.TRAIN`), the probability that a finite chunk
    size will be used.
    In the other case, any chunk can attend to the full past and future
    context."""

    chunk_size_min: int
    """When sampling a random chunk size, the minimum chunk size that can be
    picked."""

    chunk_size_max: int
    """When sampling a random chunk size, the maximum chunk size that can be
    picked."""

    limited_left_context_prob: float
    """When sampling a random chunk size, the probability that the left context
    will be limited.
    In the other case, any chunk can attend to the full past context."""

    left_context_chunks_min: int
    """When sampling a random left context size, the minimum number of left
    context chunks that can be picked."""

    left_context_chunks_max: int
    """When sampling a random left context size, the maximum number of left
    context chunks that can be picked."""

    test_config: Optional[DCTConfig] = None
    """The configuration that should be used for `Stage.TEST`.
    When `None`, evaluation is done with full context (i.e. non-streaming)."""

    valid_config: Optional[DCTConfig] = None
    """The configuration that should be used for `Stage.VALID`.
    When `None`, evaluation is done with full context (i.e. non-streaming)."""

    def _sample_bool(prob: float) -> bool:
        """Samples a random boolean with a probability, in a way that depends on
        PyTorch's RNG seed.

        Arguments
        ---------
        prob : float
            Probability (0..1) to return True (False otherwise)."""
        return torch.rand((1,)).item() < prob

    def __call__(self, stage: Stage) -> DCTConfig:
        """Samples a random (or not) DCT configuration depending on the current
        stage.

        Arguments
        ---------
        stage : speechbrain.core.Stage
            Current stage of training or evaluation.
            In training mode, a random DCTConfig will be sampled according to
            the specified probabilities and ranges.
            In evaluation, the relevant DCTConfig attribute will be picked.
        """
        if stage == Stage.TRAIN:
            # When training for streaming, for each batch, we have a
            # `dynamic_chunk_prob` probability of sampling a chunk size
            # between `dynamic_chunk_min` and `_max`, otherwise output
            # frames can see anywhere in the future.
            if self._sample_bool(self.dct_prob):
                chunk_size = torch.randint(
                    self.chunk_size_min, self.chunk_size_max + 1, (1,),
                ).item()

                if self._sample_bool(self.limited_left_context_prob):
                    left_context_chunks = torch.randint(
                        self.left_context_chunks_min,
                        self.left_context_chunks_max + 1,
                        (1,),
                    ).item()
                else:
                    left_context_chunks = None

                return DCTConfig(chunk_size, left_context_chunks)
            return None
        elif stage == Stage.TEST:
            return self.test_config
        elif stage == Stage.VALID:
            return self.valid_config
        else:
            raise AttributeError(f"Unsupported stage found {stage}")
