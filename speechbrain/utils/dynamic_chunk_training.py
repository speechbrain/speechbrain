"""Configuration and utility classes for classes for Dynamic Chunk Training, as
often used for the training of streaming-capable models in speech recognition.

The definition of Dynamic Chunk Training is based on that of the following
paper, though a lot of the literature refers to the same definition:
https://arxiv.org/abs/2012.05481

Authors
* Sylvain de Langen 2023
"""

from dataclasses import dataclass
from typing import Optional

import torch

import speechbrain as sb


# NOTE: this configuration object is intended to be relatively specific to
# Dynamic Chunk Training; if you want to implement a different similar type of
# chunking different from that, you should consider using a different object.
@dataclass
class DynChunkTrainConfig:
    """Dynamic Chunk Training configuration object for use with transformers,
    often in ASR for streaming.

    This object may be used both to configure masking at training time and for
    run-time configuration of DynChunkTrain-ready models.
    """

    chunk_size: int
    """Size in frames of a single chunk, always `>0`.
    If chunkwise streaming should be disabled at some point, pass an optional
    streaming config parameter."""

    left_context_size: Optional[int] = None
    """Number of *chunks* (not frames) visible to the left, always `>=0`.
    If zero, then chunks can never attend to any past chunk.
    If `None`, the left context is infinite (but use
    `.is_infinite_left_context` for such a check)."""

    def is_infinite_left_context(self) -> bool:
        """Returns true if the left context is infinite (i.e. any chunk can
        attend to any past frame).
        """
        return self.left_context_size is None

    def left_context_size_frames(self) -> Optional[int]:
        """Returns the number of left context *frames* (not chunks).
        If ``None``, the left context is infinite.
        See also the ``left_context_size`` field.
        """
        if self.left_context_size is None:
            return None

        return self.chunk_size * self.left_context_size


@dataclass
class DynChunkTrainConfigRandomSampler:
    """Helper class to generate a DynChunkTrainConfig at runtime depending on the current
    stage.

    Example
    -------
    >>> from speechbrain.core import Stage
    >>> from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
    >>> from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfigRandomSampler
    >>> # for the purpose of this example, we test a scenario with a 100%
    >>> # chance of the (24, None) scenario to occur
    >>> sampler = DynChunkTrainConfigRandomSampler(
    ...     chunkwise_prob=1.0,
    ...     chunk_size_min=24,
    ...     chunk_size_max=24,
    ...     limited_left_context_prob=0.0,
    ...     left_context_chunks_min=16,
    ...     left_context_chunks_max=16,
    ...     test_config=DynChunkTrainConfig(32, 16),
    ...     valid_config=None
    ... )
    >>> one_train_config = sampler(Stage.TRAIN)
    >>> one_train_config
    DynChunkTrainConfig(chunk_size=24, left_context_size=None)
    >>> one_train_config.is_infinite_left_context()
    True
    >>> sampler(Stage.TEST)
    DynChunkTrainConfig(chunk_size=32, left_context_size=16)
    """

    chunkwise_prob: float
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

    test_config: Optional[DynChunkTrainConfig] = None
    """The configuration that should be used for `Stage.TEST`.
    When `None`, evaluation is done with full context (i.e. non-streaming)."""

    valid_config: Optional[DynChunkTrainConfig] = None
    """The configuration that should be used for `Stage.VALID`.
    When `None`, evaluation is done with full context (i.e. non-streaming)."""

    def _sample_bool(self, prob):
        """Samples a random boolean with a probability, in a way that depends on
        PyTorch's RNG seed.

        Arguments
        ---------
        prob : float
            Probability (0..1) to return True (False otherwise).

        Returns
        -------
        The sampled boolean
        """
        return torch.rand((1,)).item() < prob

    def __call__(self, stage):
        """In training stage, samples a random DynChunkTrain configuration.
        During validation or testing, returns the relevant configuration.

        Arguments
        ---------
        stage : speechbrain.core.Stage
            Current stage of training or evaluation.
            In training mode, a random DynChunkTrainConfig will be sampled
            according to the specified probabilities and ranges.
            During evaluation, the relevant DynChunkTrainConfig attribute will
            be picked.

        Returns
        -------
        The appropriate configuration
        """
        if stage == sb.core.Stage.TRAIN:
            # When training for streaming, for each batch, we have a
            # `dynamic_chunk_prob` probability of sampling a chunk size
            # between `dynamic_chunk_min` and `_max`, otherwise output
            # frames can see anywhere in the future.
            if self._sample_bool(self.chunkwise_prob):
                chunk_size = torch.randint(
                    self.chunk_size_min,
                    self.chunk_size_max + 1,
                    (1,),
                ).item()

                if self._sample_bool(self.limited_left_context_prob):
                    left_context_chunks = torch.randint(
                        self.left_context_chunks_min,
                        self.left_context_chunks_max + 1,
                        (1,),
                    ).item()
                else:
                    left_context_chunks = None

                return DynChunkTrainConfig(chunk_size, left_context_chunks)
            return None
        elif stage == sb.core.Stage.TEST:
            return self.test_config
        elif stage == sb.core.Stage.VALID:
            return self.valid_config
        else:
            raise AttributeError(f"Unsupported stage found {stage}")
