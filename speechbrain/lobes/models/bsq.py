"""Binary spherical quantizer.

Authors
 * Luca Della Libera 2025
"""

# Adapted from:
# https://github.com/lucidrains/vector-quantize-pytorch/blob/8f5b428949feb4bca52264f253377188f2c21a23/vector_quantize_pytorch/lookup_free_quantization.py

from typing import Tuple

import torch
from torch import nn

__all__ = ["BinarySphericalQuantizer"]


class BinarySphericalQuantizer(nn.Module):
    """Binary spherical quantizer.

    This module implements a binary quantizer over the unit hypersphere.
    Given a continuous input vector x ∈ R^{D}, it:
      1. Projects x onto the unit sphere.
      2. Quantizes each dimension to {-1/sqrt(D), +1/sqrt(D)} based on its sign.
      3. Interprets the resulting sign pattern as a binary code index.
      4. Computes an auxiliary entropy/diversity loss to encourage
         confident assignments and uniform codebook usage.

    Parameters
    ----------
    code_dim : int
        Dimensionality of the code / number of bits per code vector.
        The codebook size is 2 ** code_dim.
    entropy_loss_weight : float, optional
        Weight for the entropy-based auxiliary loss term.
    diversity_gamma : float, optional
        Coefficient for the codebook entropy term in the auxiliary loss.
        Larger values encourage more uniform usage of all codes.

    Example
    -------
    >>> import torch
    >>> code_dim = 13
    >>> x = torch.randn(2, 50, code_dim)
    >>> quantizer = BinarySphericalQuantizer(code_dim)
    >>> quant, indices, aux_loss = quantizer(x)

    """

    def __init__(
        self,
        code_dim: "int",
        entropy_loss_weight: "float" = 0.1,
        diversity_gamma: "float" = 1.0,
    ) -> "None":
        super().__init__()
        self.code_dim = code_dim
        self.entropy_loss_weight = entropy_loss_weight
        self.diversity_gamma = diversity_gamma

        codebook_size = 2**code_dim

        # Bit mask used to convert a {0, 1} bit pattern into an integer index
        self.register_buffer("mask", 2 ** torch.arange(code_dim - 1, -1, -1))
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

        # Precompute all possible codes on the binary sphere
        all_codes = torch.arange(codebook_size)
        bits = ((all_codes[..., None].int() & self.mask) != 0).float()
        codebook = self.bits_to_codes(bits)
        self.register_buffer("codebook", codebook.float(), persistent=False)

    def bits_to_codes(self, bits: "torch.Tensor") -> "torch.Tensor":
        """Convert {0, 1} bits to {-1, +1} codes.

        Parameters
        ----------
        bits : torch.Tensor
            Tensor of bits in {0, 1} with shape [..., code_dim].

        Returns
        -------
        torch.Tensor
            Tensor of codes in {-1, +1} with the same shape as `bits`.

        """
        return bits * 2 - 1

    def forward(
        self,
        x: "torch.Tensor",
        inv_temperature: "float" = 100.0,
    ) -> "Tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        """Quantize continuous vectors on the binary sphere.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [..., code_dim]. The last dimension
            must match `self.code_dim`. It is L2-normalized internally.
        inv_temperature : float, optional
            Inverse temperature for the softmax over codebook distances
            used to compute the entropy-based auxiliary loss.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple (quantized, indices, aux_loss) where:
            - quantized: torch.Tensor
                Quantized version of the input with the same shape as `x`,
                lying on the unit sphere with values approximately in {-1, +1}.
            - indices: torch.Tensor
                Integer code indices of shape [...], obtained by interpreting
                the sign pattern of each vector as a binary code.
            - aux_loss: torch.Tensor
                Scalar auxiliary loss combining per-sample entropy and
                codebook-diversity regularization, scaled by
                `entropy_loss_weight`.

        """
        # Normalize input on the last dimension
        x = nn.functional.normalize(x, dim=-1)
        original_input = x

        # Hard sign quantization to {-1, +1}
        codebook_value = torch.ones_like(x)
        quantized = torch.where(x > 0, codebook_value, -codebook_value)

        # Compute integer indices from sign pattern
        indices = ((quantized > 0).int() * self.mask.int()).sum(dim=-1)

        # Normalize quantized vectors on the last dimension
        quantized = nn.functional.normalize(quantized, dim=-1)

        # Straight-through estimator: gradient flows through `x`,
        # but forward value is `quantized`
        x = x + (quantized - x).detach()

        # Normalized codebook on the unit sphere
        codebook = self.codebook.float()
        codebook = nn.functional.normalize(codebook, dim=-1)

        # ------------------------
        # Entropy-based aux loss
        # ------------------------

        # Same as Euclidean distance up to a constant
        distance = -2 * torch.einsum(
            "... i d, j d -> ... i j", original_input, codebook
        )

        # Soft assignment probabilities over codebook entries
        prob = (-distance * inv_temperature).softmax(dim=-1)

        # Flatten over all but the codebook dimension
        prob = prob.flatten(end_dim=1)
        per_sample_probs = prob

        # Per-sample entropy (encourages confident assignments)
        per_sample_entropy = (
            (-per_sample_probs * per_sample_probs.clamp(min=1e-5).log())
            .sum(dim=-1)
            .mean()
        )

        # Average distribution over the codebook (encourages diversity)
        avg_prob = per_sample_probs.mean(dim=0)
        codebook_entropy = (-avg_prob * avg_prob.clamp(min=1e-5).log()).sum(
            dim=-1
        )

        # 1. Per-sample entropy is pushed low -> confident predictions
        # 2. Codebook entropy is pushed high -> uniform code usage
        entropy_aux_loss = (
            per_sample_entropy - self.diversity_gamma * codebook_entropy
        )

        # Final auxiliary loss
        aux_loss = entropy_aux_loss * self.entropy_loss_weight

        return x, indices, aux_loss
