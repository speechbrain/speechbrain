"""Custom transformations for the diffusion model

Authors
 * Artem Ploujnikov 2022
"""

import torch
from torch import nn


class Normalize(nn.Module):
    def __init__(self, min_level_db):
        super().__init__()
        self.min_level_db = min_level_db

    def forward(self, x):
        """Normalizes audio features in decibels (usually spectrograms)

        Arguments
        ---------
        x: torch.Tensor
            input features
        Returns
        -------
        normalized_features: torch.Tensor
            the normalized features
        """
        
        x = ((x - self.min_level_db) / -self.min_level_db)
        x *= 2.
        x = x - 1.
        x = torch.clip(x, -1, 1)
        return x


class Denormalize(nn.Module):
    def __init__(self, min_level_db):
        super().__init__()
        self.min_level_db = min_level_db

    def forward(self, x):
        """Denormalizes audio features in decibels

        Arguments
        ---------
        x: torch.Tensor
            normalzied features

        Returns
        -------
        denormalized_features: torch.Tensor
            denormalized features

        """
        x = torch.clip(x, -1, 1)
        x = (x + 1.)/2.
        x *= -self.min_level_db
        x += self.min_level_db
        return x

class DynamicRangeCompression(nn.Module):
        """Dynamic range compression for audio signals
        Arguments
        ---------
        multiplier: float
            the multiplier constant
        clip_val: float
            the minimum accepted value (values below this
            minimum will be clipped)
        """
        def __init__(self, multiplier=1, clip_val=1e-5):
            super().__init__()
            self.multiplier = multiplier
            self.clip_val = clip_val

        def forward(self, x):
            return torch.log(torch.clamp(x, min=self.clip_val) * self.multiplier)
