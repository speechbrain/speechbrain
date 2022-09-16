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

    def forward(self, feat_db):
        """Normalizes audio features in decibels (usually spectrograms)

        Arguments
        ---------
        feat_db: torch.Tensor
            input features
        Returns
        -------
        normalized_features: torch.Tensor
            the normalized features
        """

        return torch.clip((((feat_db - self.min_level_db) / -self.min_level_db)*2.)-1., -1, 1)


class Denormalize(nn.Module):
    def __init__(self, min_level_db):
        super().__init__()
        self.min_level_db = min_level_db

    def forward(self, feat_db):
        """Denormalizes audio features in decibels

        Arguments
        ---------
        feat_db: torch.Tensor
            normalzied features

        Returns
        -------
        denormalized_features: torch.Tensor
            denormalized features

        """
        return (((torch.clip(feat_db, -1, 1)+1.)/2.) * -self.min_level_db) + self.min_level_db
