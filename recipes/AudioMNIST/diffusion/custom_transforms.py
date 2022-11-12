"""Custom transformations for the diffusion model

Authors
 * Artem Ploujnikov 2022
"""

import torch
from torch import nn
from speechbrain.dataio.dataio import length_to_mask


class MinLevelNorm(nn.Module):
    """A normalization for the decibel scale

    The scheme is as follows

    x_norm = (x - min_level_db)/-min_level_db * 2 - 1

    Arguments
    ---------
    min_level_db: float
        the minimum level
    """

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

        x = (x - self.min_level_db) / -self.min_level_db
        x *= 2.0
        x = x - 1.0
        x = torch.clip(x, -1, 1)
        return x

    def denormalize(self, x):
        """Reverses the min level normalization process

        Arguments
        ---------
        x: torch.Tensor
            the normalized tensor

        Returns
        -------
        result: torch.Tensor
            the denormalized tensor
        """
        x = torch.clip(x, -1, 1)
        x = (x + 1.0) / 2.0
        x *= -self.min_level_db
        x += self.min_level_db
        return x


class GlobalNorm(nn.Module):
    """
    A global normalization module - computes a single mean and standard deviation
    for the entire batch across unmasked positions and uses it to normalize the
    inputs to the desired mean and standard deviation

    Arguments
    ---------
    norm_mean: float
        the desired normalized mean
    norm_std: float
        the desired normalized standard deviation
    update_steps: float
        the number of steps over which statistics will be collected
    length_dim: int
        the dimension used to represent the length
    mask_value: float
        the value with which to fill masked positions
        without a mask_value, the masked positions would be normalized,
        which might not be desired
    """

    def __init__(
        self,
        norm_mean=0.0,
        norm_std=1.0,
        update_steps=None,
        length_dim=2,
        mask_value=0.0,
    ):
        super().__init__()

        running_mean = torch.tensor(0.0)
        running_std = torch.tensor(0.0)
        weight = torch.tensor(0.0)
        self.register_buffer("running_mean", running_mean)
        self.register_buffer("running_std", running_std)
        self.register_buffer("weight", weight)
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.mask_value = mask_value
        self.step_count = 0
        self.update_steps = update_steps
        self.length_dim = length_dim
        self.frozen = False

    def forward(self, x, lengths=None, mask_value=None, skip_update=False):
        """Normalizes the tensor provided

        Arguments
        ---------
        x: torch.Tensor
            the tensor to normalize
        lengths: torch.Tensor
            a tensor of relative lengths (padding will not
            count towards normalization)
        mask_value: float
            the value to use for masked positions
        skip_update: false
            whether to skip updates to the norm

        Returns
        -------
        result: torch.Tensor
            the normalized tensor
        """
        if lengths is None:
            lengths = torch.ones(len(x))
        if mask_value is None:
            mask_value = self.mask_value

        mask = self.get_mask(x, lengths)

        if (
            not skip_update
            and not self.frozen
            and (
                self.update_steps is None or self.step_count < self.update_steps
            )
        ):
            x_masked = x.masked_select(mask)
            mean = x_masked.mean()
            std = x_masked.std()
            weight = lengths.sum()

            # TODO: Numerical stability
            new_weight = self.weight + weight
            self.running_mean.data = (
                self.weight * self.running_mean + weight * mean
            ) / new_weight
            self.running_std.data = (
                self.weight * self.running_std + weight * std
            ) / new_weight
            self.weight.data = new_weight
        x = self.normalize(x)
        if not torch.is_tensor(mask_value):
            mask_value = torch.tensor(mask_value, device=x.device)
        mask_value_norm = self.normalize(mask_value)
        x = x.masked_fill(~mask, mask_value_norm)
        self.step_count += 1
        return x

    def normalize(self, x):
        x = (x - self.running_mean) / self.running_std
        x = (x * self.norm_std) + self.norm_mean
        return x

    def get_mask(self, x, lengths):
        max_len = x.size(self.length_dim)
        mask = length_to_mask(lengths * max_len, max_len)
        for dim in range(1, x.dim()):
            if dim != self.length_dim:
                mask = mask.unsqueeze(dim)
        mask = mask.expand_as(x).bool()
        return mask

    def denormalize(self, x):
        """Reverses the normalization proces

        Arguments
        ---------
        x: torch.Tensor
            a normalized tensor

        Returns
        -------
        result: torch.Tensor
            a denormalized version of x
        """
        x = (x - self.norm_mean) / self.norm_std
        x = x * self.running_std + self.running_mean
        return x

    def freeze(self):
        """Stops updates to the running mean/std"""
        self.frozen = True

    def unfreeze(self):
        """Resumes updates to the running mean/std"""
        self.frozen = False


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
