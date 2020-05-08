"""
An approximation of the SpecAugment algorithm, carried out in the time domain.

Author: Peter Plantinga 2020
"""
import os
import torch
from speechbrain.yaml import load_extended_yaml
from speechbrain.processing.features import spectral_magnitude


class SpecAugment(torch.nn.Module):
    """A time-domain approximation of the SpecAugment algorithm.

    Arguments
    ---------
    filterbank : bool
        Whether to apply a filterbank to the spectral features.
    log : bool
        Whether to apply log to the outputs (regardless of type).
    **overrides
        A set of overrides to use for the `spec_augment.yaml` file.

    Example
    -------
    >>> inputs = torch.randn([10, 16000])
    >>> feature_maker = SpecAugment(speed_perturb={'speeds':[8]})
    >>> feats = feature_maker(inputs, torch.ones(10), init_params=True)
    >>> feats.shape
    torch.Size([10, 81, 40])
    """

    def __init__(self, filterbank=True, log=True, **overrides):
        super().__init__()
        self.filterbank = filterbank
        self.log = log
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_dir, "spec_augment.yaml")) as f:
            self.params = load_extended_yaml(f, overrides)

    def forward(self, waveforms, lengths, init_params=False):
        """Returns the distorted waveforms.

        Arguments
        ---------
        waveforms : torch.Tensor
            The waveforms to distort
        """
        # Augmentation
        if self.training:
            waveforms = self.params.speed_perturb(waveforms)
            waveforms = self.params.drop_freq(waveforms)
            waveforms = self.params.drop_chunk(waveforms, lengths)

        # Feature generation
        features = self.params.compute_STFT(waveforms)
        features = spectral_magnitude(features)
        if self.filterbank:
            features = self.params.compute_fbanks(features, init_params)
        if self.log:
            features = torch.log(features + 1e-10)

        return features
