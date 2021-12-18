"""
Utilities for TTS recipes

Authors
* Artem Ploujnikov 2021
"""
from speechbrain.utils.checkpoints import torch_save
import os

class PretrainedModelMixin:
    """
    A brain mixin that provides a function to save models and other
    artefacts for pretrained models
    """

    def save_for_pretrained(self):
        """
        Saves the necessary files for the pretrained model
        """
        pretrainer = self.hparams.pretrainer
        for key, value in pretrainer.loadables.items():
            path = pretrainer.paths[key]
            os.makedirs(os.path.dirname(path))
            torch_save(value, path)
