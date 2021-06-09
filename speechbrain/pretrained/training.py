"""
Utilities for TTS recipes

Authors
* Artem Ploujnikov 2021
"""
from speechbrain.utils.checkpoints import torch_save


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
            torch_save(value, path)
