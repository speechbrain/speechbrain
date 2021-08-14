"""
Utilities for model training

Authors
* Artem Ploujnikov 2021
"""

import torch
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


def _detach(value):
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu()
    elif isinstance(value, dict):
        result = {key: _detach(item_value) for key, item_value in value.items()}
    else:
        result = value
    return result
