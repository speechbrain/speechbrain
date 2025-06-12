"""Reproducibility tools

Author:
    * Artem Ploujnikov 2025
"""

import re
import speechbrain as sb
import torch

from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


@sb.utils.checkpoints.register_checkpoint_hooks
class SaveableGenerator:
    """A wrapper that can be used to store the state of
    the random number generator in a checkpoint. It helps
    with reproducibility in long-running experiments.

    Currently, this only supports CPU and Cuda devices
    natively. If you need training on other architectures,
    consider implementing a custom generator.

    Running it on an unsupported device not using the Torch
    generator interface will simply fail to restore the
    state but will not cause an error.

    Sample usage in hparams:
    ```yaml
    generator: !new:model.custom_model.SaveableGenerator
    checkpointer: !new:speechbrain.utils.checkpoints.Checkpointer
        checkpoints_dir: !ref <save_folder>
        recoverables:
            model: !ref <model>
            lr_scheduler: !ref <lr_annealing>
            counter: !ref <epoch_counter>
            generator: !ref <generator>*+
    ```

    Arguments
    ---------
    generators : list, optional
        A list of generator objects. If not provided, all
        default generators for CPU and Cuda will be used
    """

    def __init__(self, generators=None):
        if generators is None:
            generators = {
                "default": torch.default_generator
            }
            if torch.cuda.is_available():
                for idx, generator in torch.cuda.default_generators:
                    generators[f"cuda:{idx}"] = generator
        self.generators = generators

    @sb.utils.checkpoints.mark_as_saver
    def _save(self, path):
        save_dict = {
            key: generator.get_state()
            for key, generator in self.generators.items()
        }
        torch.save(save_dict, path)

    @sb.utils.checkpoints.mark_as_loader
    def _recover(self, path, end_of_epoch):
        del end_of_epoch
        save_dict = torch.load(path)
        for key, state in save_dict.items():
            match = re.match(r"cuda:(\d+)", key)
            if match:
                if not torch.cuda.is_available():
                    logger.warning("Unable to restore RNG for %s, CUDA unavailable", key)
                    continue
                idx = match.group(1)
                if idx > torch.cuda.device_count() - 1:
                    logger.warning("Unable to restore RNG for %s, device not found", key)
                    continue
            self.generators[key].set_state(state)
