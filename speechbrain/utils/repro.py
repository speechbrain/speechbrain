"""Reproducibility tools

Author:
    * Artem Ploujnikov 2025
"""

import re

import torch

import speechbrain as sb
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

    Typical in hparams:
    ```yaml
    generator: !new:model.custom_model.SaveableGenerator # <-- Include the wrapper

    checkpointer: !new:speechbrain.utils.checkpoints.Checkpointer
        checkpoints_dir: !ref <save_folder>
        recoverables:
            model: !ref <model>
            lr_scheduler: !ref <lr_annealing>
            counter: !ref <epoch_counter>
            generator: !ref <generator>
    ```

    Arguments
    ---------
    generators : Mapping[str, Generator], optional
        A dictionary of named generator objects. If not provided,
        the default generators for CPU and Cuda will be used

    Examples
    --------
    >>> import torch
    >>> from speechbrain.utils.repro import SaveableGenerator
    >>> from speechbrain.utils.checkpoints import Checkpointer
    >>> gena, genb = [torch.Generator().manual_seed(x) for x in [42, 24]]
    >>> saveable_gen = SaveableGenerator(
    ...     generators={"a": gena, "b": genb}
    ... )
    >>> tempdir = getfixture('tmpdir')
    >>> checkpointer = Checkpointer(
    ...     tempdir,
    ...     recoverables={"generator": saveable_gen})
    >>> torch.randint(0, 10, (1,), generator=gena).item()
    2
    >>> torch.randint(0, 10, (1,), generator=genb).item()
    4
    >>> _ = checkpointer.save_checkpoint()
    >>> torch.randint(0, 10, (1,), generator=gena).item()
    7
    >>> torch.randint(0, 10, (1,), generator=genb).item()
    5
    >>> _ = checkpointer.recover_if_possible()
    >>> torch.randint(0, 10, (1,), generator=gena).item()
    7
    >>> torch.randint(0, 10, (1,), generator=genb).item()
    5
    """

    def __init__(self, generators=None):
        if generators is None:
            generators = {"default": torch.default_generator}
            if torch.cuda.is_available():
                for idx in range(torch.cuda.device_count()):
                    generators[f"cuda:{idx}"] = _CudaDefaultGeneratorWrapper(
                        idx
                    )

        self.generators = generators

    @sb.utils.checkpoints.mark_as_saver
    def save(self, path):
        """Save the generator state for later recovery

        Arguments
        ---------
        path : str, Path
            Where to save. Will overwrite.
        """
        save_dict = {
            key: generator.get_state()
            for key, generator in self.generators.items()
        }
        torch.save(save_dict, path)

    @sb.utils.checkpoints.mark_as_loader
    def load(self, path, end_of_epoch):
        """
        Loads the generator state if the corresponding devices are
        present

        Arguments
        ---------
        path : str, Path
            Where to load from.
        end_of_epoch : bool
            Whether the checkpoint was end-of-epoch or not.
        """
        del end_of_epoch
        save_dict = torch.load(path)
        for key, state in save_dict.items():
            if key == "default":
                torch.default_generator.set_state(state)
                continue
            match = re.match(r"cuda:(\d+)", key)
            if match:
                if not torch.cuda.is_available():
                    logger.warning(
                        "Unable to restore RNG for %s, CUDA unavailable", key
                    )
                    continue
                idx = int(match.group(1))
                if idx > torch.cuda.device_count() - 1:
                    logger.warning(
                        "Unable to restore RNG for %s, device not found", key
                    )
                    continue
            self.generators[key].set_state(state)


class _CudaDefaultGeneratorWrapper:
    """A generator wrapper for default generators - because torch no longer
    exposes default_generators

    This class should not be used outside of SaveableGenerator

    Arguments
    ---------
    device : int|str
        The device index or identifier"""

    def __init__(self, device):
        self.device = device

    def get_state(self):
        """Returns the generator state

        Returns
        -------
        result : torch.Tensor
            The generator state
        """
        return torch.cuda.get_rng_state(self.device)

    def set_state(self, new_state):
        """ "Sets the generator state

        Arguments
        ---------
        new_state : dict
            The new state
        """
        torch.cuda.set_rng_state(new_state, self.device)
