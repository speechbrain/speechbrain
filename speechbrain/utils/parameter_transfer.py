"""Convenience functions for the simplest parameter transfer cases.

Use `speechbrain.utils.checkpoints.Checkpointer` to find a checkpoint
and the path to the parameter file.

Authors
 * Aku Rouhe 2020
"""
import logging

from speechbrain.utils.checkpoints import (
    DEFAULT_LOAD_HOOKS,
    DEFAULT_TRANSFER_HOOKS,
)
from speechbrain.utils.checkpoints import get_default_hook

logger = logging.getLogger(__name__)


class Pretrainer:
    """A pretrainig equivalent to Checkpointer"""

    def __init__(self, loadables=None, custom_load_hooks=None):
        self.loadables = {}
        if loadables is not None:
            self.add_loadables(loadables)
        self.custom_load_hooks = {}
        if custom_load_hooks is not None:
            self.custom_load_hooks.update(custom_load_hooks)

    def add_loadables(self, loadables):
        """Update the loadables dict from the given mapping.

        Arguments
        ---------
        loadables : mapping
            The objects to pretrain.
        """
        self.loadables.update(loadables)

    def fetch_parameters(self, source):
        """Fetches from given source (path, huggingface_hub code etc.)

        Returns
        -------
        dict
            Mapping from loadable names to parameter files
        """
        # TODO
        pass

    def fetch_and_load(self, source, device=None):
        paramfiles = self.fetch_parameters(source)
        logger.info(f"Loading pretrained weights from {source}")
        self._call_load_hooks(paramfiles, device)

    def _call_load_hooks(self, paramfiles, device=None):
        # This internal function finds the correct hook to call for every
        # recoverable, and calls it.
        for name, obj in self.loadables.items():
            loadpath = paramfiles[name]

            # First see if object has custom load hook:
            if name in self.custom_load_hooks:
                self.custom_load_hooks[name](obj, loadpath, device)
                continue
            # Try the default transfer hook:
            default_hook = get_default_hook(obj, DEFAULT_TRANSFER_HOOKS)
            if default_hook is not None:
                default_hook(obj, loadpath, device)
                continue
            # Otherwise find the default loader for that type:
            default_hook = get_default_hook(obj, DEFAULT_LOAD_HOOKS)
            if default_hook is not None:
                # Need to fake end-of-epoch:
                end_of_epoch = False
                default_hook(obj, loadpath, end_of_epoch, device)
                continue
            # If we got here, no custom hook or registered default hook exists
            MSG = f"Don't know how to load {type(obj)}. Register default hook \
                    or add custom hook for this object."
            raise RuntimeError(MSG)
