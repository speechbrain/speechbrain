"""Convenience functions for the simplest parameter transfer cases.

Use `speechbrain.utils.checkpoints.Checkpointer` to find a checkpoint
and the path to the parameter file.

Authors
 * Aku Rouhe 2020
 * Andreas Nautsch 2023
 * Adel Moumen 2023
"""
import logging
import pathlib
from speechbrain.utils.distributed import run_on_main
from speechbrain.pretrained.fetching import fetch, FetchFrom, FetchSource
from speechbrain.utils.checkpoints import (
    DEFAULT_LOAD_HOOKS,
    DEFAULT_TRANSFER_HOOKS,
    PARAMFILE_EXT,
    get_default_hook,
)

logger = logging.getLogger(__name__)


class Pretrainer:
    """Orchestrates pretraining

    First collects parameter file symlinks into the given directory. Then
    calls load hooks for each of those parameter files.

    Arguments
    ---------
    collect_in : str or Path
        Path to directory where the parameter file symlinks are collected.
    loadables : mapping
        Mapping from loadable key to object. This connects the keys to
        the actual object instances.
    paths : mapping
        Mapping from loadable key to filepath. The last part
        of the path is treated as file name, the rest of it
        is treated as a "source" which can be either a directory
        path or a magic source like Huggingface hub ID.
        e.g. sb/asr-crdnn-libri/lm.ckpt
        -> source=sb/asr-crdnn-libri, file=lm.ckpt
        Note that when collecting, you can specify a default source,
        which is used for all loadables that don't have a path specified.
    custom_hooks : mapping
        Mapping from loadable key to parameter transfer hook function. If you
        want to use a custom loading function, specify it here.
    conditions: mapping
        An optional mapping from loadable keys to condition values,
        useful for loading certain elements only if a flag is turned on
    """

    def __init__(
        self,
        collect_in="./model_checkpoints",
        loadables=None,
        paths=None,
        custom_hooks=None,
        conditions=None,
    ):
        self.loadables = {}
        self.loadable_paths = {}
        self.collect_in = pathlib.Path(collect_in)
        if loadables is not None:
            self.add_loadables(loadables)
        self.paths = {}
        if paths is not None:
            self.add_paths(paths)
        self.custom_hooks = {}
        if custom_hooks is not None:
            self.add_custom_hooks(custom_hooks)
        self.conditions = {}
        if conditions is not None:
            self.add_conditions(conditions)
        self.is_local = []

    def set_collect_in(self, path):
        """Change the collecting path"""
        self.collect_in = pathlib.Path(path)

    def add_loadables(self, loadables):
        """Update the loadables dict from the given mapping.

        Arguments
        ---------
        loadables : mapping
            Mapping from loadable key to object
        """
        self.loadables.update(loadables)

    def add_paths(self, paths):
        """Update the paths for different loadables.

        When collecting parameters, paths here are preferred. Note that when
        collecting, you can specify a default source, which is used for all
        loadables that don't have a path specified.

        Arguments
        ---------
        paths : mapping
            Mapping from loadable key to filepath. The last part
            of the path is treated as file name, the rest of it
            is treated as a "source" which can be either a directory
            path or a magic source like Huggingface hub ID.
            e.g. sb/asr-crdnn-libri/lm.ckpt
            -> source=sb/asr-crdnn-libri, file=lm.ckpt
        """
        self.paths.update(paths)

    def add_custom_hooks(self, custom_hooks):
        """Update the custom hooks.

        When loading parameters, hooks here are preferred over class defaults.

        Arguments
        ---------
        custom_hooks : mapping
            Mapping from loadable key to parameter transfer hook function. If
            you want to use a custom loading function, specify it here.

        """
        self.custom_hooks.update(custom_hooks)

    def add_conditions(self, conditions):
        """Update the conditions.

        Arguments
        ---------
        conditions: mapping
            Mapping from loadable keys to condition values,
            useful for loading certain elements only if a flag is turned on

        """
        self.conditions.update(conditions)

    @staticmethod
    def split_path(path):
        """Splits a path to source and filename

        This also handles URLs and Huggingface hub paths, in addition to
        regular paths.

        Arguments
        ---------
        path : str

        Returns
        -------
        str
            Source
        str
            Filename
        """

        def split(src):
            """Core function to split path.
            """
            if "/" in src:
                return src.rsplit("/", maxsplit=1)
            else:
                # Interpret as path to file in current directory.
                return "./", src

        if isinstance(path, FetchSource):
            fetch_from, fetch_path = path
            source, filename = split(fetch_path)
            return FetchSource(fetch_from, source), filename
        else:
            return split(path)

    def collect_files(
        self, default_source=None, internal_ddp_handling=False,
    ):
        """Fetches parameters from known paths with fallback default_source

        The actual parameter files may reside elsewhere, but this ensures a
        symlink in the self.collect_in directory. The symlink always uses the
        loadable key in the filename. This standardization makes it easier to
        orchestrate pretraining on e.g. distributed setups.

        Use the default_source if you have everything organized neatly into one
        location, like a Huggingface hub repo.

        Arguments
        ---------
        default_source : str or Path or FetchSource
            This is used for each loadable which doesn't have a path already
            specified. If the loadable has key "asr", then the file to look for is
            default_source/asr.ckpt
        internal_ddp_handling : bool
            Whether/not the function should handle DDP i.e. `run_on_main`.
            (Default: False)

        Returns
        -------
        dict
            Mapping from loadable key to a local path from which loadable's
            parameters can be loaded. This is not used in this class, but
            can possibly be helpful.
        """
        logger.debug(
            f"Collecting files (or symlinks) for pretraining in {self.collect_in}."
        )
        self.collect_in.mkdir(exist_ok=True)
        for name in self.loadables:
            if not self.is_loadable(name):
                continue
            save_filename = name + PARAMFILE_EXT
            if name in self.paths:
                source, filename = self.split_path(self.paths[name])
            elif default_source is not None:
                filename = save_filename
                source = default_source
            else:
                raise ValueError(
                    f"Path not specified for '{name}', "
                    "and no default_source given!"
                )
            if internal_ddp_handling:
                # path needs to be available only if it is a local source w/o symlink
                run_on_main(
                    fetch,
                    kwargs={
                        "filename": filename,
                        "source": source,
                        "overwrite": False,
                        "save_filename": save_filename,
                        "use_auth_token": False,
                        "revision": None,
                    },
                )

                # we need the path; regardless of rank
                path = fetch(
                    filename=filename,
                    source=source,
                    savedir=self.collect_in,
                    overwrite=False,
                    save_filename=save_filename,
                    use_auth_token=False,
                    revision=None,
                )
            else:
                # main node is the only one calling this, so path is available
                path = fetch(
                    filename=filename,
                    source=source,
                    savedir=self.collect_in,
                    overwrite=False,
                    save_filename=save_filename,
                    use_auth_token=False,
                    revision=None,
                )
            self.loadable_paths[name] = path
            fetch_from = None
            if isinstance(source, FetchSource):
                fetch_from, source = source
            if fetch_from is FetchFrom.LOCAL or str(path) == str(
                source
            ) + "/" + str(filename):
                logger.info(f"Set local path in self.paths[{name}] = {path}")
                self.paths[name] = str(path)
                self.is_local.append(name)
        return self.loadable_paths

    def is_loadable(self, name):
        """Returns True if no condition is defined or for the specified
        loadable or if the condition is true

        Arguments
        ---------
        name: str
            the name of the loadable

        Returns
        -------
        is_loadable: bool
            whether the item should be loaded
        """
        if name not in self.conditions:
            return True
        condition = self.conditions[name]
        if callable(condition):
            return condition()
        else:
            return bool(condition)

    def load_collected(self, device=None):
        """Loads the files that have been collected.

        Arguments
        ---------
        device : str
            Device on which to load, if you want to load to a specific device
            directly ( otherwise just leave it to None ).
        """
        logger.info(
            f"Loading pretrained files for: {', '.join(self.loadables)}"
        )
        paramfiles = {}
        for name, path in self.loadable_paths.items():
            if not self.is_loadable(name):
                continue
            filename = name + PARAMFILE_EXT
            paramfiles[name] = self.collect_in / filename
            if not paramfiles[name].exists():
                # fallback to the original path; this is useful if a relative path was given
                logger.info(
                    f"Redirecting (loading from original path): {paramfiles[name]} -> {path}"
                )
                paramfiles[name] = path

            if name in self.is_local:
                logger.info(
                    f"Redirecting (loading from local path): {paramfiles[name]} -> {self.paths[name]}"
                )
                paramfiles[name] = self.paths[name]
        self._call_load_hooks(paramfiles, device)

    def _call_load_hooks(self, paramfiles, device=None):
        # This internal function finds the correct hook to call for every
        # recoverable, and calls it.
        for name, obj in self.loadables.items():
            if not self.is_loadable(name):
                continue
            loadpath = paramfiles[name]

            # First see if object has custom load hook:
            if name in self.custom_hooks:
                self.custom_hooks[name](obj, loadpath, device=device)
                continue
            # Try the default transfer hook:
            default_hook = get_default_hook(obj, DEFAULT_TRANSFER_HOOKS)
            if default_hook is not None:
                default_hook(obj, loadpath, device=device)
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
