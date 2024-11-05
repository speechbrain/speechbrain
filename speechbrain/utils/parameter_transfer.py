"""Convenience functions for the simplest parameter transfer cases.

Use `speechbrain.utils.checkpoints.Checkpointer` to find a checkpoint
and the path to the parameter file.

Authors
 * Aku Rouhe 2020
 * Andreas Nautsch 2023
 * Adel Moumen 2023
"""

import pathlib
import platform
import warnings

from speechbrain.utils.checkpoints import (
    DEFAULT_LOAD_HOOKS,
    DEFAULT_TRANSFER_HOOKS,
    PARAMFILE_EXT,
    get_default_hook,
)
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.fetching import FetchSource, LocalStrategy, fetch
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class Pretrainer:
    """Orchestrates pretraining

    First optionally collects files from some source (local directory,
    HuggingFace repository, base URL), into the `collect_in` directory, if
    specified.

    Then, calls load hooks for each of those files.

    Arguments
    ---------
    collect_in : str or Path, optional
        Path to directory where the files are to be collected.
        If `None`, then files will be referred to from cache or directly, if
        possible (URLs will fail). There will not be a centralized target
        directory with all the files.

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
        collect_in=None,
        loadables=None,
        paths=None,
        custom_hooks=None,
        conditions=None,
    ):
        self.loadables = {}

        self.set_collect_in(collect_in)

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
        self.collect_in = pathlib.Path(path) if path is not None else None

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
            """Core function to split path."""
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
        self,
        default_source=None,
        use_auth_token=False,
        local_strategy: LocalStrategy = LocalStrategy.SYMLINK,
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
            specified.
            e.g. if the loadable has key `"asr"`, then the file to look for is
            `<default_source>/asr.ckpt`
        use_auth_token : bool (default: False)
            If true Huggingface's auth_token will be used to load private models from the HuggingFace Hub,
            default is False because the majority of models are public.
        local_strategy : speechbrain.utils.fetching.LocalStrategy
            The fetching strategy to use, which controls the behavior of remote file
            fetching with regards to symlinking and copying.
            Ignored if a `collect_in` directory was not specified.
            See :func:`speechbrain.utils.fetching.fetch` for further details.

        Returns
        -------
        dict
            Mapping from loadable key to a local path from which loadable's
            parameters can be loaded. This is not used in this class, but
            can possibly be helpful.
        """

        if self.collect_in is not None:
            logger.debug(
                f"Collecting files (or symlinks) for pretraining in {self.collect_in}."
            )
            self.collect_in.mkdir(exist_ok=True)

            if (
                platform.system() == "Windows"
                and local_strategy == LocalStrategy.SYMLINK
            ):
                warnings.warn(
                    "Requested Pretrainer collection using symlinks on Windows. This might not work; see `LocalStrategy` documentation. Consider unsetting `collect_in` in Pretrainer to avoid symlinking altogether."
                )
        else:
            logger.debug(
                "Fetching files for pretraining (no collection directory set)"
            )

        loadable_paths = {}
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

            fetch_kwargs = {
                "filename": filename,
                "source": source,
                "savedir": self.collect_in,
                "overwrite": False,
                "save_filename": save_filename,
                "use_auth_token": use_auth_token,
                "revision": None,
                "local_strategy": local_strategy,
            }

            path = None

            def run_fetch(**kwargs):
                """Very basic local wrapper to fetch to store the path in a
                local of collect_files

                Arguments
                ---------
                **kwargs : dict
                    Arguments to forward to fetch"""
                nonlocal path
                path = fetch(**kwargs)

            # run fetch() on the main process, potentially performing downloading
            # which we do NOT want to happen concurrently.
            #
            # then, if there are any non-main processes, run fetch() on them to
            # resolve the path.
            #
            # path needs to be available only if it is a local source w/o symlink
            run_on_main(
                run_fetch,
                kwargs=fetch_kwargs,
                post_func=run_fetch,
                post_kwargs=fetch_kwargs,
            )

            loadable_paths[name] = path
            if isinstance(source, FetchSource):
                _fetch_from, source = source

            logger.debug(f'Set local path in self.paths["{name}"] = {path}')
            self.paths[name] = str(path)
            self.is_local.append(name)
        return loadable_paths

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

    def load_collected(self):
        """Loads the files that have been collected."""
        logger.info(
            f"Loading pretrained files for: {', '.join(self.loadables)}"
        )
        paramfiles = {}
        for name in self.loadables:
            if not self.is_loadable(name):
                continue
            filename = name + PARAMFILE_EXT

            if name in self.is_local:
                logger.debug(
                    f"Redirecting (loading from local path): {name} -> {self.paths[name]}"
                )
                paramfiles[name] = self.paths[name]
            elif self.collect_in is not None:
                paramfiles[name] = self.collect_in / filename
            else:
                raise ValueError(
                    f'Pretrainer has never collected `{name}`, did you forget a call to `collect_files`? Could not fall back to `collect_in`, as it was not specified (default is no longer "model_checkpoints").'
                )
        self._call_load_hooks(paramfiles)

    def _call_load_hooks(self, paramfiles):
        # This internal function finds the correct hook to call for every
        # recoverable, and calls it.
        for name, obj in self.loadables.items():
            if not self.is_loadable(name):
                continue
            loadpath = paramfiles[name]

            # First see if object has custom load hook:
            if name in self.custom_hooks:
                self.custom_hooks[name](obj, loadpath)
                continue
            # Try the default transfer hook:
            default_hook = get_default_hook(obj, DEFAULT_TRANSFER_HOOKS)
            if default_hook is not None:
                default_hook(obj, loadpath)
                continue
            # Otherwise find the default loader for that type:
            default_hook = get_default_hook(obj, DEFAULT_LOAD_HOOKS)
            if default_hook is not None:
                # Need to fake end-of-epoch:
                end_of_epoch = False
                default_hook(obj, loadpath, end_of_epoch)
                continue
            # If we got here, no custom hook or registered default hook exists
            MSG = f"Don't know how to load {type(obj)}. Register default hook \
                    or add custom hook for this object."
            raise RuntimeError(MSG)
