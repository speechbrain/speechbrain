"""Downloads or otherwise fetches pretrained models

Authors:
 * Aku Rouhe 2021
 * Samuele Cornell 2021
 * Andreas Nautsch 2022, 2023
"""

import pathlib
import platform
import shutil
import urllib.error
import urllib.request
import warnings
from collections import namedtuple
from enum import Enum
from typing import Optional, Union

import huggingface_hub
from requests.exceptions import HTTPError

from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class FetchFrom(Enum):
    """Designator where to fetch models/audios from.

    Note: HuggingFace repository sources and local folder sources may be confused if their source type is undefined.
    """

    LOCAL = 1
    HUGGING_FACE = 2
    URI = 3


# For easier use
FetchSource = namedtuple("FetchSource", ["FetchFrom", "path"])
FetchSource.__doc__ = (
    """NamedTuple describing a source path and how to fetch it"""
)
FetchSource.__hash__ = lambda self: hash(self.path)
FetchSource.encode = lambda self, *args, **kwargs: "_".join(
    (str(self.path), str(self.FetchFrom))
).encode(*args, **kwargs)
# FetchSource.__str__ = lambda self: str(self.path)


class LocalStrategy(Enum):
    """Describes what strategy should be chosen for fetching and linking to
    local files when using :func:`~fetch`."""

    SYMLINK = 1
    """If the file is remote and not in cache, fetch it (potentially to cache).

    Then, create a symbolic link in the destination folder to the local file,
    if necessary.

    .. warning::
        Windows requires extra configuration to enable symbolic links, as it is
        a potential security risk on this platform.
        You either need to run Python as an administrator, or to enable
        developer mode. See `MS docs <https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development>`_.
        Additionally, the `huggingface_hub` library makes a use of symlinks that
        is independently controlled. See
        `HF hub docs <https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations>`_
        for reference.
    """

    COPY = 2
    """If the file is remote and not in cache, fetch it (potentially to cache).

    Then, create a copy of the local file in the destination folder, if
    necessary.
    """

    COPY_SKIP_CACHE = 3
    """If the file is remote and not in cache, fetch it, preferably directly to
    the destination directory.

    Then, create a copy in the destination folder to the local file, if
    necessary."""

    NO_LINK = 4
    """If the file is remote and not in cache, fetch it (potentially to cache).

    Then, return the local path to it, even if it is not the destination folder
    (e.g. it might be located in a cache directory).

    .. note::
        **This strategy may break code that does not expect this behavior,**
        since the destination folder is no longer guaranteed to contain a copy
        or link to the file.
    """


def link_with_strategy(
    src: pathlib.Path, dst: pathlib.Path, local_strategy: LocalStrategy
) -> pathlib.Path:
    """If `src == dst` **or** if using `LocalStrategy.NO_LINK`, returns `src`
    with no change to the files.

    If using `LocalStrategy.COPY` or `LocalStrategy.COPY_SKIP_CACHE`, destroy
    the file or symlink at `dst` if present and creates a copy from `src` to
    `dst`.

    If using `LocalStrategy.SYMLINK`, destroy the file or symlink at `dst` if
    present and creates a symlink from `src` to `dst`.

    Arguments
    ---------
    src : pathlib.Path
        Path to the source file to link to. Must be a valid path.
    dst : pathlib.Path
        Path of the final destination file. The file might not already exist,
        but the directory leading up to it must exist.
    local_strategy : LocalStrategy
        Strategy to adopt for linking.

    Returns
    -------
    pathlib.Path
        Path to the final file on disk, after linking/copying (if any).
    """

    src = src.absolute()
    dst = dst.absolute()

    if src == dst:
        if src.is_symlink():
            raise ValueError(
                f"Fetch: Found local symlink '{src}' pointing to itself. "
                "This may require manual removal to recover. "
                "Did you maybe incorrectly call fetch() with `source==savedir`?"
            )

        logger.debug(
            "Fetch: Source and destination '%s' are identical, returning assuming this is intended",
            src,
        )

        return dst

    if local_strategy == LocalStrategy.SYMLINK:
        if platform.system() == "Windows":
            warnings.warn(
                "Using SYMLINK strategy on Windows for fetching potentially "
                "requires elevated privileges and is not recommended. See "
                "`LocalStrategy` documentation."
            )

        logger.debug(
            "Fetch: Local file found, creating symlink '%s' -> '%s'", src, dst
        )

        dst.unlink(missing_ok=True)  # remove link or delete file
        dst.symlink_to(src)
        return dst

    if local_strategy in (LocalStrategy.COPY, LocalStrategy.COPY_SKIP_CACHE):
        logger.info("Fetch: Local file found, copying '%s' -> '%s'", src, dst)

        dst.unlink(missing_ok=True)  # remove link or delete file
        shutil.copy(str(src), str(dst))
        return dst

    if local_strategy == LocalStrategy.NO_LINK:
        logger.debug(
            "Fetch: Local file found, returning '%s' (NOT linking to '%s' because NO_LINK was passed)",
            src,
            dst,
        )

        return src


def guess_source(source: Union[str, FetchSource]) -> FetchSource:
    """From a given `FetchSource` or string source identifier, attempts to guess
    the matching :class:`~FetchFrom` (e.g. local or URI).

    If `source` is already a `FetchSource`, it is returned as-is.

    Arguments
    ---------
    source : str or FetchSource
        Where to look for the file. This is interpreted in special ways:
        - First, if the source begins with "http://" or "https://", it is
          interpreted as a web address and the file is downloaded.
        - Second, if the source is a valid directory path, a symlink is
          created to the file.
        - Otherwise, the source is interpreted as a Huggingface model hub ID,
          and the file is downloaded from there (potentially with caching).

    Returns
    -------
    tuple of (FetchFrom, str)"""

    if isinstance(source, FetchSource):
        return source

    if pathlib.Path(source).is_dir():
        return FetchFrom.LOCAL, source

    uri_supported_schemes = (
        "http:",
        "https:",
    )
    if source.startswith(uri_supported_schemes):
        return FetchFrom.URI, source

    return FetchFrom.HUGGING_FACE, source


def fetch(
    filename,
    source: Union[str, FetchSource],
    savedir: Union[str, pathlib.Path] = "./pretrained_model_checkpoints",
    overwrite: bool = False,
    allow_updates: bool = True,
    allow_network: bool = True,
    save_filename: Optional[str] = None,
    use_auth_token: bool = False,
    revision: Optional[str] = None,
    huggingface_cache_dir: Optional[Union[str, pathlib.Path]] = None,
    local_strategy: LocalStrategy = LocalStrategy.NO_LINK,
):
    """Ensures you have a local copy of the file and returns its path.

    If the file already exists under `<savedir>/<save_filename>` **and**
    `overwrite == False`, then that path is always returned and no other action
    is performed.

    Unless the `local_strategy` is `LocalStrategy.NO_LINK`, the file will be
    available under `<savedir>/<save_filename>`.

    .. note::
        If `source` is an URI but found locally, `fetch` will not attempt
        downloading again, unless `override` is `True`.

    Effect of local file strategies when the fetch source is a **local file**:

    - `LocalStrategy.SYMLINK`: Create symlink to the file in the destination.
    - `LocalStrategy.COPY`: Create copy of the file in the destination.
    - `LocalStrategy.COPY_SKIP_CACHE`: Create copy of the file in the
      destination.
    - `LocalStrategy.NO_LINK`: Returns the path to the source file.

    Effect of local file strategies when the fetch source is an **URI**:
    Downloads the file to the destination.

    Effect of local file strategies when the fetch source is the **HF Hub**:

    - `LocalStrategy.SYMLINK`: Create symlink to the file in the cache.
    - `LocalStrategy.COPY`: Downloads to cache if necessary, then copies to the
      destination.
    - `LocalStrategy.COPY_SKIP_CACHE`: Copies from cache if available, else
      downloads directly to the destination.
    - `LocalStrategy.NO_LINK`: Returns the path to the cache.

    Arguments
    ---------
    filename : str
        Name of the file including extensions.
    source : str or FetchSource
        Where to look for the file. See :func:`~guess_source` for how the fetch
        type is interpreted (e.g. local/URI/HF).
    savedir : str
        Directory path where the file will be reachable (unless using
        `LocalStrategy.NO_LINK`, in which case it may or may not end up in this
        directory).
    overwrite : bool (default: `False`)
        If `True`, always overwrite existing savedir/filename file and download
        the file over again, even if it was unchanged.
        If False (the default), if savedir/filename exists, assume it is
        correct and don't download/relink.
        Note that Huggingface local cache is always used - with overwrite=True
        you may just relink from the local cache.
    allow_updates : bool (default: `True`)
        If `True`, for a remote file on HF, check for updates and download newer
        revisions if available.
        If `False`, when the requested files are available locally, load them
        without fetching from HF.
    allow_network : bool (default: `True`)
        If `True`, network accesses are allowed. If `False`, then remote URLs
        or HF won't be fetched, regardless of any other parameter.
    save_filename : str, optional (default: `None`)
        The filename to use for saving this file. Defaults to the `filename`
        argument if not given or `None`.
    use_auth_token : bool (default: `False`)
        If true Huggingface's auth_token will be used to load private models from the HuggingFace Hub,
        default is False because majority of models are public.
    revision : str, optional (default: `None`)
        The model revision corresponding to the HuggingFace Hub model revision,
        in the form of a Git revision (branch name, tag, commit hash).
        This allows to pin your code to a particular version of a model hosted
        on HuggingFace.
        When changing the revision while local files still exist,
        `allow_updates` must be `True`.
    huggingface_cache_dir: str, optional (default: `None`)
        Path to HuggingFace cache; if `None`, the default cache directory is
        used: `~/.cache/huggingface` unless overridden by environment variables.
        See `huggingface_hub documentation <https://huggingface.co/docs/huggingface_hub/guides/manage-cache#manage-huggingfacehub-cache-system>`_
        Ignored if the local strategy is `LocalStrategy.COPY_SKIP_CACHE`.
        (default: None)
    local_strategy : LocalStrategy, optional
        Which strategy to use to deal with files locally. (default:
        `LocalStrategy.NO_LINK`)

    Returns
    -------
    pathlib.Path
        Path to file on local file system.

    Raises
    ------
    ValueError
        If file is not found
    """

    if save_filename is None:
        save_filename = filename

    savedir = pathlib.Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    fetch_from, source = guess_source(source)

    sourcefile = f"{source}/{filename}"
    destination = (savedir / save_filename).absolute()

    # only HF supports updates
    should_try_update = overwrite or (
        fetch_from == FetchFrom.HUGGING_FACE and allow_updates
    )

    if destination.exists() and not should_try_update:
        MSG = f"Fetch {filename}: Using existing file/symlink in {str(destination)}"
        logger.info(MSG)
        return destination

    if fetch_from == FetchFrom.LOCAL:
        return link_with_strategy(
            pathlib.Path(sourcefile).absolute(), destination, local_strategy
        )

    if fetch_from == FetchFrom.URI:
        if not allow_network:
            # TODO: streamline exceptions?
            raise ValueError(
                f"Fetch {filename}: File was not found locally and "
                "`allow_network` was disabled."
            )

        logger.info(
            "Fetch %s: Downloading from URL '%s'", filename, str(sourcefile)
        )
        # Download
        try:
            urllib.request.urlretrieve(sourcefile, destination)
        except urllib.error.URLError as e:
            raise ValueError(
                f"Interpreted {source} as web address, but could not download."
            ) from e

        return destination

    assert fetch_from == FetchFrom.HUGGING_FACE

    # Assume we are fetching from HF at this point
    # Interpret source as huggingface hub ID
    try:
        kwargs = {
            "repo_id": source,
            "filename": filename,
            "use_auth_token": use_auth_token,
            "revision": revision,
            "local_files_only": not allow_network,
        }

        # COPY strategy? Directly attempt saving to destination
        if local_strategy == LocalStrategy.COPY_SKIP_CACHE:
            logger.info(
                "Fetch %s: Fetching from HuggingFace Hub '%s' to '%s' if not cached",
                str(filename),
                str(source),
                str(destination),
            )
            fetched_file = huggingface_hub.hf_hub_download(
                **kwargs,
                local_dir=savedir,
                local_dir_use_symlinks=False,
                force_filename=save_filename,
            )

            fetched_file = pathlib.Path(fetched_file).absolute()
            assert fetched_file == destination, (
                "Downloaded file unexpectedly in wrong location "
                "because of the COPY strategy, this is a bug"
            )

            return pathlib.Path(fetched_file).absolute()

        # Otherwise, normal fetch to cache
        logger.info(
            "Fetch %s: Fetching from HuggingFace Hub '%s' if not cached",
            str(filename),
            str(source),
        )
        fetched_file = huggingface_hub.hf_hub_download(
            **kwargs,
            cache_dir=huggingface_cache_dir,
        )
        fetched_file = pathlib.Path(fetched_file).absolute()
    except HTTPError as e:
        if "404 Client Error" in str(e):
            raise ValueError("File not found on HF hub") from e
        raise

    return link_with_strategy(fetched_file, destination, local_strategy)
