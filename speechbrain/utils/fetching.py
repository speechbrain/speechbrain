"""Downloads or otherwise fetches pretrained models

Authors:
 * Aku Rouhe 2021
 * Samuele Cornell 2021
 * Andreas Nautsch 2022, 2023
 * Sylvain de Langen 2024
 * Peter Plantinga 2024
"""

import pathlib
import platform
import shutil
import urllib.error
import urllib.request
import warnings
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import huggingface_hub
from requests.exceptions import HTTPError

from speechbrain.utils.distributed import main_process_only
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
    """If using `LocalStrategy.COPY` or `LocalStrategy.COPY_SKIP_CACHE`, destroy
    the file or symlink at `dst` if present and creates a copy from `src` to
    `dst`.

    If using `LocalStrategy.SYMLINK`, destroy the file or symlink at `dst` if
    present and creates a symlink from `src` to `dst`.

    If `LocalStrategy.NO_LINK` is passed, the src path is returned.

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

    if local_strategy == LocalStrategy.NO_LINK:
        return src

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

    raise ValueError(
        f"Illegal local strategy {local_strategy} passed for linking"
    )


def guess_source(source: Union[str, FetchSource]) -> tuple[FetchFrom, str]:
    """From a given `FetchSource` or string source identifier, attempts to guess
    the matching :class:`~FetchFrom` (e.g. local or URI).

    If `source` is already a `FetchSource`, it is returned as-is.

    Arguments
    ---------
    source : str or FetchSource
        Where to look for the file. :func:`~fetch` interprets this path using
        the following logic:

        - First, if the source begins with "http://" or "https://", it is
          interpreted as a web address and the file is downloaded.
        - Second, if the source is a valid directory path, the file is either
          linked, copied or directly returned depending on the local strategy.
        - Otherwise, the source is interpreted as a HuggingFace model hub ID,
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


@dataclass(frozen=True)
class FetchConfig:
    """A dataclass containing all the configurations for fetching, such as caching strategy.

    Attributes
    ----------
    overwrite : bool, defaults to `False`
        Allows the destination to be recreated by copy/symlink/fetch.
        This does **not** skip the HuggingFace cache (see `allow_updates`).
    allow_updates : bool, defaults to `False`
        If `True`, for a remote file on HF, check for updates and download newer
        revisions if available.
        If `False`, when the requested files are available locally, load them
        without fetching from HF.
    allow_network : bool, defaults to `True`
        If `True`, network accesses are allowed. If `False`, then remote URLs
        or HF won't be fetched, regardless of any other parameter.
    use_auth_token : bool, defaults to  `False`
        If `True`, use HuggingFace's `auth_token` to enable loading private
        models from the Hub.
    revision : Optional[str] defaults to `None`
        HuggingFace Hub model revision (Git branch name/tag/commit hash) to pin
        to a specific version.
        When changing the revision while local files might still exist,
        `allow_updates` must be `True`.
    huggingface_cache_dir: Optional[str] defaults to `None`
        Path to HuggingFace cache; if `None`, assumes the default cache location
        `<https://huggingface.co/docs/huggingface_hub/guides/manage-cache#manage-huggingfacehub-cache-system>`.
        Ignored if using `LocalStrategy.COPY_SKIP_CACHE`.
        Please prefer to let the user specify the cache directory themselves
        through the environment.
    """

    overwrite: bool = False
    allow_updates: bool = False
    allow_network: bool = True
    use_auth_token: bool = False
    revision: str = None
    huggingface_cache_dir: str = None


@main_process_only
def download_file(source, source_path, destination):
    """Download a source path to a destination"""
    try:
        urllib.request.urlretrieve(source_path, destination)
    except urllib.error.URLError as e:
        raise ValueError(
            f"Interpreted '{source}' as web address, but could not download."
        ) from e


@main_process_only
def download_file_hf(hf_kwargs, destination, local_strategy):
    """Download a source file from huggingface to local"""
    try:
        fetched_file = huggingface_hub.hf_hub_download(**hf_kwargs)
        fetched_file = pathlib.Path(fetched_file)
        if local_strategy != LocalStrategy.COPY_SKIP_CACHE:
            link_with_strategy(fetched_file, destination, local_strategy)

    except HTTPError as e:
        if "404 Client Error" in str(e):
            raise ValueError("File not found on HF hub") from e
        raise


def fetch(
    filename,
    source: Union[str, FetchSource],
    savedir: Optional[Union[str, pathlib.Path]] = None,
    save_filename: Optional[str] = None,
    local_strategy: LocalStrategy = LocalStrategy.SYMLINK,
    fetch_config: FetchConfig = FetchConfig(),
):
    """Fetches a local path, remote URL or remote HuggingFace file, downloading
    it locally if necessary and returns the local path.

    When a `savedir` is specified, but the file already exists locally
    elsewhere, the specified :class:`~LocalStrategy` chooses whether to copy or
    symlink it.

    If `<savedir>/<save_filename>` exists locally, it is returned as is (unless using `overwrite` or `allow_updates`).

    The `HF_HOME` environment (default: `~/.cache/huggingface`) `selects the cache directory for HF <https://huggingface.co/docs/huggingface_hub/guides/manage-cache#manage-huggingfacehub-cache-system>`__.
    To prefer directly downloading to `savedir`, specify `local_strategy=LocalStrategy.COPY_SKIP_CACHE`.
    **HF cache is always looked up first if possible.**

    Arguments
    ---------
    filename : str
        Name of the file including extensions.
    source : str or FetchSource
        Local or remote root path for the filename. The final path is
        determined by `<source>/<filename>`.
        See :func:`~guess_source` for how the path kind is deduced.
    savedir : str, optional
        If specified, directory under which the files will be available
        (possibly as a copy or symlink depending on `local_strategy`).
        Must be specified when downloading from an URL.
    save_filename : str, optional, defaults to `None`
        The filename to use for saving this file. Defaults to the `filename`
        argument if not given or `None`.
    local_strategy : LocalStrategy
        Which strategy to use for local file storage -- see `LocalStrategy` for options.
        Ignored by `fetch` unless `savedir` is provided, default is `LocalStrategy.SYMLINK` which
        adds a link to the downloaded/cached file in the `savedir`.
    fetch_config : FetchConfig
        A configuration for how to perform fetching, see `FetchConfig` dataclass for details.

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

    fetch_from, source = guess_source(source)
    source_path = f"{source}/{filename}"

    # If savedir is specified, ensure folder exists and use as destination
    # for downloaded files. Otherwise, note that no link should be made.
    if savedir is not None:
        savedir = pathlib.Path(savedir)
        savedir.mkdir(parents=True, exist_ok=True)
        destination = (savedir / save_filename).absolute()
    else:
        destination = None
        local_strategy = LocalStrategy.NO_LINK

    # Check fetch_config type
    assert isinstance(fetch_config, FetchConfig)

    # HF is the only download method that supports updates
    should_try_update = fetch_config.overwrite or (
        fetch_from == FetchFrom.HUGGING_FACE and fetch_config.allow_updates
    )

    # Check if file is already present at destination
    if (
        destination is not None
        and destination.exists()
        and not should_try_update
    ):
        file_kind = "symlink" if destination.is_symlink() else "file"
        logger.info(
            "Fetch %s: Using %s found at '%s'",
            filename,
            file_kind,
            str(destination),
        )
        return destination

    if fetch_from == FetchFrom.LOCAL:
        source_path = pathlib.Path(source_path).absolute()
        return link_with_strategy(source_path, destination, local_strategy)

    if fetch_from == FetchFrom.URI:
        if destination is None:
            raise ValueError(
                f"Fetch {filename}: `savedir` must be specified for URI fetches"
            )

        if not fetch_config.allow_network:
            # TODO: streamline exceptions?
            raise ValueError(
                f"Fetch {filename}: File was not found locally and "
                "`allow_network` was disabled."
            )

        # Finally, we have to download, which is done on main process only
        logger.info("Fetch %s: Downloading '%s'", filename, str(source_path))
        download_file(source, source_path, destination)
        return destination

    # Only available option left is Huggingface, download on main
    assert fetch_from == FetchFrom.HUGGING_FACE

    logger.info(
        "Fetch %s: Fetching from HuggingFace Hub '%s' if not cached",
        str(filename),
        str(source),
    )

    # Assemble the arguments needed for `hf_hub_download`
    hf_kwargs = {
        "repo_id": source,
        "filename": filename,
        "use_auth_token": fetch_config.use_auth_token,
        "revision": fetch_config.revision,
        "local_files_only": not fetch_config.allow_network,
    }
    if local_strategy == LocalStrategy.COPY_SKIP_CACHE:
        hf_kwargs.update(
            {
                "local_dir": savedir,
                "local_dir_use_symlinks": False,
                "force_filename": save_filename,
            }
        )
    else:
        hf_kwargs["cache_dir"] = fetch_config.huggingface_cache_dir

    # Download is done on the main process only
    download_file_hf(hf_kwargs, destination, local_strategy)

    # destination can be None if local_strategy is NO_LINK
    # In this case, we call the hub download once more to get the file
    if destination is None:
        destination = pathlib.Path(huggingface_hub.hf_hub_download(**hf_kwargs))

    return destination
