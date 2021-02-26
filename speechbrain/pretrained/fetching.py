"""Downloads or otherwise fetches pretrained models

Authors:
 * Aku Rouhe 2021
 * Samuele Cornell 2021
"""
import urllib.request
import urllib.error
import pathlib
import logging

logger = logging.getLogger(__name__)


def fetch(filename, source, savedir="./data"):
    """Ensures you have a local copy of the file, returns its path

    In case the source is an external location, downloads the file.
    In case the source is already accessible on the filesystem,
    creates a symlink in the savedir. Thus, the side effects of
    this function always look similar: savedir/filename can be used
    to access the file.

    Arguments
    ---------
    filename : str
        Name of the file including extensions.
    source : str
        Where to look for the file. This is interpreted in special ways:
        First, if the source begins with "http://" or "https://", it is
        interpreted as a web address and the file is downloaded.
        Second, if the source is a valid directory path, a symlink is
        created to the file.
        Third, if huggingface-hub is installed and the source is a valid
        Huggingface model hub ID, the file is downloaded from there.
    savedir : str
        Path where to save downloads/symlinks.

    Returns
    -------
    pathlib.Path
        Path to file on local file system.

    NOTE
    ----
    This will overwrite savedir/filename
    """
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True)
    sourcefile = f"{source}/{filename}"
    destination = savedir / filename
    if source.startswith("http://") or source.startswith("https://"):
        # Interpret source as web address.
        # Download
        try:
            urllib.request.urlretrieve(sourcefile, destination)
        except urllib.error.URLError:
            raise ValueError(
                f"Interpreted {source} as web address, but could not download."
            )
    elif pathlib.Path(source).is_dir():
        # Interpret source as local directory path
        # Just symlink
        sourcepath = pathlib.Path(sourcefile).absolute()
        destination.unlink(missing_ok=True)
        destination.symlink_to(sourcepath)
    else:
        # Interpret source as huggingface hub ID
        # Use huggingface hub's fancy cached download.
        try:
            import huggingface_hub
        except ImportError:
            # Extra tools pattern:
            raise ValueError(
                f"Interpreted {source} as Huggingface hub ID, but Huggingface-hub"
                "is not installed. Please install with pip install huggingface-hub"
            )
        url = huggingface_hub.hf_hub_url(source, filename)
        fetched_file = huggingface_hub.cached_download(url, cache_dir=savedir)
        # Huggingface hub downloads to etag filename, symlink to the expected one:
        sourcepath = pathlib.Path(fetched_file).absolute()
        destination.unlink(missing_ok=True)
        destination.symlink_to(sourcepath)
    return destination
