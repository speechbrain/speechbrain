"""A pipeline for caching data transformations into hdf5 files.

Authors:
 * Peter Plantinga, 2025
 * Adel Moumen, 2025
"""

from pathlib import Path

import h5py

from speechbrain.utils.data_pipeline import CachedDynamicItem, DynamicItem


class CachedHDF5DynamicItem(CachedDynamicItem):
    """CachedDynamicItem that uses HDF5 to store the cache. This performant
    data storage format only creates a single file, which may be faster or
    more efficient than the default storage (one torch file per id).

    Arguments
    ---------
    cache_location : os.PathLike
        Storage folder for containing HDF5 cached output file.
    file_mode : str
        The mode to use when opening the HDF5 file. When creating the
        cache, writing must be allowed, but when reading from multiple
        processes, writing should not be allowed.
    cache_filename : str
        The name of the HDF5 file to store the cache in.
    compression : str or int, optional
        Compression to use for the HDF5 file. Valid values are "gzip", "lzf", "szip", or an integer 0-9 (for gzip compression level).
        See h5py documentation for details. Example: compression="gzip" or compression=4.
    *args
    **kwargs
        Forwarded to DynamicItem constructor
    """

    def __init__(
        self,
        cache_location,
        file_mode="a",
        cache_filename="cache.hdf5",
        compression=None,
        *args,
        **kwargs,
    ):
        super().__init__(cache_location, *args, **kwargs)

        # Open connection to HDF5 file
        self.file_mode = file_mode
        self.compression = compression
        # cache_location in the parent is a directory; keep filename separate.
        self.cache_filename = Path(cache_filename)
        self.hdf5file = h5py.File(self.hdf5_path, file_mode)

    def _is_cached(self, uid):
        """Test whether uid is cached."""
        return uid in self.hdf5file

    def _load(self, uid):
        """Load result from cache"""
        return self.hdf5file[uid][:]

    def _cache(self, result, uid):
        """Save the result to the cache"""
        self.hdf5file.create_dataset(uid, data=result)

    @property
    def hdf5_path(self):
        """Compute the full path to the HDF5 file from cache_location and cache_filename."""
        return Path(self.cache_location) / self.cache_filename

    def __getstate__(self):
        """Get the state of the object for pickling. In case of pickling, we need to close the HDF5 file."""
        state = self.__dict__.copy()
        # h5py objects can't be pickled; drop the live handle
        h5_handle = state.pop("hdf5file", None)
        if h5_handle is not None:
            h5_handle.close()
        return state

    def __setstate__(self, state):
        """Set the state of the object for unpickling."""
        self.__dict__ = state
        # Reopen the file lazily in the same mode using the directory and filename.
        self.hdf5file = h5py.File(self.hdf5_path, self.file_mode)

    def change_file_mode(self, new_file_mode):
        """Change mode that the hdf5 file is opened with. Usually used to convert from
        writing format (building cache) to read-only format (multi-process loading)."""
        self.hdf5file.close()
        self.file_mode = new_file_mode
        self.hdf5file = h5py.File(self.hdf5_path, new_file_mode)

    @classmethod
    def cache(
        cls,
        cache_location,
        file_mode="a",
        cache_filename="cache.hdf5",
        compression=None,
    ):
        """Decorator which takes a DynamicItem and creates a CachedHDF5DynamicItem

        Arguments
        ---------
        cache_location : os.PathLike
            Storage folder for containing HDF5 cached output file.
        file_mode : str
            The mode to use when opening the HDF5 file. When creating the
            cache, writing must be allowed, but when reading from multiple
            processes, writing should not be allowed.
        cache_filename : str
            The name of the HDF5 file to store the cache in.
        compression : str
            The compression algorithm to use for the HDF5 file.

        Example
        -------
        >>> import os, numpy
        >>> from speechbrain.utils.data_pipeline import takes, provides
        >>> tempdir = getfixture("tmpdir")
        >>> @CachedHDF5DynamicItem.cache(tempdir)
        ... @takes("id", "text")
        ... @provides("tokenized")
        ... def count_to(id, limit):
        ...     return numpy.arange(limit)
        >>> "utt_id" in count_to.hdf5file
        False
        >>> count_to("utt_id", 5)
        array([0, 1, 2, 3, 4])
        >>> "utt_id" in count_to.hdf5file
        True
        >>> # The output shouldn't change on the second call
        >>> count_to("utt_id", 5)
        array([0, 1, 2, 3, 4])
        >>> # NOTE: NO INVALID CACHE DETECTION
        >>> count_to("utt_id", 10)
        array([0, 1, 2, 3, 4])
        """

        def decorator(obj):
            """Decorator definition."""
            if not isinstance(obj, DynamicItem):
                raise ValueError("Can only cache a DynamicItem")
            return cls(
                cache_location,
                file_mode,
                cache_filename=cache_filename,
                compression=compression,
                takes=obj.takes,
                func=obj.func,
                provides=obj.provides,
            )

        return decorator
