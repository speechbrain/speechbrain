"""This file contains utilities for preprocessing of features, particularly
using neural models

Authors
 * Artem Ploujnikov 2025
"""

import csv
import math
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import numpy as np
import torch
from tqdm.auto import tqdm

from speechbrain.dataio.batch import undo_batch
from speechbrain.dataio.dataloader import make_dataloader
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.utils.data_pipeline import DataPipeline, provides, takes
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)

has_h5 = False
try:
    import h5py

    has_h5 = True
except ImportError:
    logger.warning("h5py not installed, h5 is not supported")


class FeatureExtractor:
    """A utility class for pipeline-based feature extraction

    Arguments
    ---------
    src_keys : list
        The keys from the source dataset
    storage : Storage | str
        The storage provider instance of a storage URI
        Example: numpy:/some/path
    storage_opts : dict
        Storage options (optional)
    id_key: str
        The key within the batch that will be used as an identifier
    device : str|torch.device
        The device on which operations will be run
    dataloader_opts : dict
        Parameters to be passed to the data loader (batch size, etc)
    dynamic_items : list
        Configuration for the dynamic items produced when fetching an example.
        List of DynamicItems or dicts with the format:
            func: <callable> # To be called
            takes: <list> # key or list of keys of args this takes
            provides: key # key or list of keys that this provides
    description : str
        The description for the progress indicator
    """

    def __init__(
        self,
        src_keys,
        storage,
        storage_opts=None,
        id_key="id",
        device="cpu",
        dataloader_opts=None,
        dynamic_items=None,
        description=None,
    ):
        if not dataloader_opts:
            dataloader_opts = {}
        self.id_key = id_key
        self.src_keys = src_keys
        self.storage = resolve_storage(storage, storage_opts, mode="w")
        self.id_key = id_key
        self.dataloader_opts = dataloader_opts
        self.device = device
        self.pipeline = DataPipeline(
            static_data_keys=src_keys, dynamic_items=dynamic_items or []
        )
        self.description = description

    def extract(self, dataset, data=None):
        """Runs the preprocessing operation

        Arguments
        ---------
        dataset : dict|speechbrain.dataio.dataset.DynamicItemDataset
            the dataset to be saved
        data : dict
            the raw data dictionary (to update with extra features)
        """
        if isinstance(dataset, dict):
            dataset = DynamicItemDataset(dataset)
        dataset.set_output_keys(self.src_keys + [self.id_key])
        try:
            dataloader = make_dataloader(dataset, **self.dataloader_opts)
            batch_size = self.dataloader_opts.get("batch_size", 1)
            batch_count = int(math.ceil(len(dataset) / batch_size))
            for batch in tqdm(
                dataloader, total=batch_count, desc=self.description
            ):
                batch = batch.to(self.device)
                self.process_batch(batch, data)
        finally:
            self.storage.close()

    def process_batch(self, batch, data):
        """Processes a batch of data

        Arguments
        ---------
        batch: speechbrain.dataio.batch.PaddedBatch
            a batch
        data : dict
            the raw data dictionary (to update with extra features)
        """
        batch_dict = batch.as_dict()
        ids = batch_dict[self.id_key]
        features = self.pipeline.compute_outputs(batch_dict)

        for item_id, item_features in zip(ids, undo_batch(features)):
            for key, value in item_features.items():
                self.storage.save(item_id, key, value)
            self._add_inline_features(item_id, item_features, data)

    def _add_inline_features(self, item_id, item_features, data):
        item_data = data.get(item_id) if data is not None else None
        for key in self.inline_keys:
            if item_data is not None:
                item_data[key] = item_features[key]
            del item_features[key]
        return item_features

    def add_dynamic_item(self, func, takes=None, provides=None):
        """Adds a dynamic item to be output

        Two calling conventions. For DynamicItem objects, just use:
        add_dynamic_item(dynamic_item).
        But otherwise, should use:
        add_dynamic_item(func, takes, provides).

        See `speechbrain.utils.data_pipeline`.

        Arguments
        ---------
        func : callable, DynamicItem
            If a DynamicItem is given, adds that directly. Otherwise a
            DynamicItem is created, and this specifies the callable to use. If
            a generator function is given, then create a GeneratorDynamicItem.
            Otherwise creates a normal DynamicItem.
        takes : list, str
            List of keys. When func is called, each key is resolved to
            either an entry in the data or the output of another dynamic_item.
            The func is then called with these as positional arguments,
            in the same order as specified here.
            A single arg can be given directly.
        provides : str
            Unique key or keys that this provides.
        """
        self.pipeline.add_dynamic_item(func, takes, provides)

    def set_output_features(self, keys, inline_keys=None):
        """Sets the features to be output

        Arguments
        ---------
        keys : list
            Keys to be output / saved
        inline_keys : list, optional
            The keys to be used inline (added to the data dictionary
            rather than saved in flies)"""
        self.inline_keys = inline_keys or []
        self.pipeline.set_output_keys(keys + self.inline_keys)


class Storage:
    handlers = {}

    """A base class for storage providers"""

    def save(self, data_id, key, data):
        """Saves a data element

        Arguments
        ---------
        data_id : object
            The identifier of the data sample (usually a string or an integer)
        key : str
            The key, identifying the type of data being saved
        data : object
            The data to be saved - what data is allowed will
            depend on the storage
        """
        raise NotImplementedError()

    def load(self, data_id, key):  # noqa: DOC202
        """Loads a data element

        Arguments
        ---------
        data_id : object
            The identifier of the data sample (usually a string or an integer)
        key : str
            The key, identifying the type of data being saved

        Returns
        -------
        data : object
            The saved data
        """
        raise NotImplementedError()

    def flush(self):
        """Saves any pre-cahced data to permanent storage"""
        pass

    def close(self):
        """Closes any open resources"""
        pass

    @classmethod
    def from_uri(cls, uri, mode, options):
        """Instantiates a new storage from a URI

        Arguments
        ---------
        uri: str
            A URI
        mode : str
            The access mode: "r" or "w"
        options : dict
            Additional storage-specific options

        Returns
        -------
        storage : Storage
            The storage instance
        """
        if options is None:
            options = {}
        parse_result = urlparse(uri)
        storage_cls = cls.handlers.get(parse_result.scheme)
        options = options.get(parse_result.scheme, options)
        if storage_cls is None:
            raise ValueError(f"Invalid storage URI {uri}. Unsupported scheme")
        return storage_cls.from_uri(uri, mode, options)

    @classmethod
    def register(cls, scheme, storage_cls):
        """Registers a new storage class"""
        cls.handlers[scheme] = storage_cls


def storage_uri_scheme(scheme):
    """A convenience decorator associating a storage provider with
    a URI scheme

    Arguments
    ---------
    scheme : str
        The URI scheme

    Returns
    -------
    f: callable
        The wrapper function
    """

    def f(cls):
        Storage.register(scheme, cls)
        return cls

    return f


@storage_uri_scheme("numpy")
class NumpyStorage(Storage):
    """A simple shareded Numpy-based storage implementation

    Arguments
    ---------
    path : str
        The storage path (a folder)
    mode : str
        The access mode ("r" or "w")
    shard_size : int
        The number of data samples per shard
    async_save : bool
        Whether to save files asynchronously.
        This can be useful on compute clusters where
        I/O speed is a bottleneck
    async_save_concurrency : int
        The number of concurrent processes for I/O
    """

    def __init__(
        self,
        path,
        mode="r",
        shard_size=1000,
        async_save=False,
        async_save_concurrency=5,
    ):
        self.path = Path(path).expanduser()
        self.manifest_file_name = self.path / "manifest.csv"
        self.shard_size = shard_size
        self.manifest = {}
        self.mode = mode
        self.current_shard = {}
        self.current_read_ids = None
        self.current_read_shard = None
        if self.manifest_file_name.exists():
            self._read_manifest()
        self.async_save = async_save
        self.async_save_concurrency = async_save_concurrency
        self._async_save_futures = {}
        if self.async_save:
            self.save_executor = ThreadPoolExecutor(
                max_workers=self.async_save_concurrency
            )
        if mode == "w":
            self.path.mkdir(exist_ok=True, parents=True)
            self._open_manifest()
        self.is_closed = False

    def _read_manifest(self):
        with open(
            self.manifest_file_name, "r", encoding="utf-8"
        ) as manifest_file:
            reader = csv.reader(manifest_file)
            for data_id, shard in reader:
                self.manifest[data_id] = shard

    def _open_manifest(self):
        self._manifest_file = open(
            self.manifest_file_name, "a+", encoding="utf-8"
        )
        self._manifest_writer = csv.writer(self._manifest_file)

    def save(self, data_id, key, data):
        """Saves a data element

        Arguments
        ---------
        data_id : object
            The identifier of the data sample (usually a string or an integer)
        key : str
            The key, identifying the type of data being saved
        data : object
            The data to be saved - what data is allowed will
            depend on the storage
        """
        if self.mode == "r":
            raise ValueError("Read-only file")
        if (
            len(self.current_shard) >= self.shard_size
            and data_id not in self.current_shard
        ):
            self.flush()
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
        if data_id not in self.current_shard:
            self.current_shard[data_id] = {}
        self.current_shard[data_id][key] = data

    def load(self, data_id, key):
        """Loads a data element

        Arguments
        ---------
        data_id : object
            The identifier of the data sample (usually a string or an integer)
        key : str
            The key, identifying the type of data being saved

        Returns
        -------
        data : object
            The saved data
        """
        if (
            self.current_read_ids is not None
            and data_id in self.current_read_ids
        ):
            return self.current_read_shard[data_id].get(key)
        if data_id in self.current_shard and data_id not in self.manifest:
            # Being written now
            return self.current_shard[data_id][key]

        if data_id not in self.manifest:
            # Does not exist
            return None

        file_name = self.manifest[data_id]
        shard_data = np.load(self.path / file_name)
        data_ids = shard_data["data_ids"]
        keys = shard_data["keys"]
        self.current_read_ids = data_ids
        shard = {}
        for read_data_id in data_ids:
            shard[read_data_id] = {}
            for read_key in keys:
                shard_key = f"{read_data_id}__{read_key}"
                if shard_key in shard_data:
                    shard[read_data_id][read_key] = shard_data[shard_key]
        self.current_read_shard = shard
        result = shard[data_id].get(key)
        if result is not None:
            result = torch.from_numpy(result)
        return result

    def flush(self):
        """Saves any pre-cahced data to permanent storage"""
        self._save_current()
        self._manifest_file.flush()
        if self.async_save:
            self._wait()

    def _save_current(self):
        """Saves the current shard to disk"""
        if not self.current_shard:
            return
        data_ids = list(self.current_shard.keys())
        keys = set([])
        out_data = {}
        file_name = f"{str(uuid4())}.npz"
        manifest_rows = []
        for data_id, item_data in self.current_shard.items():
            out_data[data_id] = {}
            for key, key_data in item_data.items():
                keys.add(key)
                out_data[f"{data_id}__{key}"] = key_data
                self.manifest[data_id] = file_name
            manifest_rows.append((data_id, file_name))
        self._manifest_writer.writerows(manifest_rows)
        self._save_file(
            self.path / file_name,
            data_ids=data_ids,
            keys=list(keys),
            **out_data,
        )
        self.current_shard = {}

    def _save_file(self, file_name, **data):
        """Saves a numpy file (synchronously or asynchronously, depending on settings)"""
        if self.async_save:
            future = self.save_executor.submit(np.savez, file_name, **data)
            self._async_save_futures[file_name] = future
        else:
            np.savez(file_name, **data)

    def _wait(self):
        wait(list(self._async_save_futures.values()))
        for file_name, future in self._async_save_futures.items():
            exc = future.exception()
            if exc is not None:
                exc_info = (type(exc), exc, exc.__traceback__)
                logger.warn(
                    "Saving extracted features for %s could not be completed: %s",
                    file_name,
                    str(exc),
                    exc_info=exc_info,
                )

    def close(self):
        """Closes any open resources"""
        if self.mode == "w":
            self.flush()
            if self.async_save:
                self.save_executor.shutdown(wait=True)
            self._manifest_file.close()
        self.is_closed = True

    @classmethod
    def from_uri(cls, uri, mode, options=None):
        """Instantiates a new storage from a URI

        Arguments
        ---------
        uri: str
            A URI
        mode : str
            The access mode: "r" for reading or "w" for writing
        options : dict
            Additional storage-specific options

        Returns
        -------
        storage : Storage
            The storage instance
        """
        if isinstance(uri, str):
            uri = urlparse(uri)
        query = parse_qs(uri.query)
        if options is None:
            options = {}
        if "shard_size" in query:
            options["shard_size"] = int(query["shard_size"])
        if "async_save" in query:
            options["async_save"] = _to_bool(query["async_save"])
        if "async_save_concurrency" in query:
            options["async_save_concurrency"] = int(
                query["async_save_concurrency"]
            )
        return cls(path=uri.path, mode=mode, **options)


@storage_uri_scheme("h5")
class H5Storage(Storage):
    """A storage wrapper for h5py

    Arguments
    ---------
    path : str | path-like
        The storage path (a file or a directory)
    mode : str
        The access mode ("r" or "w")
    write_batch_size : int
        The batch size
    compression : str
        the compression mode
    """

    def __init__(self, path, mode, write_batch_size=10, compression=None):
        if not has_h5:
            raise ValueError("h5 not supported, please install h5py")
        self.mode = mode
        path = Path(path)
        if not path.name.endswith(".h5"):
            path = path / "data.h5"
        self.path = path
        self.compression = compression
        self.data = {}
        self.data_ids = {}
        self.data_index = {}
        self.write_batch_size = write_batch_size
        self.file = None

    def save(self, data_id, key, data):
        """Saves a data element

        Arguments
        ---------
        data_id : object
            The identifier of the data sample (usually a string or an integer)
        key : str
            The key, identifying the type of data being saved
        data : object
            The data to be saved - what data is allowed will
            depend on the storage
        """
        if self.mode != "w":
            raise ValueError("The file was not opened in write mode")
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
        if key not in self.data:
            self.data[key] = []
            self.data_ids[key] = []
        elif len(self.data[key]) >= self.write_batch_size:
            self._flush_key(key)
        self.data[key].append(data)
        self.data_ids[key].append(data_id)

    def load(self, data_id, key):  # noqa: DOC202
        """Loads a data element

        Arguments
        ---------
        data_id : object
            The identifier of the data sample (usually a string or an integer)
        key : str
            The key, identifying the type of data being saved

        Returns
        -------
        data : object
            The saved data
        """
        if self.mode != "r":
            raise ValueError("The file was not opened in read mode")
        self._ensure_open()
        key_data = f"{key}_data"
        if key_data not in self.file:
            return None
        if key not in self.data_index:
            self._read_index(key)
        idx = self.data_index[key].get(data_id)
        if idx is None:
            return None
        data_flat = self.file[key_data][idx]
        key_shapes = f"{key}_shapes"
        shape = self.file[key_shapes][idx]
        result = data_flat.reshape(shape)
        return torch.from_numpy(result)

    def flush(self):
        """Saves any pre-cahced data to permanent storage"""
        if self.file is not None:
            keys = list(self.data.keys())
            for key in keys:
                self._flush_key(key)

    def close(self):
        """Closes any open resources"""
        if self.file is not None:
            self.flush()
            self.file.close()

    def _ensure_open(self):
        if self.file is None:
            self.path.parent.mkdir(exist_ok=True, parents=True)
            self.file = h5py.File(self.path, self.mode)

    def _read_index(self, key):
        dataset_idx = self.file[f"{key}_idx"]
        self.data_index[key] = {
            data_id.decode("utf-8"): idx
            for idx, data_id in enumerate(dataset_idx)
        }

    def _flush_key(self, key):
        self._ensure_open()
        key_data = f"{key}_data"
        key_shapes = f"{key}_shapes"
        key_idx = f"{key}_idx"
        batch = self.data[key]
        if not batch:
            return  # Nothing to do
        data = batch[0]
        dim = data.ndim
        if key_data not in self.file:
            dt = h5py.vlen_dtype(data.dtype)
            dataset = self.file.create_dataset(
                key_data,
                shape=(0,),
                maxshape=(None,),
                dtype=dt,
                compression=self.compression,
            )
            dataset_shapes = self.file.create_dataset(
                key_shapes, shape=(0, dim), maxshape=(None, dim), dtype=int
            )
            dataset_idx = self.file.create_dataset(
                key_idx,
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
        else:
            dataset = self.file[key_data]
            dataset_shapes = self.file[key_shapes]
            dataset_idx = self.file[key_idx]
        offset = len(dataset)
        end = offset + len(batch)
        dataset.resize(size=(end,))
        dataset_shapes.resize(size=(end, dim))
        dataset_idx.resize(size=(end,))
        data_shapes = [item.shape for item in batch]
        batch_flat = [item.flatten() for item in batch]
        dataset[offset:end] = batch_flat
        dataset_shapes[offset:end] = data_shapes
        dataset_idx[offset:end] = self.data_ids[key]
        self.data[key], self.data_ids[key] = [], []

    @classmethod
    def from_uri(cls, uri, mode, options=None):
        """Instantiates a new storage from a URI

        Arguments
        ---------
        uri: str
            A URI
        mode : str
            The access mode: "r" or "w"
        options : dict
            Additional storage-specific options

        Returns
        -------
        storage : Storage
            The storage instance
        """
        if isinstance(uri, str):
            uri = urlparse(uri)
        query = parse_qs(uri.query)
        if options is None:
            options = {}
        if "write_batch_size" in query:
            options["write_batch_size"] = int(query["write_batch_size"])
        if "compression" in query:
            options["compression"] = _to_bool(query["compression"])
        return cls(path=uri.path, mode=mode, **options)


_bool_map = {
    "0": False,
    "1": True,
    "true": True,
    "false": False,
}


def _to_bool(value, default=False):
    bool_value = default
    if value is not None:
        bool_value = _bool_map.get(value)
        if bool_value is None:
            raise ValueError(f"Invalid Boolean {bool_value}")
    return bool_value


def resolve_storage(storage, storage_opts=None, mode="r"):
    """A helper function that returns the storage as is if it is a Storage instance
    or resolves it from a URL if one is passed

    Arguments
    ---------
    storage : Storage | str
        The storage provider instance of a storage URI
        Example: numpy:/some/path
    storage_opts : dict
        Storage options (optional)
    mode : str
        The access mode: "r" for reading or "w" for writing

    Returns
    -------
    result : Storage
        a storage instance
    """
    if isinstance(storage, Storage):
        result = storage
    elif isinstance(storage, str):
        result = Storage.from_uri(storage, mode, storage_opts)
    else:
        raise ValueError(f"Invalid storage: {storage}")
    return result


@contextmanager
def prepared_features(datasets, keys, storage, storage_opts=None, id_key="id"):
    """Adds previously extracted features

    Argumengts
    ----------
    datasets : speechbrain.dataio.dataset.DynamicItemDataset | dict | enumerable
        A dataset
    keys : list
        A list of keys to be made available. Please note that
        this does not automatically load them into batches - this will
        be controlled by set_output_keys
    storage : Storage | str
        A storage instance or URI
    storage_opts : dict
        Storage options
    id_key : str
        The key to be used for the ID (optional)
    """
    storage = resolve_storage(storage, storage_opts)
    if isinstance(datasets, DynamicItemDataset):
        _add_prepared_features(datasets, keys, storage, id_key)
    elif isinstance(datasets, dict):
        for dataset in datasets.values():
            _add_prepared_features(dataset, keys, storage, id_key)
    else:
        for dataset in datasets:
            _add_prepared_features(dataset, keys, storage, id_key)
    yield datasets
    storage.close()


def _add_prepared_features(dataset, keys, storage, id_key):
    def storage_pipeline(key, storage, data_id):
        data = storage.load(data_id, key)
        return data

    for key in keys:
        key_storage_pipeline = partial(storage_pipeline, key, storage)
        key_storage_pipeline = takes(id_key)(key_storage_pipeline)
        key_storage_pipeline = provides(key)(key_storage_pipeline)
        dataset.add_dynamic_item(key_storage_pipeline)
