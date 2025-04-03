"""This file contains utilities for preprocessing of features, particularly
using neural models

Authors
 * Artem Ploujnikov 2025
"""

import csv
import logging
import math
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import numpy as np
from tqdm.auto import tqdm

from speechbrain.dataio.batch import undo_batch
from speechbrain.dataio.dataloader import make_dataloader
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.utils.data_pipeline import DataPipeline

logger = logging.getLogger(__name__)


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
        if isinstance(storage, Storage):
            self.storage = storage
        elif isinstance(storage, str):
            self.storage = Storage.from_uri(
                storage, mode="w", options=storage_opts
            )
        else:
            raise ValueError(f"Invalid storage: {storage}")
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
    """A simple shareded Numpy-based storage implementation"""

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
        return shard[data_id].get(key)

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
        if "shard_size" in query:
            options["shard_size"] = int(query["shard_size"])
        if "async_save" in query:
            options["async_save"] = _to_bool(query["async_save"])
        if "async_save_concurrency" in query:
            options["async_save_concurrency"] = int(
                query["async_save_concurrency"]
            )
        path = uri.path
        if path.startswith("/~"):
            path = path.lstrip("/")
        return cls(path=path, mode=mode, **options)


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
