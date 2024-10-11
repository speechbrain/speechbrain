"""This file contains utilities for preprocessing of features, particularly
using neural models

Authors
 * Artem Ploujnikov 2023
"""

import concurrent.futures
import logging
import math
import os
import re
import shutil
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

import speechbrain as sb
from speechbrain.dataio.batch import undo_batch
from speechbrain.dataio.dataloader import make_dataloader
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.utils.data_pipeline import DataPipeline

logger = logging.getLogger(__name__)

variable_finder = re.compile(r"\$([\w.]+)")


class FeatureExtractor:
    """A utility class for pipeline-based feature extraction

    Arguments
    ---------
    save_path : str|path-like
        The path where the preprocessed features will be saved

    src_keys : list
        The keys from the source dataset

    id_key: str
        The key within the batch that will be used as an identifier

    save_format : str|callable
        The format in which prepared features will be saved

    device : str|torch.device
        The device on which operations will be run

    dataloader_opts : dict
        Parameters to be passed to the data loader (batch size, etc)

    dynamic_items : list
        Configuration for the dynamic items produced when fetching an example.
        List of DynamicItems or dicts with the format::
            func: <callable> # To be called
            takes: <list> # key or list of keys of args this takes
            provides: key # key or list of keys that this provides

    description : str
        The description for the progress indicator

    async_save : bool
        Where or not to save files asynchronously. This can be useful
        on clusters with slow shared filesystems

    async_save_batch_size : int
        The batch size to use when saving asynchronously.
        Ignored if async_save == False

    async_save_concurrency : int
        The concurrency level (number of simultaneous operations)
        when saving asynchronously.

        Ignored if async_save == False
    """

    def __init__(
        self,
        save_path,
        src_keys,
        id_key="id",
        save_format="npy",
        device="cpu",
        dataloader_opts=None,
        dynamic_items=None,
        description=None,
        async_save=True,
        async_save_batch_size=16,
        async_save_concurrency=8,
    ):
        if not dataloader_opts:
            dataloader_opts = {}
        self.id_key = id_key
        self.save_path = save_path
        self.src_keys = src_keys
        self.id_key = id_key
        self.dataloader_opts = dataloader_opts
        if callable(save_format):
            self.save_fn = save_format
        elif save_format in SAVE_FORMATS:
            self.save_fn = SAVE_FORMATS[save_format]
        else:
            raise ValueError(f"Unsupported save_format: {save_format}")
        self.device = device
        self.pipeline = DataPipeline(
            static_data_keys=src_keys, dynamic_items=dynamic_items or []
        )
        self.async_save = async_save
        self._async_save_futures = {}
        self.async_save_batch_size = async_save_batch_size
        self.async_save_concurrency = async_save_concurrency
        self.save_executor = None
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
        if self.async_save:
            self._init_async_save()
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
            if self.async_save:
                self._finish_async_save()

    def _init_async_save(self):
        self.save_executor = ThreadPoolExecutor(
            max_workers=self.async_save_concurrency
        )

    def _finish_async_save(self):
        try:
            self.flush()
        finally:
            self.save_executor.shutdown()
            self.save_executor = None

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

        for idx, (item_id, item_features) in enumerate(
            zip(ids, undo_batch(features)), start=1
        ):
            self._add_inline_features(item_id, item_features, data)
            if self.async_save:
                future = self.save_executor.submit(
                    self.save_fn,
                    item_id,
                    item_features,
                    save_path=self.save_path,
                )
                self._async_save_futures[item_id] = future
                if idx % self.async_save_batch_size == 0:
                    self.flush()
            else:
                self.save_fn(item_id, item_features, save_path=self.save_path)

    def flush(self):
        """Flushes all futures that have been accumulated"""
        concurrent.futures.wait(self._async_save_futures.values())
        for item_id, future in self._async_save_futures.items():
            exc = future.exception()
            if exc is not None:
                exc_info = (type(exc), exc, exc.__traceback__)
                logger.warn(
                    "Saving extracted features for %s could not be completed: %s",
                    item_id,
                    str(exc),
                    exc_info=exc_info,
                )
        self._async_save_futures.clear()

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


def save_pt(item_id, data, save_path):
    """Saves the data in the PyTorch format (one file per sample)

    Arguments
    ---------
    item_id: str
        the ID of the item to be saved

    data: dict
        the data to be saved

    save_path: path-like
        the destination path
    """
    file_path = save_path / f"{item_id}.pt"
    torch.save(data, file_path)


def save_npy(item_id, data, save_path):
    """Saves the data in numpy format (one file per sample per feature)

    Arguments
    ---------
    item_id: str
        the ID of the item to be saved

    data: dict
        the data to be saved

    save_path: path-like
        the destination path
    """
    for key, value in data.items():
        file_path = save_path / f"{key}_{item_id}.npy"
        np.save(file_path, value.detach().cpu().numpy())


def load_pt(save_path, item_id, features):
    """Loads a PyTorch pickled file

    Arguments
    ---------
    save_path : path-like
        The storage path
    item_id : object
        The item identifier
    features : enumerable
        Not used

    Returns
    -------
    result : object
        the contents of the file
    """
    file_path = save_path / f"{item_id}.pt"
    return torch.load(file_path)


def load_npy(save_path, item_id, features):
    """Loads a raw NumPy array

    Arguments
    ---------
    save_path : path-like
        The storage path
    item_id : object
        The item identifier
    features : enumerable
        The features to be loaded

    Returns
    -------
    result : any
        The contents
    """
    return {
        key: np.load(save_path / f"{key}_{item_id}.npy") for key in features
    }


SAVE_FORMATS = {
    "pt": save_pt,
    "npy": save_npy,
}

LOAD_FORMATS = {
    "pt": load_pt,
    "npy": load_npy,
}


def add_prepared_features(
    dataset, save_path, features, id_key="id", save_format="npy"
):
    """Adds prepared features to a pipeline

    Arguments
    ---------
    dataset : speechbrains.dataio.dataset.DynamicItemDataset
        a dataset
    save_path : str|path-like
        the path where prepared features are saved
    features : list
        the list of features to be added
    id_key : str
        the ID of the pipeline elements used as the item ID
    save_format : str | callable
        One of the known formats (pt or npy) or a custom
        function to load prepared features for a data sample"""
    load_fn = LOAD_FORMATS.get(save_format, save_format)
    save_path = Path(save_path)

    @sb.utils.data_pipeline.takes(id_key)
    @sb.utils.data_pipeline.provides(*features)
    def prepared_features_pipeline(item_id):
        """A pipeline function that provides the features defined with
        registered loaders

        Arguments
        ---------
        item_id : object
            The item identifier

        Yields
        ------
        feature : any
            The features being extracted
        """
        data = load_fn(save_path, item_id, features)
        for feature in features:
            yield data[feature]

    dataset.add_dynamic_item(prepared_features_pipeline)


DEFAULT_PATTERNS = ["*.csv", "*.json", "features", "*_prepare.pkl"]


class Freezer:
    """A utility class that helps archive and restore prepared
    data. This is particularly useful on compute clusters where
    preparation needs to be done on non-permanent storage

    Arguments
    ---------
    save_path : str|path-like
        the path where prepared data is saved
    archive_path : str|path-like
        the path to the archive
    patterns : enumerable
        a list of glob patterns with prepared files
    """

    def __init__(self, save_path, archive_path, patterns=None):
        self.save_path = Path(save_path)
        self.archive_path = Path(archive_path) if archive_path else None
        self.patterns = patterns or DEFAULT_PATTERNS

    def freeze(self):
        """Archives pretrained files"""
        if self.archive_path is None:
            logger.info("Prepared data archiving is unavailable")
            return
        if self.archive_path.exists():
            logger.info(
                "The prepared dataset has already been archived in %s",
                self.archive_path,
            )
            return
        file_names = self.get_files()
        logger.info(
            "Archiving %d files from the prepared dataset in %s",
            len(file_names),
            self.archive_path,
        )
        mode = self._get_archive_mode("w")
        tmp_archive_path = self.save_path / self.archive_path.name
        logger.info("Creating a temporary archive: %s", tmp_archive_path)
        with tarfile.open(tmp_archive_path, mode) as tar_file:
            for file_name in file_names:
                tar_file.add(
                    name=file_name,
                    arcname=file_name.relative_to(self.save_path),
                )
        logger.info("Copying %s to %s", tmp_archive_path, self.archive_path)
        shutil.copy(tmp_archive_path, self.archive_path)
        logger.info("Done copying, removing %s", tmp_archive_path)
        os.remove(tmp_archive_path)

    def _get_archive_mode(self, mode):
        """Adds a suffix to the archive mode"""
        if self.archive_path.name.endswith(".gz"):
            mode = f"{mode}:gz"
        return mode

    def unfreeze(self):
        """Unarchives pretrained files into save_path

        Returns
        -------
        result: bool
            True if the archive exists and has been unpacked,
            False otherwise."""
        if self.archive_path is None:
            logger.info("Prepared dataset freezing is disabled")
            result = False
        elif self.archive_path.exists():
            logger.info(
                "Unpacking prepared dataset %s into %s",
                self.archive_path,
                self.save_path,
            )
            mode = self._get_archive_mode("r")
            with tarfile.open(self.archive_path, mode) as tar_file:
                tar_file.extractall(self.save_path)
            logger.info("Prepared dataset unpacked")
            result = True
        else:
            logger.info(
                "No frozen prepared dataset exists in %s", self.archive_path
            )
            result = False
        return result

    def get_files(self):
        """Returns the list of prepared files available
        to be archived

        Returns
        -------
        result: list
            A list of file names"""
        return [
            file_name
            for pattern in self.patterns
            for file_name in self.save_path.glob(pattern)
        ]

    def __enter__(self):
        self.unfreeze()

    def __exit__(self, exc_type, exc_value, traceback):
        self.freeze()
