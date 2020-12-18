"""Datasets load individual data points (examples)

Authors
  * Aku Rouhe 2020
  * Samuele Cornell 2020
  * Peter Plantinga 2020
"""

import contextlib
from torch.utils.data import Dataset
from speechbrain.data_io.data_io import load_data_json, load_data_csv
import logging

logger = logging.getLogger(__name__)


class TransformDataset(Dataset):
    """Dataset that reads, wrangles and produces dicts

    Arguments
    ---------
    data : dict
        Dictionary containing single data points (e.g. utterances).
    item_transforms : dict, optional
        Configuration for the dynamic items produced when fetchin an example.
        Nested dict with the format (in YAML notation):
        <key>: func to be called on data in key
        <key2>: func to be called on data in key2
        functions should return a dict with all items that need to be accessed
    """

    def __init__(
        self, data, output_keys, item_transforms=None,
    ):
        self.data = data
        self.data_ids = list(self.data.keys())
        static_keys = self.data[self.data_ids[0]]
        if "id" in static_keys:
            raise ValueError("The key 'id' is reserved for the data point id.")

        self.item_transforms = item_transforms or {}
        self.output_keys = output_keys

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, index):
        data_id = self.data_ids[index]
        data_point = self.data[data_id]
        data_point["id"] = data_id

        output_keys_computed = set(self.output_keys) - set(data_point.keys())

        # Apply user-defined transform
        for key in self.item_transforms:

            # Stop applying transforms if we've computed all outputs
            if not output_keys_computed:
                break

            # Compute next transform
            transformed = self.item_transforms[key](data_point[key])
            output_keys_computed -= set(transformed.keys())
            data_point.update(transformed)

        return {k: v for k, v in data_point.items() if k in self.output_keys}

    @contextlib.contextmanager
    def output_keys_as(self, keys):
        """Context manager to temporarily set output keys

        NOTE
        ----
        Not thread-safe. While in this context manager, the output keys
        are affected for any call.
        """
        saved_keys = self.output_keys
        self.output_keys = keys
        yield self
        self.output_keys = saved_keys

    @classmethod
    def from_json(
        cls, json_path, output_keys, replacements={}, item_transforms=None
    ):
        """Load a data prep JSON file and create a Dataset based on it."""
        data = load_data_json(json_path, replacements)
        return cls(data, output_keys, item_transforms)

    @classmethod
    def from_csv(
        cls, csv_path, output_keys, replacements={}, item_transforms=None
    ):
        """Load a data prep CSV file and create a Dataset based on it."""
        data = load_data_csv(csv_path, replacements)
        return cls(data, output_keys, item_transforms)
