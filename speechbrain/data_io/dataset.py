"""Datasets load individual data points (examples)

Authors
  * Samuele Cornell 2020
  * Aku Rouhe 2020
"""

from torch.utils.data import Dataset
from speechbrain.utils.data_pipeline import DataPipeline
import logging

logger = logging.getLogger(__name__)


class DynamicItemDataset(Dataset):
    """Dataset that reads, wrangles and produces dicts

    Each data point dict provides some items (by key), for example a path to a
    wavefile with the key "wav_file". When a data point is fetched from this
    Dataset, more items are produced dynamically, based on pre-existing items
    and other dynamic created items. For example, a dynamic item could take the
    wavfile path and load the audio from disk.

    The dynamic items can depend on other dynamic items: a suitable evaluation
    order is used automatically,  as long as there are no circular dependencies.

    A specified list of keys is collected in the output dict. These can be items
    in the original data or dynamic items. If some dynamic items are not
    requested, nor depended on by other requested items, they won't be computed.
    So for example if a user simply wants to iterate over the text, the
    time-consuming audio loading can be skipped.

    About the format:
    Takes a dict of dicts as the collection of data points to read/wrangle.
    The top level keys are data point IDs.
    Each data point (example) dict should have the same keys, corresponding to
    different items in that data point.

    Altogether the data collection could look like this:
    >>> data = {
    ...  "spk1utt1": {
    ...      "wav_file": "/path/to/spk1utt1.wav",
    ...      "text": ["hello world"],
    ...      "speaker": "spk1",
    ...      },
    ...  "spk1utt2": {
    ...      "wav_file": "/path/to/spk1utt2.wav",
    ...      "text": ["how are you world"],
    ...      "speaker": "spk1",
    ...      }
    ... }

    NOTE
    ----
        The top level key, the data point id, is implicitly added as an item
        in the data point, with the key "id"

    Each dynamic item is configured by three things: a key, a func, and a list
    of argkeys. The key should be unique among all the items (dynamic or not) in
    each data point. The func is any callable, and it returns the dynamic item's
    value. The callable is called with the values of other items as specified
    by the argkeys list (as positional args, passed in the order specified by
    argkeys).

    The dynamic_items configuration could look like this:
    >>> import torch
    >>> dynamic_items = {
    ...  "wav": {
    ...      "func": lambda l: torch.Tensor(l),
    ...      "argkeys": ["wav_loaded"] },
    ...  "wav_loaded": {
    ...      "func": lambda path: [ord(c)/100 for c in path],  # Fake "loading"
    ...      "argkeys": ["wav_file"] },
    ...  "words": {
    ...      "func": lambda t: t.split(),
    ...      "argkeys": ["text"] }, }

    Arguments
    ---------
    data : dict
        Dictionary containing single data points (e.g. utterances).
    dynamic_items : dict, optional
        Configuration for the dynamic items produced when fetchin an example.
        Nested dict with the format (in YAML notation):
        <key>:
            func: <callable> # To be called
            argkeys: <list> # keys of args, either other funcs or in data
        <key2>: ...
    output_keys : list, optional
        List of keys (either directly available in data or dynamic items)
        to include in the output dict when data points are fetched.
    """

    def __init__(
        self, data, dynamic_elements=None, output_keys=None,
    ):

        self.ex_ids = list(self.examples.keys())
        self.pipeline = DataPipeline()

    def __len__(self):
        return len(self.ex_ids)

    def __getitem__(self, item):
        ex_id = self.ex_ids[item]
        c_ex = self.examples[ex_id]
        out = {"id": ex_id}

        for k in c_ex.keys():
            for t_pipeline in self.data_transforms:
                if t_pipeline.target == k:
                    pip_name = t_pipeline.name
                    out[pip_name] = t_pipeline(c_ex[k])

        return out

    @classmethod
    def from_extended_yaml(cls, examples, data_transforms, **kwargs):
        return cls(examples, data_transforms, **kwargs)
