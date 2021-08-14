"""
A helper class for reading HuggingFace datasets

Authors
* Artem Ploujnikov 2021
"""
import speechbrain as sb
from datasets import load_dataset
from speechbrain.dataio.dataset import DynamicItemDataset


def load(path, mappings=None, split=None, **kwargs):
    """
    A loading wrapper for HuggingFace datasets

    Arguments
    ---------
    path: str
        the path to the dataset

    mappings: dict
        Optional - a dictionary of {<dataset_key>: <target_key>}

    Returns
    -------
    dataset: DynamicItemDataset
        a SpeechBrain dataset
    """

    dataset_hs = load_dataset(path, **kwargs)
    if split:
        dataset_hs = dataset_hs[split]
    dataset_sb = DynamicItemDataset.from_arrow_dataset(dataset_hs)

    if mappings is not None:
        mapping_to = list(mappings.keys())
        mapping_from = list(mappings.values())

        @sb.utils.data_pipeline.takes(*mapping_from)
        @sb.utils.data_pipeline.provides(*mapping_to)
        def mapper(*args):
            return args

        dataset_sb.add_dynamic_item(mapper)

    return dataset_sb
