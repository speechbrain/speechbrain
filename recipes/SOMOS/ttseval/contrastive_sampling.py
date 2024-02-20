"""
Contrastive sampling utilities

Authors
 * Artem Ploujnikov 2023
"""

import torch
from functools import partial
from speechbrain.utils.data_pipeline import takes, provides
from speechbrain.utils import checkpoints


@checkpoints.register_checkpoint_hooks
class RegressionContrastiveEnhancement:
    """An dataset enhancement for contrastive learning on simple regression
    tasks with a single metric, useful for cases where consistently estimating
    differences is easier than evaluating the metric for an individual sample
    and where differences that are too small are not considered useful. Originally,
    this was developed for MOS estimation

    For any given sample, a paired sample will be selected by first finding
    all "allowed" pairings (by excluding those less than min_delta units away),
    sorting them by distance - and then sampling from the uniform distribution
    (of indices, not of distances).

    Arguments
    ---------
    metric_key : str
        The key in the original dataset corresponding to the metric

    min_delta : float
        The minimum metric distance (absolute value) for a sample to be pairable
        with any given sample

    Examples
    --------
    >>> data = {
    ...     1: {"score": 3.5},
    ...     2: {"score": 2.7},
    ...     3: {"score": 5.0},
    ...     4: {"score": 1.2},
    ...     5: {"score": 2.5},
    ...     6: {"score": 3.2},
    ...     7: {"score": 3.8},
    ...     8: {"score": 1.7},
    ...     9: {"score": 1.2},
    ...     10: {"score": 4.2},
    >>> }
    >>> from speechbrain.dataio.dataset import DynamicItemDataset
    >>> dataset = DynamicItemDataset(data)
    >>> dataset.set_output_keys(["id", "score"])
    >>> from contrastive_sampling import RegressionContrastiveEnhancement
    >>> sampling = RegressionContrastiveEnhancement(
    ...     metric_key="score",
    ...     min_delta=0.5,
    ...     seed=42
    ... )
    ... sampling.bind(dataset)
    >>> from speechbrain.dataio.dataloader import make_dataloader
    >>> loader = make_dataloader(dataset)
    >>> loader_it = iter(loader)
    >>> batch = next(loader_it)
    >>> batch.score.item()
    >>> batch.contrast_score.item()
    """

    def __init__(self, metric_key, min_delta, seed=None):
        self.metric_key = metric_key
        self.min_delta = min_delta
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def bind(self, dataset):
        """Binds the enhancement to a dataset, adding the contrastive pairings
        to its pipeline

        Arguments
        ---------
        dataset : speechbrain.dataio.dataset.DynamicItemDataset
            the target dataset
        """
        metric_values = torch.tensor(
            [item[self.metric_key] for item in dataset]
        )
        size = len(dataset)
        self.indexes = torch.arange(size)
        metric_values_sorted, self.indexes_sorted = metric_values.sort()
        metric_diff_abs = (
            metric_values_sorted[None, :] - metric_values_sorted[:, None]
        ).abs()
        selection_blocked = metric_diff_abs < self.min_delta
        min_shift_right = selection_blocked.triu().sum(-1)
        self.min_shift_left = selection_blocked.tril().sum(-1)
        self.indexes_sorted_mirror = torch.cat(
            [self.indexes_sorted.flip(0)[1:], self.indexes_sorted,]
        )
        self.indexes_sorted_mirror = self.indexes_sorted_mirror.unsqueeze(
            0
        ).expand(len(dataset), self.indexes_sorted_mirror.size(-1))
        for idx in range(len(dataset)):
            self.indexes_sorted_mirror[idx] = self.indexes_sorted_mirror[
                idx
            ].roll(-idx)

        self.shift_max = (2 * size - 1) - self.min_shift_left - min_shift_right
        keys = list(dataset.pipeline.output_mapping.keys())
        self.data_ids = dataset.data_ids
        pairings_map = self._get_pairings_map()
        if not keys:
            raise ValueError("Output keys must be set before binding")
        contrastive_keys = [f"contrast_{key}" for key in keys]
        self.pipeline = ContrastivePairingPipeline(keys, pairings_map)
        pipeline_element = partial(self.pipeline, dataset)
        pipeline_element = takes("id")(pipeline_element)
        pipeline_element = provides(*contrastive_keys)(pipeline_element)
        dataset.add_dynamic_item(pipeline_element)
        dataset.set_output_keys(keys + contrastive_keys)

    def _get_pairings_map(self):
        """Builds a returns a dictionary of item pairings"""
        shift_rel = torch.rand(len(self.indexes), generator=self.generator)
        shift_abs = (
            self.min_shift_left + (self.shift_max * shift_rel).floor().int()
        )
        indexes_selected = self.indexes_sorted_mirror[
            torch.arange(len(shift_abs)), shift_abs
        ]
        pairings = torch.zeros_like(indexes_selected)
        pairings[self.indexes_sorted] = indexes_selected
        return {
            data_id: pairing_idx
            for data_id, pairing_idx in zip(self.data_ids, pairings)
        }

    def shuffle(self):
        """Re-samples the pairings"""
        pairings_map = self._get_pairings_map()
        self.pipeline.pairings = pairings_map

    @checkpoints.mark_as_saver
    def save(self, path):
        """Saves the current metrics on the specified path."""
        data = {
            "generator_state": self.generator.get_state(),
        }
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch=False, device=None):
        """Loads the needed information."""
        del end_of_epoch
        del device
        data = torch.load(path)
        self.generator.set_state(data["generator_state"])


class ContrastivePairingPipeline:
    """A helper callable that adds keys from the paired samples
    to dataset elements. Instances of this class are intended to
    be used as dynamic items in a DynamicItemDataset.

    Arguments
    ---------
    keys : list
        a list of keys in the original dataset to be enhanced

    parirings : dict
        a dictionary indicating how IDs are paired - with keys
        corresponding to anchor items and values to the paired items
    """

    def __init__(self, keys, pairings):
        self.keys = keys
        self.pairings = pairings

    def __call__(self, dataset, data_id):
        """Provides the data keys from the paired data sample

        Arguments
        ---------
        dataset : speechbrain.dataio.dataio.DynamicItemDataset
            a dataset
        data_id : object
            the ID of the item

        Returns
        -------
        result: generator
            the values corresponding to the specified keys from
            the paired item"""
        pairing_id = self.pairings[data_id]
        with dataset.output_keys_as(self.keys):
            pairing = dataset[pairing_id]
        for key in self.keys:
            yield pairing[key]
