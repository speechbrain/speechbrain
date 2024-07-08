import torch


def test_ConcatDatasetBatchSampler(device):
    import numpy as np
    from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

    from speechbrain.dataio.sampler import (
        ConcatDatasetBatchSampler,
        ReproducibleRandomSampler,
    )

    datasets = []
    for i in range(3):
        if i == 0:
            datasets.append(
                TensorDataset(torch.arange(i * 10, (i + 1) * 10, device=device))
            )
        else:
            datasets.append(
                TensorDataset(torch.arange(i * 6, (i + 1) * 6, device=device))
            )

    samplers = [ReproducibleRandomSampler(x) for x in datasets]
    dataset = ConcatDataset(datasets)
    loader = DataLoader(
        dataset, batch_sampler=ConcatDatasetBatchSampler(samplers, [1, 1, 1])
    )

    concat_data = []

    for data in loader:
        concat_data.append([x.item() for x in data[0]])
    concat_data = np.array(concat_data)

    non_cat_data = []
    for i in range(len(samplers)):
        c_data = []
        loader = DataLoader(dataset.datasets[i], sampler=samplers[i])

        for data in loader:
            c_data.append(data[0].item())

        non_cat_data.append(c_data)

    minlen = min([len(x) for x in non_cat_data])
    non_cat_data = [x[:minlen] for x in non_cat_data]
    non_cat_data = np.array(non_cat_data)
    np.testing.assert_array_equal(non_cat_data.T, concat_data)

    # check DynamicBatchSampler
    from speechbrain.dataio.batch import PaddedBatch
    from speechbrain.dataio.dataloader import SaveableDataLoader
    from speechbrain.dataio.dataset import DynamicItemDataset
    from speechbrain.dataio.sampler import DynamicBatchSampler

    max_batch_length = 4
    num_buckets = 5

    item_lengths = [1, 2, 3, 4, 5, 6, 7]
    items = [[length] * length for length in item_lengths]

    dataset = {
        "ex_{}".format(length): {"wav": torch.tensor(item), "duration": length}
        for item, length in zip(items, item_lengths)
    }
    dataset = DynamicItemDataset(dataset)
    dataset.set_output_keys(["wav"])

    bsampler = DynamicBatchSampler(
        dataset,
        max_batch_length,
        num_buckets,
        lambda x: x["duration"],
        shuffle=False,
        batch_ordering="ascending",
    )

    dataloader = SaveableDataLoader(
        dataset, batch_sampler=bsampler, collate_fn=PaddedBatch
    )

    assert next(iter(dataloader))["wav"].data.shape == torch.Size([1, 1])

    bsampler = DynamicBatchSampler(
        dataset,
        max_batch_length,
        num_buckets,
        lambda x: x["duration"],
        shuffle=False,
        batch_ordering="descending",
    )

    dataloader = SaveableDataLoader(
        dataset, batch_sampler=bsampler, collate_fn=PaddedBatch
    )
    assert next(iter(dataloader))["wav"].data.shape == torch.Size([1, 7])

    max_batch_length = 10
    num_buckets = 5

    bsampler = DynamicBatchSampler(
        dataset,
        max_batch_length,
        num_buckets,
        lambda x: x["duration"],
        shuffle=False,
        batch_ordering="ascending",
    )

    dataloader = SaveableDataLoader(
        dataset, batch_sampler=bsampler, collate_fn=PaddedBatch
    )

    for b in dataloader:
        assert b["wav"].data.shape[1] <= max_batch_length
