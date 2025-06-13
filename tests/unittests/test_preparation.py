"""Unit tests for the preparation utilities: tensor storage and
feature extraction"""

import torch


def test_storage_registration():
    from speechbrain.dataio.preparation import Storage, storage_uri_scheme

    @storage_uri_scheme("custom")
    class CustomStorage(Storage):
        @classmethod
        def from_uri(cls, uri, mode, options=None):
            return cls()

    storage = Storage.from_uri("custom:/foo/bar", "r")
    assert isinstance(storage, CustomStorage)


def test_h5_storage(tmpdir):
    from pathlib import Path

    from speechbrain.dataio.preparation import Storage

    path = Path(tmpdir) / "test.h5"

    sample_data = [
        {
            "tokens": torch.randint(0, 1000, (100 + idx, 6)),
            "feats": torch.randn((100 + idx, 128)),
        }
        for idx in range(100)
    ]

    with Storage.from_uri(f"h5:{str(path)}", "w") as storage:
        for idx, item in enumerate(sample_data):
            item_id = f"TST{idx}"
            for key, value in item.items():
                storage.save(item_id, key, value)

    with Storage.from_uri(f"h5:{str(path)}", "r") as storage:
        for idx, item in enumerate(sample_data):
            item_id = f"TST{idx}"
            for key, value in item.items():
                stored_value = storage.load(item_id, key)
                assert torch.allclose(stored_value, value)


def test_feature_extractor(tmpdir):
    from pathlib import Path

    import speechbrain as sb
    from speechbrain.dataio.batch import PaddedData
    from speechbrain.dataio.dataset import DynamicItemDataset
    from speechbrain.dataio.preparation import (
        FeatureExtractor,
        prepared_features,
    )
    from speechbrain.utils.data_utils import batch_pad_right

    path = Path(tmpdir) / "test.h5"
    sample_data = {
        f"TST{idx}": {"tokens": torch.randint(0, 1000, (100 + idx, 6))}
        for idx in range(100)
    }
    features = {
        f"TST{idx}": torch.randn((100 + idx, 128)) for idx in range(100)
    }

    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("features")
    def features_pipeline(ids):
        out, lengths = batch_pad_right([features[id] for id in ids])
        return PaddedData(out, lengths)

    storage = f"h5:{str(path)}"
    feature_extractor = FeatureExtractor(
        device="cpu",
        storage=storage,
        dynamic_items=[
            features_pipeline,
        ],
        src_keys=["id", "tokens"],
        dataloader_opts={"batch_size": 4},
    )
    feature_extractor.set_output_features(["features", "tokens"])
    feature_extractor.extract(sample_data)

    test_data = {f"TST{idx}": {"label": f"label{idx}"} for idx in range(100)}
    dataset = DynamicItemDataset(test_data)
    with prepared_features(
        dataset, keys=["tokens", "features"], storage=storage
    ):
        dataset.set_output_keys(["tokens", "features"])
        for idx, item_id in enumerate(dataset.data_ids):
            item_data = dataset[idx]
            assert (item_data["tokens"] == sample_data[item_id]["tokens"]).all()
            assert torch.allclose(item_data["features"], features[item_id])
