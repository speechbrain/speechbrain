"""An feature extraction script for LibriTTS

Authors
 * Artem Ploujnikov 2025
"""

import sys
from pathlib import Path

from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.preparation import prepared_features
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Dataset prep (parsing Librispeech
    from libritts_prepare import prepare_libritts  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_libritts,
        kwargs={
            "data_folder": hparams["data_folder"],
            "train_split": hparams["train_splits"],
            "valid_split": hparams["dev_splits"],
            "test_split": hparams["test_splits"],
            "save_json_train": hparams["train_json"],
            "save_json_valid": hparams["valid_json"],
            "save_json_test": hparams["test_json"],
            "sample_rate": hparams["sample_rate"],
            "skip_prep": hparams["skip_prep"],
            "skip_resample": hparams["skip_resample"],
        },
    )

    data_folder = hparams["data_folder"]
    datasets = {}
    splits = ["train", "valid", "test"]
    for split in splits:
        json_path = hparams[f"{split}_json"]
        name = Path(json_path).stem
        dataset = DynamicItemDataset.from_json(
            json_path=json_path,
            replacements={"data_root": data_folder},
        )
        if hparams["data_count"]:
            dataset.data_ids = dataset.data_ids[: hparams["data_count"]]
        datasets[split] = dataset

    features = hparams["extract_features"]
    id_key = hparams["id_key"]
    with prepared_features(
        datasets,
        keys=features,
        storage=hparams["storage"],
        storage_opts=hparams["storage_opts"],
    ):
        for split in splits:
            print(f"Sanity checking {split}")
            dataset = datasets[split]
            dataset.set_output_keys([id_key] + features)
            loader = sb.dataio.dataloader.make_dataloader(
                dataset, **hparams["dataloader_opts"]
            )
            for batch in loader:
                data_ids = getattr(batch, id_key)
                print("ID", data_ids)
                for feature in features:
                    feature_data = getattr(batch, feature)
                    print(f"{feature}: {feature_data.data.shape}")
                    print(f"{feature} lengths: {feature_data.lengths.tolist()}")
                print("-" * 10)
