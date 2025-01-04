#!/usr/bin/env/python3
"""Recipe for extracting a discrete tokens with librispeech.

Authors
 * Jarod Duret 2024
"""

import os
import sys
import logging
import time
import pathlib as pl
import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(base_dir)

print(base_dir)

logger = logging.getLogger(__name__)


def wait_for_data_preparation(output_folder, timeout=3600):
    marker_file = pl.Path(output_folder) / ".data_prep_done"
    start_time = time.time()

    while not marker_file.exists():
        if time.time() - start_time > timeout:
            print(
                f"Timeout waiting for data preparation after {timeout} seconds"
            )
            return False
        time.sleep(5)

    return True


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    rank = hparams["rank"]
    output_folder = pl.Path(hparams["output_folder"])

    if rank == 0:
        # Create experiment directory
        sb.create_experiment_directory(
            experiment_directory=hparams["output_folder"],
            hyperparams_to_save=hparams_file,
            overrides=overrides,
        )

        # Dataset prep (parsing Librispeech)
        from librispeech_prepare import prepare_librispeech  # noqa

        # multi-gpu (ddp) save data preparation
        run_on_main(
            prepare_librispeech,
            kwargs={
                "data_folder": hparams["data_folder"],
                "tr_splits": hparams["train_splits"],
                "dev_splits": hparams["dev_splits"],
                "te_splits": hparams["test_splits"],
                "save_folder": hparams["output_folder"],
                "merge_lst": hparams["train_splits"],
                "merge_name": "train.csv",
                "skip_prep": hparams["skip_prep"],
            },
        )

        # Create marker file to signal completion
        marker_file = output_folder / ".data_prep_done"
        marker_file.touch()
        logger.info(f"Rank {rank}: Data preparation complete")

    else:
        # Other ranks wait for data preparation to complete
        logger.info(f"Rank {rank}: Waiting for data preparation...")
        if wait_for_data_preparation(output_folder):
            logger.info(
                f"Rank {rank}: Data preparation detected, continuing..."
            )
        else:
            logger.error(f"Rank {rank}: Data preparation timeout, exiting")
            sys.exit(1)

    tokens_extractor = hparams["tokens_extractor"]
    data_folder = hparams["data_folder"]
    datasets = []
    for split in ["train", "valid"]:
        csv_path = hparams[f"{split}_csv"]
        name = pl.Path(csv_path).stem
        dataset = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_path,
            replacements={"data_root": data_folder},
        )
        datasets.append(dataset)

    for split in hparams["test_csv"]:
        name = pl.Path(split).stem
        dataset = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=split,
            replacements={"data_root": data_folder},
        )
        datasets.append(dataset)

    merged_data = {
        key: value
        for dataset in datasets
        for key, value in dataset.data.items()
    }
    merged_dataset = DynamicItemDataset(merged_data)

    save_folder = pl.Path(hparams["save_folder"])
    logger.info("Extracting dataset tokens ...")
    tokens_extractor.extract_tokens(
        merged_dataset,
        hparams["num_codebooks"],
        (save_folder / "librispeech").as_posix(),
    )

    if rank == 0:
        if hparams["save_embedding"]:
            save_folder = pl.Path(hparams["save_folder"])
            logger.info("Saving embeddings ...")
            tokens_extractor.save_pretrained_embeddings(
                (save_folder / "embeddings").as_posix(),
                vocab_size=hparams["vocab_size"],
                num_codebooks=hparams["num_codebooks"],
            )