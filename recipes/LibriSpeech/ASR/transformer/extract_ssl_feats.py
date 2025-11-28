#!/usr/bin/env python3
"""

Authors
 * Adel Moumen 2025
"""

import os
import sys
from pathlib import Path

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger
from speechbrain.integrations.hdf5.cached_item import CachedHDF5DynamicItem

logger = get_logger(__name__)


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    
    # todo yaml this
    normalizer = hparams["normalize"].to("cuda").eval()
    ssl_encoder = hparams["ssl"].to("cuda").eval()

    # Base compute function used by all cached wrappers (no file bound yet)
    @CachedHDF5DynamicItem.cache(hparams["feats_cache_dir"])
    @sb.utils.data_pipeline.takes("id", "sig")
    @sb.utils.data_pipeline.provides("feats")
    def compute_feats(uid, sig):
        sig = sig.to("cuda").unsqueeze(0)
        length = torch.ones(1, device="cuda")
        # todo allow mixed precision call
        with torch.no_grad(), torch.cuda.amp.autocast(
            dtype=torch.float16
        ):
            feats = normalizer(sig, length)
            feats = ssl_encoder(feats, length)
        return feats.squeeze(0).cpu()

    dynamic_items = [audio_pipeline, compute_feats]
    output_keys = ["id", "sig", "feats"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": data_folder},
        dynamic_items=dynamic_items,
        output_keys=output_keys,
    )
    
    # Build valid dataset with its own cached wrapper
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"data_root": data_folder},
        dynamic_items=dynamic_items,
        output_keys=output_keys,
    )

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, 
            replacements={"data_root": data_folder},
            dynamic_items=dynamic_items,
            output_keys=output_keys,
        )

    datasets = {"train": train_data, "valid": valid_data} | {k: v for k, v in test_datasets.items()}

    for stage, dataset in datasets.items():
        print(f"Iterating {stage} dataset to warm the cache.")
        dataset.iterate_once(output_keys=["feats"])


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # 1.  # Dataset prep (parsing Librispeech)
    from librispeech_prepare import prepare_librispeech  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

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
    print("Preparing data...")
    dataio_prepare(hparams)
    print("Done preparing data")
