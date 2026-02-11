#!/usr/bin/env python3
"""Script to extract SSL features from the audio waveforms.

The script uses the `speechbrain.integrations.hdf5.cached_item` module to cache the features.
The cached features are used in the `train_speechllm.py` script to train the SpeechLLM ASR system.

Since we do the extractions within the pipeline in the dataloader, we must place
our hparams elements directly on device, and use a default bsize of 1.

Example
-------
python extract_ssl_feats.py hparams/extract_ssl_feats.yaml
    --data_folder path/to/LibriSpeech \
    --output_folder path/to/feats_cache \
    --ssl_hub path/to/wavlm-large \
    --feats_cache_dir path/to/feats_cache
    ...other_hparams...

Authors
-------
 * Adel Moumen, 2025
"""

import sys
from pathlib import Path

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.integrations.hdf5.cached_item import CachedHDF5DynamicItem
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger

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

    normalizer = hparams["normalize"].to(hparams["device"]).eval()
    ssl_encoder = hparams["ssl"].to(hparams["device"]).eval()

    # Base compute function used by all cached wrappers (no file bound yet)
    @CachedHDF5DynamicItem.cache(hparams["feats_cache_dir"], compression="gzip")
    @sb.utils.data_pipeline.takes("id", "sig")
    @sb.utils.data_pipeline.provides("feats")
    def compute_feats(uid, sig):
        sig = sig.to(hparams["device"]).unsqueeze(0)
        length = torch.ones(1, device=hparams["device"])
        with torch.no_grad(), torch.amp.autocast(
            hparams["device"].type, dtype=hparams["dtype"]
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

    datasets = {"train": train_data, "valid": valid_data} | {
        k: v for k, v in test_datasets.items()
    }

    for stage, dataset in datasets.items():
        logger.info(f"Iterating {stage} dataset to warm the cache.")
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
    logger.info("Preparing data...")
    dataio_prepare(hparams)
    logger.info("Done preparing data")
