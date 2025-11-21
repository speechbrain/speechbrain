"""Common utilities.

Authors
 * Luca Della Libera 2025
"""

import os
import random

import torch
import torchaudio

import speechbrain as sb
from speechbrain.dataio.sampler import DynamicBatchSampler
from speechbrain.utils.distributed import run_on_main

__all__ = ["download_wavlm6", "prepare_recipe"]


def download_wavlm6(cache_dir: "str") -> "str":
    """Download WavLM6 checkpoint to cache and return the path.

    Parameters
    ----------
    cache_dir:
        Cache directory where the checkpoint will be saved.

    Returns
    -------
        Path to the saved checkpoint.

    """
    os.makedirs(cache_dir, exist_ok=True)
    checkpoint_path = os.path.join(cache_dir, "wavlm6.pt")

    # If already cached, return immediately
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    # Load FocalCodec model
    codec = torch.hub.load(
        repo_or_dir="lucadellalib/focalcodec",
        model="focalcodec",
        config="lucadellalib/focalcodec_50hz",
    )

    # Save WavLM6 checkpoint
    torch.save(codec.encoder.state_dict(), checkpoint_path)

    return checkpoint_path


def prepare_recipe(hparams, run_opts):
    # Dataset preparation
    import libritts_prepare

    # Due to DDP, do the preparation ONLY on the main Python process
    run_on_main(
        libritts_prepare.prepare_libritts,
        kwargs={
            "data_folder": hparams["data_folder"],
            "train_split": hparams["train_split"],
            "valid_split": hparams["valid_split"],
            "test_split": hparams["test_split"],
            "save_json_train": hparams["train_json"],
            "save_json_valid": hparams["valid_json"],
            "save_json_test": hparams["test_json"],
            "sample_rate": hparams["sample_rate"],
            "skip_prep": hparams["skip_prep"],
            "model_name": "HiFi-GAN",
        },
    )

    # Create the datasets objects
    train_data, valid_data, test_data = dataio_prepare(
        debug=run_opts["debug"], **hparams
    )

    # Dynamic batching
    hparams["train_dataloader_kwargs"] = {
        "num_workers": hparams.get("dataloader_workers", 0)
    }
    if hparams.get("dynamic_batching", False) or hparams.get(
        "train_dynamic_batching", False
    ):
        hparams["train_dataloader_kwargs"]["batch_sampler"] = (
            DynamicBatchSampler(
                train_data,
                hparams["train_max_batch_length"],
                num_buckets=hparams.get("num_buckets"),
                length_func=lambda x: x["duration"],
                shuffle=False,
                batch_ordering=hparams.get("sorting", "batch_ordering"),
                max_batch_ex=hparams.get("max_batch_size"),
                bucket_boundaries=hparams.get("bucket_boundaries", []),
                lengths_list=hparams.get("lengths_list"),
            )
        )
    else:
        hparams["train_dataloader_kwargs"]["batch_size"] = hparams[
            "train_batch_size"
        ]
        hparams["train_dataloader_kwargs"]["shuffle"] = (
            hparams["sorting"] == "random"
        )
        hparams["train_dataloader_kwargs"]["pin_memory"] = (
            run_opts["device"] != "cpu"
        )
        hparams["train_dataloader_kwargs"]["drop_last"] = hparams.get(
            "segment_size", None
        ) is not None and hparams.get("segment_pad", False)

    hparams["valid_dataloader_kwargs"] = {
        "num_workers": hparams.get("dataloader_workers", 0)
    }
    if hparams.get("dynamic_batching", False) or hparams.get(
        "valid_dynamic_batching", False
    ):
        hparams["valid_dataloader_kwargs"]["batch_sampler"] = (
            DynamicBatchSampler(
                valid_data,
                hparams["valid_max_batch_length"],
                num_buckets=hparams.get("num_buckets"),
                length_func=lambda x: x["duration"],
                shuffle=False,
                batch_ordering="descending",
                max_batch_ex=hparams.get("max_batch_size"),
                bucket_boundaries=hparams.get("bucket_boundaries", []),
                lengths_list=hparams.get("lengths_list"),
            )
        )
    else:
        hparams["valid_dataloader_kwargs"]["batch_size"] = hparams[
            "valid_batch_size"
        ]
        hparams["valid_dataloader_kwargs"]["pin_memory"] = (
            run_opts["device"] != "cpu"
        )

    hparams["test_dataloader_kwargs"] = {
        "num_workers": hparams.get("dataloader_workers", 0)
    }
    if hparams.get("dynamic_batching", False) or hparams.get(
        "test_dynamic_batching", False
    ):
        hparams["test_dataloader_kwargs"]["batch_sampler"] = (
            DynamicBatchSampler(
                test_data,
                hparams["test_max_batch_length"],
                num_buckets=hparams.get("num_buckets"),
                length_func=lambda x: x["duration"],
                shuffle=False,
                batch_ordering="descending",
                max_batch_ex=hparams.get("max_batch_size"),
                bucket_boundaries=hparams.get("bucket_boundaries", []),
                lengths_list=hparams.get("lengths_list"),
            )
        )
    else:
        hparams["test_dataloader_kwargs"]["batch_size"] = hparams[
            "test_batch_size"
        ]
        hparams["test_dataloader_kwargs"]["pin_memory"] = (
            run_opts["device"] != "cpu"
        )

    # Pretrain the specified modules
    if "pretrainer" in hparams:
        run_on_main(hparams["pretrainer"].collect_files)
        run_on_main(hparams["pretrainer"].load_collected)

    return hparams, train_data, valid_data, test_data


def dataio_prepare(
    data_folder,
    train_json,
    valid_json,
    test_json,
    sample_rate=16000,
    train_remove_if_longer=60.0,
    valid_remove_if_longer=60.0,
    test_remove_if_longer=60.0,
    sorting="ascending",
    debug=False,
    segment_size=None,
    segment_pad=False,
    audio_backend="soundfile",
    **kwargs,
):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.

    """
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=train_json,
        replacements={"data_root": data_folder},
    )
    # Sort training data to speed up training
    train_data = train_data.filtered_sorted(
        sort_key="duration",
        reverse=sorting == "descending",
        key_max_value={"duration": train_remove_if_longer},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=valid_json,
        replacements={"data_root": data_folder},
    )
    # Sort validation data to speed up validation
    valid_data = valid_data.filtered_sorted(
        sort_key="duration",
        reverse=not debug,
        key_max_value={"duration": valid_remove_if_longer},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=test_json,
        replacements={"data_root": data_folder},
    )
    # Sort the test data to speed up testing
    test_data = test_data.filtered_sorted(
        sort_key="duration",
        reverse=not debug,
        key_max_value={"duration": test_remove_if_longer},
    )

    datasets = [train_data, valid_data, test_data]

    # Define audio pipeline
    takes = ["wav"]
    provides = ["sig"]

    def audio_pipeline_train(wav):
        original_sample_rate = sb.dataio.dataio.read_audio_info(wav).sample_rate
        sig = sb.dataio.dataio.read_audio(wav, backend=audio_backend)
        sig = torchaudio.functional.resample(
            sig, original_sample_rate, sample_rate
        )

        if segment_size is not None:
            delta_length = segment_size - len(sig)
            if delta_length > 0 and segment_pad:
                sig = torch.nn.functional.pad(sig, [0, delta_length])
            elif delta_length < 0:
                start = random.randint(0, -delta_length)
                sig = sig[start : start + segment_size]

        yield sig

    def audio_pipeline_eval(wav):
        original_sample_rate = sb.dataio.dataio.read_audio_info(wav).sample_rate
        sig = sb.dataio.dataio.read_audio(wav, backend=audio_backend)
        sig = torchaudio.functional.resample(
            sig, original_sample_rate, sample_rate
        )
        yield sig

    sb.dataio.dataset.add_dynamic_item(
        [train_data], audio_pipeline_train, takes, provides
    )
    sb.dataio.dataset.add_dynamic_item(
        [valid_data, test_data], audio_pipeline_eval, takes, provides
    )

    # Set output
    sb.dataio.dataset.set_output_keys(datasets, ["id"] + provides)

    return train_data, valid_data, test_data
