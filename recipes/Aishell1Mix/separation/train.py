#!/usr/bin/env/python3
"""Recipe for training a neural speech separation system on Aishell1Mix2/3 datasets.
The system employs an encoder, a decoder, and a masking network.

To run this recipe, do the following:
> python train.py hparams/sepformer-aishell1mix2.yaml
> python train.py hparams/sepformer-aishell1mix3.yaml


The experiment file is flexible enough to support different neural
networks. By properly changing the parameter files, you can try
different architectures. The script supports both aishell1mix2 and
aishell1mix3.


Authors
 * Cem Subakan 2020
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
 * Mirko Bronzi 2020
 * Jianyuan Zhong 2020
"""

import os
import sys
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
import logging


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    run_opts["auto_mix_prec"] = hparams["auto_mix_prec"]

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Logger info
    logger = logging.getLogger(__name__)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Check if storage folder for dataset exists
    if not hparams["data_folder"]:
        print("Please, specify a valid data_folder for dataset storage")
        sys.exit(1)

    # Data preparation
    from recipes.Aishell1Mix.prepare_data import prepare_aishell1mix

    run_on_main(
        prepare_aishell1mix,
        kwargs={
            "datapath": hparams["data_folder"],
            "savepath": hparams["save_folder"],
            "n_spks": hparams["num_spks"],
            "skip_prep": hparams["skip_prep"],
            "aishell1mix_addnoise": hparams["use_wham_noise"],
            "fs": hparams["sample_rate"],
            "datafreqs": hparams["data_freqs"],
            "datamodes": hparams["data_modes"],
        },
    )
    hparams["data_folder"] += f'/aishell1mix/Aishell1Mix{hparams["num_spks"]}'

    # Create dataset objects
    from recipes.LibriMix.separation.train import dataio_prep

    if hparams["dynamic_mixing"]:
        from dynamic_mixing import (
            dynamic_mix_data_prep_aishell1mix as dynamic_mix_data_prep,
        )

        # if the base_folder for dm is not processed, preprocess them
        if "processed" not in hparams["base_folder_dm"]:
            # if the processed folder already exists we just use it otherwise we do the preprocessing
            if not os.path.exists(
                os.path.normpath(hparams["base_folder_dm"]) + "_processed"
            ):
                from recipes.Aishell1Mix.meta.preprocess_dynamic_mixing import (
                    resample_folder,
                )

                print("Resampling the base folder")
                run_on_main(
                    resample_folder,
                    kwargs={
                        "input_folder": hparams["base_folder_dm"],
                        "output_folder": os.path.normpath(
                            hparams["base_folder_dm"]
                        )
                        + "_processed",
                        "fs": hparams["sample_rate"],
                        "regex": "**/*.wav",
                    },
                )
                # adjust the base_folder_dm path
                hparams["base_folder_dm"] = (
                    os.path.normpath(hparams["base_folder_dm"]) + "_processed"
                )
            else:
                print(
                    "Using the existing processed folder on the same directory as base_folder_dm"
                )
                hparams["base_folder_dm"] = (
                    os.path.normpath(hparams["base_folder_dm"]) + "_processed"
                )

        # Colleting the hparams for dynamic batching
        dm_hparams = {
            "train_data": hparams["train_data"],
            "data_folder": hparams["data_folder"],
            "base_folder_dm": hparams["base_folder_dm"],
            "sample_rate": hparams["sample_rate"],
            "num_spks": hparams["num_spks"],
            "training_signal_len": hparams["training_signal_len"],
            "dataloader_opts": hparams["dataloader_opts"],
        }

        train_data = dynamic_mix_data_prep(dm_hparams)

        # Inheriting data preparation from librimix. It uses these variables:
        # hparams["data_folder"]
        # hparams["train_data"]
        # hparams["valid_data"]
        # hparams["test_data"]
        # hparams["num_spks"]
        # hparams["use_wham_noise"]
        _, valid_data, test_data = dataio_prep(hparams)
    else:
        train_data, valid_data, test_data = dataio_prep(hparams)

    # Load pretrained model if pretrained_separator is present in the yaml
    if "pretrained_separator" in hparams:
        run_on_main(hparams["pretrained_separator"].collect_files)
        hparams["pretrained_separator"].load_collected()

    from recipes.LibriMix.separation.train import Separation

    # Brain class initialization
    # Inheriting the Separation class from librimix. It uses these variables:
    # hparams["num_spks"]
    # hparams["use_speedperturb"]
    # hparams["use_rand_shift"]
    # hparams["use_wham_noise"]
    # hparams["use_wavedrop"]
    # hparams["wavedrop"]
    # hparams["limit_training_signal_len"]
    # hparams["Encoder"]
    # hparams["MaskNet"]
    # hparams["Decoder"]
    # hparams["loss"]
    # hparams["threshold_byloss"]
    # hparams["threshold"]
    # hparams["loss_upper_lim"]
    # hparams["clip_grad_norm"]
    # hparams["save_audio"]
    # hparams["n_audio_to_save"]
    # hparams["lr_scheduler"]
    # hparams["optimizer"]
    # hparams["train_logger"]
    # hparams["epoch_counter"]
    # hparams["speedperturb"]
    # hparams["min_shift"]
    # hparams["max_shift"]
    # hparams["output_folder"]
    # hparams["dataloader_opts"]
    # hparams["save_folder"]
    # hparams["sample_rate"]

    separator = Separation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # re-initialize the parameters if we don't use a pretrained model
    if "pretrained_separator" not in hparams:
        for module in separator.modules.values():
            separator.reset_layer_recursively(module)
    if not hparams["test_only"]:
        # Training
        separator.fit(
            separator.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["dataloader_opts"],
            valid_loader_kwargs=hparams["dataloader_opts"],
        )

    # Eval
    separator.evaluate(test_data, min_key="si-snr")
    separator.save_results(test_data)
