#!/usr/bin/python3
"""Recipe for training sound class embeddings (e.g, ecapa_tdnn) using the UrbanSound8k.
We employ an encoder followed by a sound classifier.

To run this recipe, use the following command:
> python train_10_fold.py {hyperparameter_file}

Using your own hyperparameter file or the following:
    hparams/train_ecapa_tdnn.yaml (for the ecapa+tdnn system)

Authors
    * David Whipps 2021
    * Ala Eddine Limame 2021

Based on VoxCeleb By:
    * Mirco Ravanelli 2020
    * Hwidong Na 2020
    * Nauman Dawalatabad 2020
"""
import os
import pathlib
import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from urbansound8k_prepare import prepare_urban_sound_8k
from train import UrbanSound8kBrain, dataio_prep

if __name__ == "__main__":

    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, OVERRIDES_BASE = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    ALL_FOLDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for valid_fold in ALL_FOLDS:
        overrides = OVERRIDES_BASE

        train_fold_nums = ALL_FOLDS.copy()
        train_fold_nums.remove(valid_fold)
        overrides = overrides + " train_fold_nums: " + str(train_fold_nums) + "\n"

        valid_fold_nums = [valid_fold]
        overrides = overrides + " valid_fold_nums: " + str(valid_fold_nums) + "\n"

        test_fold_nums = [valid_fold]  # test and validation are same, here
        overrides = overrides + " test_fold_nums: " + str(test_fold_nums) + "\n"

        # Load hyperparameters file with overrides
        with open(hparams_file) as fin:
            hparams = load_hyperpyyaml(fin, overrides)

        OUTPUT_FOLDER_BASE = hparams["output_folder"]
        output_folder = OUTPUT_FOLDER_BASE + "/valid_fold_" + str(valid_fold)
        hparams["output_folder"] = output_folder

        SAVE_FOLDER_BASE = hparams["save_folder"]
        save_folder = SAVE_FOLDER_BASE + "/fold_" + str(valid_fold)
        overrides = overrides + " save_folder: " + save_folder + "\n"
        checkpoints_dir = pathlib.Path(os.path.abspath(save_folder))
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        checkpointer_hparams = hparams["checkpointer"]
        checkpointer_hparams.checkpoints_dir = checkpoints_dir

        # Create experiment directory
        sb.create_experiment_directory(
            experiment_directory=output_folder,
            hyperparams_to_save=hparams_file,
            overrides=overrides,
        )

        # Tensorboard logging
        if hparams["use_tensorboard"]:
            from speechbrain.utils.train_logger import TensorboardLogger

            hparams["tensorboard_train_logger"] = TensorboardLogger(
                hparams["tensorboard_logs_folder"]
            )

        run_on_main(
            prepare_urban_sound_8k,
            kwargs={
                "data_folder": hparams["data_folder"],
                "audio_data_folder": hparams["audio_data_folder"],
                "save_json_train": hparams["train_annotation"],
                "save_json_valid": hparams["valid_annotation"],
                "save_json_test": hparams["test_annotation"],
                "train_fold_nums": hparams["train_fold_nums"],
                "valid_fold_nums": hparams["valid_fold_nums"],
                "test_fold_nums": hparams["test_fold_nums"],
                "skip_manifest_creation": hparams["skip_manifest_creation"],
            },
        )

        # Dataset IO prep: creating Dataset objects and proper encodings for phones
        datasets, label_encoder = dataio_prep(hparams)
        hparams["label_encoder"] = label_encoder

        urban_sound_8k_brain = UrbanSound8kBrain(
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )

        # The `fit()` method iterates the training loop, calling the methods
        # necessary to update the parameters of the model. Since all objects
        # with changing state are managed by the Checkpointer, training can be
        # stopped at any point, and will be resumed on next call.
        urban_sound_8k_brain.fit(
            epoch_counter=urban_sound_8k_brain.hparams.epoch_counter,
            train_set=datasets["train"],
            valid_set=datasets["valid"],
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )

        # Load the best checkpoint for evaluation
        test_stats = urban_sound_8k_brain.evaluate(
            test_set=datasets["test"],
            min_key="error",
            progressbar=True,
            test_loader_kwargs=hparams["dataloader_options"],
        )
