"""This recipe generates The LargeScaleASR Set one action at a time. It will be
necessary to call it multiple times, with different arguments, to build everything that is necessary:

1. Generate each dataset individually:
    ACTION_TO_PERFORM values must be:
        voxpopuli or commonvoice or libriheavy or yodas or people's speech or librispeech (for val/test)
2. Aggregate the manifest.
    ACTION_TO_PERFORM values must be:
        val_test_sets or large_set or medium_set or small_set or clean_set
3. (optional) generate the parquet files for HuggingFace.


Authors
 * Titouan Parcollet 2024
"""

import sys

from hyperpyyaml import load_hyperpyyaml
from preparation_scripts.huggingface_utilities import clean_the_parquet

import speechbrain as sb
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


if __name__ == "__main__":

    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    if hparams["ACTION_TO_PERFORM"] == "voxpopuli":
        from preparation_scripts.voxpopuli_prepare import prepare_voxpopuli

        prepare_voxpopuli(hparams["VOXPOPULI_PATH"], hparams["HF_DATASET_ROOT"])
    elif hparams["ACTION_TO_PERFORM"] == "librispeech":
        from preparation_scripts.librispeech_prepare import prepare_librispeech

        prepare_librispeech(
            hparams["LIBRISPEECH_PATH"],
            hparams["HF_DATASET_ROOT"],
            dev_splits=["dev-other"],
            te_splits=["test-other"],
        )
    elif hparams["ACTION_TO_PERFORM"] == "commonvoice":
        from preparation_scripts.common_voice_prepare import (
            prepare_common_voice,
        )

        prepare_common_voice(
            hparams["CV_PATH"],
            hparams["HF_DATASET_ROOT"],
        )
    elif hparams["ACTION_TO_PERFORM"] == "libriheavy":
        from preparation_scripts.libriheavy_prepare import prepare_libriheavy

        prepare_libriheavy(
            hparams["LIBRILIGHT_PATH"],
            hparams["HF_DATASET_ROOT"],
            hparams["LIBRIHEAVY_TRAIN_JSON"],
            hparams["LIBRIHEAVY_DEV_JSON"],
            hparams["LIBRIHEAVY_TEST_JSON"],
            hparams["LIBRIHEAVY_CUTOFF_HOURS"],
        )
    elif hparams["ACTION_TO_PERFORM"] == "peoples_speech":
        from preparation_scripts.peoples_speech_prepare import (
            prepare_peoples_speech,
        )

        prepare_peoples_speech(
            hparams["PEOPLES_SPEECH_PATH"],
            hparams["HF_DATASET_ROOT"],
            subsets=["clean"],
            audio_decoding=False,
        )
    elif hparams["ACTION_TO_PERFORM"] == "yodas":
        from preparation_scripts.yodas_prepare import prepare_yodas

        prepare_yodas(
            hparams["YODAS_PATH"],
            hparams["HF_DATASET_ROOT"],
            train_subsets=["en000", "en001"],
            dev_test_subset=["en003"],
        )
    elif hparams["ACTION_TO_PERFORM"] == "val_test_sets":

        from preparation_scripts.merge_csv_manifests import merge_csv_files

        merge_csv_files(
            [
                hparams["VOX_VAL_PATH"],
                hparams["LIBRI_VAL_PATH"],
                hparams["CV_VAL_PATH"],
                hparams["YODAS_VAL_PATH"],
            ],
            hparams["TLS_VAL_PATH"],
            hours_per_csv=5,
            duration_column_indice=1,
        )
        merge_csv_files(
            [
                hparams["VOX_TEST_PATH"],
                hparams["LIBRI_TEST_PATH"],
                hparams["CV_TEST_PATH"],
                hparams["YODAS_TEST_PATH"],
            ],
            hparams["TLS_TEST_PATH"],
            hours_per_csv=5,
            duration_column_indice=1,
        )
    elif hparams["ACTION_TO_PERFORM"] == "large_set":
        from preparation_scripts.merge_csv_manifests import merge_csv_files

        merge_csv_files(
            [
                hparams["VOX_TRAIN_PATH"],
                hparams["LIBRIHEAVY_TRAIN_PATH"],
                hparams["CV_TRAIN_PATH"],
                hparams["PEOPLE_TRAIN_PATH"],
                hparams["YODAS_TRAIN_PATH"],
            ],
            hparams["TLS_TRAIN_LARGE_PATH"],
            duration_column_indice=1,
        )
    elif hparams["ACTION_TO_PERFORM"] == "medium_set":
        from preparation_scripts.merge_csv_manifests import merge_csv_files

        merge_csv_files(
            [
                hparams["VOX_TRAIN_PATH"],
                hparams["LIBRIHEAVY_TRAIN_PATH"],
                hparams["CV_TRAIN_PATH"],
                hparams["PEOPLE_TRAIN_PATH"],
                hparams["YODAS_TRAIN_PATH"],
            ],
            hparams["TLS_TRAIN_MEDIUM_PATH"],
            hours_per_csv=500,
            duration_column_indice=1,
        )
    elif hparams["ACTION_TO_PERFORM"] == "small_set":
        from preparation_scripts.merge_csv_manifests import merge_csv_files

        merge_csv_files(
            [
                hparams["VOX_TRAIN_PATH"],
                hparams["LIBRIHEAVY_TRAIN_PATH"],
                hparams["CV_TRAIN_PATH"],
                hparams["PEOPLE_TRAIN_PATH"],
                hparams["YODAS_TRAIN_PATH"],
            ],
            hparams["TLS_TRAIN_SMALL_PATH"],
            hours_per_csv=50,
            duration_column_indice=1,
        )
    elif hparams["ACTION_TO_PERFORM"] == "clean_set":
        from preparation_scripts.merge_csv_manifests import merge_csv_files

        merge_csv_files(
            [
                hparams["VOX_TRAIN_PATH"],
                hparams["LIBRIHEAVY_TRAIN_PATH"],
                hparams["CV_TRAIN_PATH"],
            ],
            hparams["TLS_TRAIN_CLEAN_PATH"],
            duration_column_indice=1,
        )
    elif hparams["ACTION_TO_PERFORM"] == "export_to_parquet":
        from preparation_scripts.merge_csv_manifests import (
            copy_and_remove_path_csvs,
        )

        clean_the_parquet(hparams)
        copy_and_remove_path_csvs(
            hparams["PARQUET_ORIG_CSV"], hparams["PARQUET_CSV_OUTPUT_FOLDER"]
        )
    else:
        msg = "The entered ACTION_TO_PERFORM is not valid! Valid options are: "
        msg += "voxpopuli or commonvoice or libriheavy or yodas, people_speech"
        msg += "or librispeech or val_test_sets or large_set or "
        msg += "medium_set or small_set or clean_set"
        logger.error(msg)
