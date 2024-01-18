"""
Recipe to train kenlm ngram model.

To run this recipe, do the following:
> python train.py hparams/train.yaml --data_folder=/path/to/LibriSpeech

Authors
 * Adel Moumen 2024
"""

import os
import csv
import sys
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
import speechbrain.k2_integration as sbk2


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Librispeech)
    import librispeech_prepare

    # multi-gpu (ddp) save data preparation
    run_on_main(
        librispeech_prepare.prepare_librispeech,
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

    # Download the vocabulary file for librispeech
    librispeech_prepare.download_librispeech_vocab_text(
        destination=hparams["vocab_file"]
    )

    # Create the lexicon.txt for k2
    run_on_main(
        sbk2.lexicon.prepare_char_lexicon,
        kwargs={
            "lang_dir": hparams["lang_dir"],
            "vocab_files": [hparams["vocab_file"]],
            "extra_csv_files": [hparams["output_folder"] + "/train.csv"]
            if not hparams["skip_prep"]
            else [],
            "add_word_boundary": hparams["add_word_boundary"],
        },
    )

    caching = (
        {"cache": False}
        if "caching" in hparams and hparams["caching"] is False
        else {}
    )

    # Create the lang directory for k2
    run_on_main(
        sbk2.prepare_lang.prepare_lang,
        kwargs={
            "lang_dir": hparams["lang_dir"],
            "sil_prob": hparams["sil_prob"],
            **caching,
        },
    )

    librispeech_prepare.dataprep_lm_training(
        lm_dir=hparams["output_folder"],
        output_arpa=hparams["output_arpa"],
        csv_files=[hparams["train_csv"]],
        external_lm_corpus=[
            os.path.join(hparams["output_folder"], "librispeech-lm-norm.txt")
        ],
        vocab_file=os.path.join(hparams["lang_dir"], "words.txt"),
    )