#!/usr/bin/env/python3
"""Recipe for training a BPE tokenizer with MuST-C version 1.
The tokenizer coverts words into sub-word units that can
be used to train a language (LM) or an acoustic model (AM).

To run this recipe, do the following:
> python train.py hparams/train_bpe_1k.yaml
Authors
 * YAO-FEI, CHENG 2021
"""

import sys

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from mustc_v1_prepare import prepare_mustc_v1

if __name__ == "__main__":
    # Load hyperparametrs file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation, to be run on only one process.
    prepare_mustc_v1(
        data_folder=hparams["original_data_folder"],
        save_folder=hparams["data_folder"],
        source_font_case=hparams["source_font_case"],
        target_font_case=hparams["target_font_case"],
        is_accented_letters=hparams["is_accented_letters"],
        is_remove_punctuation=hparams["is_remove_punctuation"],
        is_remove_verbal=hparams["is_remove_non_verbal"],
        target_language=hparams["target_language"],
    )

    # Train tokenizer
    hparams["tokenizer"]()
