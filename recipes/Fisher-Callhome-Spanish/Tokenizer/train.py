#!/usr/bin/env/python3
"""Recipe for training a BPE tokenizer with librispeech.
The tokenizer coverts words into sub-word units that can
be used to train a language (LM) or an acoustic model (AM).
When doing a speech recognition experiment you have to make
sure that the acoustic and language models are trained with
the same tokenizer. Otherwise, a token mismatch is introduced
and beamsearch will produce bas results when combining AM and LM.

To run this recipe, do the following:
> python train.py hparams/train_bpe_1k.yaml
Authors
 * YAO-FEI, CHENG 2021
"""

import sys

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from fisher_callhome_prepare import prepare_fisher_callhome_spanish

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

    # Data preparation, to be run on only one process.
    if not hparams["skip_prep"]:
        prepare_fisher_callhome_spanish(
            data_folder=hparams["original_data_folder"],
            save_folder=hparams["data_folder"],
            device=hparams["device"],
        )

    # Train tokenizer
    hparams["tokenizer"]()
