#!/usr/bin/env/python3
"""
Recipe for training a BPE tokenizer with mTEDx data.
The tokenizer coverts words into sub-word units that can
be used to train a language (LM) or an acoustic model (AM).
When doing a speech recognition experiment you have to make
sure that the acoustic and language models are trained with
the same tokenizer. Otherwise, a token mismatch is introduced
and beamsearch will produce bad results.

To run this recipe, do the following:
> python train.py hyperparams/1K_unigram_subword_bpe.yaml


Authors
 * Abdel Heba 2021
 * Mohamed Anwar 2022
"""

import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Dataset prep (parsing Librispeech)
    # from ..mtedx_prepare import prepare_mtedx  # noqa
    from recipes.mTEDx.mtedx_prepare import prepare_mtedx

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_mtedx,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["data_folder"],
            "langs": hparams["langs"],
        },
    )

    # Train tokenizer
    hparams["tokenizer"]()
